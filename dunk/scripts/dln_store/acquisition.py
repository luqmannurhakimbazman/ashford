"""Bounded local and explicit-HTTPS syllabus acquisition."""

from __future__ import annotations

import ipaddress
import multiprocessing
import os
import re
import socket
import ssl
import stat
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol, Sequence
from urllib.parse import quote, unquote, urljoin, urlsplit, urlunsplit

from .schema import SyllabusIntakeError, sha256_bytes

MAX_SOURCE_BYTES = 16 * 1024 * 1024
MAX_HEADERS = 100
MAX_HEADER_BYTES = 64 * 1024
MAX_HEADER_VALUE_BYTES = 8 * 1024
MAX_DNS_ANSWERS = 64
MAX_REDIRECTS = 3
CONNECT_TIMEOUT = 5.0
READ_TIMEOUT = 10.0
TOTAL_TIMEOUT = 30.0
CHUNK_SIZE = 64 * 1024
USER_AGENT = "Dunk-Syllabus-Intake/1"


@dataclass(frozen=True)
class LocalFileSource:
    """A local path acquired by descriptor, without following a final symlink."""

    path: Path


@dataclass(frozen=True)
class HttpsSource:
    """An HTTPS source with explicit network and redirect consent."""

    url: str
    network_consent: bool = False
    allow_redirects: bool = False


@dataclass(frozen=True)
class ResolvedAddress:
    """One resolver answer."""

    family: int
    address: str


@dataclass(frozen=True)
class HttpsRequest:
    """A direct-IP request retaining the verified hostname for TLS and Host."""

    url: str
    hostname: str
    selected_address: str
    path: str
    headers: tuple[tuple[str, str], ...]
    connect_timeout: float
    read_timeout: float
    total_timeout: float = TOTAL_TIMEOUT


class HttpsResponse(Protocol):
    """Minimal bounded streaming response interface used by scripted tests."""

    status: int
    headers: Sequence[tuple[str, str]]
    connected_peer: str

    def read(self, size: int) -> bytes:
        """Return at most ``size`` further body bytes, or empty bytes at end of stream."""
        ...

    def close(self) -> None:
        """Release the underlying connection without draining the remaining body."""
        ...


class Resolver(Protocol):
    """Injected hostname resolver with an explicit wall deadline."""

    def resolve(self, hostname: str, port: int, timeout: float) -> Sequence[ResolvedAddress]:
        """Return every resolver answer for one hostname within the wall deadline."""
        ...


class HttpsTransport(Protocol):
    """Injected direct-IP HTTPS transport."""

    def open(self, request: HttpsRequest) -> HttpsResponse:
        """Open one bounded response for an already validated direct-address request."""
        ...


def _resolve_worker(hostname: str, port: int, connection: object) -> None:
    try:
        answers = socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
        if len(answers) > MAX_DNS_ANSWERS:
            payload = (False, [], "answer_limit")
        else:
            payload = (True, [(family, sockaddr[0]) for family, _, _, _, sockaddr in answers], None)
    except Exception:
        payload = (False, [], "resolution_failed")
    try:
        getattr(connection, "send")(payload)
    except Exception:
        pass
    finally:
        getattr(connection, "close")()


class SystemResolver:
    """Production resolver isolated in a killable child with a wall deadline."""

    def resolve(self, hostname: str, port: int, timeout: float) -> Sequence[ResolvedAddress]:
        """Resolve one hostname in a spawned child and reject empty or oversized answers."""
        parent: object | None = None
        child: object | None = None
        process: object | None = None
        try:
            context = multiprocessing.get_context("spawn")
            parent, child = context.Pipe(duplex=False)
            process = context.Process(
                target=_resolve_worker, args=(hostname, port, child), daemon=True
            )
            process.start()
        except Exception as exc:
            for endpoint in (parent, child):
                if endpoint is not None:
                    try:
                        getattr(endpoint, "close")()
                    except Exception:
                        pass
            raise SyllabusIntakeError(
                "dns_resolution_failed",
                "HTTPS hostname resolution could not be started",
                phase="acquisition",
            ) from exc
        try:
            getattr(child, "close")()
            if not getattr(parent, "poll")(max(0.001, timeout)):
                getattr(process, "terminate")()
                getattr(process, "join")(timeout=1)
                raise SyllabusIntakeError(
                    "dns_resolution_failed",
                    "HTTPS hostname resolution timed out",
                    phase="acquisition",
                )
            ok, answers, failure = getattr(parent, "recv")()
            getattr(process, "join")(timeout=1)
            if not ok:
                if failure == "answer_limit":
                    raise SyllabusIntakeError(
                        "unsafe_dns",
                        "HTTPS hostname returned too many addresses",
                        phase="acquisition",
                    )
                raise SyllabusIntakeError(
                    "dns_resolution_failed",
                    "HTTPS hostname could not be resolved",
                    phase="acquisition",
                )
            if not isinstance(answers, list):
                raise TypeError("resolver answers must be a list")
            try:
                return [ResolvedAddress(family, address) for family, address in answers]
            except (TypeError, ValueError) as exc:
                raise SyllabusIntakeError(
                    "dns_resolution_failed",
                    "HTTPS hostname resolver returned an invalid result",
                    phase="acquisition",
                ) from exc
        except SyllabusIntakeError:
            raise
        except Exception as exc:
            raise SyllabusIntakeError(
                "dns_resolution_failed",
                "HTTPS hostname resolver returned an invalid result",
                phase="acquisition",
            ) from exc
        finally:
            try:
                getattr(parent, "close")()
            except Exception:
                pass
            try:
                if getattr(process, "is_alive")():
                    getattr(process, "kill")()
                    getattr(process, "join")()
            except Exception:
                pass


class _DirectResponse:
    def __init__(
        self,
        sock: ssl.SSLSocket,
        status: int,
        headers: list[tuple[str, str]],
        peer: str,
        buffered: bytes,
        deadline: float,
        read_timeout: float,
    ) -> None:
        self._sock = sock
        self.status = status
        self.headers = headers
        self.connected_peer = peer
        self._buffered = buffered
        self._deadline = deadline
        self._read_timeout = read_timeout

    def read(self, size: int) -> bytes:
        if self._buffered:
            result, self._buffered = self._buffered[:size], self._buffered[size:]
            return result
        remaining = self._deadline - time.monotonic()
        if remaining <= 0:
            raise _error("total_timeout", "HTTPS acquisition exceeded its total deadline")
        self._sock.settimeout(min(self._read_timeout, remaining))
        return self._sock.recv(size)

    def close(self) -> None:
        self._sock.close()


class DirectHttpsTransport:
    """Production transport that never consults proxies or re-resolves the host."""

    def __init__(
        self,
        *,
        socket_factory: object = socket.create_connection,
        ssl_context_factory: object = ssl.create_default_context,
    ) -> None:
        """Store the socket and TLS context factories, which tests replace with scripted doubles."""
        self._socket_factory = socket_factory
        self._ssl_context_factory = ssl_context_factory

    def open(self, request: HttpsRequest) -> HttpsResponse:
        """Connect to the pre-validated address, complete TLS/SNI, and return a bounded response."""
        started = time.monotonic()
        deadline = started + request.total_timeout
        raw = self._socket_factory((request.selected_address, 443), timeout=request.connect_timeout)  # type: ignore[operator]
        tls: ssl.SSLSocket | None = None
        try:
            context = self._ssl_context_factory()  # type: ignore[operator]
            tls = context.wrap_socket(raw, server_hostname=request.hostname)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise _error("total_timeout", "HTTPS acquisition exceeded its total deadline")
            tls.settimeout(min(request.read_timeout, remaining))
            request_lines = [f"GET {request.path} HTTP/1.1"]
            request_lines.extend(f"{name}: {value}" for name, value in request.headers)
            request_lines.extend(("", ""))
            tls.sendall("\r\n".join(request_lines).encode("ascii"))
            header_bytes = bytearray()
            while b"\r\n\r\n" not in header_bytes:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise _error("total_timeout", "HTTPS acquisition exceeded its total deadline")
                tls.settimeout(min(request.read_timeout, remaining))
                try:
                    chunk = tls.recv(4096)
                except TimeoutError as exc:
                    raise _error("read_timeout", "HTTPS response headers timed out") from exc
                if not chunk:
                    raise _error("http_status", "HTTPS response ended before headers")
                header_bytes.extend(chunk)
                if len(header_bytes) > MAX_HEADER_BYTES + 4:
                    raise _error("header_limit", "HTTPS response headers exceed bounds")
            raw_headers, buffered = bytes(header_bytes).split(b"\r\n\r\n", 1)
            lines = raw_headers.split(b"\r\n")
            if not lines or len(lines[0]) > MAX_HEADER_VALUE_BYTES:
                raise _error("http_status", "HTTPS status line is invalid")
            try:
                version, status_text, _ = lines[0].decode("ascii").split(" ", 2)
                status = int(status_text)
            except (UnicodeError, ValueError) as exc:
                raise _error("http_status", "HTTPS status line is invalid") from exc
            if version not in {"HTTP/1.0", "HTTP/1.1"} or not 100 <= status <= 599:
                raise _error("http_status", "HTTPS status line is invalid")
            headers: list[tuple[str, str]] = []
            for line in lines[1:]:
                if (
                    len(line) > MAX_HEADER_VALUE_BYTES
                    or line[:1] in {b" ", b"\t"}
                    or b":" not in line
                ):
                    raise _error("header_limit", "HTTPS response contains an invalid header")
                name, value = line.split(b":", 1)
                try:
                    headers.append((name.decode("ascii"), value.decode("latin-1").strip()))
                except UnicodeError as exc:
                    raise _error(
                        "header_limit", "HTTPS response contains an invalid header"
                    ) from exc
                if len(headers) > MAX_HEADERS:
                    raise _error("header_limit", "HTTPS response has too many headers")
            peer = tls.getpeername()[0]
            return _DirectResponse(
                tls, status, headers, peer, buffered, deadline, request.read_timeout
            )
        except Exception:
            if tls is not None:
                tls.close()
            else:
                raw.close()
            raise


@dataclass(frozen=True)
class AcquiredSource:
    """Bounded bytes plus store-generated acquisition provenance."""

    body: bytes
    acquisition: dict[str, object]
    content_type: str | None


def _error(code: str, message: str) -> SyllabusIntakeError:
    return SyllabusIntakeError(code, message, phase="acquisition")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _normalize_ip(value: str) -> str:
    try:
        address = ipaddress.ip_address(value)
    except ValueError as exc:
        raise _error("unsafe_dns", "resolver returned an invalid address") from exc
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped:
        address = address.ipv4_mapped
    return address.compressed


def _canonical_ip(value: str) -> str:
    normalized = _normalize_ip(value)
    if not ipaddress.ip_address(normalized).is_global:
        raise _error("unsafe_dns", "HTTPS hostname resolved to a forbidden address")
    return normalized


def _validated_answers(answers: Sequence[ResolvedAddress]) -> list[str]:
    if not answers:
        raise _error("dns_resolution_failed", "HTTPS hostname returned no addresses")
    if len(answers) > MAX_DNS_ANSWERS:
        raise _error("unsafe_dns", "HTTPS hostname returned too many addresses")
    result: set[str] = set()
    for item in answers:
        if (
            not isinstance(item, ResolvedAddress)
            or item.family not in {socket.AF_INET, socket.AF_INET6}
            or not isinstance(item.address, str)
        ):
            raise _error("unsafe_dns", "resolver returned an invalid address family")
        normalized = _canonical_ip(item.address)
        if ipaddress.ip_address(normalized).version != (4 if item.family == socket.AF_INET else 6):
            raise _error("unsafe_dns", "resolver address family did not match its address")
        result.add(normalized)
    return sorted(
        result,
        key=lambda value: (ipaddress.ip_address(value).version, ipaddress.ip_address(value).packed),
    )


def _canonical_url(value: str, *, redirect: bool = False) -> tuple[str, str, str]:
    code = "unsafe_redirect" if redirect else "unsafe_url"
    if (
        not isinstance(value, str)
        or not value
        or any(
            character.isspace() or ord(character) < 32 or character == "\\" for character in value
        )
        or re.search(r"%(?![0-9A-Fa-f]{2})", value)
    ):
        raise _error(code, "HTTPS URL has forbidden syntax")
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as exc:
        raise _error(code, "HTTPS URL has an invalid authority") from exc
    if parsed.scheme.lower() != "https" or not parsed.netloc or parsed.username is not None:
        raise _error(code, "only credential-free HTTPS URLs are allowed")
    if parsed.password is not None or parsed.query or parsed.fragment or port not in {None, 443}:
        raise _error(code, "HTTPS URL contains a forbidden component")
    hostname = parsed.hostname
    if not hostname or "%" in hostname:
        raise _error(code, "HTTPS URL hostname is invalid")
    try:
        try:
            host = ipaddress.ip_address(hostname).compressed
        except ValueError:
            host = hostname.encode("idna").decode("ascii").lower()
            if len(host) > 253 or any(not label or len(label) > 63 for label in host.split(".")):
                raise ValueError("invalid DNS name")
    except (UnicodeError, ValueError) as exc:
        raise _error(code, "HTTPS URL hostname is invalid") from exc
    path = parsed.path or "/"
    try:
        decoded = unquote(path, errors="strict")
    except UnicodeError as exc:
        raise _error(code, "HTTPS URL path encoding is invalid") from exc
    if any(ord(character) < 32 or character == "\\" for character in decoded):
        raise _error(code, "HTTPS URL path is unsafe")
    path = quote(decoded, safe="/%:@!$&'()*+,;=-._~")
    authority = f"[{host}]" if ":" in host else host
    return urlunsplit(("https", authority, path, "", "")), host, path


def _header_map(headers: Sequence[tuple[str, str]]) -> dict[str, list[str]]:
    if len(headers) > MAX_HEADERS:
        raise _error("header_limit", "HTTPS response has too many headers")
    total = 0
    mapped: dict[str, list[str]] = {}
    for name, value in headers:
        if (
            any(ord(character) < 32 and character != "\t" for character in name + value)
            or "\r" in name + value
            or "\n" in name + value
        ):
            raise _error("header_limit", "HTTPS response contains an invalid header")
        try:
            encoded = (name + ":" + value).encode("latin-1")
        except UnicodeEncodeError as exc:
            raise _error("header_limit", "HTTPS response contains an invalid header") from exc
        total += len(encoded)
        if len(value.encode("latin-1")) > MAX_HEADER_VALUE_BYTES or total > MAX_HEADER_BYTES:
            raise _error("header_limit", "HTTPS response headers exceed bounds")
        mapped.setdefault(name.lower(), []).append(value.strip())
    return mapped


def acquire_local(source: LocalFileSource, *, acquired_at: str | None = None) -> AcquiredSource:
    """Read one regular local file, stopping at the first byte above 16 MiB."""
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        if stat.S_ISLNK(os.stat(source.path, follow_symlinks=False).st_mode):
            raise _error("unsafe_local_source", "local source must not be a symlink")
        descriptor = os.open(source.path, flags)
    except OSError as exc:
        code = (
            "unsafe_local_source"
            if getattr(exc, "errno", None) in {40, 62}
            else "source_unreadable"
        )
        raise _error(code, "local source could not be opened safely") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise _error("unsafe_local_source", "local source must be a regular file")
        chunks: list[bytes] = []
        size = 0
        try:
            while True:
                chunk = os.read(descriptor, min(CHUNK_SIZE, MAX_SOURCE_BYTES + 1 - size))
                if not chunk:
                    break
                chunks.append(chunk)
                size += len(chunk)
                if size > MAX_SOURCE_BYTES:
                    raise _error("source_too_large", "syllabus source exceeds 16 MiB")
        except OSError as exc:
            raise _error("source_unreadable", "local source could not be read safely") from exc
        if size == 0:
            raise _error("media_mismatch", "syllabus source is empty")
        body = b"".join(chunks)
    finally:
        try:
            os.close(descriptor)
        except OSError:
            pass
    digest = sha256_bytes(body)
    display = source.path.name
    return AcquiredSource(
        body,
        {
            "kind": "local_file",
            "policy_id": "syllabus-acquisition-v1",
            "trust": "store_invoked",
            "declared_source": display,
            "provenance": {
                "acquired_at": acquired_at or _utc_now(),
                "byte_count": len(body),
                "source_sha256": digest,
                "descriptor_policy": "nofollow-regular-v1",
            },
        },
        None,
    )


def acquire_https(
    source: HttpsSource,
    *,
    resolver: Resolver | None = None,
    transport: HttpsTransport | None = None,
    monotonic: callable = time.monotonic,
    acquired_at: str | None = None,
) -> AcquiredSource:
    """Fetch one explicitly consented HTTPS resource through a revalidated direct-IP path."""
    if not source.network_consent:
        raise _error("network_consent_required", "HTTPS acquisition requires explicit consent")
    resolver = resolver or SystemResolver()
    transport = transport or DirectHttpsTransport()
    initial, _, _ = _canonical_url(source.url)
    current = initial
    started = monotonic()
    hops: list[dict[str, object]] = []
    redirects = 0
    while True:
        if monotonic() - started > TOTAL_TIMEOUT:
            raise _error("total_timeout", "HTTPS acquisition exceeded its total deadline")
        current, hostname, path = _canonical_url(current, redirect=bool(hops))
        try:
            remaining = TOTAL_TIMEOUT - (monotonic() - started)
            if remaining <= 0:
                raise _error("total_timeout", "HTTPS acquisition exceeded its total deadline")
            answers = _validated_answers(
                resolver.resolve(hostname, 443, min(CONNECT_TIMEOUT, remaining))
            )
        except SyllabusIntakeError:
            raise
        except Exception as exc:
            raise _error("dns_resolution_failed", "HTTPS hostname could not be resolved") from exc
        selected = answers[0]
        remaining = TOTAL_TIMEOUT - (monotonic() - started)
        if remaining <= 0:
            raise _error("total_timeout", "HTTPS acquisition exceeded its total deadline")
        host_header = f"[{hostname}]" if ":" in hostname else hostname
        headers = (
            ("Host", host_header),
            ("User-Agent", USER_AGENT),
            ("Accept", "application/pdf, text/html"),
            ("Accept-Encoding", "identity"),
            ("Connection", "close"),
        )
        request = HttpsRequest(
            current,
            hostname,
            selected,
            path,
            headers,
            min(CONNECT_TIMEOUT, remaining),
            min(READ_TIMEOUT, remaining),
            remaining,
        )
        response: HttpsResponse | None = None
        try:
            try:
                response = transport.open(request)
            except TimeoutError as exc:
                raise _error("connect_timeout", "HTTPS connection timed out") from exc
            except ssl.SSLError as exc:
                raise _error("tls_error", "HTTPS TLS verification failed") from exc
            except OSError as exc:
                raise _error("tls_error", "HTTPS connection failed") from exc
            if monotonic() - started > TOTAL_TIMEOUT:
                raise _error("total_timeout", "HTTPS acquisition exceeded its total deadline")
            try:
                peer = _normalize_ip(response.connected_peer)
            except SyllabusIntakeError as exc:
                raise _error("peer_mismatch", "connected HTTPS peer address was invalid") from exc
            if peer != selected:
                raise _error(
                    "peer_mismatch", "connected HTTPS peer did not match the selected address"
                )
            mapped = _header_map(response.headers)
            transfer_encodings = mapped.get("transfer-encoding", [])
            if transfer_encodings:
                raise _error(
                    "unsupported_transfer_encoding", "HTTPS Transfer-Encoding is not supported"
                )
            hop: dict[str, object] = {
                "request_url": current,
                "resolved_addresses": answers,
                "selected_address": selected,
                "connected_peer": peer,
                "status": response.status,
            }
            if response.status in {301, 302, 303, 307, 308}:
                if not source.allow_redirects:
                    raise _error("redirect_not_allowed", "HTTPS redirect requires explicit consent")
                locations = mapped.get("location", [])
                if len(locations) != 1:
                    raise _error("unsafe_redirect", "HTTPS redirect must contain one Location")
                if redirects >= MAX_REDIRECTS:
                    raise _error("redirect_limit", "HTTPS redirect limit exceeded")
                target = urljoin(current, locations[0])
                target, _, _ = _canonical_url(target, redirect=True)
                hop["redirect_url"] = target
                hops.append(hop)
                redirects += 1
                current = target
                continue
            if response.status != 200:
                raise _error("http_status", "HTTPS source returned a non-success status")
            encodings = mapped.get("content-encoding", [])
            if len(encodings) > 1 or (encodings and encodings[0].lower() != "identity"):
                raise _error(
                    "unsupported_content_encoding", "compressed HTTPS bodies are not allowed"
                )
            lengths = mapped.get("content-length", [])
            if len(lengths) > 1:
                raise _error("invalid_content_length", "HTTPS Content-Length is ambiguous")
            if lengths:
                try:
                    declared_length = int(lengths[0], 10)
                except ValueError as exc:
                    raise _error(
                        "invalid_content_length", "HTTPS Content-Length is invalid"
                    ) from exc
                if declared_length < 0:
                    raise _error("invalid_content_length", "HTTPS Content-Length is invalid")
                if declared_length > MAX_SOURCE_BYTES:
                    raise _error("source_too_large", "syllabus source exceeds 16 MiB")
            chunks: list[bytes] = []
            size = 0
            while True:
                if monotonic() - started > TOTAL_TIMEOUT:
                    raise _error("total_timeout", "HTTPS acquisition exceeded its total deadline")
                try:
                    chunk = response.read(min(CHUNK_SIZE, MAX_SOURCE_BYTES + 1 - size))
                except TimeoutError as exc:
                    raise _error("read_timeout", "HTTPS body read timed out") from exc
                except OSError as exc:
                    raise _error("read_timeout", "HTTPS body read failed") from exc
                if not chunk:
                    break
                chunks.append(chunk)
                size += len(chunk)
                if size > MAX_SOURCE_BYTES:
                    raise _error("source_too_large", "syllabus source exceeds 16 MiB")
            body = b"".join(chunks)
            if lengths and len(body) != declared_length:
                raise _error(
                    "invalid_content_length", "HTTPS body length disagrees with Content-Length"
                )
            if not body:
                raise _error("media_mismatch", "syllabus source is empty")
            hop["status"] = response.status
            hops.append(hop)
            content_types = mapped.get("content-type", [])
            if len(content_types) > 1:
                raise _error("media_mismatch", "HTTPS Content-Type is ambiguous")
            content_type = content_types[0] if content_types else None
            digest = sha256_bytes(body)
            return AcquiredSource(
                body,
                {
                    "kind": "https",
                    "policy_id": "syllabus-acquisition-v1",
                    "trust": "store_invoked",
                    "declared_source": initial,
                    "provenance": {
                        "acquired_at": acquired_at or _utc_now(),
                        "byte_count": len(body),
                        "source_sha256": digest,
                        "initial_url": initial,
                        "final_url": current,
                        "redirects_followed": redirects,
                        "content_type": content_type or "",
                        "content_encoding": "identity",
                        "hops": hops,
                    },
                },
                content_type,
            )
        except SyllabusIntakeError:
            raise
        except Exception as exc:
            raise _error("tls_error", "HTTPS transport failed") from exc
        finally:
            if response is not None:
                try:
                    response.close()
                except Exception:
                    pass
