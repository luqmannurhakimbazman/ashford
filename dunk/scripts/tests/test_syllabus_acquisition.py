"""Portable syllabus acquisition, extraction, security, and offline lifecycle tests."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import dln_store.acquisition as acquisition_module
from dln_store.acquisition import (
    MAX_DNS_ANSWERS,
    MAX_SOURCE_BYTES,
    DirectHttpsTransport,
    HttpsRequest,
    HttpsSource,
    LocalFileSource,
    ResolvedAddress,
    SystemResolver,
    acquire_https,
    acquire_local,
)
from dln_store.cli import MAX_REQUEST_BYTES, _read_bounded_object, build_parser, main
import dln_store.extraction as extraction_module
import dln_store.pdf_worker as pdf_worker
from dln_store.extraction import PYPDF_VERSION, PreparationService, extract_html, extract_pdf
from dln_store.schema import SyllabusIntakeError, ValidationError
from dln_store.store import LocalStore

FIXTURES = Path(__file__).parent / "fixtures" / "syllabus"
GENERIC = FIXTURES / "generic"
GLOBAL_IP = "93.184.216.34"


@dataclass
class ScriptedResponse:
    status: int = 200
    headers: tuple[tuple[str, str], ...] = (("Content-Type", "application/pdf"),)
    connected_peer: str = GLOBAL_IP
    body: bytes = b""
    read_error: Exception | None = None

    def __post_init__(self) -> None:
        self.offset = 0
        self.closed = False

    def read(self, size: int) -> bytes:
        if self.read_error is not None:
            raise self.read_error
        chunk = self.body[self.offset : self.offset + size]
        self.offset += len(chunk)
        return chunk

    def close(self) -> None:
        self.closed = True


class ScriptedResolver:
    def __init__(self, answers: list[list[str]] | None = None) -> None:
        self.answers = answers or [[GLOBAL_IP]]
        self.calls: list[tuple[str, int]] = []
        self.timeouts: list[float] = []

    def resolve(self, hostname: str, port: int, timeout: float) -> list[ResolvedAddress]:
        self.calls.append((hostname, port))
        self.timeouts.append(timeout)
        values = self.answers[min(len(self.calls) - 1, len(self.answers) - 1)]
        return [ResolvedAddress(socket.AF_INET6 if ":" in value else socket.AF_INET, value) for value in values]


class ScriptedTransport:
    def __init__(self, responses: list[ScriptedResponse] | None = None, error: Exception | None = None) -> None:
        self.responses = responses or []
        self.error = error
        self.requests: list[HttpsRequest] = []

    def open(self, request: HttpsRequest) -> ScriptedResponse:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return self.responses[len(self.requests) - 1]


def initialized(tmp_path: Path, *, service: PreparationService | None = None) -> tuple[LocalStore, str, Path]:
    root = tmp_path / "vault"
    store = LocalStore(root, _preparation_service=service)
    result = store.init("Generic Systems", "Learn portable systems")
    domain_id = result["domain_id"]
    return store, domain_id, root / "domains" / domain_id


def tree(path: Path) -> dict[str, bytes | str]:
    result: dict[str, bytes | str] = {}
    for item in sorted(path.rglob("*")):
        relative = item.relative_to(path).as_posix()
        result[relative] = f"symlink:{item.readlink()}" if item.is_symlink() else (item.read_bytes() if item.is_file() else "dir")
    return result


def located_proposal(document: dict[str, object], needle: str) -> dict[str, object]:
    unit = next(unit for unit in document["units"] if needle in unit["text"])
    start = unit["text"].index(needle)
    return {
        "predicate": "coverage.topic",
        "label": "Grounded topic",
        "semantic_roles": ["planning_topic"],
        "value_type": "text",
        "status": "specified",
        "value": needle,
        "locators": [{"unit_id": unit["unit_id"], "start_char": start, "end_char": start + len(needle), "quote": needle}],
    }


def complete_lifecycle(store: LocalStore, domain_id: str, source: LocalFileSource | HttpsSource, media: str, needle: str) -> tuple[dict[str, object], dict[str, object]]:
    prepared = store.prepare_syllabus(
        domain_id,
        0,
        source=source,
        media_type=media,
        role="authoritative",
        display_name="generic-syllabus" + (".pdf" if media == "application/pdf" else ".html"),
        occurred_at="2026-08-19T00:00:00Z",
    )
    content = store.syllabus_content(domain_id, prepared["source_event_id"])
    proposal = store.propose_syllabus(
        domain_id,
        1,
        prepared_event_id=prepared["source_event_id"],
        occurred_at="2026-08-19T00:01:00Z",
        producer={"trust": "external_unverified", "name": "test-proposer", "version": "1"},
        proposals=[located_proposal(content["prepared_document"], needle)],
    )
    store.decide_syllabus(
        domain_id,
        2,
        proposal_event_id=proposal["proposal_event_id"],
        occurred_at="2026-08-19T00:02:00Z",
        accepted_proposal_ids=proposal["proposal_ids"],
        deferred_proposal_ids=[],
        rejected_proposal_ids=[],
        corrections=[],
    )
    return prepared, content


def test_pinned_pdf_golden_extraction_and_generic_ambiguity() -> None:
    for source_name, golden_name in (
        ("two-page-syllabus.pdf", "expected-two-page-extraction.json"),
        ("ambiguous-columns.pdf", "expected-ambiguous-extraction.json"),
    ):
        document, extraction = extract_pdf((GENERIC / source_name).read_bytes())
        golden = json.loads((GENERIC / golden_name).read_text())
        assert PYPDF_VERSION == golden["pypdf_version"] == "6.14.2"
        assert extraction["provenance"]["engine_options"] == golden["engine_options"]
        assert document == golden["prepared_document"]
    ambiguous, _ = extract_pdf((GENERIC / "ambiguous-columns.pdf").read_bytes())
    assert "Week 4        Week 5" in ambiguous["units"][0]["text"]
    assert set(ambiguous) == {"prepared_schema_version", "media_type", "normalization", "units"}


def test_st5201x_is_only_an_adversarial_input_to_the_same_extractor() -> None:
    document, extraction = extract_pdf((FIXTURES / "st5201x" / "syllabus2026.pdf").read_bytes())
    assert extraction["producer"] == {"trust": "store_invoked", "name": "pypdf", "version": PYPDF_VERSION}
    text = document["units"][0]["text"]
    assert "Week7" in text and "Week13" in text
    assert "assertion" not in json.dumps(document).lower()


def test_local_pdf_and_html_complete_lifecycles_and_rebuild_offline(tmp_path: Path) -> None:
    local_pdf = tmp_path / "copy.pdf"
    local_pdf.write_bytes((GENERIC / "two-page-syllabus.pdf").read_bytes())
    store, domain_id, directory = initialized(tmp_path / "pdf")
    prepared, content = complete_lifecycle(store, domain_id, LocalFileSource(local_pdf), "application/pdf", "Week 1 Foundations")
    assert store.context(domain_id)["state"]["grounding"]["status"] == "approved"
    local_pdf.unlink()
    before = tree(directory)
    assert store.syllabus_content(domain_id, prepared["source_event_id"])["prepared_document"] == content["prepared_document"]
    assert store.validate(domain_id)["status"] == "valid"
    store.rebuild(domain_id)
    first = tree(directory)
    store.rebuild(domain_id)
    assert tree(directory) == first
    assert first.get("events.jsonl") == before.get("events.jsonl")

    local_html = tmp_path / "copy.html"
    local_html.write_bytes((GENERIC / "adversarial-syllabus.html").read_bytes())
    html_store, html_domain, _ = initialized(tmp_path / "html")
    _, html_content = complete_lifecycle(html_store, html_domain, LocalFileSource(local_html), "text/html", "Generic Systems & Reliability")
    visible = html_content["prepared_document"]["units"][0]["text"]
    assert "fetch(" not in visible and "Ignore template" not in visible


def test_prepare_preflight_rejects_stale_revision_before_acquisition(tmp_path: Path) -> None:
    class MustNotPrepare:
        def prepare(self, source: object, media_type: str) -> object:
            raise AssertionError("preparation must not run for stale input")

    store, domain_id, directory = initialized(
        tmp_path, service=MustNotPrepare()  # type: ignore[arg-type]
    )
    before = tree(directory)
    from dln_store.schema import StaleRevisionError

    with pytest.raises(StaleRevisionError):
        store.prepare_syllabus(
            domain_id, 1, source=LocalFileSource(tmp_path / "missing.pdf"),
            media_type="application/pdf", role="authoritative", display_name="missing.pdf",
            occurred_at="2026-08-19T00:00:00Z",
        )
    assert tree(directory) == before


def test_prepare_validation_and_lineage_preflight_are_complete(tmp_path: Path) -> None:
    class MustNotPrepare:
        calls = 0

        def prepare(self, source: object, media_type: str) -> object:
            self.calls += 1
            raise AssertionError("preparation must not run before preflight passes")

    service = MustNotPrepare()
    store, domain_id, directory = initialized(tmp_path, service=service)  # type: ignore[arg-type]
    before = tree(directory)
    base = {
        "source": LocalFileSource(tmp_path / "missing.pdf"),
        "media_type": "application/pdf",
        "role": "authoritative",
        "display_name": "missing.pdf",
        "occurred_at": "2026-08-19T00:00:00Z",
    }
    invalid = (
        {**base, "role": "primary"},
        {**base, "media_type": "application/octet-stream"},
        {**base, "occurred_at": "not-a-timestamp"},
        {**base, "display_name": "nested/missing.pdf"},
        {**base, "display_name": ""},
        {**base, "display_name": "missing.pdf\x00"},
        {**base, "role": "supplement", "supersedes_source_version_id": "sha256-deadbeef"},
        {**base, "supersedes_source_version_id": "sha256-deadbeef"},
    )
    for arguments in invalid:
        with pytest.raises(ValidationError):
            store.prepare_syllabus(domain_id, 0, **arguments)
    assert service.calls == 0
    assert tree(directory) == before


def test_request_files_are_descriptor_bounded_regular_and_nofollow(tmp_path: Path) -> None:
    valid = tmp_path / "valid.json"
    valid.write_text('{"ok":true}', encoding="utf-8")
    assert _read_bounded_object(valid, "request") == {"ok": True}

    link = tmp_path / "request-link.json"
    link.symlink_to(valid)
    with pytest.raises(ValidationError, match="opened safely"):
        _read_bounded_object(link, "request")

    directory = tmp_path / "request-directory"
    directory.mkdir()
    with pytest.raises(ValidationError, match="regular file"):
        _read_bounded_object(directory, "request")

    replacement = tmp_path / "replacement.json"
    replacement.write_text('{"ok":false}', encoding="utf-8")
    original_open = os.open
    swapped = False

    def swap_after_open(path: object, flags: int) -> int:
        nonlocal swapped
        descriptor = original_open(path, flags)
        if not swapped:
            swapped = True
            os.replace(replacement, valid)
        return descriptor

    from dln_store import cli as cli_module
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(cli_module.os, "open", swap_after_open)
        assert _read_bounded_object(valid, "request") == {"ok": True}
    assert json.loads(valid.read_text()) == {"ok": False}

    oversized = tmp_path / "oversized.json"
    oversized.write_bytes(b"x" * (MAX_REQUEST_BYTES + 1))
    with pytest.raises(ValidationError, match="exceeds 2 MiB"):
        _read_bounded_object(oversized, "request")


def test_precommit_local_failures_leave_domain_byte_identical(tmp_path: Path) -> None:
    store, domain_id, directory = initialized(tmp_path)
    wrong = tmp_path / "wrong.pdf"
    wrong.write_text("<!doctype html><html><body>not pdf</body></html>")
    before = tree(directory)
    with pytest.raises(SyllabusIntakeError) as caught:
        store.prepare_syllabus(
            domain_id, 0, source=LocalFileSource(wrong), media_type="application/pdf", role="authoritative",
            display_name="wrong.pdf", occurred_at="2026-08-19T00:00:00Z"
        )
    assert caught.value.code == "media_mismatch"
    assert tree(directory) == before

    link = tmp_path / "link.pdf"
    link.symlink_to(wrong)
    with pytest.raises(SyllabusIntakeError) as caught:
        store.prepare_syllabus(
            domain_id, 0, source=LocalFileSource(link), media_type="application/pdf", role="authoritative",
            display_name="link.pdf", occurred_at="2026-08-19T00:00:00Z"
        )
    assert caught.value.code in {"unsafe_local_source", "source_unreadable"}
    assert tree(directory) == before


def test_precommit_https_failure_has_stable_code_and_preserves_domain(tmp_path: Path) -> None:
    service = PreparationService(
        resolver=ScriptedResolver(),
        transport=ScriptedTransport(error=RuntimeError("injected transport failure")),
    )
    store, domain_id, directory = initialized(tmp_path, service=service)
    before = tree(directory)
    with pytest.raises(SyllabusIntakeError) as caught:
        store.prepare_syllabus(
            domain_id,
            0,
            source=HttpsSource("https://example.com/syllabus.pdf", network_consent=True),
            media_type="application/pdf",
            role="authoritative",
            display_name="syllabus.pdf",
            occurred_at="2026-08-19T00:00:00Z",
        )
    assert caught.value.code == "tls_error"
    assert tree(directory) == before


def test_local_reader_accepts_exact_limit_and_stops_above_it(tmp_path: Path) -> None:
    exact = tmp_path / "exact.bin"
    exact.write_bytes(b"x" * MAX_SOURCE_BYTES)
    assert len(acquire_local(LocalFileSource(exact)).body) == MAX_SOURCE_BYTES
    oversized = tmp_path / "oversized.pdf"
    oversized.write_bytes(b"%PDF-" + b"x" * MAX_SOURCE_BYTES)
    store, domain_id, directory = initialized(tmp_path / "domain")
    before = tree(directory)
    with pytest.raises(SyllabusIntakeError) as caught:
        store.prepare_syllabus(
            domain_id, 0, source=LocalFileSource(oversized), media_type="application/pdf", role="authoritative",
            display_name="oversized.pdf", occurred_at="2026-08-19T00:00:00Z"
        )
    assert caught.value.code == "source_too_large"
    assert tree(directory) == before


@pytest.mark.parametrize("url", [
    "http://example.com/a.pdf", "https://user@example.com/a.pdf", "https://example.com/a.pdf?q=secret",
    "https://example.com/a.pdf#part", "https://example.com:444/a.pdf", "https://example.com\\a.pdf",
    "https://example.com/%ZZ.pdf",
])
def test_https_rejects_forbidden_url_forms(url: str) -> None:
    with pytest.raises(SyllabusIntakeError) as caught:
        acquire_https(HttpsSource(url, network_consent=True), resolver=ScriptedResolver(), transport=ScriptedTransport())
    assert caught.value.code == "unsafe_url"


def test_cli_exposes_generic_lifecycle_and_stable_intake_envelope(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    choices = build_parser()._subparsers._group_actions[0].choices
    assert {"prepare-syllabus", "propose-syllabus", "decide-syllabus", "syllabus-content"} <= set(choices)
    assert "ingest-syllabus" not in choices and "approve-syllabus" not in choices
    store, domain_id, _ = initialized(tmp_path)
    code = main([
        "prepare-syllabus", "--root", str(store.root), "--domain-id", domain_id,
        "--expected-revision", "0", "--url", "https://example.com/a.pdf",
        "--media-type", "application/pdf", "--role", "authoritative",
        "--display-name", "a.pdf", "--occurred-at", "2026-08-19T00:00:00Z",
    ])
    error = json.loads(capsys.readouterr().err)
    assert code == 2
    assert error == {
        "code": "network_consent_required", "error": "SyllabusIntakeError",
        "message": "HTTPS acquisition requires --network-consent", "phase": "acquisition",
    }


def test_https_requires_explicit_consent() -> None:
    with pytest.raises(SyllabusIntakeError) as caught:
        acquire_https(HttpsSource("https://example.com/a.pdf"), resolver=ScriptedResolver(), transport=ScriptedTransport())
    assert caught.value.code == "network_consent_required"


def test_system_resolver_deadline_terminates_and_reaps_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: dict[str, object] = {}

    class Parent:
        def poll(self, timeout: float) -> bool:
            observed["timeout"] = timeout
            return False

        def close(self) -> None:
            observed["parent_closed"] = True

    class Child:
        def close(self) -> None:
            observed["child_closed"] = True

    class Process:
        def start(self) -> None:
            observed["started"] = True

        def terminate(self) -> None:
            observed["terminated"] = True

        def join(self, timeout: float | None = None) -> None:
            observed.setdefault("joins", []).append(timeout)  # type: ignore[union-attr]

        def is_alive(self) -> bool:
            return False

        def kill(self) -> None:
            raise AssertionError("terminated resolver must not need a kill")

    class Context:
        def Pipe(self, *, duplex: bool) -> tuple[Parent, Child]:
            assert duplex is False
            return Parent(), Child()

        def Process(self, **kwargs: object) -> Process:
            assert kwargs["target"] is acquisition_module._resolve_worker
            assert kwargs["daemon"] is True
            return Process()

    monkeypatch.setattr(acquisition_module.multiprocessing, "get_context", lambda method: Context())
    with pytest.raises(SyllabusIntakeError) as caught:
        SystemResolver().resolve("example.com", 443, 0.25)
    assert caught.value.code == "dns_resolution_failed"
    assert observed == {
        "started": True,
        "child_closed": True,
        "timeout": 0.25,
        "terminated": True,
        "joins": [1],
        "parent_closed": True,
    }


def test_https_propagates_bounded_resolver_and_request_deadlines() -> None:
    resolver = ScriptedResolver()
    transport = ScriptedTransport([
        ScriptedResponse(body=(GENERIC / "two-page-syllabus.pdf").read_bytes())
    ])
    acquire_https(
        HttpsSource("https://example.com/a.pdf", network_consent=True),
        resolver=resolver,
        transport=transport,
    )
    assert len(resolver.timeouts) == 1
    assert 0 < resolver.timeouts[0] <= 5.0
    request = transport.requests[0]
    assert 0 < request.connect_timeout <= 5.0
    assert 0 < request.read_timeout <= 10.0
    assert 0 < request.total_timeout <= 30.0


@pytest.mark.parametrize("answers", [[], ["127.0.0.1"], ["10.0.0.1"], ["169.254.169.254"], ["::1"], ["fc00::1"], [GLOBAL_IP, "10.0.0.1"], [GLOBAL_IP] * (MAX_DNS_ANSWERS + 1)])
def test_https_rejects_empty_non_global_and_mixed_dns(answers: list[str]) -> None:
    with pytest.raises(SyllabusIntakeError) as caught:
        acquire_https(
            HttpsSource("https://example.com/a.pdf", network_consent=True),
            resolver=ScriptedResolver([answers]),
            transport=ScriptedTransport(),
        )
    assert caught.value.code == ("dns_resolution_failed" if not answers else "unsafe_dns")


def test_production_transport_dials_selected_ip_uses_sni_and_bounds_headers() -> None:
    class Raw:
        def close(self) -> None:
            self.closed = True

    class TLS:
        def __init__(self, response: bytes) -> None:
            self.response = response
            self.sent = b""
            self.closed = False

        def settimeout(self, timeout: float) -> None:
            self.timeout = timeout

        def sendall(self, data: bytes) -> None:
            self.sent += data

        def recv(self, size: int) -> bytes:
            result, self.response = self.response[:size], self.response[size:]
            return result

        def getpeername(self) -> tuple[str, int]:
            return GLOBAL_IP, 443

        def close(self) -> None:
            self.closed = True

    raw = Raw()
    tls = TLS(b"HTTP/1.1 200 OK\r\nContent-Type: application/pdf\r\n\r\n%PDF-body")
    calls: list[object] = []

    def socket_factory(address: tuple[str, int], *, timeout: float) -> Raw:
        calls.append((address, timeout))
        return raw

    class Context:
        def wrap_socket(self, supplied: Raw, *, server_hostname: str) -> TLS:
            calls.append((supplied, server_hostname))
            return tls

    transport = DirectHttpsTransport(socket_factory=socket_factory, ssl_context_factory=Context)
    request = HttpsRequest(
        "https://example.com/a.pdf", "example.com", GLOBAL_IP, "/a.pdf",
        (("Host", "example.com"), ("Connection", "close")), 5.0, 10.0,
    )
    response = transport.open(request)
    assert calls == [((GLOBAL_IP, 443), 5.0), (raw, "example.com")]
    assert b"GET /a.pdf HTTP/1.1\r\nHost: example.com" in tls.sent
    assert response.read(9) == b"%PDF-body"
    response.close()

    oversized_tls = TLS(b"HTTP/1.1 200 OK\r\nX: " + b"a" * 66000 + b"\r\n\r\n")
    transport = DirectHttpsTransport(
        socket_factory=lambda address, timeout: Raw(),
        ssl_context_factory=lambda: type("C", (), {"wrap_socket": lambda self, raw, server_hostname: oversized_tls})(),
    )
    with pytest.raises(SyllabusIntakeError) as caught:
        transport.open(request)
    assert caught.value.code == "header_limit"

    class HeaderTimeoutTLS(TLS):
        def recv(self, size: int) -> bytes:
            raise TimeoutError("injected header timeout")

    timed_out_tls = HeaderTimeoutTLS(b"")
    transport = DirectHttpsTransport(
        socket_factory=lambda address, timeout: Raw(),
        ssl_context_factory=lambda: type("C", (), {"wrap_socket": lambda self, raw, server_hostname: timed_out_tls})(),
    )
    with pytest.raises(SyllabusIntakeError) as caught:
        transport.open(request)
    assert caught.value.code == "read_timeout"
    assert 0 < timed_out_tls.timeout <= request.read_timeout


def test_https_ipv6_host_and_non_ascii_path_are_canonicalized() -> None:
    address = "2606:4700:4700::1111"
    response = ScriptedResponse(
        200,
        (("Content-Type", "application/pdf"),),
        connected_peer=address,
        body=(GENERIC / "two-page-syllabus.pdf").read_bytes(),
    )
    transport = ScriptedTransport([response])
    acquire_https(
        HttpsSource("https://[2606:4700:4700::1111]/café.pdf", network_consent=True),
        resolver=ScriptedResolver([[address]]),
        transport=transport,
    )
    request = transport.requests[0]
    assert ("Host", "[2606:4700:4700::1111]") in request.headers
    assert request.path == "/caf%C3%A9.pdf"


def test_https_peer_redirect_and_redirect_revalidation() -> None:
    first = ScriptedResponse(302, (("Location", "https://cdn.example.com/final.pdf"),), body=b"")
    second = ScriptedResponse(200, (("Content-Type", "application/pdf"),), body=(GENERIC / "two-page-syllabus.pdf").read_bytes())
    resolver = ScriptedResolver([[GLOBAL_IP], ["93.184.216.35"]])
    second.connected_peer = "93.184.216.35"
    transport = ScriptedTransport([first, second])
    acquired = acquire_https(
        HttpsSource("https://example.com/start.pdf", network_consent=True, allow_redirects=True),
        resolver=resolver,
        transport=transport,
    )
    assert acquired.acquisition["provenance"]["redirects_followed"] == 1
    assert resolver.calls == [("example.com", 443), ("cdn.example.com", 443)]
    assert [request.selected_address for request in transport.requests] == [GLOBAL_IP, "93.184.216.35"]


def test_https_redirect_without_consent_and_overflow() -> None:
    redirect = lambda: ScriptedResponse(302, (("Location", "https://example.com/next.pdf"),), body=b"")
    with pytest.raises(SyllabusIntakeError) as caught:
        acquire_https(
            HttpsSource("https://example.com/a.pdf", network_consent=True),
            resolver=ScriptedResolver(), transport=ScriptedTransport([redirect()]),
        )
    assert caught.value.code == "redirect_not_allowed"
    with pytest.raises(SyllabusIntakeError) as caught:
        acquire_https(
            HttpsSource("https://example.com/a.pdf", network_consent=True, allow_redirects=True),
            resolver=ScriptedResolver(), transport=ScriptedTransport([redirect(), redirect(), redirect(), redirect()]),
        )
    assert caught.value.code == "redirect_limit"


@pytest.mark.parametrize(("response", "transport_error", "code"), [
    (ScriptedResponse(200, (("Content-Type", "application/pdf"),), connected_peer="93.184.216.35"), None, "peer_mismatch"),
    (ScriptedResponse(200, tuple((f"X-{i}", "v") for i in range(101))), None, "header_limit"),
    (ScriptedResponse(200, (("Content-Encoding", "gzip"),)), None, "unsupported_content_encoding"),
    (ScriptedResponse(200, (("Transfer-Encoding", "chunked"),)), None, "unsupported_transfer_encoding"),
    (ScriptedResponse(200, (("Transfer-Encoding", "identity"), ("Content-Length", "0"))), None, "unsupported_transfer_encoding"),
    (ScriptedResponse(200, (("Content-Length", str(MAX_SOURCE_BYTES + 1)),)), None, "source_too_large"),
    (ScriptedResponse(200, (("Content-Length", "1"), ("Content-Length", "1"))), None, "invalid_content_length"),
    (ScriptedResponse(503, ()), None, "http_status"),
    (ScriptedResponse(200, (("Content-Type", "application/pdf"),), read_error=TimeoutError()), None, "read_timeout"),
    (None, TimeoutError(), "connect_timeout"),
    (None, RuntimeError("unstable transport detail"), "tls_error"),
])
def test_https_stable_peer_header_body_encoding_status_and_timeout_codes(response: ScriptedResponse | None, transport_error: Exception | None, code: str) -> None:
    responses = [] if response is None else [response]
    with pytest.raises(SyllabusIntakeError) as caught:
        acquire_https(
            HttpsSource("https://example.com/a.pdf", network_consent=True),
            resolver=ScriptedResolver(), transport=ScriptedTransport(responses, transport_error),
        )
    assert caught.value.code == code


def test_https_streamed_body_limit_and_total_timeout() -> None:
    response = ScriptedResponse(200, (("Content-Type", "application/pdf"),), body=b"x" * (MAX_SOURCE_BYTES + 1))
    with pytest.raises(SyllabusIntakeError) as caught:
        acquire_https(HttpsSource("https://example.com/a.pdf", network_consent=True), resolver=ScriptedResolver(), transport=ScriptedTransport([response]))
    assert caught.value.code == "source_too_large"
    ticks = iter((0.0, 31.0))
    with pytest.raises(SyllabusIntakeError) as caught:
        acquire_https(HttpsSource("https://example.com/a.pdf", network_consent=True), resolver=ScriptedResolver(), transport=ScriptedTransport(), monotonic=lambda: next(ticks))
    assert caught.value.code == "total_timeout"


def test_https_transfer_encoding_errors_are_stable_and_bounded() -> None:
    for headers in (
        (("Transfer-Encoding", "chunked"),),
        (("Transfer-Encoding", "chunked"), ("Content-Length", "12")),
    ):
        with pytest.raises(SyllabusIntakeError) as caught:
            acquire_https(
                HttpsSource("https://example.com/a.pdf", network_consent=True),
                resolver=ScriptedResolver(),
                transport=ScriptedTransport([ScriptedResponse(200, headers)]),
            )
        assert caught.value.code == "unsupported_transfer_encoding"
        assert str(caught.value) == "HTTPS Transfer-Encoding is not supported"


def test_html_suppression_uses_matching_stack_and_self_closing_semantics() -> None:
    mismatched = b"<!doctype html><html><body>Visible<template><div>SECRET</template>LEAK</div>ALSO LEAK</body></html>"
    document, _ = extract_html(mismatched, "utf-8")
    text = document["units"][0]["text"]
    assert text == "Visible"
    assert "SECRET" not in text and "LEAK" not in text

    self_closing = b"<!doctype html><html><body><template/><script/><style/>Visible</body></html>"
    document, _ = extract_html(self_closing, "utf-8")
    assert document["units"][0]["text"] == "Visible"


def test_scripted_https_html_completes_lifecycle_without_subresources(tmp_path: Path) -> None:
    body = (GENERIC / "adversarial-syllabus.html").read_bytes()
    transport = ScriptedTransport([ScriptedResponse(200, (("Content-Type", "text/html; charset=utf-8"),), body=body)])
    store, domain_id, _ = initialized(tmp_path, service=PreparationService(resolver=ScriptedResolver(), transport=transport))
    _, content = complete_lifecycle(
        store, domain_id, HttpsSource("https://example.com/syllabus.html", network_consent=True),
        "text/html", "Generic Systems & Reliability",
    )
    assert "Ignore template" not in content["prepared_document"]["units"][0]["text"]
    assert len(transport.requests) == 1


def test_ambiguous_fixture_cannot_be_accepted(tmp_path: Path) -> None:
    source_file = tmp_path / "ambiguous.pdf"
    source_file.write_bytes((GENERIC / "ambiguous-columns.pdf").read_bytes())
    store, domain_id, _ = initialized(tmp_path / "domain")
    prepared = store.prepare_syllabus(
        domain_id, 0, source=LocalFileSource(source_file), media_type="application/pdf", role="authoritative",
        display_name="ambiguous.pdf", occurred_at="2026-08-19T00:00:00Z",
    )
    request = json.loads((GENERIC / "ambiguous-proposal-request.json").read_text())
    request["prepared_event_id"] = prepared["source_event_id"]
    proposal = store.propose_syllabus(domain_id, 1, **request)
    with pytest.raises(ValidationError, match="ambiguous"):
        store.decide_syllabus(
            domain_id, 2, proposal_event_id=proposal["proposal_event_id"], occurred_at="2026-08-19T00:02:00Z",
            accepted_proposal_ids=proposal["proposal_ids"], deferred_proposal_ids=[], rejected_proposal_ids=[], corrections=[],
        )


def test_pdf_parent_reaps_timed_out_child_and_handles_short_input_writes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class HungProcess:
        pid = 424242
        returncode: int | None = None

        def wait(self, timeout: float | None = None) -> int:
            raise subprocess.TimeoutExpired("pdf-worker", timeout)

        def poll(self) -> int | None:
            return self.returncode

    process = HungProcess()
    killed: list[int] = []

    def kill_group(candidate: object) -> None:
        assert candidate is process
        killed.append(process.pid)
        process.returncode = -15

    monkeypatch.setattr(extraction_module, "_kill_group", kill_group)
    with pytest.raises(SyllabusIntakeError) as caught:
        extract_pdf(b"%PDF-timeout", popen=lambda *args, **kwargs: process)  # type: ignore[arg-type]
    assert caught.value.code == "extraction_timeout"
    assert killed == [process.pid]

    original_write = os.write
    partial = {"used": False}

    def short_once(descriptor: int, data: object) -> int:
        view = memoryview(data)
        if not partial["used"] and len(view) > 1:
            partial["used"] = True
            count = max(1, len(view) // 2)
            return original_write(descriptor, view[:count])
        return original_write(descriptor, view)

    monkeypatch.setattr(extraction_module.os, "write", short_once)
    document, _ = extract_pdf((GENERIC / "two-page-syllabus.pdf").read_bytes())
    assert partial["used"] is True
    assert len(document["units"]) == 2


def test_pdf_worker_output_handles_short_and_zero_progress_writes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    original_write = os.write
    output = tmp_path / "worker.json"

    def short_write(descriptor: int, data: object) -> int:
        view = memoryview(data)
        count = max(1, len(view) // 2)
        return original_write(descriptor, view[:count])

    monkeypatch.setattr(pdf_worker.os, "write", short_write)
    pdf_worker._write(output, {"protocol_version": 1, "ok": True})
    assert json.loads(output.read_text()) == {"ok": True, "protocol_version": 1}

    monkeypatch.setattr(pdf_worker.os, "write", lambda descriptor, data: 0)
    with pytest.raises(OSError, match="no progress"):
        pdf_worker._write(tmp_path / "zero.json", {"protocol_version": 1, "ok": True})


def test_pdf_worker_reports_no_text_encrypted_and_page_limit(tmp_path: Path) -> None:
    from pypdf import PdfWriter

    blank = PdfWriter()
    blank.add_blank_page(width=72, height=72)
    blank_path = tmp_path / "blank.pdf"
    with blank_path.open("wb") as stream:
        blank.write(stream)
    with pytest.raises(SyllabusIntakeError) as caught:
        extract_pdf(blank_path.read_bytes())
    assert caught.value.code == "no_text"

    encrypted = PdfWriter()
    encrypted.add_blank_page(width=72, height=72)
    encrypted.encrypt("secret", algorithm="RC4-40")
    encrypted_path = tmp_path / "encrypted.pdf"
    with encrypted_path.open("wb") as stream:
        encrypted.write(stream)
    with pytest.raises(SyllabusIntakeError) as caught:
        extract_pdf(encrypted_path.read_bytes())
    assert caught.value.code == "encrypted"

    many = PdfWriter()
    for _ in range(501):
        many.add_blank_page(width=72, height=72)
    many_path = tmp_path / "many.pdf"
    with many_path.open("wb") as stream:
        many.write(stream)
    with pytest.raises(SyllabusIntakeError) as caught:
        extract_pdf(many_path.read_bytes())
    assert caught.value.code == "page_limit"


def test_scripted_https_lifecycle_and_local_https_hash_equivalence(tmp_path: Path) -> None:
    body = (GENERIC / "two-page-syllabus.pdf").read_bytes()
    local = tmp_path / "same.pdf"
    local.write_bytes(body)
    local_store, local_domain, _ = initialized(tmp_path / "local")
    local_prepared, _ = complete_lifecycle(local_store, local_domain, LocalFileSource(local), "application/pdf", "Week 1 Foundations")

    resolver = ScriptedResolver()
    transport = ScriptedTransport([ScriptedResponse(200, (("Content-Type", "application/pdf"),), body=body)])
    service = PreparationService(resolver=resolver, transport=transport)
    https_store, https_domain, _ = initialized(tmp_path / "https", service=service)
    https_prepared, _ = complete_lifecycle(
        https_store, https_domain,
        HttpsSource("https://example.com/generic.pdf", network_consent=True),
        "application/pdf", "Week 1 Foundations",
    )
    assert local_prepared["source_sha256"] == https_prepared["source_sha256"]
    assert local_prepared["prepared_document_sha256"] == https_prepared["prepared_document_sha256"]
    local_event = local_store.syllabus_content(local_domain, local_prepared["source_event_id"])
    https_event = https_store.syllabus_content(https_domain, https_prepared["source_event_id"])
    assert local_event["acquisition"]["kind"] == "local_file"
    assert https_event["acquisition"]["kind"] == "https"
    assert len(transport.requests) == 1
