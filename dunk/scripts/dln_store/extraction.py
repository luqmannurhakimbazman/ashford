"""Media verification and bounded syllabus text extraction."""

from __future__ import annotations

import json
import os
import platform
import re
import signal
import subprocess
import sys
import tempfile
import unicodedata
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Callable

from .acquisition import (
    AcquiredSource,
    HttpsSource,
    HttpsTransport,
    LocalFileSource,
    Resolver,
    acquire_https,
    acquire_local,
)
from .schema import SyllabusIntakeError, sha256_bytes, validate_prepared_document

PYPDF_VERSION = "6.14.2"
MAX_TEXT_BYTES = 8 * 1024 * 1024
MAX_HTML_TOKENS = 200_000
MAX_HTML_DEPTH = 256
WORKER_TIMEOUT = 30.0
WORKER_RESULT_LIMIT = 16 * 1024 * 1024
BLOCK_TAGS = {
    "address", "article", "aside", "blockquote", "br", "dd", "div", "dl", "dt", "fieldset",
    "figcaption", "figure", "footer", "form", "h1", "h2", "h3", "h4", "h5", "h6", "header",
    "hr", "li", "main", "nav", "ol", "p", "pre", "section", "table", "tbody", "td", "tfoot",
    "th", "thead", "tr", "ul",
}
SUPPRESSED_TAGS = {"script", "style", "template"}
VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}


@dataclass(frozen=True)
class PreparedSource:
    """Validated store-generated preparation ready for atomic installation."""

    raw_bytes: bytes
    prepared_document: dict[str, object]
    acquisition: dict[str, object]
    extraction: dict[str, object]


def _error(code: str, message: str, *, phase: str = "extraction") -> SyllabusIntakeError:
    return SyllabusIntakeError(code, message, phase=phase)


def _normalize_pdf_text(text: str) -> str:
    return unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))


def _content_type_base(value: str | None) -> str | None:
    if value is None:
        return None
    return value.split(";", 1)[0].strip().lower()


def _verify_media(acquired: AcquiredSource, media_type: str) -> str:
    if media_type not in {"application/pdf", "text/html"}:
        raise _error("media_mismatch", "media type must be application/pdf or text/html", phase="media")
    header_type = _content_type_base(acquired.content_type)
    if acquired.acquisition.get("kind") == "https" and header_type is None:
        raise _error("media_mismatch", "HTTPS source omitted Content-Type", phase="media")
    if header_type is not None and header_type != media_type:
        raise _error("media_mismatch", "declared and HTTPS media types disagree", phase="media")
    body = acquired.body
    if media_type == "application/pdf":
        if not body.startswith(b"%PDF-"):
            raise _error("media_mismatch", "PDF source does not have PDF magic", phase="media")
        return "utf-8"
    if body.startswith(b"%PDF-") or b"\x00" in body[:4096]:
        raise _error("media_mismatch", "HTML source appears to be binary or PDF", phase="media")
    charset = "utf-8"
    if acquired.content_type and ";" in acquired.content_type:
        match = re.search(r"charset\s*=\s*[\"']?([^;\s\"']+)", acquired.content_type, re.I)
        if match:
            charset = match.group(1).lower()
    if charset in {"utf8", "utf-8-sig"}:
        charset = "utf-8"
    if charset not in {"utf-8", "us-ascii", "ascii"}:
        raise _error("unsupported_charset", "HTML charset is not UTF-8 or US-ASCII", phase="media")
    try:
        preview = body.decode("utf-8-sig" if charset == "utf-8" else "ascii")[:4096]
    except UnicodeDecodeError as exc:
        raise _error("unsupported_charset", "HTML bytes do not match the declared charset", phase="media") from exc
    meta = re.search(r"<meta\s+[^>]*charset\s*=\s*[\"']?([^\s\"'/>;]+)", preview, re.I)
    if meta:
        meta_charset = meta.group(1).lower().replace("_", "-")
        if meta_charset not in {"utf-8", "utf8", "us-ascii", "ascii"}:
            raise _error("unsupported_charset", "HTML meta charset is not supported", phase="media")
        normalized_meta = "utf-8" if meta_charset in {"utf-8", "utf8"} else "ascii"
        normalized_header = "utf-8" if charset == "utf-8" else "ascii"
        if acquired.content_type and normalized_meta != normalized_header:
            raise _error("unsupported_charset", "HTML header and meta charsets disagree", phase="media")
    lead = preview.lstrip().lower()
    if not any(marker in lead[:4096] for marker in ("<!doctype html", "<html", "<head", "<body")):
        raise _error("media_mismatch", "HTML source lacks recognizable document markup", phase="media")
    return "utf-8-sig" if charset == "utf-8" else "ascii"


class _VisibleHtmlParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.open_tags: list[str] = []
        self.tokens = 0
        self.bytes_seen = 0

    def _token(self) -> None:
        self.tokens += 1
        if self.tokens > MAX_HTML_TOKENS:
            raise _error("parse_error", "HTML token limit exceeded")

    def _append(self, value: str) -> None:
        if not value:
            return
        self.bytes_seen += len(value.encode("utf-8"))
        if self.bytes_seen > MAX_TEXT_BYTES * 2:
            raise _error("text_limit", "normalized syllabus text exceeds 8 MiB")
        self.parts.append(value)

    def _suppressed(self) -> bool:
        return any(tag in SUPPRESSED_TAGS for tag in self.open_tags)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._token()
        if tag not in VOID_TAGS:
            self.open_tags.append(tag)
            if len(self.open_tags) > MAX_HTML_DEPTH:
                raise _error("parse_error", "HTML nesting limit exceeded")
        if tag not in SUPPRESSED_TAGS and not self._suppressed() and tag in BLOCK_TAGS:
            self._append("\n")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._token()
        if tag not in SUPPRESSED_TAGS and not self._suppressed() and tag in BLOCK_TAGS:
            self._append("\n")

    def handle_endtag(self, tag: str) -> None:
        self._token()
        if tag in VOID_TAGS or not self.open_tags or self.open_tags[-1] != tag:
            return
        was_suppressed = self._suppressed()
        self.open_tags.pop()
        if not was_suppressed and tag in BLOCK_TAGS:
            self._append("\n")

    def handle_decl(self, decl: str) -> None:
        self._token()

    def handle_data(self, data: str) -> None:
        self._token()
        if not self._suppressed():
            self._append(data)

    def handle_comment(self, data: str) -> None:
        self._token()


def _normalize_html(parts: list[str]) -> str:
    text = "".join(parts).replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[\t\f\v ]+", " ", line).strip() for line in text.split("\n")]
    output: list[str] = []
    blank = True
    for line in lines:
        if line:
            output.append(line)
            blank = False
        elif not blank and output:
            output.append("")
            blank = True
    while output and output[-1] == "":
        output.pop()
    return unicodedata.normalize("NFC", "\n".join(output))


def extract_html(body: bytes, encoding: str) -> tuple[dict[str, object], dict[str, object]]:
    try:
        text = body.decode(encoding)
    except UnicodeDecodeError as exc:
        raise _error("unsupported_charset", "HTML bytes could not be decoded", phase="media") from exc
    parser = _VisibleHtmlParser()
    try:
        parser.feed(text)
        parser.close()
    except SyllabusIntakeError:
        raise
    except Exception as exc:
        raise _error("parse_error", "HTML could not be parsed") from exc
    normalized = _normalize_html(parser.parts)
    size = len(normalized.encode("utf-8"))
    if size > MAX_TEXT_BYTES:
        raise _error("text_limit", "normalized syllabus text exceeds 8 MiB")
    if not normalized.strip():
        raise _error("no_text", "HTML contains no visible text")
    document = {
        "prepared_schema_version": 1,
        "media_type": "text/html",
        "normalization": {"policy_id": "html-visible-text-nfc-lf-v1", "unicode": "NFC", "line_endings": "LF"},
        "units": [{"unit_id": "document:1", "kind": "document", "label": "HTML document", "text": normalized, "text_sha256": sha256_bytes(normalized.encode("utf-8"))}],
    }
    extraction = {
        "kind": "stdlib_html",
        "policy_id": "syllabus-extraction-v1",
        "producer": {"trust": "store_invoked", "name": "html.parser", "version": platform.python_version()},
        "warnings": [],
        "provenance": {
            "normalization_policy_id": "html-visible-text-nfc-lf-v1",
            "engine_options": {"convert_charrefs": True, "subresources": False},
            "unit_count": 1,
            "text_byte_size": size,
            "python_version": platform.python_version(),
            "platform_system": platform.system(),
            "platform_machine": platform.machine(),
        },
    }
    validate_prepared_document(document)
    return document, extraction


def _kill_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=1)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def extract_pdf(
    body: bytes,
    *,
    popen: Callable[..., subprocess.Popen[bytes]] = subprocess.Popen,
) -> tuple[dict[str, object], dict[str, object]]:
    worker = Path(__file__).with_name("pdf_worker.py").resolve()
    with tempfile.TemporaryDirectory(prefix="dln-pdf-") as temporary:
        directory = Path(temporary)
        os.chmod(directory, 0o700)
        source = directory / "source.pdf"
        result = directory / "result.json"
        descriptor = os.open(source, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            view = memoryview(body)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise _error("resource_limit", "PDF worker input could not be written")
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        environment = {"PYTHONUTF8": "1", "LC_ALL": "C", "TMPDIR": temporary}
        try:
            process = popen(
                [os.path.abspath(sys.executable), "-I", str(worker), str(source), str(result), PYPDF_VERSION],
                cwd=temporary,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                shell=False,
                start_new_session=True,
            )
        except OSError as exc:
            raise _error("extractor_unavailable", "PDF worker could not be started") from exc
        try:
            try:
                process.wait(timeout=WORKER_TIMEOUT)
            except subprocess.TimeoutExpired as exc:
                _kill_group(process)
                raise _error("extraction_timeout", "PDF extraction timed out") from exc
            if process.returncode != 0:
                code = "resource_limit" if process.returncode < 0 else "worker_protocol_error"
                raise _error(code, "PDF worker terminated without a valid result")
        finally:
            if process.poll() is None:
                _kill_group(process)
        try:
            if result.stat().st_size > WORKER_RESULT_LIMIT:
                raise _error("resource_limit", "PDF worker result exceeded its limit")
            payload = json.loads(result.read_text(encoding="utf-8"))
        except SyllabusIntakeError:
            raise
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise _error("worker_protocol_error", "PDF worker returned an invalid result") from exc
    if not isinstance(payload, dict) or payload.get("protocol_version") != 1:
        raise _error("worker_protocol_error", "PDF worker protocol version is invalid")
    if payload.get("ok") is not True:
        code = payload.get("code")
        allowed = {"extractor_unavailable", "extractor_version_mismatch", "encrypted", "page_limit", "text_limit", "parse_error", "resource_limit"}
        if code not in allowed:
            code = "worker_protocol_error"
        raise _error(str(code), "PDF text-layer extraction failed")
    pages = payload.get("pages")
    if payload.get("engine") != "pypdf" or payload.get("version") != PYPDF_VERSION or not isinstance(pages, list) or not all(isinstance(page, str) for page in pages):
        raise _error("worker_protocol_error", "PDF worker result fields are invalid")
    units: list[dict[str, object]] = []
    total = 0
    nonblank = False
    for index, page in enumerate(pages, 1):
        normalized = _normalize_pdf_text(page)
        total += len(normalized.encode("utf-8"))
        if total > MAX_TEXT_BYTES:
            raise _error("text_limit", "normalized syllabus text exceeds 8 MiB")
        nonblank = nonblank or bool(normalized.strip())
        units.append({"unit_id": f"page:{index}", "kind": "page", "label": f"Page {index}", "text": normalized, "text_sha256": sha256_bytes(normalized.encode("utf-8"))})
    if not units or not nonblank:
        raise _error("no_text", "PDF contains no extractable text layer")
    document = {
        "prepared_schema_version": 1,
        "media_type": "application/pdf",
        "normalization": {"policy_id": "pypdf-plain-nfc-lf-v1", "unicode": "NFC", "line_endings": "LF"},
        "units": units,
    }
    extraction = {
        "kind": "pypdf_worker",
        "policy_id": "syllabus-extraction-v1",
        "producer": {"trust": "store_invoked", "name": "pypdf", "version": PYPDF_VERSION},
        "warnings": [],
        "provenance": {
            "normalization_policy_id": "pypdf-plain-nfc-lf-v1",
            "engine_options": payload.get("options"),
            "unit_count": len(units),
            "text_byte_size": total,
            "python_version": platform.python_version(),
            "platform_system": platform.system(),
            "platform_machine": platform.machine(),
            "worker_policy_id": "pypdf-resource-worker-v1",
        },
    }
    validate_prepared_document(document)
    return document, extraction


class PreparationService:
    """Compose bounded acquisition, media agreement, and deterministic extraction."""

    def __init__(
        self,
        *,
        resolver: Resolver | None = None,
        transport: HttpsTransport | None = None,
        monotonic: Callable[[], float] | None = None,
    ) -> None:
        self.resolver = resolver
        self.transport = transport
        self.monotonic = monotonic

    def prepare(self, source: LocalFileSource | HttpsSource, media_type: str) -> PreparedSource:
        if isinstance(source, LocalFileSource):
            acquired = acquire_local(source)
        elif isinstance(source, HttpsSource):
            kwargs: dict[str, object] = {"resolver": self.resolver, "transport": self.transport}
            if self.monotonic is not None:
                kwargs["monotonic"] = self.monotonic
            acquired = acquire_https(source, **kwargs)  # type: ignore[arg-type]
        else:
            raise _error("invalid_source_options", "unsupported syllabus source", phase="acquisition")
        encoding = _verify_media(acquired, media_type)
        if media_type == "application/pdf":
            document, extraction = extract_pdf(acquired.body)
        else:
            document, extraction = extract_html(acquired.body, encoding)
        acquisition = dict(acquired.acquisition)
        provenance = dict(acquisition["provenance"])
        provenance["declared_media_type"] = media_type
        acquisition["provenance"] = provenance
        return PreparedSource(acquired.body, document, acquisition, extraction)
