"""Isolated pinned-pypdf text-layer extraction worker.

The parent invokes this file directly with fixed arguments. It writes one bounded JSON
result and never writes to the domain store.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
from pathlib import Path

PROTOCOL_VERSION = 1
MAX_PAGES = 500
MAX_RESULT_BYTES = 16 * 1024 * 1024


def _limits() -> None:
    limits = (
        (resource.RLIMIT_CPU, 20, 20),
        (resource.RLIMIT_FSIZE, MAX_RESULT_BYTES, MAX_RESULT_BYTES),
        (resource.RLIMIT_NOFILE, 32, 32),
        (resource.RLIMIT_CORE, 0, 0),
    )
    for kind, soft, hard in limits:
        resource.setrlimit(kind, (soft, hard))
    try:
        resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))
    except ValueError:
        # Darwin exposes address-space limits but rejects finite values in some
        # sandboxed runtimes. Linux (the CI/production contract) must install it;
        # Darwin still enforces CPU, file-size, descriptor, core, and wall limits.
        if sys.platform != "darwin":
            raise


def _write(path: Path, payload: dict[str, object]) -> None:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    if len(encoded) > MAX_RESULT_BYTES:
        payload = {"protocol_version": PROTOCOL_VERSION, "ok": False, "code": "text_limit"}
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("PDF worker output write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main() -> int:
    """Extract bounded page text for one fixed-argument invocation and return its exit code."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("expected_version")
    args = parser.parse_args()
    try:
        _limits()
    except (OSError, ValueError):
        _write(
            args.output,
            {"protocol_version": PROTOCOL_VERSION, "ok": False, "code": "resource_limit"},
        )
        return 0
    try:
        import pypdf
    except Exception:
        _write(
            args.output,
            {"protocol_version": PROTOCOL_VERSION, "ok": False, "code": "extractor_unavailable"},
        )
        return 0
    if pypdf.__version__ != args.expected_version:
        _write(
            args.output,
            {
                "protocol_version": PROTOCOL_VERSION,
                "ok": False,
                "code": "extractor_version_mismatch",
            },
        )
        return 0
    try:
        reader = pypdf.PdfReader(str(args.input), strict=False)
        if reader.is_encrypted:
            _write(
                args.output,
                {"protocol_version": PROTOCOL_VERSION, "ok": False, "code": "encrypted"},
            )
            return 0
        if len(reader.pages) > MAX_PAGES:
            _write(
                args.output,
                {"protocol_version": PROTOCOL_VERSION, "ok": False, "code": "page_limit"},
            )
            return 0
        pages: list[str] = []
        approximate = 0
        for page in reader.pages:
            text = page.extract_text(extraction_mode="plain") or ""
            approximate += len(text.encode("utf-8"))
            if approximate > MAX_RESULT_BYTES - 65536:
                _write(
                    args.output,
                    {"protocol_version": PROTOCOL_VERSION, "ok": False, "code": "text_limit"},
                )
                return 0
            pages.append(text)
    except MemoryError:
        _write(
            args.output,
            {"protocol_version": PROTOCOL_VERSION, "ok": False, "code": "resource_limit"},
        )
        return 0
    except Exception:
        _write(
            args.output, {"protocol_version": PROTOCOL_VERSION, "ok": False, "code": "parse_error"}
        )
        return 0
    _write(
        args.output,
        {
            "protocol_version": PROTOCOL_VERSION,
            "ok": True,
            "engine": "pypdf",
            "version": pypdf.__version__,
            "options": {"strict": False, "extraction_mode": "plain"},
            "pages": pages,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
