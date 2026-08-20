"""Command-line interface for Dunk's local authoritative store."""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any, Sequence

from .acquisition import HttpsSource, LocalFileSource
from .schema import StoreError, ValidationError
from .store import LocalStore, resolve_root

MAX_REQUEST_BYTES = 2 * 1024 * 1024


def _add_root(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--root", help="DLN vault root (overrides DLN_VAULT_ROOT and CLAUDE_PLUGIN_DATA)"
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser covering every dln-store subcommand."""
    parser = argparse.ArgumentParser(
        prog="dln-store",
        description="Authoritative local learning store with portable syllabus intake",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    command = subparsers.add_parser("init", help="initialize a learning domain")
    _add_root(command)
    command.add_argument("--domain", required=True)
    command.add_argument("--goal", required=True)

    command = subparsers.add_parser("list", help="list local learning domains")
    _add_root(command)

    command = subparsers.add_parser("context", help="return validated profile and derived context")
    _add_root(command)
    command.add_argument("--domain-id", required=True)

    command = subparsers.add_parser(
        "commit", help="append events and atomically publish projections"
    )
    _add_root(command)
    command.add_argument("--domain-id", required=True)
    command.add_argument("--expected-revision", required=True, type=int)
    command.add_argument("--request", required=True, type=Path)

    command = subparsers.add_parser(
        "prepare-syllabus", help="acquire, extract, and atomically prepare a syllabus"
    )
    _add_root(command)
    command.add_argument("--domain-id", required=True)
    command.add_argument("--expected-revision", required=True, type=int)
    source = command.add_mutually_exclusive_group(required=True)
    source.add_argument("--file", type=Path)
    source.add_argument("--url")
    command.add_argument("--network-consent", action="store_true")
    command.add_argument("--allow-redirects", action="store_true")
    command.add_argument("--media-type", choices=("application/pdf", "text/html"), required=True)
    command.add_argument("--role", choices=("authoritative", "supplement"), required=True)
    command.add_argument("--display-name", required=True)
    command.add_argument("--occurred-at", required=True)
    command.add_argument("--supersedes-source-version-id")

    command = subparsers.add_parser(
        "propose-syllabus", help="seal bounded proposals against prepared content"
    )
    _add_root(command)
    command.add_argument("--domain-id", required=True)
    command.add_argument("--expected-revision", required=True, type=int)
    command.add_argument("--request", required=True, type=Path)

    command = subparsers.add_parser("decide-syllabus", help="record one complete learner decision")
    _add_root(command)
    command.add_argument("--domain-id", required=True)
    command.add_argument("--expected-revision", required=True, type=int)
    command.add_argument("--request", required=True, type=Path)

    command = subparsers.add_parser(
        "syllabus-content", help="return verified canonical prepared content"
    )
    _add_root(command)
    command.add_argument("--domain-id", required=True)
    command.add_argument("--source-event-id", required=True)

    command = subparsers.add_parser("rebuild", help="rebuild all derived projections")
    _add_root(command)
    command.add_argument("--domain-id", required=True)

    command = subparsers.add_parser(
        "validate", help="validate canonical sources and projection drift"
    )
    _add_root(command)
    command.add_argument("--domain-id", required=True)

    command = subparsers.add_parser("doctor", help="diagnose locks and interrupted transactions")
    _add_root(command)
    command.add_argument("--domain-id", required=True)
    command.add_argument("--recover", action="store_true")
    command.add_argument("--break-stale-lock", action="store_true")

    command = subparsers.add_parser(
        "import-legacy-ks", help="import one manually exported marker-delimited legacy KS"
    )
    _add_root(command)
    command.add_argument("--domain", required=True)
    command.add_argument("--input", required=True, type=Path)
    return parser


def _read_bounded_object(path: Path, label: str) -> dict[str, Any]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValidationError(f"{label} could not be opened safely") from exc
    try:
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise ValidationError(f"{label} must be a regular file")
            chunks: list[bytes] = []
            size = 0
            while True:
                chunk = os.read(descriptor, min(65536, MAX_REQUEST_BYTES + 1 - size))
                if not chunk:
                    break
                chunks.append(chunk)
                size += len(chunk)
                if size > MAX_REQUEST_BYTES:
                    raise ValidationError(f"{label} exceeds 2 MiB")
        except OSError as exc:
            raise ValidationError(f"{label} could not be read safely") from exc
    finally:
        try:
            os.close(descriptor)
        except OSError:
            pass
    try:
        request = json.loads(b"".join(chunks).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValidationError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(request, dict):
        raise ValidationError(f"{label} must be a JSON object")
    return request


def _bounded_request(
    path: Path, label: str, required: set[str], optional: set[str] | None = None
) -> dict[str, Any]:
    request = _read_bounded_object(path, label)
    keys = set(request)
    unknown = keys - required - (optional or set())
    missing = required - keys
    if unknown or missing:
        raise ValidationError(f"{label} has unknown or missing keys")
    return request


def _content_view(content: dict[str, Any]) -> dict[str, Any]:
    result = {
        key: value for key, value in content.items() if key not in {"raw_bytes", "prepared_bytes"}
    }
    raw = content.get("raw_bytes")
    if isinstance(raw, bytes):
        source = content.get("source", {})
        result["raw"] = {"available": True, "byte_count": len(raw), "sha256": source.get("sha256")}
    else:
        result["raw"] = {"available": False}
    return result


def _run(args: argparse.Namespace) -> dict[str, Any]:
    store = LocalStore(resolve_root(args.root))
    if args.command == "init":
        return store.init(args.domain, args.goal)
    if args.command == "list":
        return store.list_domains()
    if args.command == "context":
        return store.context(args.domain_id)
    if args.command == "commit":
        return store.commit(
            args.domain_id,
            args.expected_revision,
            _read_bounded_object(args.request, "commit request"),
        )
    if args.command == "prepare-syllabus":
        if args.file is not None:
            if args.network_consent or args.allow_redirects:
                raise ValidationError("network consent and redirects are valid only with --url")
            source = LocalFileSource(args.file)
        else:
            if not args.network_consent:
                from .schema import SyllabusIntakeError

                raise SyllabusIntakeError(
                    "network_consent_required",
                    "HTTPS acquisition requires --network-consent",
                    phase="acquisition",
                )
            source = HttpsSource(
                args.url, network_consent=True, allow_redirects=args.allow_redirects
            )
        return store.prepare_syllabus(
            args.domain_id,
            args.expected_revision,
            source=source,
            media_type=args.media_type,
            role=args.role,
            display_name=args.display_name,
            occurred_at=args.occurred_at,
            supersedes_source_version_id=args.supersedes_source_version_id,
        )
    if args.command == "propose-syllabus":
        request = _bounded_request(
            args.request,
            "syllabus proposal request",
            {"prepared_event_id", "occurred_at", "producer", "proposals"},
        )
        return store.propose_syllabus(args.domain_id, args.expected_revision, **request)
    if args.command == "decide-syllabus":
        required = {
            "proposal_event_id",
            "occurred_at",
            "accepted_proposal_ids",
            "deferred_proposal_ids",
            "rejected_proposal_ids",
            "corrections",
        }
        request = _bounded_request(
            args.request,
            "syllabus decision request",
            required,
            {"actor", "supersedes_decision_event_id"},
        )
        return store.decide_syllabus(args.domain_id, args.expected_revision, **request)
    if args.command == "syllabus-content":
        return _content_view(store.syllabus_content(args.domain_id, args.source_event_id))
    if args.command == "rebuild":
        return store.rebuild(args.domain_id)
    if args.command == "validate":
        return store.validate(args.domain_id)
    if args.command == "doctor":
        return store.doctor(
            args.domain_id, recover=args.recover, break_stale_lock=args.break_stale_lock
        )
    if args.command == "import-legacy-ks":
        return store.import_legacy(args.domain, args.input)
    raise ValidationError(f"unknown command: {args.command}")


def _emit(value: Any, stream: Any) -> None:
    json.dump(value, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    stream.write("\n")


def main(argv: Sequence[str] | None = None) -> int:
    """Run one CLI invocation and return its process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = _run(args)
    except StoreError as exc:
        payload: dict[str, Any] = {
            "error": exc.__class__.__name__,
            "code": exc.code,
            "message": str(exc),
        }
        if hasattr(exc, "phase"):
            payload["phase"] = exc.phase
        _emit(payload, sys.stderr)
        return exc.exit_code
    except (OSError, RuntimeError) as exc:
        _emit(
            {"error": exc.__class__.__name__, "code": "runtime_error", "message": str(exc)},
            sys.stderr,
        )
        return 1
    _emit(result, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
