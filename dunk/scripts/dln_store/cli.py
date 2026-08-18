"""Command-line interface for Dunk's local authoritative store."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from .schema import StoreError, ValidationError, parse_json_file
from .store import LocalStore, resolve_root


def _add_root(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--root",
        help="DLN vault root (overrides DLN_VAULT_ROOT and CLAUDE_PLUGIN_DATA)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dln-store",
        description="Authoritative stdlib-only local learning event store",
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

    command = subparsers.add_parser("commit", help="append events and atomically publish projections")
    _add_root(command)
    command.add_argument("--domain-id", required=True)
    command.add_argument("--expected-revision", required=True, type=int)
    command.add_argument("--request", required=True, type=Path)

    command = subparsers.add_parser("rebuild", help="rebuild all derived projections")
    _add_root(command)
    command.add_argument("--domain-id", required=True)

    command = subparsers.add_parser("validate", help="validate canonical sources and projection drift")
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


def _run(args: argparse.Namespace) -> dict[str, Any]:
    store = LocalStore(resolve_root(args.root))
    if args.command == "init":
        return store.init(args.domain, args.goal)
    if args.command == "list":
        return store.list_domains()
    if args.command == "context":
        return store.context(args.domain_id)
    if args.command == "commit":
        request = parse_json_file(args.request, "commit request")
        if not isinstance(request, dict):
            raise ValidationError("commit request must be a JSON object")
        return store.commit(args.domain_id, args.expected_revision, request)
    if args.command == "rebuild":
        return store.rebuild(args.domain_id)
    if args.command == "validate":
        return store.validate(args.domain_id)
    if args.command == "doctor":
        return store.doctor(
            args.domain_id,
            recover=args.recover,
            break_stale_lock=args.break_stale_lock,
        )
    if args.command == "import-legacy-ks":
        return store.import_legacy(args.domain, args.input)
    raise ValidationError(f"unknown command: {args.command}")


def _emit(value: Any, stream: Any) -> None:
    json.dump(value, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    stream.write("\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = _run(args)
    except StoreError as exc:
        _emit({"error": exc.__class__.__name__, "message": str(exc)}, sys.stderr)
        return exc.exit_code
    except (OSError, RuntimeError) as exc:
        _emit({"error": exc.__class__.__name__, "message": str(exc)}, sys.stderr)
        return 1
    _emit(result, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
