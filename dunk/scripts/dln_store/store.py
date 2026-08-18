"""Authoritative filesystem store with locking, optimistic revisions, and recovery."""

from __future__ import annotations

import errno
import fcntl
import json
import os
import re
import shutil
import socket
import stat
import tempfile
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterator

from .grounding import reduce_grounding_timeline
from .legacy import legacy_event
from .projector import project_state
from .render import render_all_receipts, render_all_syllabus_receipts, render_dashboard
from .schema import (
    RESERVED_SYLLABUS_EVENT_KINDS,
    LockError,
    RecoveryRequiredError,
    StaleRevisionError,
    StoreError,
    ValidationError,
    build_syllabus_approval_event,
    canonical_json,
    encode_event_lines,
    initial_profile,
    load_profile,
    make_domain_id,
    parse_events_bytes,
    pretty_json,
    sha256_bytes,
    validate_commit_request,
    validate_profile,
)
from .st5201x_syllabus import build_ingestion_event

TXN_NAME = ".dln-transaction"


@dataclass(frozen=True)
class DomainPaths:
    """Canonical filesystem locations for one domain inside a vault root."""

    root: Path
    domain_id: str

    @property
    def directory(self) -> Path:
        """Return the domain directory holding canonical and derived files."""
        return self.root / "domains" / self.domain_id

    @property
    def profile(self) -> Path:
        """Return the user-editable profile path."""
        return self.directory / "profile.yaml"

    @property
    def events(self) -> Path:
        """Return the append-only event log path."""
        return self.directory / "events.jsonl"

    @property
    def state(self) -> Path:
        """Return the generated state projection path."""
        return self.directory / "state.json"

    @property
    def dashboard(self) -> Path:
        """Return the generated dashboard path."""
        return self.directory / "dashboard.md"

    @property
    def sessions(self) -> Path:
        """Return the directory holding generated Session Receipts."""
        return self.directory / "sessions"

    @property
    def syllabus(self) -> Path:
        """Return the directory holding generated Syllabus Intake Receipts."""
        return self.directory / "syllabus"

    @property
    def lock(self) -> Path:
        """Return the advisory lock file guarding writes to this domain."""
        return self.root / ".locks" / f"{self.domain_id}.lock"

    @property
    def transaction(self) -> Path:
        """Return the in-progress transaction directory path."""
        return self.directory / TXN_NAME


def resolve_root(explicit: str | None = None) -> Path:
    """Resolve the vault root from --root, DLN_VAULT_ROOT, then CLAUDE_PLUGIN_DATA."""
    if explicit:
        return Path(explicit).expanduser().resolve()
    configured = os.environ.get("DLN_VAULT_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    plugin_data = os.environ.get("CLAUDE_PLUGIN_DATA")
    if plugin_data:
        return (Path(plugin_data).expanduser() / "dln-vault").resolve()
    raise ValidationError(
        "vault root is not configured; pass --root, set DLN_VAULT_ROOT, "
        "or make CLAUDE_PLUGIN_DATA available"
    )


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError as exc:
        if exc.errno not in {errno.EINVAL, errno.ENOTSUP, errno.EBADF}:
            raise
    finally:
        os.close(fd)


def _write_fsynced(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _write_json_fsynced(path: Path, value: Any) -> None:
    _write_fsynced(path, pretty_json(value))


def _safe_relative(value: str) -> Path:
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise RecoveryRequiredError(f"transaction contains unsafe target path {value!r}")
    return Path(*pure.parts)


def _reject_symlink(path: Path, label: str) -> None:
    if path.is_symlink():
        raise ValidationError(f"{label} must not be a symlink: {path}")


def _reject_target_symlinks(base: Path, relative: Path) -> None:
    current = base
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValidationError(f"generated target path must not contain symlinks: {current}")


def _is_editor_metadata(base: Path, path: Path) -> bool:
    return any(part.startswith(".") for part in path.relative_to(base).parts)


def _hash_file(path: Path) -> str | None:
    try:
        return sha256_bytes(path.read_bytes())
    except FileNotFoundError:
        return None


def _lock_metadata() -> dict[str, Any]:
    return {
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "started_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }


def _read_lock(handle: Any) -> tuple[dict[str, Any] | None, str | None]:
    try:
        handle.seek(0)
        raw = handle.read().decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return None, f"cannot read lock: {exc}"
    if not raw.strip():
        return None, None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, f"invalid lock metadata: {exc.msg}"
    if not isinstance(value, dict):
        return None, "invalid lock metadata: expected object"
    return value, None


def _open_lock(path: Path) -> Any:
    """Open a regular lock file without following a caller-supplied symlink."""
    try:
        if stat.S_ISLNK(os.lstat(path).st_mode):
            raise ValidationError(f"lock file must not be a symlink: {path}")
    except FileNotFoundError:
        pass
    flags = os.O_CREAT | os.O_RDWR
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags, 0o600)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise ValidationError(f"lock file must not be a symlink: {path}") from exc
        raise
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValidationError(f"lock path must be a regular file: {path}")
        return os.fdopen(fd, "r+b", buffering=0)
    except BaseException:
        os.close(fd)
        raise


def _lock_directory(paths: DomainPaths) -> Path:
    directory = paths.lock.parent
    if directory.is_symlink():
        raise ValidationError(f"lock directory must not be a symlink: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    return directory


@contextmanager
def domain_lock(paths: DomainPaths) -> Iterator[None]:
    """Hold a kernel-owned advisory lock; the path is never unlinked."""
    lock_directory = _lock_directory(paths)
    handle = _open_lock(paths.lock)
    acquired = False
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except BlockingIOError as exc:
            existing, error = _read_lock(handle)
            detail = error or json.dumps(existing, sort_keys=True)
            raise LockError(
                f"active writer lock at {paths.lock} (status=active, metadata={detail})"
            ) from exc
        handle.seek(0)
        handle.truncate()
        handle.write(pretty_json(_lock_metadata()))
        os.fsync(handle.fileno())
        _fsync_directory(lock_directory)
        yield
    finally:
        if acquired:
            handle.seek(0)
            handle.truncate()
            os.fsync(handle.fileno())
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _failpoint(name: str) -> None:
    if os.environ.get("DLN_STORE_CRASH_AT") == name:
        os._exit(91)
    if os.environ.get("DLN_STORE_FAIL_AT") == name:
        raise OSError(f"injected failure at {name}")


def _load_journal(paths: DomainPaths) -> dict[str, Any]:
    _reject_symlink(paths.transaction, "transaction directory")
    journal_path = paths.transaction / "journal.json"
    _reject_symlink(journal_path, "transaction journal")
    try:
        value = json.loads(journal_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RecoveryRequiredError(
            f"interrupted transaction at {paths.transaction} has no journal; "
            "preserve it for diagnosis"
        ) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise RecoveryRequiredError(f"cannot read transaction journal: {exc}") from exc
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise RecoveryRequiredError("transaction journal has an unsupported schema")
    preconditions = value.get("source_preconditions")
    if not isinstance(preconditions, dict) or not all(
        isinstance(path, str) and isinstance(digest, str) for path, digest in preconditions.items()
    ):
        raise RecoveryRequiredError("transaction journal has invalid source preconditions")
    for source_path in preconditions:
        _safe_relative(source_path)
    targets = value.get("targets")
    if not isinstance(targets, list) or not targets:
        raise RecoveryRequiredError("transaction journal has no targets")
    for entry in targets:
        if not isinstance(entry, dict):
            raise RecoveryRequiredError("transaction journal target must be an object")
        for field in ("path", "candidate_sha256", "had_original"):
            if field not in entry:
                raise RecoveryRequiredError(f"transaction journal target is missing {field}")
        _safe_relative(entry["path"])
    return value


def _journal_write(paths: DomainPaths, journal: dict[str, Any]) -> None:
    target = paths.transaction / "journal.json"
    temporary = paths.transaction / "journal.json.tmp"
    _write_json_fsynced(temporary, journal)
    os.replace(temporary, target)
    _fsync_directory(paths.transaction)


def _restore_transaction(paths: DomainPaths, journal: dict[str, Any]) -> None:
    _reject_symlink(paths.transaction, "transaction directory")
    backups = paths.transaction / "backups"
    _reject_symlink(backups, "transaction backups directory")

    # Refuse a partial rollback before changing any target when another writer/editor
    # has installed bytes that are neither the recorded original nor our candidate.
    for entry in journal["targets"]:
        relative = _safe_relative(entry["path"])
        _reject_target_symlinks(paths.directory, relative)
        target_hash = _hash_file(paths.directory / relative)
        original_hash = entry.get("backup_sha256") if entry["had_original"] else None
        if target_hash not in {original_hash, entry["candidate_sha256"]}:
            raise RecoveryRequiredError(
                f"cannot restore {relative}: target changed outside this transaction; "
                "preserve the journal for diagnosis"
            )

    for entry in journal["targets"]:
        relative = _safe_relative(entry["path"])
        target = paths.directory / relative
        backup = backups / relative
        if entry["had_original"]:
            expected = entry.get("backup_sha256")
            if not backup.is_file() or _hash_file(backup) != expected:
                raise RecoveryRequiredError(
                    f"cannot restore {relative}: transaction backup is missing or corrupt"
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.parent / f".{target.name}.restore-{uuid.uuid4().hex}"
            shutil.copyfile(backup, temporary)
            _fsync_file(temporary)
            os.replace(temporary, target)
        else:
            try:
                target.unlink()
            except FileNotFoundError:
                pass
        _fsync_directory(target.parent)


def _remove_transaction(paths: DomainPaths) -> None:
    _reject_symlink(paths.transaction, "transaction directory")
    shutil.rmtree(paths.transaction)
    _fsync_directory(paths.directory)


def recover_transaction(paths: DomainPaths) -> str:
    """Discard uncommitted preparation or finish a validated installing transaction."""
    _reject_symlink(paths.transaction, "transaction directory")
    if not paths.transaction.exists():
        return "none"
    if not paths.transaction.joinpath("journal.json").is_file():
        _remove_transaction(paths)
        return "discarded-unjournaled"
    journal = _load_journal(paths)
    if journal.get("phase") == "prepared":
        _remove_transaction(paths)
        return "discarded-prepared"
    if journal.get("phase") != "installing":
        raise RecoveryRequiredError(f"unknown transaction phase {journal.get('phase')!r}")
    target_entries = {entry["path"]: entry for entry in journal["targets"]}
    for relative_text, original_hash in journal["source_preconditions"].items():
        relative = _safe_relative(relative_text)
        current_hash = _hash_file(paths.directory / relative)
        candidate_entry = target_entries.get(relative_text)
        allowed = {original_hash}
        if candidate_entry is not None:
            allowed.add(candidate_entry["candidate_sha256"])
        if current_hash not in allowed:
            raise RecoveryRequiredError(
                f"authoritative source {relative_text} changed after the crash; "
                "recovery refused to overwrite it"
            )
    stage = paths.transaction / "stage"
    _reject_symlink(stage, "transaction stage directory")
    available = True
    for entry in journal["targets"]:
        relative = _safe_relative(entry["path"])
        _reject_target_symlinks(paths.directory, relative)
        _reject_target_symlinks(stage, relative)
        target = paths.directory / relative
        staged = stage / relative
        expected = entry["candidate_sha256"]
        if _hash_file(target) != expected and _hash_file(staged) != expected:
            available = False
            break
    if available:
        ordered = sorted(
            journal["targets"], key=lambda item: (item["path"] == "profile.yaml", item["path"])
        )
        for entry in ordered:
            relative = _safe_relative(entry["path"])
            target = paths.directory / relative
            if _hash_file(target) == entry["candidate_sha256"]:
                continue
            staged = stage / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staged, target)
            _fsync_directory(target.parent)
        _remove_transaction(paths)
        return "rolled-forward"
    _restore_transaction(paths, journal)
    _remove_transaction(paths)
    return "restored-backups"


def _install_transaction(
    paths: DomainPaths,
    targets: dict[str, bytes],
    *,
    source_preconditions: dict[str, str],
) -> None:
    _reject_symlink(paths.transaction, "transaction directory")
    if paths.transaction.exists():
        raise RecoveryRequiredError(
            f"interrupted transaction exists at {paths.transaction}; run doctor --recover"
        )
    pending: dict[str, bytes] = {}
    for relative_text in sorted(targets):
        relative = _safe_relative(relative_text)
        _reject_target_symlinks(paths.directory, relative)
        candidate = targets[relative_text]
        if _hash_file(paths.directory / relative) != sha256_bytes(candidate):
            pending[relative_text] = candidate
    if not pending:
        return
    paths.transaction.mkdir(mode=0o700)
    stage = paths.transaction / "stage"
    backups = paths.transaction / "backups"
    stage.mkdir()
    backups.mkdir()
    journal_targets: list[dict[str, Any]] = []
    written_directories: set[Path] = {stage, backups}
    try:
        for relative_text in sorted(pending):
            relative = _safe_relative(relative_text)
            target = paths.directory / relative
            staged = stage / relative
            _write_fsynced(staged, pending[relative_text])
            written_directories.add(staged.parent)
            _failpoint(f"stage:{relative.as_posix()}")
            entry: dict[str, Any] = {
                "candidate_sha256": sha256_bytes(pending[relative_text]),
                "had_original": target.is_file(),
                "path": relative.as_posix(),
            }
            if target.is_file():
                backup = backups / relative
                backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(target, backup)
                _fsync_file(backup)
                written_directories.add(backup.parent)
                entry["backup_sha256"] = _hash_file(backup)
            journal_targets.append(entry)
        for directory in sorted(
            written_directories, key=lambda item: (-len(item.parts), item.as_posix())
        ):
            _fsync_directory(directory)
        journal = {
            "phase": "prepared",
            "schema_version": 1,
            "source_preconditions": dict(sorted(source_preconditions.items())),
            "targets": journal_targets,
            "transaction_id": uuid.uuid4().hex,
        }
        _journal_write(paths, journal)
        _failpoint("before_install")
        for relative_text, expected_hash in source_preconditions.items():
            relative = _safe_relative(relative_text)
            if _hash_file(paths.directory / relative) != expected_hash:
                raise StaleRevisionError(
                    f"{relative_text} changed during commit/rebuild; reload context and retry"
                )
        journal["phase"] = "installing"
        _journal_write(paths, journal)
        ordered = sorted(
            journal_targets, key=lambda item: (item["path"] == "profile.yaml", item["path"])
        )
        for entry in ordered:
            relative = _safe_relative(entry["path"])
            target = paths.directory / relative
            staged = stage / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staged, target)
            _fsync_directory(target.parent)
            _failpoint(f"install:{entry['path']}")
        _remove_transaction(paths)
    except BaseException:
        if paths.transaction.exists():
            try:
                if not paths.transaction.joinpath("journal.json").is_file():
                    _remove_transaction(paths)
                else:
                    journal = _load_journal(paths)
                    if journal.get("phase") != "prepared":
                        _restore_transaction(paths, journal)
                    _remove_transaction(paths)
            except RecoveryRequiredError:
                pass
        raise


def _load_sources(paths: DomainPaths) -> tuple[dict[str, Any], bytes, list[dict[str, Any]], bytes]:
    _reject_symlink(paths.profile, "profile.yaml")
    _reject_symlink(paths.events, "events.jsonl")
    _reject_symlink(paths.sessions, "sessions directory")
    _reject_symlink(paths.syllabus, "syllabus directory")
    if paths.syllabus.exists() and not paths.syllabus.is_dir():
        raise ValidationError(f"syllabus path must be a directory: {paths.syllabus}")
    _reject_symlink(paths.transaction, "transaction directory")
    profile, profile_bytes = load_profile(paths.profile)
    if profile["domain_id"] != paths.domain_id:
        raise ValidationError("profile domain_id does not match its directory")
    try:
        events_bytes = paths.events.read_bytes()
    except FileNotFoundError as exc:
        raise ValidationError(f"{paths.events}: canonical event log is missing") from exc
    events = parse_events_bytes(events_bytes, str(paths.events))
    return profile, profile_bytes, events, events_bytes


def _projection_targets(
    profile: dict[str, Any],
    profile_bytes: bytes,
    events: list[dict[str, Any]],
    events_bytes: bytes,
) -> tuple[dict[str, bytes], dict[str, Any]]:
    timeline = reduce_grounding_timeline(events)
    state = project_state(
        profile,
        events,
        profile_bytes=profile_bytes,
        events_bytes=events_bytes,
        timeline=timeline,
    )
    targets = {
        "dashboard.md": render_dashboard(state),
        "state.json": pretty_json(state),
    }
    targets.update(render_all_receipts(profile, events, timeline))
    targets.update(render_all_syllabus_receipts(events, timeline))
    return targets, state


class LocalStore:
    """Authoritative local store that owns every write beneath a vault root."""

    def __init__(self, root: Path):
        """Bind the store to a vault root."""
        self.root = root.resolve()

    def paths(self, domain_id: str) -> DomainPaths:
        """Return the paths for a domain id, rejecting ids the store did not generate."""
        if not isinstance(domain_id, str) or not re.fullmatch(
            r"[a-z0-9][a-z0-9-]{0,48}-[0-9a-f]{8}", domain_id
        ):
            raise ValidationError(
                "domain-id must be a generated lowercase slug plus SHA-256 suffix"
            )
        return DomainPaths(self.root, domain_id)

    def init(self, domain: str, goal: str) -> dict[str, Any]:
        """Create a new domain directory atomically and return its identity."""
        profile = initial_profile(domain, goal)
        domain_id = profile["domain_id"]
        final = self.paths(domain_id)
        domains = self.root / "domains"
        _reject_symlink(domains, "domains directory")
        domains.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=f".init-{domain_id}-", dir=domains))
        try:
            (stage / "sessions").mkdir()
            profile_bytes = pretty_json(profile)
            events_bytes = b""
            state = project_state(
                profile, [], profile_bytes=profile_bytes, events_bytes=events_bytes
            )
            _write_fsynced(stage / "profile.yaml", profile_bytes)
            _write_fsynced(stage / "events.jsonl", events_bytes)
            _write_fsynced(stage / "state.json", pretty_json(state))
            _write_fsynced(stage / "dashboard.md", render_dashboard(state))
            _fsync_directory(stage / "sessions")
            _fsync_directory(stage)
            if os.path.lexists(final.directory):
                raise ValidationError(f"domain {domain_id!r} already exists; use context or commit")
            try:
                os.rename(stage, final.directory)
            except OSError as exc:
                if exc.errno in {errno.EEXIST, errno.ENOTEMPTY, errno.ENOTDIR}:
                    raise ValidationError(
                        f"domain {domain_id!r} already exists; use context or commit"
                    ) from exc
                raise
            _fsync_directory(domains)
        finally:
            if stage.exists():
                shutil.rmtree(stage)
        return {"domain_id": domain_id, "revision": 0, "status": "initialized"}

    def _require_domain(self, domain_id: str) -> DomainPaths:
        paths = self.paths(domain_id)
        _reject_symlink(self.root / "domains", "domains directory")
        _reject_symlink(paths.directory, "domain directory")
        if not paths.directory.is_dir():
            raise ValidationError(f"unknown domain-id {domain_id!r}")
        return paths

    def _recover_before_write(self, paths: DomainPaths) -> str:
        _reject_symlink(paths.transaction, "transaction directory")
        if paths.transaction.exists():
            return recover_transaction(paths)
        return "none"

    @staticmethod
    def _syllabus_snapshot_matches(existing: dict[str, Any], incoming: dict[str, Any]) -> bool:
        """Compare immutable verified content while allowing a different audit filename/time."""
        existing_source = dict(existing["source"])
        incoming_source = dict(incoming["source"])
        existing_source.pop("original_filename", None)
        incoming_source.pop("original_filename", None)
        existing_snapshot = {
            "assertion_set_sha256": existing["assertion_set_sha256"],
            "assertions": existing["assertions"],
            "extraction": existing["extraction"],
            "pages": existing["pages"],
            "source": existing_source,
        }
        incoming_snapshot = {
            "assertion_set_sha256": incoming["assertion_set_sha256"],
            "assertions": incoming["assertions"],
            "extraction": incoming["extraction"],
            "pages": incoming["pages"],
            "source": incoming_source,
        }
        return canonical_json(existing_snapshot, newline=False) == canonical_json(
            incoming_snapshot, newline=False
        )

    def _commit_request(
        self,
        domain_id: str,
        expected_revision: int,
        request: dict[str, Any],
        *,
        allowed_reserved_kinds: set[str],
        digest_idempotence: bool = False,
    ) -> dict[str, Any]:
        """Run every mutation through one lock, candidate projection, and transaction path."""
        paths = self._require_domain(domain_id)
        validate_commit_request(request)
        reserved = {
            event["kind"]
            for event in request.get("events", [])
            if event["kind"] in RESERVED_SYLLABUS_EVENT_KINDS
        }
        disallowed = reserved - allowed_reserved_kinds
        if disallowed:
            kinds = ", ".join(sorted(disallowed))
            raise ValidationError(
                f"reserved syllabus event kind(s) {kinds} require ingest-syllabus or "
                "approve-syllabus"
            )
        with domain_lock(paths):
            recovery = self._recover_before_write(paths)
            profile, profile_bytes, events, events_bytes = _load_sources(paths)
            if profile["domain_id"] != domain_id:
                raise ValidationError("profile domain_id does not match its directory")
            if profile["revision"] != expected_revision:
                raise StaleRevisionError(
                    f"stale revision: expected {expected_revision}, current {profile['revision']}"
                )
            current_state = project_state(
                profile,
                events,
                profile_bytes=profile_bytes,
                events_bytes=events_bytes,
            )
            grounded = current_state["grounding"]["status"] in {
                "approved",
                "approved_update_pending",
            }
            if grounded and "syllabus" in request.get("profile_patch", {}):
                raise ValidationError(
                    "request.profile_patch.syllabus: flat topics are a legacy ungrounded "
                    "fallback and cannot change approved course coverage; record a superseding "
                    "snapshot with approve-syllabus instead"
                )

            if digest_idempotence:
                incoming_sources = [
                    event
                    for event in request.get("events", [])
                    if event["kind"] == "syllabus_source_ingested"
                ]
                if len(incoming_sources) != 1 or len(request.get("events", [])) != 1:
                    raise ValidationError("ingest-syllabus must contain exactly one source event")
                incoming = incoming_sources[0]
                matching = [
                    event
                    for event in events
                    if event["kind"] == "syllabus_source_ingested"
                    and event["source"]["sha256"] == incoming["source"]["sha256"]
                ]
                if matching:
                    existing = matching[0]
                    if not self._syllabus_snapshot_matches(existing, incoming):
                        raise ValidationError(
                            "existing syllabus digest conflicts with the verified adapter snapshot"
                        )
                    approval_status = (
                        "approved"
                        if any(
                            event["kind"] == "syllabus_approval_recorded"
                            and event["source_version_id"]
                            == incoming["source"]["source_version_id"]
                            for event in events
                        )
                        else "approval_required"
                    )
                    return {
                        "appended_events": 0,
                        "approval_status": approval_status,
                        "domain_id": domain_id,
                        "recovery": recovery,
                        "revision": profile["revision"],
                        "status": "noop",
                    }

            existing_by_id = {event["event_id"]: event for event in events}
            incoming_by_id: dict[str, dict[str, Any]] = {}
            new_events: list[dict[str, Any]] = []
            for event in request.get("events", []):
                event_id = event["event_id"]
                prior_in_request = incoming_by_id.get(event_id)
                if prior_in_request is not None:
                    if canonical_json(prior_in_request, newline=False) != canonical_json(
                        event, newline=False
                    ):
                        raise ValidationError(
                            f"request.events: duplicate ID {event_id!r} has conflicting bodies"
                        )
                    continue
                incoming_by_id[event_id] = event
                existing = existing_by_id.get(event_id)
                if existing is not None:
                    if canonical_json(existing, newline=False) != canonical_json(
                        event, newline=False
                    ):
                        raise ValidationError(
                            f"event ID {event_id!r} already exists with different content"
                        )
                    continue
                new_events.append(event)

            candidate_profile = dict(profile)
            patch = request.get("profile_patch", {})
            for key, value in patch.items():
                candidate_profile[key] = value
            profile_changed = any(profile.get(key) != value for key, value in patch.items())
            if not new_events and not profile_changed:
                return {
                    "appended_events": 0,
                    "domain_id": domain_id,
                    "recovery": recovery,
                    "revision": profile["revision"],
                    "status": "noop",
                }

            candidate_profile["revision"] = profile["revision"] + 1
            validate_profile(candidate_profile)
            candidate_profile_bytes = pretty_json(candidate_profile)
            appended_bytes = encode_event_lines(new_events)
            candidate_events_bytes = events_bytes + appended_bytes
            if not candidate_events_bytes.startswith(events_bytes):
                raise StoreError("append-only prefix invariant failed")
            candidate_events = events + new_events
            projection_targets, state = _projection_targets(
                candidate_profile,
                candidate_profile_bytes,
                candidate_events,
                candidate_events_bytes,
            )
            targets = {
                "events.jsonl": candidate_events_bytes,
                "profile.yaml": candidate_profile_bytes,
                **projection_targets,
            }
            _install_transaction(
                paths,
                targets,
                source_preconditions={
                    "events.jsonl": sha256_bytes(events_bytes),
                    "profile.yaml": sha256_bytes(profile_bytes),
                },
            )
            return {
                "appended_events": len(new_events),
                "domain_id": domain_id,
                "event_count": state["source"]["event_count"],
                "recovery": recovery,
                "revision": candidate_profile["revision"],
                "status": "committed",
            }

    def commit(
        self, domain_id: str, expected_revision: int, request: dict[str, Any]
    ) -> dict[str, Any]:
        """Append ordinary learning events/profile edits; syllabus kinds are reserved."""
        return self._commit_request(
            domain_id,
            expected_revision,
            request,
            allowed_reserved_kinds=set(),
        )

    def ingest_syllabus(
        self,
        domain_id: str,
        expected_revision: int,
        document_path: Path,
        *,
        original_filename: str,
        media_type: str,
        adapter_id: str,
        occurred_at: str,
        supersedes_source_version_id: str | None = None,
    ) -> dict[str, Any]:
        """Verify and append the exact supported ST5201X source snapshot."""
        event = build_ingestion_event(
            document_path,
            original_filename=original_filename,
            media_type=media_type,
            adapter_id=adapter_id,
            occurred_at=occurred_at,
            supersedes_source_version_id=supersedes_source_version_id,
        )
        result = self._commit_request(
            domain_id,
            expected_revision,
            {"events": [event]},
            allowed_reserved_kinds={"syllabus_source_ingested"},
            digest_idempotence=True,
        )
        result.setdefault("approval_status", "approval_required")
        result.update(
            {
                "assertion_set_sha256": event["assertion_set_sha256"],
                "source_event_id": event["event_id"],
                "source_id": event["source"]["source_id"],
                "source_sha256": event["source"]["sha256"],
                "source_version_id": event["source"]["source_version_id"],
            }
        )
        return result

    def approve_syllabus(
        self, domain_id: str, expected_revision: int, request: dict[str, Any]
    ) -> dict[str, Any]:
        """Append one complete learner approval/correction snapshot."""
        event = build_syllabus_approval_event(request)
        result = self._commit_request(
            domain_id,
            expected_revision,
            {"events": [event]},
            allowed_reserved_kinds={"syllabus_approval_recorded"},
        )
        result.update(
            {
                "approval_event_id": event["event_id"],
                "approval_status": "approved",
                "approval_set_sha256": event["approval_set_sha256"],
                "source_version_id": event["source_version_id"],
            }
        )
        return result

    def rebuild(self, domain_id: str) -> dict[str, Any]:
        """Regenerate every derived projection from the canonical sources."""
        paths = self._require_domain(domain_id)
        with domain_lock(paths):
            recovery = self._recover_before_write(paths)
            profile, profile_bytes, events, events_bytes = _load_sources(paths)
            targets, state = _projection_targets(profile, profile_bytes, events, events_bytes)
            _install_transaction(
                paths,
                targets,
                source_preconditions={
                    "events.jsonl": sha256_bytes(events_bytes),
                    "profile.yaml": sha256_bytes(profile_bytes),
                },
            )
            return {
                "domain_id": domain_id,
                "event_count": len(events),
                "recovery": recovery,
                "revision": profile["revision"],
                "state_sha256": sha256_bytes(pretty_json(state)),
                "status": "rebuilt",
            }

    def context(self, domain_id: str) -> dict[str, Any]:
        """Return the profile and a freshly projected state for a domain."""
        paths = self._require_domain(domain_id)
        with domain_lock(paths):
            if paths.transaction.exists():
                raise RecoveryRequiredError(
                    f"interrupted transaction at {paths.transaction}; run doctor --recover"
                )
            profile, profile_bytes, events, events_bytes = _load_sources(paths)
            state = project_state(
                profile, events, profile_bytes=profile_bytes, events_bytes=events_bytes
            )
            return {"profile": profile, "state": state}

    def validate(self, domain_id: str) -> dict[str, Any]:
        """Check all generated projections and receipt trees against canonical sources."""
        paths = self._require_domain(domain_id)
        with domain_lock(paths):
            if paths.transaction.exists():
                raise RecoveryRequiredError(
                    f"interrupted transaction at {paths.transaction}; run doctor --recover"
                )
            profile, profile_bytes, events, events_bytes = _load_sources(paths)
            targets, _ = _projection_targets(profile, profile_bytes, events, events_bytes)
            drift: list[str] = []
            for relative, expected in targets.items():
                actual = paths.directory / relative
                if actual.is_symlink() or not actual.is_file() or actual.read_bytes() != expected:
                    drift.append(relative)
            expected_receipts = {
                relative for relative in targets if relative.startswith(("sessions/", "syllabus/"))
            }
            orphan_sessions: list[str] = []
            orphan_syllabus: list[str] = []
            for generated_directory, orphans in (
                (paths.sessions, orphan_sessions),
                (paths.syllabus, orphan_syllabus),
            ):
                if not generated_directory.is_dir():
                    continue
                for entry in generated_directory.rglob("*"):
                    if not (entry.is_file() or entry.is_symlink()):
                        continue
                    if not entry.is_symlink() and _is_editor_metadata(generated_directory, entry):
                        continue
                    relative = entry.relative_to(paths.directory).as_posix()
                    if relative not in expected_receipts:
                        orphans.append(relative)
            problems: list[str] = []
            if drift:
                problems.append(
                    "derived projection drift: "
                    + ", ".join(sorted(set(drift)))
                    + "; restore generated files from canonical sources with rebuild"
                )
            if orphan_sessions:
                problems.append(
                    "unexpected files under sessions/: "
                    + ", ".join(sorted(set(orphan_sessions)))
                    + "; every receipt must correspond to a session_completed event"
                )
            if orphan_syllabus:
                problems.append(
                    "unexpected files under syllabus/: "
                    + ", ".join(sorted(set(orphan_syllabus)))
                    + "; every receipt must correspond to a syllabus_source_ingested event"
                )
            if problems:
                raise ValidationError("; ".join(problems))
            return {
                "derived_drift": [],
                "domain_id": domain_id,
                "event_count": len(events),
                "revision": profile["revision"],
                "status": "valid",
            }

    def list_domains(self) -> dict[str, Any]:
        """Summarize every domain in the vault, reporting unavailable ones."""
        domains_directory = self.root / "domains"
        results: list[dict[str, Any]] = []
        if not domains_directory.exists():
            return {"domains": []}
        for directory in sorted(
            path
            for path in domains_directory.iterdir()
            if path.is_dir() and not path.name.startswith(".")
        ):
            paths = DomainPaths(self.root, directory.name)
            try:
                with domain_lock(paths):
                    if paths.transaction.exists():
                        raise RecoveryRequiredError("recovery required")
                    profile, _ = load_profile(paths.profile)
                    if profile["domain_id"] != directory.name:
                        raise ValidationError("profile domain_id does not match its directory")
                results.append(
                    {
                        "domain": profile["domain"],
                        "domain_id": profile["domain_id"],
                        "goal": profile["goal"],
                        "revision": profile["revision"],
                        "status": "ready",
                    }
                )
            except StoreError as exc:
                results.append(
                    {"domain_id": directory.name, "status": "unavailable", "diagnostic": str(exc)}
                )
        return {"domains": results}

    def doctor(
        self,
        domain_id: str,
        *,
        recover: bool = False,
        break_stale_lock: bool = False,
    ) -> dict[str, Any]:
        """Report lock and transaction health, optionally recovering or unlocking."""
        paths = self._require_domain(domain_id)
        lock_directory = _lock_directory(paths)
        handle = _open_lock(paths.lock)
        metadata, lock_error = _read_lock(handle)
        lock_broken = False
        try:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                liveness = "active"
                if break_stale_lock:
                    raise LockError(
                        f"refusing to break an active advisory lock: {lock_error or metadata}"
                    )
            else:
                liveness = "stale" if (metadata is not None or lock_error) else "absent"
                if break_stale_lock and liveness == "stale":
                    handle.seek(0)
                    handle.truncate()
                    os.fsync(handle.fileno())
                    _fsync_directory(lock_directory)
                    lock_broken = True
                    liveness = "absent"
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()
        recovery_status = "not-requested"
        if recover:
            with domain_lock(paths):
                recovery_status = recover_transaction(paths)
                _load_sources(paths)
        return {
            "domain_id": domain_id,
            "lock": {
                "broken": lock_broken,
                "diagnostic": lock_error,
                "metadata": metadata,
                "status": liveness,
            },
            "recovery": recovery_status,
            "transaction_present": paths.transaction.exists(),
        }

    def import_legacy(self, domain: str, input_path: Path) -> dict[str, Any]:
        """Import an exported legacy Knowledge State as non-evidence prior context."""
        try:
            source = input_path.read_bytes()
        except FileNotFoundError as exc:
            raise ValidationError(f"legacy KS input is missing: {input_path}") from exc
        event, claims = legacy_event(source)
        domain_id = make_domain_id(domain)
        paths = self.paths(domain_id)
        initialized = False
        if not paths.directory.exists():
            goal = claims.get("goal") or "Imported legacy Knowledge State"
            self.init(domain, goal)
            initialized = True
        with domain_lock(paths):
            if paths.transaction.exists():
                raise RecoveryRequiredError(
                    f"interrupted transaction at {paths.transaction}; run doctor --recover"
                )
            profile, _, events, _ = _load_sources(paths)
            if events:
                matching = [
                    item
                    for item in events
                    if item["kind"] == "legacy_snapshot_imported"
                    and item["source_sha256"] == event["source_sha256"]
                ]
                if not matching:
                    raise ValidationError(
                        "refusing a different legacy snapshot import into a non-empty domain"
                    )
        result = self.commit(
            domain_id,
            profile["revision"],
            {"events": [event]},
        )
        result["initialized"] = initialized
        result["source_sha256"] = event["source_sha256"]
        return result
