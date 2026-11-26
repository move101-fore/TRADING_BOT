import csv
import hashlib
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

LOG = Path("research") / "frozen_specs.csv"
OPEN_MARKER = "[CONFIRM]"


def spec_hash(path):
    """SHA-256 of a spec file, independent of Windows/Unix line endings."""
    text = Path(path).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(text).hexdigest()


def open_items(path):
    """Lines of the spec that still carry the open-item marker."""
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if OPEN_MARKER in line]


def _read(log):
    if not Path(log).exists():
        return []
    with open(log, newline="") as f:
        return list(csv.DictReader(f))


def _git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return ""


def freeze(path, version, log=LOG, commit=None):
    """
    Record a spec version and its hash. Refuses while open items remain, and refuses to re-freeze a version
    whose file has changed (a changed spec must get a NEW version).
    """
    open_ = open_items(path)
    if open_:
        raise ValueError(f"{len(open_)} open item(s) still marked {OPEN_MARKER}; resolve them before freezing:\n  "
                         + "\n  ".join(open_))
    digest = spec_hash(path)
    for row in _read(log):
        if row["version"] == version:
            if row["sha256"] == digest:
                return row
            raise ValueError(f"{version} is already frozen with a different hash; create a new version instead")
    row = {"version": version, "sha256": digest, "frozen_at": datetime.now(timezone.utc).isoformat(),
           "git_commit": commit if commit is not None else _git_commit()}
    Path(log).parent.mkdir(parents=True, exist_ok=True)
    new = not Path(log).exists()
    with open(log, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
    return row


def verify(path, version, log=LOG):
    """True only if the spec file is byte-for-byte (line-ending-insensitive) what was frozen."""
    for row in _read(log):
        if row["version"] == version:
            return row["sha256"] == spec_hash(path)
    return False


def main(argv=None):
    """Usage: python -m backend.research.freeze freeze|verify|status <version> <spec file>"""
    a = sys.argv[1:] if argv is None else argv
    if len(a) != 3 or a[0] not in ("freeze", "verify", "status"):
        raise SystemExit(main.__doc__)
    action, version, path = a
    if action == "status":
        print(f"{len(open_items(path))} open item(s) in {path}")
        for line in open_items(path):
            print("  ", line)
    elif action == "freeze":
        row = freeze(path, version)
        print(f"Frozen {row['version']}  sha256={row['sha256']}  commit={row['git_commit'][:10]}")
    else:
        ok = verify(path, version)
        print("MATCHES the frozen version" if ok else "DOES NOT MATCH (or was never frozen)")
        raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
