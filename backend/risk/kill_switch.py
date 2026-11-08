import json
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_PATH = Path("data") / "kill_switch.json"


def _now():
    return datetime.now(timezone.utc).isoformat()


class KillSwitch:
    """
    Global kill switch kept in a file so it survives restarts and is shared by
    every process. It FAILS CLOSED: if the file exists but cannot be read, the
    switch is treated as active. It never resets itself; a person must call
    reset() with confirm=True and a reason.
    """

    def __init__(self, path=DEFAULT_PATH):
        self.path = Path(path)

    def _read(self):
        if not self.path.exists():
            return {"active": False, "history": []}
        try:
            data = json.loads(self.path.read_text())
            if not isinstance(data, dict) or "active" not in data:
                raise ValueError("bad format")
            data.setdefault("history", [])
            return data
        except Exception:
            return {"active": True, "reason": "KILL_SWITCH_FILE_UNREADABLE",
                    "source": "system", "time": _now(), "history": []}

    def _write(self, data):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(self.path)

    def is_active(self):
        return bool(self._read().get("active", True))

    def status(self):
        return self._read()

    def activate(self, reason, source="manual"):
        data = self._read()
        if data.get("active"):
            return data                                  # keep the original trigger
        data.update({"active": True, "reason": reason, "source": source, "time": _now()})
        data["history"].append({"event": "activated", "reason": reason, "source": source, "time": _now()})
        self._write(data)
        return data

    def reset(self, reason, confirm=False):
        """Manual reset only. Requires confirm=True and a non-empty reason."""
        if confirm is not True or not str(reason).strip():
            raise ValueError("reset needs confirm=True and a reason")
        data = self._read()
        data.update({"active": False, "reason": "", "source": "", "time": _now()})
        data["history"].append({"event": "reset", "reason": reason, "time": _now()})
        self._write(data)
        return data
