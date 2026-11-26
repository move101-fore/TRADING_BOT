import pytest

from backend.research.freeze import freeze, open_items, spec_hash, verify


def write(path, text):
    path.write_text(text, encoding="utf-8")
    return path


def test_hash_ignores_line_endings(tmp_path):
    a = tmp_path / "a.md"
    b = tmp_path / "b.md"
    a.write_bytes(b"line one\nline two\n")
    b.write_bytes(b"line one\r\nline two\r\n")
    assert spec_hash(a) == spec_hash(b)


def test_open_items_block_freezing(tmp_path):
    spec = write(tmp_path / "s.md", "A fixed rule.\nA choice [CONFIRM]\nAnother [CONFIRM] choice\n")
    assert len(open_items(spec)) == 2
    with pytest.raises(ValueError, match="2 open item"):
        freeze(spec, "X-v1", log=tmp_path / "log.csv", commit="abc")
    assert not (tmp_path / "log.csv").exists()


def test_freeze_then_verify_and_detect_changes(tmp_path):
    spec = write(tmp_path / "s.md", "All rules are fixed.\n")
    log = tmp_path / "log.csv"
    row = freeze(spec, "X-v1", log=log, commit="abc123")
    assert row["sha256"] == spec_hash(spec) and verify(spec, "X-v1", log)
    write(spec, "All rules are fixed.\nA quiet edit.\n")
    assert not verify(spec, "X-v1", log)


def test_a_frozen_version_cannot_be_refrozen_with_changes(tmp_path):
    spec = write(tmp_path / "s.md", "Rules.\n")
    log = tmp_path / "log.csv"
    freeze(spec, "X-v1", log=log, commit="c")
    assert freeze(spec, "X-v1", log=log, commit="c")["version"] == "X-v1"        # same content is fine
    write(spec, "Rules, changed.\n")
    with pytest.raises(ValueError, match="different hash"):
        freeze(spec, "X-v1", log=log, commit="c")
    assert freeze(spec, "X-v2", log=log, commit="c")["version"] == "X-v2"        # a new version is allowed


def test_unfrozen_version_does_not_verify(tmp_path):
    spec = write(tmp_path / "s.md", "Rules.\n")
    assert not verify(spec, "never", tmp_path / "log.csv")
