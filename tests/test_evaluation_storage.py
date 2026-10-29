import json
from pathlib import Path

import pytest

from scripts import evaluate_translation


def windows_permission_error():
    error = PermissionError("Report is open in a Windows reader")
    error.winerror = 5
    return error


def test_report_survives_a_transient_windows_reader_lock(tmp_path, monkeypatch):
    destination = tmp_path / "result.json"
    destination.write_text('{"previous": true}', encoding="utf-8")
    replace = Path.replace
    attempts, sleeps = [], []

    def locked_then_released(path, target):
        attempts.append(path)
        if len(attempts) < 3:
            assert json.loads(destination.read_text()) == {"previous": True}
            raise windows_permission_error()
        return replace(path, target)

    monkeypatch.setattr(Path, "replace", locked_then_released)
    monkeypatch.setattr(evaluate_translation.time, "sleep", sleeps.append)
    evaluate_translation.save(destination, {"translation": "주스"})
    assert json.loads(destination.read_text(encoding="utf-8")) == {"translation": "주스"}
    assert len(attempts) == 3 and sleeps == [0.1, 0.1]
    assert not destination.with_suffix(".json.tmp").exists()


def test_permanent_lock_raises_and_preserves_both_reports(tmp_path, monkeypatch):
    destination = tmp_path / "result.json"
    destination.write_text('{"previous": true}', encoding="utf-8")
    attempts = []

    def locked(path, target):
        attempts.append(path)
        raise windows_permission_error()

    monkeypatch.setattr(Path, "replace", locked)
    monkeypatch.setattr(evaluate_translation.time, "sleep", lambda _: None)
    with pytest.raises(PermissionError):
        evaluate_translation.save(destination, {"new": True})
    assert len(attempts) == 20
    assert json.loads(destination.read_text()) == {"previous": True}
    assert json.loads(destination.with_suffix(".json.tmp").read_text()) == {"new": True}


def test_unrelated_permission_failure_is_not_retried(tmp_path, monkeypatch):
    def forbidden(path, target):
        raise PermissionError("Not a Windows sharing failure")

    sleeps = []
    monkeypatch.setattr(Path, "replace", forbidden)
    monkeypatch.setattr(evaluate_translation.time, "sleep", sleeps.append)
    with pytest.raises(PermissionError):
        evaluate_translation.save(tmp_path / "result.json", {"new": True})
    assert not sleeps
