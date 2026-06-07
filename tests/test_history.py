"""Tests for mist_autoresearch.history."""
import json
from pathlib import Path

import pytest

from mist_autoresearch.history import History


class TestHistory:

    def test_initial_state_no_file(self, tmp_path):
        h = History(tmp_path / "history.json")
        assert h.iterations == []
        assert h.best_iteration is None
        assert h.stopped_reason is None
        assert h.n_iterations == 0

    def test_loads_existing_file(self, tmp_path):
        path = tmp_path / "history.json"
        data = {
            "iterations": [{"iteration": 1, "strategy": [], "narrative": "n",
                             "results_csv": "r.csv", "mean_rank": 1.0,
                             "p_value_vs_baseline": None, "timestamp": "t"}],
            "best_iteration": 1,
            "stopped_reason": "patience",
            "started_at": "2026-01-01T00:00:00",
        }
        path.write_text(json.dumps(data))
        h = History(path)
        assert h.n_iterations == 1
        assert h.best_iteration == 1
        assert h.stopped_reason == "patience"

    def test_add_iteration_persists(self, tmp_path):
        path = tmp_path / "history.json"
        h = History(path)
        h.add_iteration(1, [{"transform": "x"}], "narrative", "r.csv", 1.5, 0.03)
        assert h.n_iterations == 1
        assert path.exists()
        reloaded = json.loads(path.read_text())
        assert len(reloaded["iterations"]) == 1
        assert reloaded["iterations"][0]["iteration"] == 1
        assert reloaded["iterations"][0]["mean_rank"] == 1.5
        assert reloaded["iterations"][0]["p_value_vs_baseline"] == 0.03

    def test_add_multiple_iterations(self, tmp_path):
        h = History(tmp_path / "history.json")
        h.add_iteration(1, [], "n1", "r1.csv", 2.0, None)
        h.add_iteration(2, [], "n2", "r2.csv", 1.0, 0.02)
        assert h.n_iterations == 2

    def test_update_best_persists(self, tmp_path):
        path = tmp_path / "history.json"
        h = History(path)
        h.update_best(3)
        assert h.best_iteration == 3
        reloaded = json.loads(path.read_text())
        assert reloaded["best_iteration"] == 3

    def test_set_stopped_reason_persists(self, tmp_path):
        path = tmp_path / "history.json"
        h = History(path)
        h.set_stopped_reason("max_iterations")
        assert h.stopped_reason == "max_iterations"
        reloaded = json.loads(path.read_text())
        assert reloaded["stopped_reason"] == "max_iterations"

    def test_creates_parent_directory(self, tmp_path):
        path = tmp_path / "nested" / "deep" / "history.json"
        h = History(path)
        h.add_iteration(1, [], "n", "r.csv", 1.0, None)
        assert path.exists()

    def test_p_value_none_serialises_as_null(self, tmp_path):
        path = tmp_path / "history.json"
        h = History(path)
        h.add_iteration(1, [], "n", "r.csv", 1.0, None)
        data = json.loads(path.read_text())
        assert data["iterations"][0]["p_value_vs_baseline"] is None

    def test_iterations_property_returns_list(self, tmp_path):
        h = History(tmp_path / "history.json")
        assert isinstance(h.iterations, list)
