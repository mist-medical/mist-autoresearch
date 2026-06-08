"""Tests for mist_autoresearch.base."""

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from mist_autoresearch.base import (
    AbstractResearcher,
    _get_mean_rank,
    _get_p_vs_baseline,
)
from mist_autoresearch.stopping import StoppingCriteria


# ---------------------------------------------------------------------------
# Concrete subclass for testing the loop
# ---------------------------------------------------------------------------


class _FakeResearcher(AbstractResearcher):
    """Concrete researcher that returns canned values for testing."""

    def __init__(self, output_dir, stopping, propose_seq=None, eval_df=None):
        super().__init__(output_dir=output_dir, stopping=stopping)
        self._propose_seq = propose_seq or [([], "narrative")]
        self._propose_idx = 0
        self._eval_df = eval_df or pd.DataFrame(
            {"id": [f"p{i}" for i in range(20)], "WT_dice": [0.9] * 20}
        )

    def propose(self, context):
        entry = self._propose_seq[self._propose_idx % len(self._propose_seq)]
        self._propose_idx += 1
        return entry

    def evaluate(self, strategy, iteration_dir):
        (iteration_dir / "postprocess_results.csv").parent.mkdir(
            parents=True, exist_ok=True
        )
        path = iteration_dir / "postprocess_results.csv"
        self._eval_df.to_csv(path, index=False)
        return self._eval_df

    def build_context(self, baseline_results, rank_df, significance_df):
        return {}


def _sc(**kwargs) -> StoppingCriteria:
    defaults = dict(
        max_iterations=3,
        patience=5,
        alpha=0.05,
        min_iterations=1,
        min_patients_for_significance=5,
    )
    defaults.update(kwargs)
    return StoppingCriteria(**defaults)


# ---------------------------------------------------------------------------
# Tests for module-level helpers
# ---------------------------------------------------------------------------


class TestGetMeanRank:
    def test_returns_rank_for_known_strategy(self):
        df = pd.DataFrame({"strategy": ["a", "b"], "average_rank": [1.0, 2.0]})
        assert _get_mean_rank(df, "a") == 1.0

    def test_returns_inf_for_unknown_strategy(self):
        df = pd.DataFrame({"strategy": ["a"], "average_rank": [1.0]})
        assert _get_mean_rank(df, "unknown") == float("inf")


class TestGetPVsBaseline:
    def _sig_df(self, p_value):
        import pandas as pd

        df = pd.DataFrame(
            {"baseline": {"iter_001": p_value, "baseline": float("nan")}},
        )
        df.index.name = "strategy"
        return df

    def test_returns_p_value(self):
        df = self._sig_df(0.03)
        result = _get_p_vs_baseline(df, "iter_001")
        assert result == pytest.approx(0.03)

    def test_returns_none_for_none_df(self):
        assert _get_p_vs_baseline(None, "iter_001") is None

    def test_returns_none_for_baseline_name(self):
        df = self._sig_df(0.03)
        assert _get_p_vs_baseline(df, "baseline") is None

    def test_returns_none_when_name_not_in_index(self):
        df = self._sig_df(0.03)
        assert _get_p_vs_baseline(df, "missing") is None

    def test_returns_none_when_baseline_not_in_columns(self):
        df = pd.DataFrame({"other": {"iter_001": 0.03}})
        assert _get_p_vs_baseline(df, "iter_001") is None

    def test_returns_none_for_nan_p_value(self):
        df = self._sig_df(float("nan"))
        assert _get_p_vs_baseline(df, "iter_001") is None


# ---------------------------------------------------------------------------
# Tests for the run() loop
# ---------------------------------------------------------------------------


class TestAbstractResearcherRun:
    def _make_rank_df(self, names, ranks):
        return pd.DataFrame({"strategy": names, "average_rank": ranks})

    def test_run_creates_output_dirs(self, tmp_path):
        r = _FakeResearcher(tmp_path / "out", _sc(max_iterations=1))
        with (
            patch("mist_autoresearch.base.rank_results") as mock_rank,
            patch("mist_autoresearch.base.compute_pairwise_significance") as mock_sig,
        ):
            mock_rank.return_value = (
                self._make_rank_df(["baseline", "iteration_001"], [1.0, 2.0]),
                None,
            )
            mock_sig.return_value = pd.DataFrame(
                {"baseline": {"iteration_001": 0.03}},
                dtype=float,
            )
            r.run()
        assert (tmp_path / "out" / "baseline").is_dir()
        assert (tmp_path / "out" / "iteration_001").is_dir()

    def test_run_writes_research_notebook(self, tmp_path):
        r = _FakeResearcher(tmp_path / "out", _sc(max_iterations=1))
        with (
            patch("mist_autoresearch.base.rank_results") as mock_rank,
            patch("mist_autoresearch.base.compute_pairwise_significance"),
        ):
            mock_rank.return_value = (
                self._make_rank_df(["baseline", "iteration_001"], [1.0, 2.0]),
                None,
            )
            r.run()
        assert (tmp_path / "out" / "research_notebook.md").exists()

    def test_run_writes_summary_json(self, tmp_path):
        r = _FakeResearcher(tmp_path / "out", _sc(max_iterations=1))
        with (
            patch("mist_autoresearch.base.rank_results") as mock_rank,
            patch("mist_autoresearch.base.compute_pairwise_significance"),
        ):
            mock_rank.return_value = (
                self._make_rank_df(["baseline", "iteration_001"], [1.0, 2.0]),
                None,
            )
            r.run()
        summary = json.loads((tmp_path / "out" / "summary.json").read_text())
        assert "best_strategy_name" in summary
        assert "best_strategy" in summary

    def test_run_stops_at_max_iterations(self, tmp_path):
        r = _FakeResearcher(tmp_path / "out", _sc(max_iterations=2))
        with (
            patch("mist_autoresearch.base.rank_results") as mock_rank,
            patch("mist_autoresearch.base.compute_pairwise_significance"),
        ):
            mock_rank.return_value = (
                self._make_rank_df(
                    ["baseline", "iteration_001", "iteration_002"],
                    [1.0, 2.0, 3.0],
                ),
                None,
            )
            r.run()
        assert r._propose_idx == 2

    def test_run_returns_best_strategy(self, tmp_path):
        winning_strategy = [
            {
                "transform": "remove_small_objects",
                "apply_to_labels": [-1],
                "per_label": False,
            }
        ]
        r = _FakeResearcher(
            tmp_path / "out",
            _sc(max_iterations=1),
            propose_seq=[(winning_strategy, "winning")],
        )
        with (
            patch("mist_autoresearch.base.rank_results") as mock_rank,
            patch("mist_autoresearch.base.compute_pairwise_significance"),
        ):
            # iteration_001 wins
            mock_rank.return_value = (
                self._make_rank_df(["iteration_001", "baseline"], [1.0, 2.0]),
                None,
            )
            result = r.run()
        assert result == winning_strategy

    def test_run_returns_none_baseline_when_best(self, tmp_path):
        r = _FakeResearcher(tmp_path / "out", _sc(max_iterations=1))
        with (
            patch("mist_autoresearch.base.rank_results") as mock_rank,
            patch("mist_autoresearch.base.compute_pairwise_significance"),
        ):
            mock_rank.return_value = (
                self._make_rank_df(["baseline", "iteration_001"], [1.0, 2.0]),
                None,
            )
            result = r.run()
        assert result is None  # baseline best → no postprocessing

    def test_run_updates_history(self, tmp_path):
        r = _FakeResearcher(tmp_path / "out", _sc(max_iterations=2))
        with (
            patch("mist_autoresearch.base.rank_results") as mock_rank,
            patch("mist_autoresearch.base.compute_pairwise_significance"),
        ):
            mock_rank.return_value = (
                self._make_rank_df(
                    ["baseline", "iteration_001", "iteration_002"],
                    [1.0, 2.0, 3.0],
                ),
                None,
            )
            r.run()
        assert r.history.n_iterations == 2

    def test_early_stop_on_patience_and_significance(self, tmp_path):
        sc = StoppingCriteria(
            max_iterations=20,
            patience=1,
            alpha=0.05,
            min_iterations=1,
            min_patients_for_significance=5,
        )
        r = _FakeResearcher(tmp_path / "out", sc)
        # iteration_001 wins at rank 1.0; iteration_002 doesn't beat it →
        # patience fires after iter_002, significance (p=0.01) confirms → stop.
        sig_df = pd.DataFrame({"baseline": {"iteration_001": 0.01}}, dtype=float)

        with (
            patch("mist_autoresearch.base.rank_results") as mock_rank,
            patch("mist_autoresearch.base.compute_pairwise_significance") as mock_sig,
        ):
            mock_rank.return_value = (
                self._make_rank_df(["iteration_001", "baseline"], [1.0, 2.0]),
                None,
            )
            mock_sig.return_value = sig_df
            r.run()

        # 2 iterations: iter_001 becomes best (no stop), iter_002 doesn't improve →
        # patience=1 met, p=0.01 < 0.05 → "patience+significance"
        assert r._propose_idx == 2
        assert r.history.stopped_reason == "patience+significance"

    def test_rank_and_significance_writes_csvs(self, tmp_path):
        r = _FakeResearcher(tmp_path / "out", _sc(max_iterations=1))
        r.output_dir.mkdir(parents=True, exist_ok=True)
        df1 = pd.DataFrame({"id": [f"p{i}" for i in range(20)], "WT_dice": [0.9] * 20})
        df2 = pd.DataFrame({"id": [f"p{i}" for i in range(20)], "WT_dice": [0.5] * 20})
        with (
            patch("mist_autoresearch.base.rank_results") as mock_rank,
            patch("mist_autoresearch.base.compute_pairwise_significance") as mock_sig,
        ):
            mock_rank.return_value = (
                pd.DataFrame({"strategy": ["a", "b"], "average_rank": [1.0, 2.0]}),
                None,
            )
            mock_sig.return_value = pd.DataFrame(
                {
                    "a": {"a": float("nan"), "b": 0.8},
                    "b": {"a": 0.2, "b": float("nan")},
                },
                dtype=float,
            )
            r._rank_and_significance([df1, df2], ["a", "b"], 20)
        assert (tmp_path / "out" / "rankings.csv").exists()
        assert (tmp_path / "out" / "significance.csv").exists()

    def test_rank_and_significance_skips_sig_with_few_patients(self, tmp_path):
        sc = StoppingCriteria(min_patients_for_significance=50)
        r = _FakeResearcher(tmp_path / "out", sc)
        r.output_dir.mkdir(parents=True, exist_ok=True)
        df1 = pd.DataFrame({"id": ["p1"], "WT_dice": [0.9]})
        df2 = pd.DataFrame({"id": ["p1"], "WT_dice": [0.5]})
        with (
            patch("mist_autoresearch.base.rank_results") as mock_rank,
            patch("mist_autoresearch.base.compute_pairwise_significance") as mock_sig,
        ):
            mock_rank.return_value = (
                pd.DataFrame({"strategy": ["a", "b"], "average_rank": [1.0, 2.0]}),
                None,
            )
            _, sig_df = r._rank_and_significance([df1, df2], ["a", "b"], 1)
        assert sig_df is None
        mock_sig.assert_not_called()

    def test_summary_json_none_rank_when_no_iterations(self, tmp_path):
        r = _FakeResearcher(tmp_path / "out", _sc())
        r.output_dir.mkdir(parents=True, exist_ok=True)
        r._write_summary("baseline", None, float("inf"))
        summary = json.loads((tmp_path / "out" / "summary.json").read_text())
        assert summary["best_overall_rank"] is None


# ---------------------------------------------------------------------------
# Tests for resume (_recover_state, _recompute_tracking, run() with history)
# ---------------------------------------------------------------------------


def _seed_run(out: Path, n_iterations: int = 1, best_iteration=None) -> None:
    """Write a partial run to *out* without executing any real evaluation."""
    out.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({"id": [f"p{i}" for i in range(20)], "WT_dice": [0.9] * 20})

    baseline_dir = out / "baseline"
    baseline_dir.mkdir()
    df.to_csv(baseline_dir / "postprocess_results.csv", index=False)

    strategy = [
        {
            "transform": "remove_small_objects",
            "apply_to_labels": [-1],
            "per_label": False,
        }
    ]
    history_data = {
        "iterations": [],
        "best_iteration": best_iteration,
        "stopped_reason": None,
        "started_at": "2026-01-01T00:00:00",
    }

    for i in range(1, n_iterations + 1):
        iter_dir = out / f"iteration_{i:03d}"
        iter_dir.mkdir()
        (iter_dir / "strategy.json").write_text(json.dumps(strategy))
        df.to_csv(iter_dir / "postprocess_results.csv", index=False)
        history_data["iterations"].append(
            {
                "iteration": i,
                "strategy": strategy,
                "narrative": "test",
                "results_csv": str(iter_dir / "postprocess_results.csv"),
                "mean_rank": 1.5,
                "p_value_vs_baseline": 0.5,
                "timestamp": "2026-01-01T00:00:00",
            }
        )

    (out / "history.json").write_text(json.dumps(history_data))


class TestResume:
    def _make_rank_df(self, names, ranks):
        return pd.DataFrame({"strategy": names, "average_rank": ranks})

    # ------------------------------------------------------------------
    # _recover_state
    # ------------------------------------------------------------------

    def test_recover_state_loads_baseline_and_iterations(self, tmp_path):
        out = tmp_path / "out"
        _seed_run(out, n_iterations=2)
        r = _FakeResearcher(out, _sc())
        baseline, all_results, all_names, all_strategies = r._recover_state()
        assert list(baseline.columns) == ["id", "WT_dice"]
        assert all_names == ["baseline", "iteration_001", "iteration_002"]
        assert len(all_results) == 3
        assert all_strategies[0] is None
        assert all_strategies[1] is not None

    def test_recover_state_raises_if_baseline_csv_missing(self, tmp_path):
        out = tmp_path / "out"
        _seed_run(out, n_iterations=1)
        (out / "baseline" / "postprocess_results.csv").unlink()
        r = _FakeResearcher(out, _sc())
        with pytest.raises(FileNotFoundError, match="baseline results not found"):
            r._recover_state()

    def test_recover_state_raises_if_iteration_csv_missing(self, tmp_path):
        out = tmp_path / "out"
        _seed_run(out, n_iterations=1)
        (out / "iteration_001" / "postprocess_results.csv").unlink()
        r = _FakeResearcher(out, _sc())
        with pytest.raises(FileNotFoundError, match="results missing for iteration"):
            r._recover_state()

    # ------------------------------------------------------------------
    # _recompute_tracking
    # ------------------------------------------------------------------

    def test_recompute_tracking_baseline_best(self, tmp_path):
        out = tmp_path / "out"
        _seed_run(out, n_iterations=2, best_iteration=None)
        r = _FakeResearcher(out, _sc())
        rank_df = self._make_rank_df(
            ["baseline", "iteration_001", "iteration_002"], [1.0, 2.0, 3.0]
        )
        best_rank, best_name, since = r._recompute_tracking(rank_df)
        assert best_name == "baseline"
        assert best_rank == 1.0
        assert since == 2  # no iteration ever beat baseline → n_iterations

    def test_recompute_tracking_iteration_best(self, tmp_path):
        out = tmp_path / "out"
        _seed_run(out, n_iterations=3, best_iteration=1)
        r = _FakeResearcher(out, _sc())
        rank_df = self._make_rank_df(
            ["iteration_001", "iteration_002", "iteration_003", "baseline"],
            [1.0, 2.0, 3.0, 4.0],
        )
        _, best_name, since = r._recompute_tracking(rank_df)
        assert best_name == "iteration_001"
        assert since == 2  # n_iterations(3) - best_iteration(1) = 2

    # ------------------------------------------------------------------
    # run() resume behaviour
    # ------------------------------------------------------------------

    def test_run_resumes_from_correct_iteration(self, tmp_path):
        out = tmp_path / "out"
        _seed_run(out, n_iterations=1)
        r = _FakeResearcher(out, _sc(max_iterations=3))
        with (
            patch("mist_autoresearch.base.rank_results") as mock_rank,
            patch("mist_autoresearch.base.compute_pairwise_significance"),
        ):
            mock_rank.return_value = (
                self._make_rank_df(
                    ["baseline", "iteration_001", "iteration_002", "iteration_003"],
                    [1.0, 2.0, 3.0, 4.0],
                ),
                None,
            )
            r.run()
        # 1 iteration already done; only iterations 002 and 003 should be proposed
        assert r._propose_idx == 2

    def test_run_does_not_call_write_header_on_resume(self, tmp_path):
        out = tmp_path / "out"
        _seed_run(out, n_iterations=1)
        r = _FakeResearcher(out, _sc(max_iterations=2))
        with (
            patch("mist_autoresearch.base.rank_results") as mock_rank,
            patch("mist_autoresearch.base.compute_pairwise_significance"),
            patch(
                "mist_autoresearch.base.ResearchNotebook.write_header"
            ) as mock_header,
        ):
            mock_rank.return_value = (
                self._make_rank_df(
                    ["baseline", "iteration_001", "iteration_002"],
                    [1.0, 2.0, 3.0],
                ),
                None,
            )
            r.run()
        mock_header.assert_not_called()

    def test_run_fresh_start_calls_write_header(self, tmp_path):
        r = _FakeResearcher(tmp_path / "out", _sc(max_iterations=1))
        with (
            patch("mist_autoresearch.base.rank_results") as mock_rank,
            patch("mist_autoresearch.base.compute_pairwise_significance"),
            patch(
                "mist_autoresearch.base.ResearchNotebook.write_header"
            ) as mock_header,
        ):
            mock_rank.return_value = (
                self._make_rank_df(["baseline", "iteration_001"], [1.0, 2.0]),
                None,
            )
            r.run()
        mock_header.assert_called_once()

    def test_run_resume_returns_best_strategy(self, tmp_path):
        out = tmp_path / "out"
        _seed_run(out, n_iterations=1)
        # The seeded iteration_001 strategy — resume should return it if it wins.
        expected = [
            {
                "transform": "remove_small_objects",
                "apply_to_labels": [-1],
                "per_label": False,
            }
        ]
        r = _FakeResearcher(out, _sc(max_iterations=1))
        with (
            patch("mist_autoresearch.base.rank_results") as mock_rank,
            patch("mist_autoresearch.base.compute_pairwise_significance"),
        ):
            # max_iterations=1 already done → loop doesn't run, best from seeded data
            mock_rank.return_value = (
                self._make_rank_df(["iteration_001", "baseline"], [1.0, 2.0]),
                None,
            )
            result = r.run()
        assert result == expected
