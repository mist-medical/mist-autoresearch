"""Tests for mist_autoresearch.postprocessing.evaluator."""

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from mist_autoresearch.postprocessing.evaluator import PostprocessingEvaluator


def _make_evaluator(tmp_path) -> PostprocessingEvaluator:
    return PostprocessingEvaluator(
        predictions_dir=tmp_path / "predictions",
        test_csv=tmp_path / "test.csv",
        config=tmp_path / "config.json",
        num_workers=1,
    )


def _write_results(output_dir: Path, df: pd.DataFrame) -> None:
    """Write a fake postprocess_results.csv as mist_postprocess would."""
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "postprocess_results.csv", index=False)


class TestPostprocessingEvaluator:
    def test_run_writes_strategy_json(self, tmp_path):
        ev = _make_evaluator(tmp_path)
        df = pd.DataFrame({"id": ["p1"], "WT_dice": [0.9]})

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = lambda cmd, check: _write_results(
                Path(cmd[cmd.index("--output") + 1]), df
            )
            ev.run([{"transform": "remove_small_objects"}], tmp_path / "iter_001")

        strategy_path = tmp_path / "iter_001" / "strategy.json"
        assert strategy_path.exists()
        loaded = json.loads(strategy_path.read_text())
        assert loaded == [{"transform": "remove_small_objects"}]

    def test_run_calls_mist_postprocess(self, tmp_path):
        ev = _make_evaluator(tmp_path)
        df = pd.DataFrame({"id": ["p1"], "WT_dice": [0.9]})
        iter_dir = tmp_path / "iter_001"

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = lambda cmd, check: _write_results(iter_dir, df)
            ev.run([], iter_dir)

        args = mock_run.call_args[0][0]
        assert "mist_postprocess" in args
        assert "--base-predictions" in args
        assert "--output" in args
        assert "--paths-csv" in args
        assert "--eval-config" in args

    def test_run_passes_check_true(self, tmp_path):
        ev = _make_evaluator(tmp_path)
        df = pd.DataFrame({"id": ["p1"], "WT_dice": [0.9]})
        iter_dir = tmp_path / "iter_001"

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = lambda cmd, check: _write_results(iter_dir, df)
            ev.run([], iter_dir)

        _, kwargs = mock_run.call_args
        assert kwargs.get("check") is True

    def test_run_returns_results_dataframe(self, tmp_path):
        ev = _make_evaluator(tmp_path)
        expected = pd.DataFrame({"id": ["p1", "p2"], "WT_dice": [0.9, 0.8]})
        iter_dir = tmp_path / "iter_001"

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = lambda cmd, check: _write_results(iter_dir, expected)
            result = ev.run([], iter_dir)

        pd.testing.assert_frame_equal(
            result.reset_index(drop=True), expected.reset_index(drop=True)
        )

    def test_run_raises_if_results_csv_missing(self, tmp_path):
        ev = _make_evaluator(tmp_path)

        with patch("subprocess.run"):  # does NOT write results CSV
            with pytest.raises(FileNotFoundError, match="postprocess_results.csv"):
                ev.run([], tmp_path / "iter_001")

    def test_run_creates_output_dir(self, tmp_path):
        ev = _make_evaluator(tmp_path)
        df = pd.DataFrame({"id": ["p1"], "WT_dice": [0.9]})
        nested = tmp_path / "nested" / "dir" / "iter"

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = lambda cmd, check: _write_results(nested, df)
            ev.run([], nested)

        assert nested.is_dir()

    def test_run_empty_strategy_writes_empty_json(self, tmp_path):
        ev = _make_evaluator(tmp_path)
        df = pd.DataFrame({"id": ["p1"], "WT_dice": [0.9]})
        iter_dir = tmp_path / "baseline"

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = lambda cmd, check: _write_results(iter_dir, df)
            ev.run([], iter_dir)

        strategy = json.loads((iter_dir / "strategy.json").read_text())
        assert strategy == []

    def test_num_workers_forwarded_to_cmd(self, tmp_path):
        ev = PostprocessingEvaluator(
            predictions_dir=tmp_path / "preds",
            test_csv=tmp_path / "test.csv",
            config=tmp_path / "config.json",
            num_workers=4,
        )
        df = pd.DataFrame({"id": ["p1"], "WT_dice": [0.9]})
        iter_dir = tmp_path / "iter_001"

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = lambda cmd, check: _write_results(iter_dir, df)
            ev.run([], iter_dir)

        args = mock_run.call_args[0][0]
        assert "--num-workers-postprocess" in args
        idx = args.index("--num-workers-postprocess")
        assert args[idx + 1] == "4"
