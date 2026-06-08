"""Tests for mist_autoresearch.cli."""

import argparse
from unittest.mock import MagicMock, patch

import pytest

from mist_autoresearch import cli
from mist_autoresearch.stopping import StoppingCriteria


# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------


class TestParser:
    def _parse(self, args):
        return cli._build_parser().parse_args(args)

    def test_postprocessing_subcommand_required_args(self, tmp_path):
        ns = self._parse(
            [
                "postprocessing",
                "--config",
                str(tmp_path / "config.json"),
                "--predictions",
                str(tmp_path / "preds"),
                "--test-csv",
                str(tmp_path / "test.csv"),
                "--output",
                str(tmp_path / "out"),
            ]
        )
        assert ns.command == "postprocessing"
        assert ns.config == str(tmp_path / "config.json")

    def test_postprocessing_defaults(self, tmp_path):
        ns = self._parse(
            [
                "postprocessing",
                "--config",
                "c.json",
                "--predictions",
                "preds/",
                "--test-csv",
                "t.csv",
                "--output",
                "out/",
            ]
        )
        assert ns.max_iterations == 50
        assert ns.patience == 10
        assert ns.alpha == 0.05
        assert ns.min_iterations == 5
        assert ns.min_patients_for_significance == 15
        assert ns.model is None
        assert ns.num_workers == 1
        assert ns.additional_prompt is None

    def test_postprocessing_custom_stopping(self, tmp_path):
        ns = self._parse(
            [
                "postprocessing",
                "--config",
                "c.json",
                "--predictions",
                "p/",
                "--test-csv",
                "t.csv",
                "--output",
                "o/",
                "--max-iterations",
                "20",
                "--patience",
                "5",
                "--alpha",
                "0.10",
                "--min-iterations",
                "3",
                "--min-patients-for-significance",
                "10",
            ]
        )
        assert ns.max_iterations == 20
        assert ns.patience == 5
        assert ns.alpha == pytest.approx(0.10)
        assert ns.min_iterations == 3
        assert ns.min_patients_for_significance == 10

    def test_missing_subcommand_exits(self):
        with pytest.raises(SystemExit):
            cli._build_parser().parse_args([])

    def test_missing_required_arg_exits(self):
        with pytest.raises(SystemExit):
            cli._build_parser().parse_args(
                [
                    "postprocessing",
                    "--predictions",
                    "p/",
                    "--test-csv",
                    "t.csv",
                    "--output",
                    "o/",
                    # --config missing
                ]
            )


# ---------------------------------------------------------------------------
# _build_stopping
# ---------------------------------------------------------------------------


class TestBuildStopping:
    def test_builds_stopping_criteria(self):
        ns = argparse.Namespace(
            max_iterations=20,
            patience=5,
            alpha=0.10,
            min_iterations=3,
            min_patients_for_significance=10,
        )
        sc = cli._build_stopping(ns)
        assert isinstance(sc, StoppingCriteria)
        assert sc.max_iterations == 20
        assert sc.patience == 5
        assert sc.alpha == pytest.approx(0.10)
        assert sc.min_iterations == 3
        assert sc.min_patients_for_significance == 10


# ---------------------------------------------------------------------------
# _run_postprocessing
# ---------------------------------------------------------------------------


class TestRunPostprocessing:
    def test_creates_researcher_and_calls_run(self, tmp_path):
        ns = argparse.Namespace(
            config=str(tmp_path / "config.json"),
            predictions=str(tmp_path / "preds"),
            test_csv=str(tmp_path / "test.csv"),
            output=str(tmp_path / "out"),
            max_iterations=2,
            patience=1,
            alpha=0.05,
            min_iterations=1,
            min_patients_for_significance=5,
            model="claude-opus-4-8",
            num_workers=1,
            additional_prompt=None,
        )
        with patch("mist_autoresearch.cli.PostprocessingResearcher") as MockResearcher:
            mock_instance = MagicMock()
            MockResearcher.return_value = mock_instance
            cli._run_postprocessing(ns)
        mock_instance.run.assert_called_once()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


class TestMain:
    def test_main_routes_postprocessing(self, tmp_path, monkeypatch):
        called = {}
        monkeypatch.setattr(
            cli, "_run_postprocessing", lambda ns: called.__setitem__("ran", True)
        )
        cli.main(
            [
                "postprocessing",
                "--config",
                "c.json",
                "--predictions",
                "p/",
                "--test-csv",
                "t.csv",
                "--output",
                "o/",
            ]
        )
        assert called.get("ran") is True
