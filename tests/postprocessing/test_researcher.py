"""Tests for mist_autoresearch.postprocessing.researcher."""

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

from mist_autoresearch.postprocessing.researcher import (
    PostprocessingResearcher,
    _load_transform_metadata,
    _parse_strategy_response,
)
from mist_autoresearch.stopping import StoppingCriteria


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(tmp_path: Path) -> Path:
    cfg = {
        "labels": [1, 2, 3],
        "final_classes": {"WT": [1, 2, 3], "TC": [1, 3], "ET": [3]},
    }
    p = tmp_path / "config.json"
    p.write_text(json.dumps(cfg))
    return p


def _make_researcher(tmp_path: Path, model=None) -> PostprocessingResearcher:
    config = _make_config(tmp_path)
    return PostprocessingResearcher(
        config=config,
        predictions_dir=tmp_path / "preds",
        test_csv=tmp_path / "test.csv",
        output_dir=tmp_path / "out",
        stopping=StoppingCriteria(max_iterations=1),
        model=model,
    )


def _fake_run(stdout: str):
    """Return a mock subprocess.CompletedProcess with the given stdout."""
    return SimpleNamespace(stdout=stdout, returncode=0)


# ---------------------------------------------------------------------------
# _parse_strategy_response
# ---------------------------------------------------------------------------


class TestParseStrategyResponse:
    def test_parses_fenced_json_block(self):
        text = (
            "Here is my proposal:\n"
            "```json\n"
            '{"steps": [{"transform": "remove_small_objects", '
            '"apply_to_labels": [-1], "per_label": false}], '
            '"narrative": "reasoning"}\n'
            "```"
        )
        steps, narrative = _parse_strategy_response(text)
        assert steps[0]["transform"] == "remove_small_objects"
        assert narrative == "reasoning"

    def test_parses_bare_json_object(self):
        text = '{"steps": [], "narrative": "no-op"}'
        steps, narrative = _parse_strategy_response(text)
        assert steps == []
        assert narrative == "no-op"

    def test_raises_when_no_json_found(self):
        with pytest.raises(RuntimeError, match="Could not find"):
            _parse_strategy_response("Sorry, I cannot propose a strategy.")

    def test_raises_on_invalid_json(self):
        with pytest.raises(RuntimeError, match="Failed to parse"):
            _parse_strategy_response("```json\n{invalid json}\n```")

    def test_raises_when_steps_key_missing(self):
        text = '{"narrative": "oops"}'
        with pytest.raises(RuntimeError, match="missing required 'steps'"):
            _parse_strategy_response(text)

    def test_narrative_defaults_to_empty_string(self):
        text = '{"steps": []}'
        _, narrative = _parse_strategy_response(text)
        assert narrative == ""

    def test_prefers_fenced_block_over_bare_json(self):
        text = (
            'Some text {"steps": [], "narrative": "bare"} here.\n'
            "```json\n"
            '{"steps": [{"transform": "fill_holes_with_label", '
            '"apply_to_labels": [1], "per_label": true}], "narrative": "fenced"}\n'
            "```"
        )
        steps, narrative = _parse_strategy_response(text)
        assert narrative == "fenced"


# ---------------------------------------------------------------------------
# PostprocessingResearcher init
# ---------------------------------------------------------------------------


class TestPostprocessingResearcherInit:
    def test_reads_config_on_init(self, tmp_path):
        r = _make_researcher(tmp_path)
        assert "labels" in r._config_data

    def test_creates_evaluator(self, tmp_path):
        r = _make_researcher(tmp_path)
        assert r.evaluator is not None

    def test_model_none_by_default(self, tmp_path):
        r = _make_researcher(tmp_path)
        assert r._model is None


# ---------------------------------------------------------------------------
# propose()
# ---------------------------------------------------------------------------


class TestPropose:
    def _valid_response(self, steps, narrative):
        payload = json.dumps({"steps": steps, "narrative": narrative})
        return f"```json\n{payload}\n```"

    def test_returns_steps_and_narrative(self, tmp_path):
        r = _make_researcher(tmp_path)
        steps = [
            {
                "transform": "remove_small_objects",
                "apply_to_labels": [-1],
                "per_label": False,
            }
        ]
        response_text = self._valid_response(steps, "reasoning")

        with (
            patch("subprocess.run", return_value=_fake_run(response_text)),
            patch(
                "mist_autoresearch.postprocessing.researcher._load_transform_metadata",
                return_value=[],
            ),
        ):
            result_steps, narrative = r.propose(
                {
                    "config": r._config_data,
                    "transforms": [],
                    "baseline_results": [],
                    "rank_df": None,
                    "significance": None,
                    "history": [],
                }
            )

        assert result_steps == steps
        assert narrative == "reasoning"

    def test_raises_on_unparseable_response(self, tmp_path):
        r = _make_researcher(tmp_path)
        with (
            patch("subprocess.run", return_value=_fake_run("no json here")),
            patch(
                "mist_autoresearch.postprocessing.researcher._load_transform_metadata",
                return_value=[],
            ),
        ):
            with pytest.raises(RuntimeError, match="Could not find"):
                r.propose(
                    {
                        "config": r._config_data,
                        "transforms": [],
                        "baseline_results": [],
                        "rank_df": None,
                        "significance": None,
                        "history": [],
                    }
                )

    def test_cmd_without_model(self, tmp_path):
        r = _make_researcher(tmp_path, model=None)
        response_text = self._valid_response([], "n")

        with (
            patch("subprocess.run", return_value=_fake_run(response_text)) as mock_run,
            patch(
                "mist_autoresearch.postprocessing.researcher._load_transform_metadata",
                return_value=[],
            ),
        ):
            r.propose(
                {
                    "config": r._config_data,
                    "transforms": [],
                    "baseline_results": [],
                    "rank_df": None,
                    "significance": None,
                    "history": [],
                }
            )

        cmd = mock_run.call_args[0][0]
        assert "--model" not in cmd

    def test_cmd_with_model(self, tmp_path):
        r = _make_researcher(tmp_path, model="claude-opus-4-8")
        response_text = self._valid_response([], "n")

        with (
            patch("subprocess.run", return_value=_fake_run(response_text)) as mock_run,
            patch(
                "mist_autoresearch.postprocessing.researcher._load_transform_metadata",
                return_value=[],
            ),
        ):
            r.propose(
                {
                    "config": r._config_data,
                    "transforms": [],
                    "baseline_results": [],
                    "rank_df": None,
                    "significance": None,
                    "history": [],
                }
            )

        cmd = mock_run.call_args[0][0]
        assert "--model" in cmd
        assert "claude-opus-4-8" in cmd

    def test_subprocess_called_with_check_true(self, tmp_path):
        r = _make_researcher(tmp_path)
        response_text = self._valid_response([], "n")

        with (
            patch("subprocess.run", return_value=_fake_run(response_text)) as mock_run,
            patch(
                "mist_autoresearch.postprocessing.researcher._load_transform_metadata",
                return_value=[],
            ),
        ):
            r.propose(
                {
                    "config": r._config_data,
                    "transforms": [],
                    "baseline_results": [],
                    "rank_df": None,
                    "significance": None,
                    "history": [],
                }
            )

        _, kwargs = mock_run.call_args
        assert kwargs.get("check") is True

    def test_propagates_subprocess_error(self, tmp_path):
        r = _make_researcher(tmp_path)
        with (
            patch(
                "subprocess.run", side_effect=subprocess.CalledProcessError(1, "claude")
            ),
            patch(
                "mist_autoresearch.postprocessing.researcher._load_transform_metadata",
                return_value=[],
            ),
        ):
            with pytest.raises(subprocess.CalledProcessError):
                r.propose(
                    {
                        "config": r._config_data,
                        "transforms": [],
                        "baseline_results": [],
                        "rank_df": None,
                        "significance": None,
                        "history": [],
                    }
                )


# ---------------------------------------------------------------------------
# build_context
# ---------------------------------------------------------------------------


class TestBuildContext:
    def test_context_contains_required_keys(self, tmp_path):
        r = _make_researcher(tmp_path)
        with patch(
            "mist_autoresearch.postprocessing.researcher._load_transform_metadata",
            return_value=[{"name": "t"}],
        ):
            ctx = r.build_context(
                pd.DataFrame({"id": ["p1"], "WT_dice": [0.9]}), None, None
            )
        assert "config" in ctx
        assert "transforms" in ctx
        assert "baseline_results" in ctx
        assert "rank_df" in ctx
        assert "significance" in ctx
        assert "history" in ctx

    def test_rank_df_serialised_when_provided(self, tmp_path):
        r = _make_researcher(tmp_path)
        rank = pd.DataFrame({"strategy": ["baseline"], "average_rank": [1.0]})
        with patch(
            "mist_autoresearch.postprocessing.researcher._load_transform_metadata",
            return_value=[],
        ):
            ctx = r.build_context(
                pd.DataFrame({"id": ["p1"], "WT_dice": [0.9]}), rank, None
            )
        assert ctx["rank_df"] is not None

    def test_significance_serialised_when_provided(self, tmp_path):
        r = _make_researcher(tmp_path)
        sig = pd.DataFrame({"baseline": {"iteration_001": 0.03}}, dtype=float)
        with patch(
            "mist_autoresearch.postprocessing.researcher._load_transform_metadata",
            return_value=[],
        ):
            ctx = r.build_context(
                pd.DataFrame({"id": ["p1"], "WT_dice": [0.9]}), None, sig
            )
        assert ctx["significance"] is not None


# ---------------------------------------------------------------------------
# _build_prompt
# ---------------------------------------------------------------------------


class TestBuildPrompt:
    def test_prompt_contains_dataset_info(self, tmp_path):
        r = _make_researcher(tmp_path)
        ctx = {
            "config": r._config_data,
            "transforms": [],
            "baseline_results": [],
            "rank_df": None,
            "significance": None,
            "history": [],
        }
        prompt = r._build_prompt(ctx)
        assert "labels" in prompt
        assert "WT" in prompt

    def test_prompt_contains_json_instruction(self, tmp_path):
        r = _make_researcher(tmp_path)
        ctx = {
            "config": r._config_data,
            "transforms": [],
            "baseline_results": [],
            "rank_df": None,
            "significance": None,
            "history": [],
        }
        prompt = r._build_prompt(ctx)
        assert "```json" in prompt
        assert "steps" in prompt
        assert "narrative" in prompt

    def test_prompt_contains_history_entries(self, tmp_path):
        r = _make_researcher(tmp_path)
        ctx = {
            "config": r._config_data,
            "transforms": [],
            "baseline_results": [],
            "rank_df": [{"strategy": "baseline", "average_rank": 1.0}],
            "significance": None,
            "history": [
                {
                    "iteration": 1,
                    "mean_rank": 1.5,
                    "p_value_vs_baseline": 0.03,
                    "strategy": [],
                }
            ],
        }
        prompt = r._build_prompt(ctx)
        assert "Iteration 1" in prompt

    def test_prompt_includes_ranking_when_provided(self, tmp_path):
        r = _make_researcher(tmp_path)
        ctx = {
            "config": r._config_data,
            "transforms": [],
            "baseline_results": [],
            "rank_df": [{"strategy": "baseline", "average_rank": 1.0}],
            "significance": None,
            "history": [],
        }
        prompt = r._build_prompt(ctx)
        assert "Rankings" in prompt

    def test_prompt_includes_significance_when_provided(self, tmp_path):
        r = _make_researcher(tmp_path)
        ctx = {
            "config": r._config_data,
            "transforms": [],
            "baseline_results": [],
            "rank_df": None,
            "significance": {"baseline": {"iteration_001": 0.03}},
            "history": [],
        }
        prompt = r._build_prompt(ctx)
        assert "Significance" in prompt


# ---------------------------------------------------------------------------
# evaluate delegate
# ---------------------------------------------------------------------------


class TestEvaluateDelegate:
    def test_evaluate_calls_evaluator_run(self, tmp_path):
        r = _make_researcher(tmp_path)
        expected = pd.DataFrame({"id": ["p1"], "WT_dice": [0.9]})
        with patch.object(r.evaluator, "run", return_value=expected) as mock_run:
            result = r.evaluate([], tmp_path / "iter")
        mock_run.assert_called_once_with([], tmp_path / "iter")
        pd.testing.assert_frame_equal(result, expected)


# ---------------------------------------------------------------------------
# _load_transform_metadata
# ---------------------------------------------------------------------------


class TestLoadTransformMetadata:
    def test_returns_list_of_dicts(self):
        result = _load_transform_metadata()
        assert isinstance(result, list)
        assert all(isinstance(entry, dict) for entry in result)

    def test_contains_known_transform(self):
        result = _load_transform_metadata()
        names = [entry.get("name") for entry in result]
        assert "remove_small_objects" in names
