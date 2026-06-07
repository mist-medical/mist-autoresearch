"""Tests for mist_autoresearch.postprocessing.researcher."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from mist_autoresearch.postprocessing.researcher import (
    PostprocessingResearcher,
    STRATEGY_TOOL,
    _load_transform_metadata,
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


def _make_researcher(tmp_path: Path, client=None) -> PostprocessingResearcher:
    config = _make_config(tmp_path)
    mock_client = client or MagicMock()
    return PostprocessingResearcher(
        config=config,
        predictions_dir=tmp_path / "preds",
        test_csv=tmp_path / "test.csv",
        output_dir=tmp_path / "out",
        stopping=StoppingCriteria(max_iterations=1),
        client=mock_client,
    )


def _make_tool_response(steps, narrative):
    block = SimpleNamespace(
        type="tool_use",
        name="submit_strategy",
        input={"steps": steps, "narrative": narrative},
    )
    return SimpleNamespace(content=[block], stop_reason="tool_use")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPostprocessingResearcherInit:

    def test_reads_config_on_init(self, tmp_path):
        r = _make_researcher(tmp_path)
        assert "labels" in r._config_data

    def test_creates_evaluator(self, tmp_path):
        r = _make_researcher(tmp_path)
        assert r.evaluator is not None


class TestPropose:

    def test_returns_steps_and_narrative(self, tmp_path):
        steps = [{"transform": "remove_small_objects", "apply_to_labels": [-1],
                  "per_label": False}]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_tool_response(steps, "reasoning")
        r = _make_researcher(tmp_path, client=mock_client)

        with patch("mist_autoresearch.postprocessing.researcher._load_transform_metadata",
                   return_value=[]):
            result_steps, narrative = r.propose({
                "config": r._config_data,
                "transforms": [],
                "baseline_results": [],
                "rank_df": None,
                "significance": None,
                "history": [],
            })

        assert result_steps == steps
        assert narrative == "reasoning"

    def test_raises_if_no_tool_use_block(self, tmp_path):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = SimpleNamespace(
            content=[SimpleNamespace(type="text", text="sorry")],
            stop_reason="end_turn",
        )
        r = _make_researcher(tmp_path, client=mock_client)

        with pytest.raises(RuntimeError, match="tool use"):
            r.propose({
                "config": r._config_data, "transforms": [],
                "baseline_results": [], "rank_df": None,
                "significance": None, "history": [],
            })

    def test_uses_correct_tool_in_api_call(self, tmp_path):
        steps = []
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_tool_response(steps, "n")
        r = _make_researcher(tmp_path, client=mock_client)

        with patch("mist_autoresearch.postprocessing.researcher._load_transform_metadata",
                   return_value=[]):
            r.propose({"config": r._config_data, "transforms": [],
                       "baseline_results": [], "rank_df": None,
                       "significance": None, "history": []})

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["tools"] == [STRATEGY_TOOL]

    def test_passes_model_to_api(self, tmp_path):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_tool_response([], "n")
        config = _make_config(tmp_path)
        r = PostprocessingResearcher(
            config=config,
            predictions_dir=tmp_path / "preds",
            test_csv=tmp_path / "test.csv",
            output_dir=tmp_path / "out",
            stopping=StoppingCriteria(),
            model="claude-haiku-4-5-20251001",
            client=mock_client,
        )
        with patch("mist_autoresearch.postprocessing.researcher._load_transform_metadata",
                   return_value=[]):
            r.propose({"config": r._config_data, "transforms": [],
                       "baseline_results": [], "rank_df": None,
                       "significance": None, "history": []})
        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["model"] == "claude-haiku-4-5-20251001"


class TestBuildContext:

    def test_context_contains_required_keys(self, tmp_path):
        r = _make_researcher(tmp_path)
        with patch("mist_autoresearch.postprocessing.researcher._load_transform_metadata",
                   return_value=[{"name": "t"}]):
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
        with patch("mist_autoresearch.postprocessing.researcher._load_transform_metadata",
                   return_value=[]):
            ctx = r.build_context(
                pd.DataFrame({"id": ["p1"], "WT_dice": [0.9]}), rank, None
            )
        assert ctx["rank_df"] is not None

    def test_significance_serialised_when_provided(self, tmp_path):
        r = _make_researcher(tmp_path)
        sig = pd.DataFrame({"baseline": {"iteration_001": 0.03}}, dtype=float)
        with patch("mist_autoresearch.postprocessing.researcher._load_transform_metadata",
                   return_value=[]):
            ctx = r.build_context(
                pd.DataFrame({"id": ["p1"], "WT_dice": [0.9]}), None, sig
            )
        assert ctx["significance"] is not None


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
        assert "final_classes" in prompt or "WT" in prompt

    def test_prompt_contains_history_entries(self, tmp_path):
        r = _make_researcher(tmp_path)
        ctx = {
            "config": r._config_data,
            "transforms": [],
            "baseline_results": [],
            "rank_df": [{"strategy": "baseline", "average_rank": 1.0}],
            "significance": None,
            "history": [{"iteration": 1, "mean_rank": 1.5,
                         "p_value_vs_baseline": 0.03, "strategy": []}],
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


class TestEvaluateDelegate:

    def test_evaluate_calls_evaluator_run(self, tmp_path):
        r = _make_researcher(tmp_path)
        expected = pd.DataFrame({"id": ["p1"], "WT_dice": [0.9]})
        with patch.object(r.evaluator, "run", return_value=expected) as mock_run:
            result = r.evaluate([], tmp_path / "iter")
        mock_run.assert_called_once_with([], tmp_path / "iter")
        pd.testing.assert_frame_equal(result, expected)


class TestLoadTransformMetadata:

    def test_returns_list_of_dicts(self):
        result = _load_transform_metadata()
        assert isinstance(result, list)
        assert all(isinstance(entry, dict) for entry in result)

    def test_contains_known_transform(self):
        result = _load_transform_metadata()
        names = [entry.get("name") for entry in result]
        assert "remove_small_objects" in names
