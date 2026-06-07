"""PostprocessingResearcher: LLM-driven postprocessing strategy search."""
import json
from pathlib import Path
from typing import Any

import anthropic
import pandas as pd

from mist_autoresearch.base import AbstractResearcher
from mist_autoresearch.stopping import StoppingCriteria

from .evaluator import PostprocessingEvaluator


# ---------------------------------------------------------------------------
# Tool schema
# ---------------------------------------------------------------------------

STRATEGY_TOOL: dict[str, Any] = {
    "name": "submit_strategy",
    "description": (
        "Submit a postprocessing strategy to evaluate. The strategy is an "
        "ordered list of steps applied sequentially to each prediction mask."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "steps": {
                "type": "array",
                "description": "Ordered list of postprocessing steps.",
                "items": {
                    "type": "object",
                    "properties": {
                        "transform": {
                            "type": "string",
                            "description": "Name of the registered transform.",
                        },
                        "apply_to_labels": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": (
                                "Label IDs to apply this step to. "
                                "Use [-1] for all foreground labels."
                            ),
                        },
                        "per_label": {
                            "type": "boolean",
                            "description": (
                                "If True, apply the transform to each label "
                                "independently. If False, apply to the grouped "
                                "binary mask of all specified labels."
                            ),
                        },
                        "kwargs": {
                            "type": "object",
                            "description": "Transform-specific keyword arguments.",
                        },
                    },
                    "required": ["transform", "apply_to_labels", "per_label"],
                },
            },
            "narrative": {
                "type": "string",
                "description": (
                    "Explain your reasoning: what you observed in prior results, "
                    "what this strategy is intended to fix, and what you expect "
                    "to happen."
                ),
            },
        },
        "required": ["steps", "narrative"],
    },
}


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _load_transform_metadata() -> list[dict[str, Any]]:
    from mist.postprocessing.transform_registry import describe_transforms
    return describe_transforms()


# ---------------------------------------------------------------------------
# Researcher
# ---------------------------------------------------------------------------

class PostprocessingResearcher(AbstractResearcher):
    """LLM-driven search for the best postprocessing strategy.

    At each iteration, Claude proposes a postprocessing strategy as structured
    JSON via tool use. The strategy is evaluated by ``PostprocessingEvaluator``
    (which calls ``mist_postprocess``). Results feed back to Claude on the next
    iteration.

    Args:
        config: Path to ``config.json`` from ``mist_analyze``.
        predictions_dir: Directory of baseline NIfTI predictions.
        test_csv: CSV with ``id`` and ``mask`` columns (ground truth paths).
        output_dir: Root directory for run outputs.
        stopping: Stopping criteria.
        model: Anthropic model ID.
        num_workers: Forwarded to the evaluator for parallel postprocessing.
        client: Optional pre-configured ``anthropic.Anthropic`` client
            (useful for testing / custom retry/timeout settings).
    """

    def __init__(
        self,
        config: Path,
        predictions_dir: Path,
        test_csv: Path,
        output_dir: Path,
        stopping: StoppingCriteria,
        model: str = "claude-opus-4-8",
        num_workers: int = 1,
        client: anthropic.Anthropic | None = None,
    ) -> None:
        super().__init__(output_dir=output_dir, stopping=stopping, model=model)
        self.config = Path(config)
        self._client = client or anthropic.Anthropic()
        self._config_data: dict = json.loads(self.config.read_text())
        self.evaluator = PostprocessingEvaluator(
            predictions_dir=Path(predictions_dir),
            test_csv=Path(test_csv),
            config=Path(config),
            num_workers=num_workers,
        )

    # ------------------------------------------------------------------
    # AbstractResearcher interface
    # ------------------------------------------------------------------

    def evaluate(self, strategy: list, iteration_dir: Path) -> pd.DataFrame:
        return self.evaluator.run(strategy, iteration_dir)

    def build_context(
        self,
        baseline_results: pd.DataFrame,
        rank_df: pd.DataFrame | None,
        significance_df: pd.DataFrame | None,
    ) -> dict:
        return {
            "config": self._config_data,
            "transforms": _load_transform_metadata(),
            "baseline_results": baseline_results.to_dict(orient="records"),
            "rank_df": (
                rank_df.to_dict(orient="records") if rank_df is not None else None
            ),
            "significance": (
                significance_df.to_dict() if significance_df is not None else None
            ),
            "history": self.history.iterations,
        }

    def propose(self, context: dict) -> tuple[list, str]:
        """Call Claude with the current context and return (steps, narrative).

        Raises:
            RuntimeError: If the model does not return a tool-use block.
        """
        prompt = self._build_prompt(context)
        response = self._client.messages.create(
            model=self.model,
            max_tokens=4096,
            tools=[STRATEGY_TOOL],
            messages=[{"role": "user", "content": prompt}],
        )

        for block in response.content:
            if block.type == "tool_use" and block.name == "submit_strategy":
                steps = block.input.get("steps", [])
                narrative = block.input.get("narrative", "")
                return steps, narrative

        raise RuntimeError(
            "Model did not return a strategy via tool use. "
            f"Stop reason: {response.stop_reason}."
        )

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_prompt(self, context: dict) -> str:
        cfg = context["config"]
        labels = cfg.get("labels", [])
        final_classes = cfg.get("final_classes", {})
        transforms = context["transforms"]
        history = context["history"]
        rank_df = context["rank_df"]
        significance = context["significance"]

        parts = [
            "You are a medical image postprocessing researcher. Your goal is to "
            "find the best postprocessing strategy to improve segmentation quality "
            "for a 3D medical image segmentation task. You propose one strategy "
            "per iteration using the submit_strategy tool.",
            "",
            "## Dataset",
            f"- Segmentation labels: {labels}",
            f"- Final classes: {json.dumps(final_classes)}",
            "",
            "## Available Transforms",
            json.dumps(transforms, indent=2),
            "",
            "## Baseline Results (No Postprocessing)",
            json.dumps(context["baseline_results"][:5], indent=2),
        ]

        if history:
            parts += ["", "## Strategies Tried So Far"]
            for entry in history:
                parts.append(
                    f"- Iteration {entry['iteration']}: "
                    f"mean_rank={entry['mean_rank']:.2f}, "
                    f"p_vs_baseline={entry.get('p_value_vs_baseline')}, "
                    f"strategy={json.dumps(entry['strategy'])}"
                )
            parts.append("")

        if rank_df is not None:
            parts += [
                "## Current Rankings (lower is better)",
                json.dumps(rank_df, indent=2),
                "",
            ]

        if significance is not None:
            parts += [
                "## Significance Matrix",
                "Entry [A, B] = p-value that strategy A is significantly better "
                "than strategy B (one-sided Wilcoxon). Lower = more significant.",
                json.dumps(significance, indent=2),
                "",
            ]

        parts += [
            "Propose the next postprocessing strategy using the submit_strategy "
            "tool. Avoid repeating strategies already tried. Explain your "
            "reasoning based on the results so far.",
        ]

        return "\n".join(parts)
