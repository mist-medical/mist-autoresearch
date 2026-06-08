"""PostprocessingResearcher: LLM-driven postprocessing strategy search."""

import json
import re
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

from mist_autoresearch.base import AbstractResearcher
from mist_autoresearch.stopping import StoppingCriteria

from .evaluator import PostprocessingEvaluator


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _load_transform_metadata() -> list[dict[str, Any]]:
    from mist.postprocessing.transform_registry import describe_transforms

    return describe_transforms()


def _parse_strategy_response(text: str) -> tuple[list, str]:
    """Extract (steps, narrative) from a claude -p response.

    Looks for a JSON code block first, then falls back to a bare JSON object.

    Raises:
        RuntimeError: If no valid JSON object containing ``steps`` is found.
    """
    # Prefer a fenced JSON block.
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if not match:
        match = re.search(r"(\{.*\})", text, re.DOTALL)
    if not match:
        raise RuntimeError(
            f"Could not find a JSON strategy in the model response:\n{text[:400]}"
        )
    try:
        data = json.loads(match.group(1))
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Failed to parse strategy JSON: {exc}\nRaw text:\n{text[:400]}"
        ) from exc

    if "steps" not in data:
        raise RuntimeError(f"Parsed JSON is missing required 'steps' key: {data}")
    return data.get("steps", []), data.get("narrative", "")


# ---------------------------------------------------------------------------
# Researcher
# ---------------------------------------------------------------------------


class PostprocessingResearcher(AbstractResearcher):
    """LLM-driven search for the best postprocessing strategy.

    Proposals are made by calling ``claude -p`` (Claude Code CLI) so no
    separate Anthropic API key or billing account is required — it runs on
    the active Claude Code session.

    Args:
        config: Path to ``config.json`` from ``mist_analyze``.
        predictions_dir: Directory of baseline NIfTI predictions.
        test_csv: CSV with ``id`` and ``mask`` columns (ground truth paths).
        output_dir: Root directory for run outputs.
        stopping: Stopping criteria.
        model: Model name forwarded to ``claude --model``. Pass ``None`` to
            use Claude Code's default model.
        num_workers: Forwarded to the evaluator for parallel postprocessing.
        additional_prompt: Optional path to a Markdown file whose contents are
            injected into every proposal prompt under "## Additional Context".
            Use it to share dataset-specific knowledge, evaluation criteria,
            hypotheses, or transform suggestions with the agent.
    """

    def __init__(
        self,
        config: Path,
        predictions_dir: Path,
        test_csv: Path,
        output_dir: Path,
        stopping: StoppingCriteria,
        model: str | None = None,
        num_workers: int = 1,
        additional_prompt: Path | str | None = None,
    ) -> None:
        super().__init__(output_dir=output_dir, stopping=stopping)
        self.config = Path(config)
        self._model = model
        self._additional_prompt: str | None = (
            Path(additional_prompt).read_text() if additional_prompt else None
        )
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
        """Call ``claude -p`` and return (steps, narrative).

        Raises:
            RuntimeError: If the response cannot be parsed as a valid strategy.
            subprocess.CalledProcessError: If the ``claude`` CLI exits non-zero.
        """
        prompt = self._build_prompt(context)
        cmd = ["claude", "-p", prompt]
        if self._model:
            cmd = ["claude", "-p", "--model", self._model, prompt]

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return _parse_strategy_response(result.stdout)

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
            "for a 3D medical image segmentation task.",
            "",
            "## Dataset",
            f"- Segmentation labels: {labels}",
            f"- Final classes: {json.dumps(final_classes)}",
        ]

        if self._additional_prompt:
            parts += [
                "",
                "## Additional Context",
                self._additional_prompt,
            ]

        parts += [
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
            "Propose the next postprocessing strategy to try. Avoid repeating "
            "strategies already tried. Explain your reasoning based on the "
            "results so far.",
            "",
            "Respond with ONLY a JSON object in a ```json code block. "
            "The object must have exactly two keys:",
            '  "steps": a list of strategy step objects (may be empty [])',
            '  "narrative": a string explaining your reasoning',
            "",
            "Example:",
            "```json",
            json.dumps(
                {
                    "steps": [
                        {
                            "transform": "remove_small_objects",
                            "apply_to_labels": [-1],
                            "per_label": False,
                            "kwargs": {"small_object_threshold": 100},
                        }
                    ],
                    "narrative": "Removing small spurious components to reduce false positives.",
                },
                indent=2,
            ),
            "```",
        ]

        return "\n".join(parts)
