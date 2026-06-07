"""Research notebook writer for the autoresearch loop."""
import json
from pathlib import Path

import pandas as pd


class ResearchNotebook:
    """Appends human-readable iteration summaries to a Markdown file.

    The notebook records what the agent tried at each iteration, the reasoning
    behind each strategy, and the resulting evaluation metrics — providing an
    auditable trail of the full research run.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def write_header(self, title: str = "Autoresearch Notebook") -> None:
        """Write the notebook header (overwrites any existing file)."""
        lines = [f"# {title}", "", "---", ""]
        self.path.write_text("\n".join(lines))

    def write_baseline(self, results: pd.DataFrame) -> None:
        """Append the baseline (no postprocessing) results section."""
        metric_cols = [c for c in results.columns if c != "id"]
        lines = [
            "## Baseline (No Postprocessing)",
            "",
            "| Metric | Mean |",
            "|--------|------|",
        ]
        for col in metric_cols:
            lines.append(f"| {col} | {results[col].mean():.4f} |")
        lines += ["", "---", ""]
        self._append("\n".join(lines))

    def write_iteration(
        self,
        iteration: int,
        strategy: list,
        narrative: str,
        results: pd.DataFrame,
        mean_rank: float,
        p_value: float | None,
        is_best: bool,
    ) -> None:
        """Append one iteration's results to the notebook."""
        metric_cols = [c for c in results.columns if c != "id"]
        lines = [
            f"## Iteration {iteration}",
            "",
            f"**Narrative:** {narrative}",
            "",
            "**Strategy:**",
            "```json",
            json.dumps(strategy, indent=2),
            "```",
            "",
            "**Results:**",
            "",
            "| Metric | Mean |",
            "|--------|------|",
        ]
        for col in metric_cols:
            lines.append(f"| {col} | {results[col].mean():.4f} |")
        lines += [""]
        lines.append(f"**Mean rank:** {mean_rank:.2f}")
        if p_value is not None:
            lines.append(f"**p-value vs baseline:** {p_value:.4f}")
        if is_best:
            lines.append("**New best!**")
        lines += ["", "---", ""]
        self._append("\n".join(lines))

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _append(self, text: str) -> None:
        with open(self.path, "a") as f:
            f.write(text)
