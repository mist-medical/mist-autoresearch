"""Abstract base class for all autoresearch loops."""

import json
import math
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd
from mist.evaluation.ranking import compute_pairwise_significance, rank_results

from .history import History
from .notebook import ResearchNotebook
from .stopping import StoppingCriteria


class AbstractResearcher(ABC):
    """Sequential LLM-driven research loop.

    Subclasses implement ``propose()``, ``evaluate()``, and ``build_context()``
    for a specific research domain (e.g. postprocessing, training config).
    The loop logic — ranking, notebook writing, stopping, history — lives here.

    Args:
        output_dir: Root directory for all run outputs.
        stopping: Stopping criteria parameters.
    """

    def __init__(
        self,
        output_dir: Path,
        stopping: StoppingCriteria,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.stopping = stopping
        self.history = History(self.output_dir / "history.json")

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def propose(self, context: dict) -> tuple[list, str]:
        """Ask the LLM for the next strategy to try.

        Args:
            context: Dict built by ``build_context()``.

        Returns:
            (strategy, narrative) where strategy is a list of steps accepted
            by the evaluator and narrative is the agent's explanation.
        """

    @abstractmethod
    def evaluate(self, strategy: list, iteration_dir: Path) -> pd.DataFrame:
        """Apply a strategy and return per-patient evaluation results.

        Passing an empty list ``[]`` must produce baseline results
        (no transformation applied).

        Args:
            strategy: List of strategy steps. Empty list = no postprocessing.
            iteration_dir: Directory where strategy JSON and results are written.

        Returns:
            DataFrame with an ``id`` column and one column per metric.
        """

    @abstractmethod
    def build_context(
        self,
        baseline_results: pd.DataFrame,
        rank_df: pd.DataFrame | None,
        significance_df: pd.DataFrame | None,
    ) -> dict:
        """Build the context dict passed to ``propose()``.

        Args:
            baseline_results: Results from the baseline (no-strategy) evaluation.
            rank_df: Current cumulative ranking DataFrame, or None before iter 1.
            significance_df: Pairwise significance matrix, or None if unavailable.

        Returns:
            A dict containing any information the LLM needs to make a proposal.
        """

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> list | None:
        """Execute the research loop and return the best strategy.

        Returns:
            The best strategy (list of steps), or ``[]`` if no postprocessing
            strategy beat the baseline.  Returns ``None`` if the loop produced
            no iterations at all.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        notebook = ResearchNotebook(self.output_dir / "research_notebook.md")
        notebook.write_header()

        # Baseline: evaluate with empty strategy (no transforms applied).
        baseline_dir = self.output_dir / "baseline"
        baseline_dir.mkdir(parents=True, exist_ok=True)
        baseline_results = self.evaluate([], baseline_dir)
        notebook.write_baseline(baseline_results)

        all_results: list[pd.DataFrame] = [baseline_results]
        all_names: list[str] = ["baseline"]
        all_strategies: list[list | None] = [None]

        best_overall_rank = float("inf")
        best_strategy_name = "baseline"
        iterations_since_improvement = 0
        rank_df: pd.DataFrame | None = None
        significance_df: pd.DataFrame | None = None

        for i in range(1, self.stopping.max_iterations + 1):
            context = self.build_context(baseline_results, rank_df, significance_df)
            strategy, narrative = self.propose(context)

            iter_dir = self.output_dir / f"iteration_{i:03d}"
            iter_dir.mkdir(parents=True, exist_ok=True)

            results = self.evaluate(strategy, iter_dir)
            all_results.append(results)
            all_names.append(f"iteration_{i:03d}")
            all_strategies.append(strategy)

            rank_df, significance_df = self._rank_and_significance(
                all_results, all_names, len(results)
            )

            iter_name = f"iteration_{i:03d}"
            iter_rank = _get_mean_rank(rank_df, iter_name)

            # Track global best across all strategies (including baseline).
            global_best_rank = float(rank_df.iloc[0]["average_rank"])
            global_best_name = str(rank_df.iloc[0]["strategy"])

            if global_best_rank < best_overall_rank:
                best_overall_rank = global_best_rank
                best_strategy_name = global_best_name
                iterations_since_improvement = 0
                if global_best_name == iter_name:
                    self.history.update_best(i)
            else:
                iterations_since_improvement += 1

            # p-value used for stopping: best strategy vs baseline.
            best_p_vs_baseline = _get_p_vs_baseline(significance_df, best_strategy_name)

            # p-value for the notebook: current iteration vs baseline.
            iter_p_vs_baseline = _get_p_vs_baseline(significance_df, iter_name)

            is_best = global_best_name == iter_name
            self.history.add_iteration(
                iteration=i,
                strategy=strategy,
                narrative=narrative,
                results_csv=str(iter_dir / "postprocess_results.csv"),
                mean_rank=iter_rank,
                p_value_vs_baseline=iter_p_vs_baseline,
            )
            notebook.write_iteration(
                i,
                strategy,
                narrative,
                results,
                iter_rank,
                iter_p_vs_baseline,
                is_best,
            )

            should_stop, reason = self.stopping.should_stop(
                i,
                iterations_since_improvement,
                len(results),
                best_p_vs_baseline,
            )
            if should_stop:
                self.history.set_stopped_reason(reason)
                break

        # Resolve the best strategy from the final rankings.
        best_idx = all_names.index(best_strategy_name)
        best_strategy = all_strategies[best_idx]
        self._write_summary(best_strategy_name, best_strategy, best_overall_rank)
        return best_strategy

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _rank_and_significance(
        self,
        all_results: list[pd.DataFrame],
        all_names: list[str],
        n_patients: int,
    ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        rank_df, _ = rank_results(results=all_results, names=all_names)
        rank_df.to_csv(self.output_dir / "rankings.csv", index=False)

        sig_df: pd.DataFrame | None = None
        if (
            len(all_results) >= 2
            and n_patients >= self.stopping.min_patients_for_significance
        ):
            sig_df = compute_pairwise_significance(results=all_results, names=all_names)
            sig_df.to_csv(self.output_dir / "significance.csv")

        return rank_df, sig_df

    def _write_summary(
        self,
        best_strategy_name: str,
        best_strategy: list | None,
        best_overall_rank: float,
    ) -> None:
        summary = {
            "best_strategy_name": best_strategy_name,
            "best_overall_rank": (
                None if best_overall_rank == float("inf") else best_overall_rank
            ),
            "best_strategy": best_strategy,
        }
        (self.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))


# ---------------------------------------------------------------------------
# Module-level helpers (testable without instantiating a researcher)
# ---------------------------------------------------------------------------


def _get_mean_rank(rank_df: pd.DataFrame, name: str) -> float:
    """Return the average_rank for a named strategy, or inf if not found."""
    row = rank_df[rank_df["strategy"] == name]
    if row.empty:
        return float("inf")
    return float(row["average_rank"].iloc[0])


def _get_p_vs_baseline(sig_df: pd.DataFrame | None, name: str) -> float | None:
    """Return p-value for strategy *name* being better than baseline.

    Returns None if the significance matrix is unavailable, the strategy is
    not in the matrix, or the strategy IS baseline.
    """
    if sig_df is None:
        return None
    if name == "baseline" or name not in sig_df.index:
        return None
    if "baseline" not in sig_df.columns:
        return None
    val = sig_df.loc[name, "baseline"]
    if math.isnan(float(val)):
        return None
    return float(val)
