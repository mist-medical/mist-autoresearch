"""Abstract base class for all autoresearch loops."""

import json
import math
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd
from mist.evaluation.ranking import (
    SUMMARY_ROW_IDS,
    compute_pairwise_significance,
    rank_results,
)

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

        If a previous run was interrupted, calling ``run()`` again on the same
        ``output_dir`` resumes from where it left off: completed iterations are
        loaded from disk and the loop continues from the next iteration number.

        Returns:
            The best strategy (list of steps), or ``None`` if baseline
            (no postprocessing) was best.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        notebook = ResearchNotebook(self.output_dir / "research_notebook.md")

        resuming = self.history.n_iterations > 0

        if resuming:
            baseline_results, all_results, all_names, all_strategies = (
                self._recover_state()
            )
            rank_df, significance_df = self._rank_and_significance(
                all_results, all_names, _count_patients(baseline_results)
            )
            best_strategy_name, iterations_since_improvement = self._recompute_tracking(
                rank_df
            )
            start_iteration = self.history.n_iterations + 1
        else:
            notebook.write_header()

            baseline_dir = self.output_dir / "baseline"
            baseline_dir.mkdir(parents=True, exist_ok=True)
            baseline_results = self.evaluate([], baseline_dir)
            notebook.write_baseline(baseline_results)

            all_results: list[pd.DataFrame] = [baseline_results]
            all_names: list[str] = ["baseline"]
            all_strategies: list[list | None] = [None]

            best_strategy_name = "baseline"
            iterations_since_improvement = 0
            rank_df = None
            significance_df = None
            start_iteration = 1

        for i in range(start_iteration, self.stopping.max_iterations + 1):
            context = self.build_context(baseline_results, rank_df, significance_df)
            strategy, narrative = self.propose(context)

            iter_dir = self.output_dir / f"iteration_{i:03d}"
            iter_dir.mkdir(parents=True, exist_ok=True)

            results = self.evaluate(strategy, iter_dir)
            all_results.append(results)
            all_names.append(f"iteration_{i:03d}")
            all_strategies.append(strategy)

            n_patients = _count_patients(results)
            rank_df, significance_df = self._rank_and_significance(
                all_results, all_names, n_patients
            )

            iter_name = f"iteration_{i:03d}"
            iter_rank = _get_mean_rank(rank_df, iter_name)

            # Track the global best across all strategies (including baseline).
            # Average ranks are pool-relative, so adding a strategy re-ranks the
            # whole pool and can reorder two previously-close strategies. Only
            # this iteration reaching the top counts as an improvement; a
            # reshuffle among strategies we already had does not reset patience,
            # but it still has to be recorded so history stays in sync with
            # rankings.csv and summary.json.
            global_best_name = str(rank_df.iloc[0]["strategy"])

            if global_best_name == iter_name:
                best_strategy_name = global_best_name
                iterations_since_improvement = 0
                self.history.update_best(i)
            else:
                iterations_since_improvement += 1
                if global_best_name != best_strategy_name:
                    best_strategy_name = global_best_name
                    self.history.update_best(_iteration_number(global_best_name))

            self.history.set_iterations_since_improvement(iterations_since_improvement)

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
                n_patients,
                best_p_vs_baseline,
            )
            if should_stop:
                self.history.set_stopped_reason(reason)
                break

        # Resolve the best strategy from the final rankings.
        best_idx = all_names.index(best_strategy_name)
        best_strategy = all_strategies[best_idx]
        best_overall_rank = (
            _get_mean_rank(rank_df, best_strategy_name)
            if rank_df is not None
            else float("inf")
        )
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

    def _recover_state(
        self,
    ) -> tuple[pd.DataFrame, list[pd.DataFrame], list[str], list[list | None]]:
        """Load results from all completed iterations for a resume.

        Raises:
            FileNotFoundError: If the baseline or any iteration results CSV is
                missing from disk.
        """
        baseline_csv = self.output_dir / "baseline" / "postprocess_results.csv"
        if not baseline_csv.exists():
            raise FileNotFoundError(
                f"Cannot resume: baseline results not found at {baseline_csv}"
            )
        baseline_results = pd.read_csv(baseline_csv)

        all_results: list[pd.DataFrame] = [baseline_results]
        all_names: list[str] = ["baseline"]
        all_strategies: list[list | None] = [None]

        for entry in self.history.iterations:
            i = entry["iteration"]
            iter_dir = self.output_dir / f"iteration_{i:03d}"
            results_csv = iter_dir / "postprocess_results.csv"
            if not results_csv.exists():
                raise FileNotFoundError(
                    f"Cannot resume: results missing for iteration {i} at {results_csv}"
                )
            all_results.append(pd.read_csv(results_csv))
            all_names.append(f"iteration_{i:03d}")
            all_strategies.append(json.loads((iter_dir / "strategy.json").read_text()))

        return baseline_results, all_results, all_names, all_strategies

    def _recompute_tracking(self, rank_df: pd.DataFrame) -> tuple[str, int]:
        """Recompute best-tracking state from rankings and history after a resume.

        The patience counter is read back from history, which records it after
        every iteration. Runs recorded before that field existed fall back to
        deriving it from ``best_iteration``.

        Returns:
            (best_strategy_name, iterations_since_improvement)
        """
        best_strategy_name = str(rank_df.iloc[0]["strategy"])

        iterations_since_improvement = self.history.iterations_since_improvement
        if iterations_since_improvement is None:
            best_iter = self.history.best_iteration
            n = self.history.n_iterations
            iterations_since_improvement = n if best_iter is None else n - best_iter

        return best_strategy_name, iterations_since_improvement

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


def _count_patients(results: pd.DataFrame, id_column: str = "id") -> int:
    """Return the number of patients in a results frame.

    ``mist_evaluate`` appends aggregate rows (Mean, Std, percentiles) to its
    results CSV. Ranking strips them, so they must not be counted here either —
    otherwise a dataset is reported as five patients larger than it is, which
    can wrongly satisfy the minimum-patients gate on the significance test.
    """
    if id_column not in results.columns:
        return len(results)
    ids = results[id_column].astype(str)
    return int((~ids.isin(SUMMARY_ROW_IDS)).sum())


def _iteration_number(name: str) -> int | None:
    """Map a strategy name to its iteration number; None for ``baseline``."""
    if not name.startswith("iteration_"):
        return None
    return int(name.removeprefix("iteration_"))


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
