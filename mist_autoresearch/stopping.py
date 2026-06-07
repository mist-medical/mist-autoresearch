"""Stopping criteria for the autoresearch loop."""
from dataclasses import dataclass


@dataclass
class StoppingCriteria:
    """Stopping criteria for the autoresearch loop.

    Attributes:
        max_iterations: Hard stop after this many iterations.
        patience: Stop early if no improvement for this many consecutive iterations.
        alpha: Significance threshold for the Wilcoxon test (best vs. baseline).
        min_iterations: Minimum iterations before early stopping is considered.
        min_patients_for_significance: Skip significance gate if dataset is smaller.
    """

    max_iterations: int = 50
    patience: int = 10
    alpha: float = 0.05
    min_iterations: int = 5
    min_patients_for_significance: int = 15

    def should_stop(
        self,
        iteration: int,
        iterations_since_improvement: int,
        n_patients: int,
        best_p_value: float | None,
    ) -> tuple[bool, str]:
        """Return (should_stop, reason).

        Args:
            iteration: Current iteration number (1-indexed).
            iterations_since_improvement: Consecutive iterations without a new best.
            n_patients: Number of patients in the dataset.
            best_p_value: p-value from Wilcoxon test (best strategy vs. baseline).
                None if the test has not been run (too few patients or strategies).

        Returns:
            (True, reason) if the loop should stop, (False, "") otherwise.
        """
        if iteration >= self.max_iterations:
            return True, "max_iterations"

        if (
            iteration >= self.min_iterations
            and iterations_since_improvement >= self.patience
        ):
            if n_patients < self.min_patients_for_significance:
                return True, "patience"
            # n_patients >= threshold: significance matrix was computed.
            # best_p_value is None when the best strategy IS baseline (no
            # meaningful self-comparison). Stop on patience alone in that case.
            if best_p_value is None:
                return True, "patience"
            if best_p_value < self.alpha:
                return True, "patience+significance"

        return False, ""
