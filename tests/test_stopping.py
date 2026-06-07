"""Tests for mist_autoresearch.stopping."""
import pytest

from mist_autoresearch.stopping import StoppingCriteria


class TestStoppingCriteria:

    def _sc(self, **kwargs) -> StoppingCriteria:
        defaults = dict(
            max_iterations=10,
            patience=3,
            alpha=0.05,
            min_iterations=2,
            min_patients_for_significance=5,
        )
        defaults.update(kwargs)
        return StoppingCriteria(**defaults)

    # ------------------------------------------------------------------
    # Hard stop
    # ------------------------------------------------------------------

    def test_hard_stop_at_max_iterations(self):
        sc = self._sc(max_iterations=5)
        stop, reason = sc.should_stop(5, 0, 20, None)
        assert stop is True
        assert reason == "max_iterations"

    def test_no_stop_below_max_iterations(self):
        sc = self._sc(max_iterations=10)
        stop, _ = sc.should_stop(9, 0, 20, 0.01)
        assert stop is False

    # ------------------------------------------------------------------
    # Patience + significance gate
    # ------------------------------------------------------------------

    def test_patience_stop_with_significance(self):
        sc = self._sc(patience=3, min_iterations=2, alpha=0.05)
        stop, reason = sc.should_stop(5, 3, 20, 0.01)
        assert stop is True
        assert reason == "patience+significance"

    def test_patience_stop_without_significance_few_patients(self):
        sc = self._sc(patience=3, min_patients_for_significance=15)
        stop, reason = sc.should_stop(5, 3, 10, None)
        assert stop is True
        assert reason == "patience"

    def test_patience_gate_not_met_below_min_iterations(self):
        sc = self._sc(patience=3, min_iterations=5)
        stop, _ = sc.should_stop(3, 3, 20, 0.01)
        assert stop is False

    def test_patience_gate_not_met_insufficient_consecutive(self):
        sc = self._sc(patience=3)
        stop, _ = sc.should_stop(5, 2, 20, 0.01)
        assert stop is False

    def test_significance_gate_blocks_when_p_value_above_alpha(self):
        sc = self._sc(patience=3, alpha=0.05, min_patients_for_significance=5)
        # patience met, enough patients, but p-value > alpha
        stop, _ = sc.should_stop(5, 3, 20, 0.10)
        assert stop is False

    def test_no_stop_when_nothing_triggers(self):
        sc = self._sc(max_iterations=50, patience=10, min_iterations=5)
        stop, reason = sc.should_stop(3, 1, 20, None)
        assert stop is False
        assert reason == ""

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_exactly_at_patience_boundary(self):
        sc = self._sc(patience=5, min_iterations=1, min_patients_for_significance=5,
                      max_iterations=100)
        stop, reason = sc.should_stop(10, 5, 20, 0.01)
        assert stop is True
        assert reason == "patience+significance"

    def test_patience_one_below_boundary(self):
        sc = self._sc(patience=5, min_iterations=1, min_patients_for_significance=5,
                      max_iterations=100)
        stop, _ = sc.should_stop(10, 4, 20, 0.01)
        assert stop is False

    def test_patience_stop_when_best_is_baseline(self):
        sc = self._sc(patience=3, min_iterations=1, min_patients_for_significance=5,
                      max_iterations=100)
        stop, reason = sc.should_stop(5, 3, 20, None)
        assert stop is True
        assert reason == "patience"

    def test_max_iterations_takes_priority_over_patience(self):
        sc = self._sc(max_iterations=3, patience=3, min_iterations=1)
        stop, reason = sc.should_stop(3, 3, 20, 0.01)
        assert stop is True
        assert reason == "max_iterations"
