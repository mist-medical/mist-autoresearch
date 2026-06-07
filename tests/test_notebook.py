"""Tests for mist_autoresearch.notebook."""

import pandas as pd

from mist_autoresearch.notebook import ResearchNotebook


def _make_results(ids=("p1", "p2"), dice=(0.9, 0.8)) -> pd.DataFrame:
    return pd.DataFrame({"id": list(ids), "WT_dice": list(dice)})


class TestResearchNotebook:
    def test_write_header_creates_file(self, tmp_path):
        nb = ResearchNotebook(tmp_path / "notebook.md")
        nb.write_header()
        text = (tmp_path / "notebook.md").read_text()
        assert "# Autoresearch Notebook" in text

    def test_write_header_custom_title(self, tmp_path):
        nb = ResearchNotebook(tmp_path / "notebook.md")
        nb.write_header(title="Custom Title")
        text = (tmp_path / "notebook.md").read_text()
        assert "# Custom Title" in text

    def test_write_header_overwrites_existing(self, tmp_path):
        path = tmp_path / "notebook.md"
        path.write_text("old content")
        nb = ResearchNotebook(path)
        nb.write_header()
        assert "old content" not in path.read_text()

    def test_write_baseline_appends_section(self, tmp_path):
        nb = ResearchNotebook(tmp_path / "notebook.md")
        nb.write_header()
        nb.write_baseline(_make_results())
        text = (tmp_path / "notebook.md").read_text()
        assert "## Baseline" in text
        assert "WT_dice" in text
        assert "0.85" in text  # mean of 0.9 and 0.8

    def test_write_baseline_includes_all_metrics(self, tmp_path):
        nb = ResearchNotebook(tmp_path / "notebook.md")
        nb.write_header()
        df = pd.DataFrame({"id": ["p1"], "WT_dice": [0.9], "ET_haus95": [5.0]})
        nb.write_baseline(df)
        text = (tmp_path / "notebook.md").read_text()
        assert "WT_dice" in text
        assert "ET_haus95" in text

    def test_write_iteration_appends_section(self, tmp_path):
        nb = ResearchNotebook(tmp_path / "notebook.md")
        nb.write_header()
        strategy = [
            {
                "transform": "remove_small_objects",
                "apply_to_labels": [-1],
                "per_label": False,
                "kwargs": {},
            }
        ]
        nb.write_iteration(
            1,
            strategy,
            "Test narrative",
            _make_results(),
            mean_rank=1.5,
            p_value=0.03,
            is_best=True,
        )
        text = (tmp_path / "notebook.md").read_text()
        assert "## Iteration 1" in text
        assert "Test narrative" in text
        assert "remove_small_objects" in text
        assert "1.50" in text
        assert "0.0300" in text
        assert "New best!" in text

    def test_write_iteration_no_p_value(self, tmp_path):
        nb = ResearchNotebook(tmp_path / "notebook.md")
        nb.write_header()
        nb.write_iteration(
            1,
            [],
            "narrative",
            _make_results(),
            mean_rank=2.0,
            p_value=None,
            is_best=False,
        )
        text = (tmp_path / "notebook.md").read_text()
        assert "p-value vs baseline" not in text

    def test_write_iteration_not_best_omits_label(self, tmp_path):
        nb = ResearchNotebook(tmp_path / "notebook.md")
        nb.write_header()
        nb.write_iteration(
            1,
            [],
            "narrative",
            _make_results(),
            mean_rank=2.0,
            p_value=None,
            is_best=False,
        )
        text = (tmp_path / "notebook.md").read_text()
        assert "New best!" not in text

    def test_multiple_iterations_all_appended(self, tmp_path):
        nb = ResearchNotebook(tmp_path / "notebook.md")
        nb.write_header()
        nb.write_baseline(_make_results())
        nb.write_iteration(1, [], "first", _make_results(), 2.0, None, False)
        nb.write_iteration(2, [], "second", _make_results(), 1.0, 0.01, True)
        text = (tmp_path / "notebook.md").read_text()
        assert "## Iteration 1" in text
        assert "## Iteration 2" in text

    def test_strategy_serialised_as_json(self, tmp_path):
        nb = ResearchNotebook(tmp_path / "notebook.md")
        nb.write_header()
        strategy = [
            {
                "transform": "fill_holes_with_label",
                "apply_to_labels": [1],
                "per_label": True,
                "kwargs": {"fill_holes_label": 0},
            }
        ]
        nb.write_iteration(1, strategy, "n", _make_results(), 1.0, None, True)
        text = (tmp_path / "notebook.md").read_text()
        assert "fill_holes_with_label" in text
        assert "fill_holes_label" in text
