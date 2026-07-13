# Usage

## Overview

`mist_autoresearch` runs sequential LLM-driven research loops on top of MIST
experiments. Each loop proposes, evaluates, and ranks candidate strategies,
feeding results back to the model until a stopping criterion is met.

## Commands

| Command                              | Description                              |
|--------------------------------------|------------------------------------------|
| `mist_autoresearch postprocessing`   | Search for the best postprocessing strategy |

## Postprocessing

```console
mist_autoresearch postprocessing \
  --config results/config.json \
  --predictions predictions/test \
  --test-csv data/test.csv \
  --output autoresearch/postprocessing/run1 \
  --max-iterations 50 \
  --patience 10 \
  --alpha 0.05 \
  --min-iterations 5
```

### Required arguments

| Flag              | Description                                              |
|-------------------|----------------------------------------------------------|
| `--config`        | Path to `config.json` from `mist_analyze`.               |
| `--predictions`   | Directory of baseline NIfTI predictions from `mist_predict`. |
| `--test-csv`      | CSV with `id` and `mask` columns (ground truth paths).   |
| `--output`        | Root output directory for the run.                       |

### Stopping criteria

| Flag                              | Default | Description                                                          |
|-----------------------------------|---------|----------------------------------------------------------------------|
| `--max-iterations`                | `50`    | Hard stop after this many iterations.                               |
| `--patience`                      | `10`    | Stop early if no improvement for this many consecutive iterations.  |
| `--alpha`                         | `0.05`  | Significance threshold (Wilcoxon p-value, best strategy vs. baseline). |
| `--min-iterations`                | `5`     | Minimum iterations before early stopping is considered.             |
| `--min-patients-for-significance` | `15`    | Skip significance gate if dataset is smaller than this.             |

### Other options

| Flag                  | Default                  | Description                                                      |
|-----------------------|--------------------------|------------------------------------------------------------------|
| `--additional-prompt` | *(none)*                 | Path to a Markdown file injected into every proposal prompt as `## Additional Context`. Use it to share dataset knowledge, evaluation criteria, or transform suggestions. |
| `--model`             | *(Claude Code default)*  | Model name forwarded to `claude --model`. Omit to use Claude Code's active model. |
| `--num-workers`       | `1`                      | Parallel workers for postprocessing and evaluation.              |

### Using `--additional-prompt`

Create a Markdown file describing anything the agent should know about your specific dataset or task:

```markdown
## Dataset Notes
This is BraTS 2026 glioma segmentation. Labels: 1=necrosis, 2=edema, 3=enhancing tumour.
Final classes: ET (label 3), TC (labels 1+3), WT (labels 1+2+3).

## Evaluation Criteria
Dice is the primary metric. Focus on ET and TC — WT tends to be high already.

## Suggestions
Small isolated components in the ET class are common false positives.
Try aggressive small-object removal on label 3 before anything else.
```

Then pass it to the run:

```console
mist_autoresearch postprocessing \
  --config results/config.json \
  --predictions predictions/test \
  --test-csv data/test.csv \
  --output autoresearch/postprocessing/run1 \
  --additional-prompt context.md
```

The file contents are included verbatim in every proposal prompt. You can update and resume the run with a revised file if you want to steer the agent mid-run.

## Output structure

```
autoresearch/postprocessing/run1/
├── research_notebook.md       # Agent reasoning at each iteration
├── history.json               # Full run log (resumable)
├── summary.json               # Best strategy + stopping reason
├── rankings.csv               # Cumulative rank table (updated each iteration)
├── significance.csv           # Pairwise Wilcoxon significance matrix
├── baseline/
│   ├── strategy.json          # Empty strategy []
│   ├── predictions/
│   └── postprocess_results.csv
├── iteration_001/
│   ├── strategy.json
│   ├── predictions/
│   └── postprocess_results.csv
└── iteration_002/
    └── ...
```

## Resuming a run

If a run is interrupted (e.g., instance restart, timeout), re-run the exact same command pointing to the same `--output` directory. The loop detects the existing `history.json` and picks up from where it left off:

- Completed iteration results are loaded from `iteration_NNN/postprocess_results.csv` on disk.
- Rankings are recomputed from the recovered results, and the patience counter is
  restored from `iterations_since_improvement` in `history.json`, so a resumed run
  stops at the same point an uninterrupted one would.
- The loop continues from the next iteration number.
- The notebook is appended to, not overwritten.

Changing `--num-workers` between runs is safe — it only affects speed, not results.

If `history.json` or any iteration CSV is missing, `run()` raises `FileNotFoundError` rather than silently producing incorrect results.

## Stopping logic

The loop stops when **any** of the following conditions is met:

1. **Hard stop** — `--max-iterations` iterations have run.
2. **Patience** — the patience counter reaches `--patience` and no significance
   test applies, either because the dataset has fewer than
   `--min-patients-for-significance` patients, or because baseline is still the
   best strategy and so there is nothing to test against it.
3. **Patience + significance** — the patience counter reaches `--patience` and
   the best strategy is significantly better than baseline (p < `--alpha`,
   one-sided Wilcoxon signed-rank test on per-patient mean ranks).

At least `--min-iterations` iterations must run before criteria 2 and 3 are checked.

Note that if patience runs out while a non-baseline strategy leads but has *not*
reached significance, the loop deliberately keeps going to `--max-iterations`
rather than stopping on a lead it cannot back up statistically. Budget compute
for the full `--max-iterations` — each iteration runs a `mist_postprocess` and
evaluation pass over the whole test CSV.

### What counts as an improvement

The patience counter resets **only** when the strategy proposed on the current
iteration takes the top of the cumulative ranking.

This distinction matters because average ranks are pool-relative: they are
recomputed over every strategy tried so far, so adding one strategy re-ranks the
whole pool and can reorder two strategies that were already close. When such a
reshuffle promotes an *older* strategy to the top, the loop records the new
leader — in `rankings.csv`, `summary.json`, and `history.json` alike — but does
not count it as an improvement, because no new strategy was found. Patience keeps
counting.

For the same reason, rank *values* are never compared across iterations. A mean
rank of 1.4 in a pool of two strategies is not better than 5.9 in a pool of
twelve; ranks inflate as the pool grows. The loop tracks the best strategy by
name, always taking the top row of the current ranking.

## Run artifacts

| File | Contents |
|---|---|
| `rankings.csv` | Cumulative BraTS-style mean rank per strategy, best first. Rewritten every iteration. |
| `significance.csv` | Pairwise one-sided Wilcoxon p-values. Entry `[A, B]` is the p-value that A is better than B. |
| `summary.json` | Winner of the run: `best_strategy_name`, its `best_overall_rank`, and the `best_strategy` steps (`null` if baseline won). Derived from the final `rankings.csv`. |
| `history.json` | Per-iteration log plus `best_iteration` (`null` if baseline is best), `iterations_since_improvement`, and `stopped_reason`. Used to resume. |
| `research_notebook.md` | The agent's narrative reasoning per iteration. |

`rankings.csv`, `summary.json`, and `history.json` always agree on which strategy
won. To act on a finished run, read `best_strategy` out of `summary.json` and feed
it to `mist_postprocess`; check `significance.csv` to confirm the winner is
significantly better than baseline before you rely on it.

## research_notebook.md

The notebook is written by the agent itself. It records:

- **Baseline** results before any postprocessing.
- For each iteration: the strategy tried, the agent's reasoning (*why* it chose
  this strategy given what it had seen), per-metric results, mean rank, and
  p-value versus baseline.

This file is intended to be reviewed by the experimenter to understand what the
agent explored and why.

## Python API

```python
from mist_autoresearch.postprocessing.researcher import PostprocessingResearcher
from mist_autoresearch.stopping import StoppingCriteria
from pathlib import Path

stopping = StoppingCriteria(max_iterations=50, patience=10, alpha=0.05)

researcher = PostprocessingResearcher(
    config=Path("results/config.json"),
    predictions_dir=Path("predictions/test"),
    test_csv=Path("data/test.csv"),
    output_dir=Path("autoresearch/postprocessing/run1"),
    stopping=stopping,
    # model="claude-opus-4-8",      # omit to use Claude Code's active model
    # additional_prompt=Path("context.md"),  # optional dataset/task notes
)

best_strategy = researcher.run()
print(best_strategy)  # list of strategy steps, or None if baseline was best
```
