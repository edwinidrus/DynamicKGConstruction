# GPT-OSS vs Minimax MINE Benchmark Report

Date: 2026-03-11

## Scope

This report compares the two available MINE benchmark result sets under `mine/`:

- `gpt-oss-120b-cloud`
- `minimax-m2.5-cloud`

The comparison is based on:

- `mine_eval_results.json` for both models
- `benchmark_summary.json` and `mine_summary.csv` for GPT-OSS

## Executive Summary

On the shared subset of 58 topics, `minimax-m2.5-cloud` outperforms `gpt-oss-120b-cloud` by a meaningful margin.

- Mean MINE score on shared topics:
  - GPT-OSS: `0.626`
  - Minimax: `0.732`
  - Absolute gap: `+0.106` in favor of Minimax
- Median MINE score on shared topics:
  - GPT-OSS: `0.633`
  - Minimax: `0.800`
- Head-to-head result over shared topics:
  - Minimax wins: `35`
  - GPT-OSS wins: `16`
  - Ties: `7`

The main qualification is coverage: GPT-OSS has a much more complete run in this folder, with `100` scored topics, while Minimax currently has only `58`. That means Minimax looks better on quality for the overlapping set, but GPT-OSS is the only one with a near-complete benchmark artifact set and broader topic coverage.

## Dataset Completeness

### GPT-OSS

- `benchmark_summary.json` is present
- `mine_summary.csv` is present
- `mine_eval_results.json` contains `100` valid scored topics
- The summary JSON reports `101` articles, but the evaluation file contains `100` valid topic entries and the CSV ends with one blank row (`0.0,0,0`), which looks like an artifact rather than a real sample

### Minimax

- `mine_eval_results.json` is present
- `extracted_kgs.json` is present
- `benchmark_summary.json` is missing
- `mine_summary.csv` is missing
- Only `58` scored topics are available

Conclusion: this is not yet a like-for-like full benchmark run. The fairest comparison is on the 58 shared topics only.

## Aggregate Comparison

### Full available runs

These numbers reflect whatever is currently present for each model, not a matched set.

| Metric | GPT-OSS | Minimax |
| --- | ---: | ---: |
| Scored topics | 100 | 58 |
| Mean score | 0.633 | 0.732 |
| Median score | 0.667 | 0.800 |
| Std. deviation | 0.216 | 0.187 |
| Min score | 0.000 | 0.000 |
| Max score | 1.000 | 1.000 |
| Scores >= 0.8 | 27 | 31 |
| Scores < 0.5 | 22 | 7 |
| Avg. covered facts | 9.49 | 10.98 |
| Avg. nodes | 34.37 | 34.88 |
| Avg. edges | 36.50 | 38.38 |

### Shared-topic comparison only

This is the clean head-to-head view over the `58` common topics.

| Metric | GPT-OSS | Minimax |
| --- | ---: | ---: |
| Shared topics | 58 | 58 |
| Mean score | 0.626 | 0.732 |
| Median score | 0.633 | 0.800 |
| Scores >= 0.8 | 15 | 31 |
| Scores < 0.5 | 15 | 7 |

Interpretation:

- Minimax is not just slightly ahead; it is ahead in both central tendency and consistency.
- Minimax produces many more strong runs (`>= 0.8`) and far fewer weak runs (`< 0.5`) on the shared set.

## Largest Minimax Advantages

These are the strongest topic-level improvements by Minimax relative to GPT-OSS.

| Topic | GPT-OSS | Minimax | Delta |
| --- | ---: | ---: | ---: |
| The Discovery of Penicillin | 0.000 | 0.800 | +0.800 |
| Ancient Egyptian Burial Practices | 0.467 | 1.000 | +0.533 |
| How the Brain Processes Language | 0.333 | 0.800 | +0.467 |
| Urban Legends and Their Origins | 0.467 | 0.933 | +0.467 |
| Why People Believe in Ghosts | 0.467 | 0.933 | +0.467 |
| A History of Fashion Trends | 0.400 | 0.800 | +0.400 |
| Unusual Animal Adaptations | 0.533 | 0.933 | +0.400 |
| How to Build a Rocket | 0.333 | 0.733 | +0.400 |
| The Role of Dreams in Psychology | 0.333 | 0.733 | +0.400 |
| The Art of Bonsai Tree Cultivation | 0.533 | 0.867 | +0.333 |

Notable pattern: many of Minimax's largest gains come from cases where GPT-OSS appears to under-extract or collapse the graph. The clearest example is `The Discovery of Penicillin`, where GPT-OSS produced only `2` nodes and `1` edge, while Minimax produced `25` nodes and `35` edges.

## Largest GPT-OSS Advantages

These are the clearest cases where GPT-OSS remains better.

| Topic | GPT-OSS | Minimax | Delta |
| --- | ---: | ---: | ---: |
| A History of Magic Tricks | 0.800 | 0.467 | +0.333 GPT-OSS |
| The Chemistry of Cooking | 1.000 | 0.733 | +0.267 GPT-OSS |
| How Satellites Work | 0.867 | 0.667 | +0.200 GPT-OSS |
| The Science Behind Sleep | 0.533 | 0.333 | +0.200 GPT-OSS |
| The Future of Renewable Energy | 0.600 | 0.400 | +0.200 GPT-OSS |

These wins are real, but fewer and smaller overall than Minimax's top improvements.

## Graph Size vs Score

The results suggest that graph size alone does not explain score quality.

- In many Minimax wins, higher score does come with larger graphs and more covered facts
- But some Minimax gains happen even with fewer nodes or edges than GPT-OSS
- Conversely, GPT-OSS sometimes builds larger graphs without better factual coverage

Examples:

- `The Discovery of Penicillin`: Minimax wins with much larger graph and much better coverage
- `A History of Fashion Trends`: Minimax improves score from `0.400` to `0.800` even with fewer edges than GPT-OSS
- `The Art of Bonsai Tree Cultivation`: Minimax improves from `0.533` to `0.867` while using substantially fewer edges

This points to extraction precision and fact alignment being more important than raw graph volume.

## Practical Takeaways

1. If the goal is best MINE score on the overlapping evaluated set, Minimax is currently the stronger model.
2. If the goal is full-run benchmark completeness and reproducible artifacts, GPT-OSS is currently ahead because its output folder is more complete.
3. The biggest weakness in GPT-OSS appears to be occasional severe under-extraction or malformed sparse graphs on some topics.
4. The biggest weakness in Minimax is not quality in this sample, but incomplete benchmark coverage and missing summary artifacts.

## Recommendation

For a fair final model decision, rerun or complete the Minimax benchmark so that both models have:

- the same topic count
- `benchmark_summary.json`
- `mine_summary.csv`
- identical benchmark configuration and post-processing

If a decision must be made from the current data only, the evidence favors Minimax on extraction quality for shared topics, while GPT-OSS remains the more complete and operationally finished benchmark run.