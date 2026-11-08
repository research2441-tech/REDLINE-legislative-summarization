# REDLINE: statute-resolved amendment execution and two-sided conformal risk control for legislative summaries

Code for the manuscript "REDLINE: Statute-Resolved Amendment Execution and Two-Sided Conformal Risk Control for Verifiable Summaries of Legislation". Every number, table and figure in the paper is produced by the scripts in this single flat folder.

## Requirements
Python 3.10+ and `pip install -r requirements.txt`. The reported experiments ran on a 2-core x86 CPU with 8 GB RAM and no GPU. The full pipeline takes several hours on that machine.

## Data
* BillSum (`FiscalNote/billsum` on the Hugging Face Hub): 18,949 training bills, 3,269 U.S. test bills, 1,237 California bills. Downloaded by `download_billsum.py`.
* Versioned U.S. Code (`dreamproit/uscode` on the Hugging Face Hub): 381 release points from 113-21 (July 2013) to 119-102 (July 2026), derived from the Office of the Law Revision Counsel release points. Downloaded by `dl_uscode.py` (about 3 GB of parquet).

## Reproduce
```bash
pip install -r requirements.txt
bash run_all.sh
```
`run_all.sh` creates `data/`, `results/`, `figs/` and `ckpt/` next to the scripts and runs every stage in order. Memory-heavy stages (`build_store.py`, the REDLINE_ADD arm) should not run concurrently on an 8 GB machine.

## Stages and paper outputs
| Script | What it does | Paper output |
|---|---|---|
| `make_concord.py` | act-to-Code concordance induced from training bills | Section III-C |
| `build_store.py`, `build_store_small.py` | version store of every cited section | Section IV-A |
| `exec_stats.py`, `exec_stats2.py`, `resolver_ablation.py` | resolution routes, executability under three version policies, instruction-noise robustness | Tables IV, V; Fig. 3 |
| `build_inputs.py`, `make_splits.py` | entity tables, executed redlines, fit/calibration/test splits (seed 7) | Table III |
| `train_gen.py ARM STEPS TAGS` | equal-budget adaptation of T5-small for B0, BILL, ENT, OPS, REDLINE, REDLINE_ADD | Tables VI, VII |
| `baselines_extractive.py`, `llm_baselines.py` | LEAD-4, TextRank, PEGASUS-BillSum, Qwen2.5-0.5B with and without evidence | Tables VI, VIII |
| `analyze.py` | ROUGE, entity factual precision, salient provision recall, Wilcoxon + Holm, bootstrap intervals, California transfer | Tables VI to VIII, X; Figs. 4, 5 |
| `crc_analysis.py` | two-sided conformal risk control, 500 re-splits, level and calibration-size sweeps | Table IX; Figs. 6 to 8 |
| `failure.py`, `timing.py` | displacement analysis, per-stage latency | Table XI; Section VI-D |
| `fig_diagrams.py`, `fig_results.py` | all figures, grayscale | Figs. 1 to 8 |

## Modules
* `redline_core.py`: citation parsing, resolver routes (explicit, code alias, anaphora, concordance), amendment compiler, executor, satisfiability-based version anchoring, redline rendering.
* `evaluate.py`: typed legal-entity grounding against bill and statute evidence, provision coverage, ROUGE.
* `crc.py`: risk and salience scorers, loss curves, conformal thresholds, policy evaluation.

## Notes
* One training seed per arm; variance is reported over bills and calibration re-splits, not over training runs.
* No 119th Congress bills are included because the official bill sources were not reachable from the experimental environment.

## License
Released for research use. BillSum and the U.S. Code dump keep their own licenses.
