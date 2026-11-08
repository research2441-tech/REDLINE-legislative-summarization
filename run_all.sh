#!/bin/bash
# End-to-end reproduction of the REDLINE experiments (CPU only; several hours on 2 cores).
set -e
mkdir -p data results figs ckpt
python3 download_billsum.py
python3 dl_uscode.py
python3 make_concord.py
python3 build_store.py
python3 build_store_small.py test
python3 exec_stats.py
python3 exec_stats2.py
python3 resolver_ablation.py
for s in test ca_test train; do python3 build_inputs.py $s; done
python3 make_splits.py
python3 baselines_extractive.py usamd
python3 baselines_extractive.py ca
export NT=2
python3 train_gen.py REDLINE 400 usamd,ca
python3 train_gen.py BILL 400 usamd,ca
python3 train_gen.py B0 0 usamd:test
python3 train_gen.py ENT 400 usamd:test
python3 train_gen.py OPS 400 usamd:test
python3 train_gen.py REDLINE_ADD 400 usamd:test
python3 llm_baselines.py usamd PEGASUS 60
python3 llm_baselines.py usamd QWEN 60
python3 analyze.py
python3 crc_analysis.py REDLINE,BILL full,conf results/summary_crc.json
python3 failure.py
python3 timing.py
python3 fig_diagrams.py
python3 -c "import fig_results as F; F.fig_exec('figs/fig3_execution.png'); F.fig_gen('figs/fig4_generation.png'); F.fig_het('figs/fig5_heterogeneity.png'); F.fig_crc_dist('figs/fig6_crc_dist.png'); F.fig_load('figs/fig7_load.png'); F.fig_ncal('figs/fig8_ncal.png')"
