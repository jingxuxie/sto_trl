# Stochastic TRL LaTeX Draft

This directory contains a self-contained conference-style LaTeX draft generated
from the verified stochastic TRL result package.

Compile from this directory:

```bash
latexmk -pdf -interaction=nonstopmode main.tex
```

Before compiling, verify that hand-written LaTeX table rows still match the
generated paper-table CSVs from the repository root:

```bash
/home/eston/anaconda3/envs/autoresearcher_sto_trl/bin/python scripts/verify_latex_claims.py
```

Primary source artifacts:

- `../../STOCHASTIC_TRL_MANUSCRIPT.md`
- `../../PAPER_CLAIM_PACKAGE.md`
- `../../results/main_claim_verification.md`
- `../../results/paper_tables/main_hard_task_results.csv`
- `../../results/paper_tables/hard_task_stress_seed0.csv`
- `../../results/paper_tables/antmaze_navigate_controller_seeds_ep20_seed012.csv`
- `../../results/paper_tables/antmaze_stitch_controller_seeds_ep20_seed012.csv`
- `../../paper/stochastic_trl/latex_claim_verification.md`

The draft intentionally describes AntMaze as a learned-controller topology
diagnostic, not an end-to-end neural stochastic TRL result.
