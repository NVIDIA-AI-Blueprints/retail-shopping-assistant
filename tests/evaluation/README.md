# Evaluation

This folder contains the planned Challenger and Judge scaffold for the Retail
Shopping Assistant.

Current scope:

- `PLAN.md` is the design source for this first scaffold.
- `eval_config.yaml` contains non-secret config references only.
- `datasets/` contains text/image shopping scenario briefs for future
  Challenger runs.
- `datasets/image_shopping/assets/` contains generated product-photo inputs
  plus YAML sidecars that own image descriptions.
- `results/` is reserved for generated run output and is ignored except for
  `.gitkeep`.
- `src/` is reserved for future evaluation-only Challenger, Judge, and report
  helpers.

Do not put runtime Shopping Assistant code here.
