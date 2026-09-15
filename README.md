# Bayesian Predictive Coding in Function Space for Continual Learning

A small, mathematically-transparent research prototype integrating
**Gaussian Processes (GP)**, **function-space variational inference
(FSVI)**, and **predictive coding (PC)** into a continual-learning
framework, following the recursion

    q_{t-1}(f) --new data D_t--> prediction --> prediction error
              --> precision weighting --> variational update --> q_t(f)

See `docs/mathematical_derivation.md` for the full derivation, including
an explicit statement of where the PC <-> VI connection is an *exact*
equivalence (Gaussian-conjugate case, proven and tested here) versus an
*interpretive/structural* connection (general case).

## Key result

On a 3-task sequential regression benchmark (`sin(x)`, `sin(x)+0.5`,
`cos(x)`), the proposed FSVI+PC continual update reproduces the exact
sequential-GP oracle's predictions to within numerical tolerance, and
its closed-form/natural-gradient optimizer is ~100x faster than generic
L-BFGS optimization of the same free energy for an equivalent answer.
See `tests/` for the checks and `experiments/continual_regression.py`
for the full comparison against baselines (independent training,
sequential exact GP, FSVI/L-BFGS, EWC-like point estimate).

## Repository layout

```
src/
  gp.py                 exact GP regression (Stage 1 / ground truth)
  variational.py         FSVI: inducing-point q(f), free energy, analytic-
                          gradient L-BFGS optimizer (Stage 2)
  predictive_coding.py    PC: precision-weighted, prediction-error-driven
                          iterative inference; closed-form conjugate
                          optimum for validation (Stage 3)
  continual_learning.py  sequential q_{t-1} -> prior_t orchestration +
                          baselines (Stage 4, Sec. 13)
  metrics.py              RMSE/MAE/NLL, calibration, forgetting/BWT
  uncertainty.py          standardized-error novelty signal (Sec. 9)
  utils.py                seeding, RBF kernel, synthetic task generator

experiments/
  toy_static.py           Stage 1: static 1D GP regression + figure
  toy_pc.py                Stage 2-3: FSVI vs exact GP vs PC comparison
  continual_regression.py Stage 4: full continual-learning benchmark
  ablations.py             Stage 5: precision/KL/beta/kernel/noise ablations

tests/                    unit tests validating every mathematical claim
docs/mathematical_derivation.md   full step-by-step derivation
configs/default.yaml       experiment hyperparameters
results/                   figures + logs written here when experiments run
```

## Setup

```bash
pip install -r requirements.txt
```

Pure NumPy/SciPy implementation (no PyTorch dependency) -- this is a
deliberate choice, not just an environment constraint: it keeps every
mathematical operation (kernel evaluation, GP conditioning, the free
energy, its gradient) visible in the code rather than hidden behind
autodiff, per the project's requirement not to obscure the mechanism.

## Running

```bash
# Stage 1: static GP regression, sanity-check figure
python experiments/toy_static.py

# Stage 2-3: FSVI (L-BFGS) vs FSVI (predictive coding) vs exact GP
python experiments/toy_pc.py

# Stage 4: full continual-learning benchmark + baselines + figures
python experiments/continual_regression.py

# Stage 5: ablation study
python experiments/ablations.py

# Tests
python tests/test_gp_fsvi.py
python tests/test_predictive_coding.py
python tests/test_continual.py
```

All experiments use fixed seeds (`configs/default.yaml`, `seed: 0`) for
reproducibility and write figures/logs to `results/`.

## Methods compared (Sec. 13)

| Method | What it is | Notes |
|---|---|---|
| Independent | fresh FSVI per task | no continual mechanism; per-task upper bound |
| Sequential exact GP | GP refit on all data seen so far | forgetting-free oracle, O(N^3), not scalable |
| FSVI continual | batch L-BFGS on `F_t`, `q_{t-1}` -> prior | Method B/D without the PC optimizer |
| **PC-FSVI continual** | **proposed**: PC iterative inference on `F_t` | matches the GP oracle in the conjugate case |
| EWC-like | point-estimate MAP + diagonal-Fisher penalty | included for contrast; loses full posterior |

## Honesty notes

- The PC<->VI equivalence proven and tested here is **exact only for the
  Gaussian-likelihood / Gaussian-`q` conjugate case** used throughout
  these experiments. Extending to non-Gaussian likelihoods (e.g.
  classification) breaks the closed-form optimum; see
  `docs/mathematical_derivation.md` Section 6 for the precise scope.
- The adaptive-`beta_t` rule in `experiments/ablations.py` is an
  explicitly-labeled heuristic, compared against a fixed-`beta` baseline
  as required by the brief (Sec. 8) -- it is not claimed to be optimal.
- No novelty claim is made beyond the scope stated in
  `docs/mathematical_derivation.md` Section 8; a full literature review
  (Sec. 20) has not been conducted as part of this deliverable.
