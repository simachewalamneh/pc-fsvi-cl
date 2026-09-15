# Mathematical Derivation: Bayesian Predictive Coding in Function Space for Continual Learning

Notation: `[E]` = established result (textbook GP/VI theory), `[D]` = our
derivation (routine but written out fully here), `[H]` = our proposed
hypothesis / interpretive claim (not a proven equivalence).

---

## 1. Generative model `[E]`

Unknown function `f : X -> Y`, GP prior:

    f ~ GP(m, k)                                   p(f) = "GP(m,k)"

Gaussian observation model for task data `D_t = {(x_i, y_i)}`:

    y_i = f(x_i) + eps_i,   eps_i ~ N(0, sigma_n^2)
    p(y_i | f) = N(y_i; f(x_i), sigma_n^2)
    p(D_t | f) = prod_i p(y_i | f)

The exact posterior `p(f | D_t)` is itself a GP (closed form, `src/gp.py`,
`ExactGP`), with posterior mean/covariance given by the standard GP
regression equations. This exact posterior is our Stage-1 ground truth.

---

## 2. Finite-dimensional function-space representation `[D]`

`q(f)` cannot be stored directly (infinite-dimensional). We use an
inducing-point (sparse-GP) representation: fix `M` locations
`Z = (z_1, ..., z_M)` and let `u = f(Z)`. Define

    q(u) = N(m, S)                                  (M-dimensional Gaussian)
    q(f) := integral p(f | u) q(u) du

`p(f | u)` is the *exact* GP conditional (no approximation), so `q(f)` is
a well-defined process, and its marginal at any `x` (including points
never used to fit the model) is available in closed form:

    A(x)      = k(x,Z) K_zz^{-1}                              (N,M)
    E_q[f(x)] = A(x) m
    Var_q[f(x)] = k(x,x) - A(x) k(Z,x) + A(x) S A(x)^T

This is why the code is truly *function-space*: `(m, S)` are the only
finite objects stored, but they induce a full distribution over `f`
evaluated anywhere, not merely at the `M` inducing inputs. This is
`FSVI.predict()` in `src/variational.py`. Contrast with a *parameter*-
space VI scheme (e.g. weight-space Bayesian linear regression), where
the stored object is a distribution over parameters with no direct,
kernel-mediated meaning at arbitrary inputs.

---

## 3. Free energy for a single task `[E]`

    F[q] = -E_{q(f)}[log p(D|f)] + KL(q(f) || p(f))

Because both `q(f)` and `p(f)` are determined by finite-dimensional
Gaussians over `u` (through the *same* GP conditional `p(f|u)`), the
function-space KL collapses to the finite-dimensional KL between `q(u)`
and `p(u) = N(0, K_zz)` (standard sparse-GP result, Titsias 2009 /
Hensman et al. 2013):

    KL(q(f) || p(f)) = KL(q(u) || p(u))
                      = 1/2 [ tr(K_zz^{-1} S) + m^T K_zz^{-1} m
                              - M + log|K_zz| - log|S| ]

---

## 4. Sequential (continual) free energy `[D]`

For task `t`, the exact sequential Bayesian update is

    p(f | D_{1:t}) proportional-to p(D_t | f) p(f | D_{1:t-1})

i.e. the posterior after task `t-1` is the correct prior for task `t`.
Approximating `p(f | D_{1:t-1}) approx q_{t-1}(f)` and repeating the
single-task VI derivation with `q_{t-1}(f)` in place of `p(f)` gives

    F_t[q] = -E_{q(f)}[log p(D_t|f)] + beta_t * KL(q(f) || q_{t-1}(f))

with `beta_t = 1` recovering the exact sequential-VI recursion; `beta_t
!= 1` is a *tempered* variant (Sec. 8) that trades off plasticity
(`beta_t` small, trust new data more) against stability (`beta_t` large,
stay close to old beliefs). Because `q_{t-1}(u)` and `q_t(u)` are
Gaussians over the *same* random vector `u = f(Z)` (we use a single,
shared set of inducing points across all tasks -- `src/continual_learning.py`),
this KL is again the finite-dimensional Gaussian-Gaussian KL, now with
`q_{t-1}`'s mean/covariance as the reference instead of the GP prior.

**Verified in `tests/test_continual.py`**: for the Gaussian-conjugate
case with sufficiently many, densely-placed inducing points, this
recursive scheme reproduces the exact batch (all-data) GP posterior to
within numerical tolerance -- confirming the "previous posterior becomes
the next prior" derivation is not just heuristic here but exact.

---

## 5. Likelihood <-> prediction error `[E]`

    -log p(y_i|f) = -log N(y_i; f(x_i), sigma_n^2)
                   = 1/2 log(2*pi*sigma_n^2) + (y_i - f(x_i))^2 / (2 sigma_n^2)

Define prediction `hat_y_i = f(x_i)` and error `eps_i = y_i - hat_y_i`,
precision `Pi_i = 1/sigma_n^2`:

    -log p(y_i|f) = 1/2 * Pi_i * eps_i^2 + C

So the Gaussian negative log-likelihood *is exactly* a precision-weighted
squared prediction error, with `C` a constant independent of `f`. Summed
and expectation-taken under `q`, this becomes the `nll` term implemented
in `FSVI.free_energy`.

---

## 6. Predictive coding as gradient descent on `F_t` `[D + H]`

For the *conjugate* case implemented here (Gaussian likelihood, Gaussian
`q`), the gradients of `F_t` w.r.t. the inducing-point mean and
covariance are exact and closed form (full derivation in
`src/predictive_coding.py` docstring):

    A = K_xz K_zz^{-1}                       (fixed "read-out" of u onto f(X))
    eps = y - A m                             (prediction error)
    grad_m F_t = -A^T (Pi * eps) + beta_t * Sigma0^{-1} (m - m0)
    Lambda*    = beta_t * Sigma0^{-1} + A^T diag(Pi) A     (posterior precision)
    S*         = Lambda*^{-1}

**`[D]` Exact claim, conjugate case only**: preconditioning the raw
gradient by `S* = Lambda*^{-1}` (a natural-gradient / Newton step) reaches
the exact optimum of `F_t` in closed form -- `tests/test_predictive_coding.py`
verifies this to `1e-6`. In this case, "predictive-coding-style iterative,
precision-weighted, prediction-error-driven inference" is *mathematically
identical* to exact function-space variational inference; the iteration
is a numerical device, not an approximation.

**`[H]` Interpretive claim, general (non-conjugate) case**: for
non-Gaussian likelihoods or non-Gaussian `q`, the same
prediction-error/precision-weighting *structure* still appears in the
gradient of the expected log-likelihood (via the chain rule through
`f(x_i)`), but the closed-form fixed point no longer exists, and
"predictive coding = one exact Newton step" no longer holds. We do not
claim exact equivalence in that regime -- only that the gradient of the
free energy is *always* expressible as a precision-weighted combination
of prediction errors, which is the sense in which PC and VI are
connected here (an approximate / structural equivalence, not identical
algorithms). This distinction is required reading before extending the
framework to classification or other non-Gaussian likelihoods.

---

## 7. Adaptive stability and novelty (Sec. 8-9) `[H]`

Standardized prediction error:

    z(x) = (y - hat_y(x)) / sqrt(Var_q[f(x)] + sigma_n^2)

Large `|z(x)|` flags observations inconsistent with the current
functional belief (`src/uncertainty.py`). We use it only as a diagnostic
and for one simple, explicitly-labeled-as-a-heuristic adaptive-`beta_t`
ablation (`experiments/ablations.py`): `beta_t` is scaled inversely with
the mean predictive variance on the incoming task's inputs, evaluated
*before* training on them. We do not claim this `g(.)` is optimal or
that prediction error is a validated task-boundary detector in general
-- only report the empirical effect on forgetting/plasticity for this
choice (Sec. 8's explicit caveat against inventing `g` without a
fixed-beta baseline for comparison).

---

## 8. Scope note on novelty (Sec. 20 discipline)

Gaussian-process continual learning, sparse-GP function-space VI, and
uncertainty-gated regularization are each established literature threads
(e.g. Titsias 2009 sparse variational GPs; Bui et al. 2017 streaming
sparse GPs / GP continual learning; Rudner et al. 2022 function-space VI
for continual learning; predictive coding as approximate inference,
Friston 2005 / Bogacz 2017; connections between PC and variational
free-energy minimization are widely discussed in the active-inference
literature). The novelty claim of *this* codebase is narrow: an explicit,
tested demonstration that for Gaussian-conjugate GP continual learning
with shared inducing points, natural-gradient predictive-coding-style
iteration and batch variational optimization provably converge to the
*same* point, and a quantified comparison of that scheme against
standard baselines. Broader novelty claims should not be made without a
dedicated literature review pass (Sec. 20) beyond the scope of this
derivation document.
