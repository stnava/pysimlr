# Tracking Summary
current phase: v0.2.19 -- `varimax` has one meaning on every multi-view entry point (linear, deep, flow, `nnh` wrappers), resolved once in `initialize_simlr` by `resolve_varimax` (default `None` rotates only a non-negative basis); `pysimlr.svd.varimax`/`varimax_rotation` dispatch explicitly to nsa_flow or the PyTorch implementation with no blanket `except`; full suite green including slow tests.
next WP: D3 -- audit the 46 remaining `except Exception` sites in `src/pysimlr` (see D2 open items); first because each is a candidate silent fallback of the kind D2 removed.
blockers: none
active work: none
device: Apple Silicon (macOS 27 arm64), CPU only; venv /Users/stnava/venvs/ants; torch 2.13.0, numpy 2.0.1, scikit-learn 1.9.0, nsa_flow 3.5.0 (editable, /Users/stnava/data/repos/nsa_flow)
last updated: 2026-10-31

---

Entries are newest first. `D<n>` are implementer deliverables; `R<n>` is reserved for reviewer notes.
This file was created at v0.2.19; earlier history lives in git (`git log --oneline`) and in the
release commit bodies.

## D2 (Remove the silent backend fallback in `svd.varimax` / `svd.varimax_rotation`, 2026-10-31)
status: done
release: v0.2.19 (tag `v0.2.19`, the commit that adds this file)
files touched: src/pysimlr/svd.py, src/pysimlr/nsa_backend.py, tests/test_varimax.py, docs/tracking.md, pyproject.toml, src/pysimlr/__init__.py
tests added: tests/test_varimax.py (8 dispatch tests; one existing test repaired)

**What and why.** Both functions wrapped the nsa_flow call in `except Exception: pass` and fell
through to the PyTorch implementation. Measured on nsa_flow 3.5.0, that catch was silently
rescuing two inputs the backend rejects -- NumPy arrays (`TypeError` inside `torch.eye`) and
batched 3-D input to `varimax` (`RuntimeError`) -- and would equally have hidden any genuine
backend failure. The two paths also disagreed on degenerate shapes (`k < 2` or `p < 2`).

Now:
- `_as_loading_matrix` converts input with `torch.as_tensor` and refuses non-2-D input with
  `ValueError` (the PyTorch path's convergence test summed the criterion across a batch, so
  batching was never supported by either path).
- Degenerate shapes return the identity rotation / unrotated input on every path.
- The PyTorch implementations are explicit private functions, `_varimax_rotation_torch` and
  `_varimax_torch`, used only when the loader resolves no backend.
- `nsa_backend._resolve_backend_function` checks with `inspect.signature(...).bind` that the
  backend accepts the keywords pysimlr passes. An incompatible backend is refused once, with a
  `RuntimeWarning` and an entry in `INCOMPATIBLE_BACKEND_FUNCTIONS`, reported by
  `backend_report(extended=True)["incompatible_functions"]`.
- Backend errors propagate.
- Same file, same pattern: `load_gradient_mapping` / `load_lbfgsb` caught `Exception` around a
  submodule import; narrowed to `ImportError`.

**User-caught gap.** The v0.2.18 report flagged the catch as "worth tightening later"; the owner
asked for it fixed now. Found while doing so:
`test_pysimlr_varimax_matches_nsa_flow_linalg_varimax_exactly` claimed to test the PyTorch
fallback but called the public `varimax_rotation`, which dispatches to nsa_flow when installed --
it compared nsa_flow with itself. It now calls `_varimax_rotation_torch` / `_varimax_torch`
directly (agreement with nsa_flow to 1e-10).

**Verification.** `pytest -m ""` (all tests including slow) and `python -m antsxqc.lint .` --
counts in the v0.2.19 commit body. Not run: a mutation check of the propagate test against the
v0.2.18 code (the new tests import private functions that did not exist there); the test fails
against a catch-and-fallback by construction.

**Open items.**
- 46 `except Exception` sites remain in `src/pysimlr` (deep 11, simlr 8, flows 6,
  sparsification 5, optimizers 5, utils 3, nnh 2, benchmarks/runner 2, benchmarks/deep_ranking 2,
  consensus 1, interpretability 1). Not audited; some may be legitimate boundary handlers that
  record provenance, others silent fallbacks. Examples seen in passing: `_ica_basis_for_view`
  falls back to the SVD basis if FastICA raises; `nnh_embed` substitutes a zero `w` if `pinv`
  raises.
- nsa_flow's `varimax_rotation` emits a torch `UserWarning` (`float()` on a tensor requiring
  grad) when given a grad-tracking input; upstream in nsa_flow, not fixed here.

## D1 (Consistent `varimax` across linear, deep and flow entry points, 2026-10-31)
status: done
release: v0.2.18 -- commit 91cd48d, tag `v0.2.18` (spec committed as a temporary note in 259c4ff, deleted in 91cd48d)
files touched: src/pysimlr/simlr.py, deep.py, flows.py, nnh.py, README.md, docs/theory/SIMLR_THEORY.md (section 9.4), 7 test files
tests added: tests/test_varimax_consistency.py (48 tests)

**What and why.** `simlr` exposed `varimax` (added in 0.2.17, 5cc5fcd / 1768d3b), but the deep
and flow entry points initialized through `initialize_deep_encoders`, which had no such argument,
and their `**kwargs` dropped it silently. Under `positivity="either"` the flag was also inert for
`pca` and `gcca` starts.

- Every entry point takes `varimax: Optional[bool] = None` and forwards it to
  `initialize_simlr`. Deep/flow entry points now warn on unrecognised keyword arguments.
- `resolve_varimax`: `None` rotates only a non-negative basis; explicit `True`/`False` wins.
- `gcca`: one rotation `R = varimax_rotation(shared_u)` of the joint MAXVAR subspace before the
  per-view regressions, so views are column-aligned by construction. `pca`/`perturbed_pca`:
  per-view Varimax then `_align_multiview_basis_columns`.
- `initialization_type="varimax"` with explicit `varimax=False` raises; mismatched sample counts
  raise `ValueError("Size mismatch ...")` before fitting.
- No post-hoc rotation of trained deep/flow bases: the decoders and RealNVP couplings are not
  O(k)-equivariant.

**Decision (2026-10-31): signed rotation is opt-in.** The spec asked for `varimax=True` as the
default for signed bases too. Measured consequences of that default led the owner to choose the
positivity-dependent default instead:
- NED latent-collapse sweep (`test_latent_collapse` generator, seeds 0-9 and 42, off-diagonal
  covariance norm >= 0.5 counts as collapsed): signed 4/11 unrotated vs 6/11 rotated;
  non-negative 5/11 unrotated vs 4/11 rotated.
- Linear `simlr` (`_coupled` fixture, 6 seeds, lars, 20 iterations): the rotated start was never
  worse (it fixed seed 3, latent recovery 0.504 -> 0.990), but at `orthox0.9x1` one seed's fit
  fell from 0.986 to 0.755.

**Pre-existing failures repaired in the same release** (all failed before Varimax existed, at
f5484c7):
- `test_basis_metric_invariance` (2): float32 fixtures put ~1e-7 roundoff into `u @ m` before the
  metric, against a 1e-9 assertion; fixtures are now float64 (worst difference 2.2e-16). Casting
  inside `cross_val_metrics` was tried and does not fix it.
- `test_positive_constraint_does_not_cripple_latent_recovery` (3, slow): the default consensus
  moved to newton and this test never pinned it; pinned to `svd` as its fast sibling is.
- `test_ned_advantage_contract` (slow): strict xfail that now passes; over 8 seeds NED latent
  recovery 0.700 vs SiMLR 0.578, held-out R^2 0.827 vs 0.725. Mark removed. The trainers still
  select the best state by training loss, not a latent-quality criterion.
- `test_nedpp_advantage_contract` (slow): R^2 margin over SiMLR +0.045 over 8 seeds against the
  claimed +0.05 (it held at 85ad3e2 by +0.064, fell to +0.024 at f4e53a6); marked strict xfail
  with these numbers rather than relaxed. NEDPP's latent-recovery lead is real (0.552 vs 0.434).

**Verification.** `pytest -m ""`: 1418 passed, 7 xfailed, 0 failed; `antsxqc.lint`: 0 findings
(recorded in the 91cd48d commit body). Mutation check: removing the deep-path forwarding fails
24 of the new tests; removing the gcca rotation fails 11.

**Open items.**
- Model selection on a latent-quality validation criterion for the deep trainers (blocks a
  durable NED/NEDPP advantage contract).
- The `orthox0.9x1` seed-1 drop after a rotated signed start is unexplained.
