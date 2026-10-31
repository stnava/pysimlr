# Prompt: Make `varimax` Support Consistent Across All `pysimlr` Tools

Make `varimax` support consistent across all `pysimlr` multi-view entry points in `/Users/stnava/code/pysimlr`.

## Context & Current Gaps
1. `simlr` (`src/pysimlr/simlr.py`) exposes `varimax: bool = True`, passing it to `initialize_simlr(...)`.
2. Deep/flow entry points — `flow_simr_v` / `flow_simlr_v` (`src/pysimlr/flows.py`), `lend_simr`, `ned_simr`, and `ned_simr_shared_private` (`src/pysimlr/deep.py`), plus any `nnh.py` dispatchers — delegate to `initialize_deep_encoders` (`src/pysimlr/simlr.py:682`), which omits `varimax` and silently drops `varimax` from `**kwargs`.
3. Under signed mode (`positivity="either"`), `varimax=True` currently has no effect when `initialization_type="pca"` (`src/pysimlr/simlr.py:800` only triggers when `(want_nonneg and varimax) or initialization_type == "varimax"`), and `_gcca_basis` (`src/pysimlr/simlr.py:515`) returns early at line 379 without honoring `varimax`.

## Requirements
1. **Unified API:** Add `varimax: bool = True` (matching `simlr`) to `initialize_deep_encoders`, `flow_simr_v`, `lend_simr`, `ned_simr`, `ned_simr_shared_private`, and `nnh.py` wrappers, forwarding it cleanly into `initialize_simlr` (and warn or error on unknown `**kwargs` rather than silently ignoring `varimax`).
2. **Consistent `initialization_type` × `varimax` semantics in `initialize_simlr`:**
   - When `varimax=False`: never apply Varimax rotation (even when `positivity="positive"`, unless `initialization_type="varimax"` was explicitly requested, in which case raise a clear `ValueError` on the contradiction `initialization_type="varimax"` + `varimax=False`).
   - When `varimax=True` with `initialization_type="gcca"`: rotate the joint MAXVAR subspace `shared_u` (`n × k`) by a **single** orthogonal Varimax rotation `R = varimax_rotation(shared_u)` (`shared_u = shared_u @ R`) *before* solving per-view least-squares `lstsq(X_m, shared_u)` and `simlr_sparseness(...)`, so all views stay column-aligned to the same localized shared axes by construction.
   - When `varimax=True` with `initialization_type="pca"` or `"perturbed_pca"`: apply per-view Varimax (for both signed and non-negative modes) followed by `_align_multiview_basis_columns`.
3. **Preserve Nonlinear Decoder/Flow Coupling:** Do **not** post-hoc rotate `V_m` inside `_finalize_bases` after training `FlowSiMRVModel` / `LENDSiMRModel` / `NEDSiMRModel`, because RealNVP coupling layers and MLP decoders `f_m(X_m V_m)` are coordinate-dependent (not $O(k)$-equivariant). Keep Varimax at basis initialization (and optionally expose `rotate_consensus_varimax: bool = False` if rotating only the final linear/aggregated `u`, leaving `model` and `v` exact).
4. **Verification:**
   - Add unit tests verifying `varimax=True` vs `varimax=False` produces distinct, column-aligned bases across `simlr`, `flow_simlr_v`, `lend_simr`, `ned_simr`, and `ned_simr_shared_private` under both `(initialization_type="gcca", positivity="either")` and `(initialization_type="pca", positivity="either" | "positive")`, and that `FlowSiMRVModel` forward/inverse reconstruction invariants still hold.
   - Ensure full `pytest` suite and `python -m antsxqc.lint .` pass with 0 findings.
   - Delete this temporary note (`docs/TEMP_VARIMAX_CONSISTENCY_PROMPT.md`) once implemented and verified.
