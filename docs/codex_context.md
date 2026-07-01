# Codex Context

## Project
DFN parameter inversion benchmark using synthetic Laxemar/Forsmark datasets.

## Current default assumptions
- Use `set-rmin-mode = effective_generation`.
- Use `center_weighting = proposal_area`.
- Use `window_mode = polygon`.
- Use `direction_mode = empirical_trace`.
- Use `p32_label = P32_r_ge_0p5m`, but note that some sets have set-specific effective support.
- Laxemar Set 4 is excluded from power-law `kr` inversion because it is exponential.

## Power-law convention
- PDF convention: `f_R(r | kr) proportional to r^-(kr+1)`
- Survival convention: `S_R(r | kr) proportional to r^-kr`
- `kr` is the survival exponent.

## Current accepted kr status
- Accepted: Laxemar Sets 1, 3, 5
- Marginal: Laxemar Set 2 due to face sampling/support-label issue
- Provisional: Forsmark Set 2 due to `kr` systematic bias
- Accepted with uncertainty: Forsmark Sets 1, 5
- Hold: Forsmark Sets 3, 4; Laxemar Set 4

## P32 status
- Use `unit_p32_forward_mc`, not `conditional_visible_trace_proxy`, for final pilot P32.
- `unit_p32_forward_mc` importance sampling oracle has passed for representative sets.
- P32 uncertainty uses `kr` bootstrap x face bootstrap.
- Forsmark Set 5 dense `C(kr)` interpolation check passed and recovered `p32_final_pilot_candidate`.

## Auxiliary diagnostics
- Kaplan-Meier trace-length correction is diagnostic only.
- Primary KM-MC consistency checks should use `observed KM` vs `mc_km_emulated_survival`.
- Do not replace `kr_hat`, `P32_hat`, or accepted/provisional/rejected statuses using Kaplan-Meier outputs alone.

## Superseded
- Global `rmin=0.5` likelihood results for Laxemar Sets 2/3 are diagnostic only.
- `conditional_visible_trace_proxy` P32 results are scaffold only.
- Forsmark Set 2 `oracle_marginal` interpretation was superseded by higher-resolution `oracle_pass`; provisional status remains because of residual `kr` systematic bias.
- `check_dense_c_interpolation.py` is now a legacy diagnostic path; prefer `build_dense_ckr_table.py` plus `estimate_p32_combined_bootstrap.py --C-interpolation-mode log_linear_dense`.

## Codex workflow
Before modifying code, read:
- `docs/current_task.md`
- `docs/codex_context.md`
- `docs/decision_log.md`

Follow the assumptions and decisions in those files.
Do not reinterpret previous benchmark conclusions unless the current task explicitly asks for it.
Update `docs/current_task.md` only if the task scope changes.
Append important new design decisions to `docs/decision_log.md`.
Update `docs/codex_context.md` only if default assumptions or final statuses changed.
Keep documentation updates concise.
