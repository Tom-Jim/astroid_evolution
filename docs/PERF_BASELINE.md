# Performance baseline (pre-upgrade)

Recorded before the code upgrade to the mathstrict_EN.md standard. Re-run the
same commands after the upgrade to compute speedups.

## Environment

- Machine: Apple M3, arm64, macOS
- Toolchain: cargo 1.99.1, rustc 1.99.1 (dev profile, optimized + debuginfo)
- Baseline commit: `9918146` ("advance docs", branch `phobos-breakup`) plus
  phase-0 comment/§-reference changes (no behavioural effect).

## Whole-suite timings

| Command | Result | Total |
|---|---|---|
| `cargo test` (default features) | 77 passed, 0 failed | ~3 s |
| `cargo test --features tier2` | 166 passed, 0 failed | ~490 s |

Per test binary under `--features tier2`:

| Binary | Tests | Time (s) |
|---|---|---|
| tier2_scf | 11 | 465.98 |
| unit (tests/unit/*.rs) | 109 | 23.32 |
| tier1_rtfp | 11 | 0.27 |
| rheo_friction | 23 | 0.05 |
| gpu_ds | 11 | 0.19 |
| sh_smooth | 1 | 0.04 |

## Representative heavy tests (tier2_scf binary)

(Individual timings recorded in the background at upgrade start; fill in when
the run finishes.)

- `relaxation_and_stage_run_on_gpu`
- `phobos_loses_mass_through_its_inner_neck`
- `a_failed_body_breaks_and_its_fragments_move`

## Notes

- `src/scf/timing.rs` already reports named per-phase spans (`rheology_tables`,
  `mesh_load`, `frame_realign`, ...); `StepRecord` carries `rheo_seconds` and
  per-step timings. Post-upgrade comparisons should use the same spans plus the
  new segment-table / dirty-ray counters of Section 3.15.
- Expected upgrade effects: segment-table reuse removes most ray traversals;
  the hyperbolic ray solver replaces the primal–dual iteration; both should
  show up in `rheo_seconds` and in the tier2_scf total.
