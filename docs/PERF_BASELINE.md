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

## Post-upgrade (through commit c0e77f5)

| Command | Result | Total |
|---|---|---|
| `cargo test --features tier2` | all pass (2 tests needed a post-fix rerun) | tier2_scf: 1109 s |

tier2_scf binary: 1109.46 s vs baseline 465.98 s (2.4x slower). Dominant new
costs: `Necks::critical_energy` evaluates ~4800 RTFP potentials per A-O step
(two 49x49 neck-plane disks), and the optical-first runout pays a full surface
ray plus an ODE rerun for every packet that falls back (all neck-region
packets). Unit suite: 118 tests / 24.68 s vs 109 / 23.32 s. tier1_rtfp: 12
tests / 1.25 s (was 11 / 0.27 s) — the added segment-table test dominates.

Speedup accounting: this upgrade adds capability (orientation-aware chords,
G1 basis, hyperbolic solver, optical runout, topology-preserving breakup) over
the frozen baseline; the tier2 integration path pays for the new per-step
diagnostics (E_crit plane sampling, Gram reports, the optical-ray pass). The
retained PD solver + `optics: false` + `basis: "enclosing-ball"` configuration
reproduces baseline behaviour; no dedicated hot-path benchmark exists yet —
profiling E_crit's disk batches is the obvious first optimization if the
integration cost matters.

## Post-migration (through commit 3141009)

Final architecture: the hyperbolic characteristic-ray solver is the sole
production rheology solver; the G1 basis is GPU-evaluated by default; PD and
the CPU/EB implementations live in tests as oracle only; the `solver`/`basis`
config keys are rejected outright.

| Command | Result | Total |
|---|---|---|
| `cargo test --features tier2` | **178 passed, 0 failed** | ~14 092 s (3 h 55 m) |

Per test binary:

| Binary | Tests | Time (s) | vs baseline | vs c0e77f5 |
|---|---|---|---|---|
| tier2_scf | 11 | 4408.18 | 465.98 → 9.5× | 1109.46 → 4.0× |
| unit | 120 | 9684.11 | 23.32 → 415× | 24.68 → 392× |
| tier1_rtfp | 12 | 0.37 | 0.27 | 1.25 |
| rheo_friction | 11 | 0.26 | (23 / 0.05 then) | — |
| sh_smooth | 23 | 0.05 | (1 / 0.04 then) | — |
| gpu_ds | 1 | 0.04 | (11 / 0.19 then) | — |

Where the time goes:

- **unit 9684 s**: essentially one test —
  `hyperbolic_ray_solver_matches_the_pd_oracle` runs the point-by-point PD
  direct-solve oracle against the ray solver at level-4 resolution (~2.5 h
  wall, internally ~2.8 cores). All other 119 unit tests finish in minutes.
  That oracle is the reference for correctness, not a hot path; if suite
  latency matters, splitting or shrinking it is the single dominant lever.
- **tier2_scf 4408 s**: every record now carries the §10.0.10 audit
  (`eps_momentum`, `eps_lambda`, `e_proj`), the certified limit bracket fires
  where the ray signature requires it, and the rest-state delivery adds the
  cached range-space equilibration backsolve per unyielded step. Slower than
  baseline and c0e77f5, but the suite is fully green for the first time on
  the hyperbolic-only + G1-GPU architecture — earlier runs either failed
  (4347 s, 4 failures) or predated the honest rest-state/collapse gates.

Note: fast-binary test counts shifted between suites during the migration
(cases moved/merged); only like-for-like totals are compared above.
