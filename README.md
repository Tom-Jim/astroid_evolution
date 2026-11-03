# Asteroid Evolution

Long-term shape evolution of rubble-pile bodies (Ryugu, Itokawa, Phobos, Deimos) under spin change (YORP) and tides. The time scales are split (mathstrict §10.4.1, hypothesis A-T1). The slow step is deep creep: the body is solved as a quasi-static viscoplastic Stokes flow, classified into a rigid core and a yielded region, and the yielded region creeps over a step of years to ten thousand years. After each creep step the surface settles quasi-statically on the frozen yield state, without advancing time: the loose layer of the yielded surface erodes and runs out over the rotating surface before it is deposited (Mode C and runout), and supercritical faces of the loose layer are relieved toward their angle of repose by a cone program (slope promotion). Every step reports the two separately, so creep and shallow slides are never confused. Gravity is ray-traced on the GPU, the friction-angle chain runs on the GPU with double-single accumulation, and the Stokes solve and the cone program run in double precision on the CPU. The desktop app is built with Tauri 2, the browser version runs on WebGPU, and both share one set of React control panels.

The mathematical model, its hypotheses and every acceptance criterion are in [`docs/mathstrict_EN.md`](docs/mathstrict_EN.md). It is a theory document: it fixes the method and leaves out code, configuration and run data, which live in `src/`, the project files and the reports. The reconstruction, forward check and forecast for the three bodies are in [`docs/evolution_report.md`](docs/evolution_report.md).

## Results at a glance

| Body | Span | Main result |
|---|---|---|
| Ryugu | 9.41 Myr ago (spin 3.40 h, the fastest spin with the whole surface bound) → today (7.6326 h) → +5 Myr | Forward run vs. measured shape: RMS 0.026 m, max 1.9 m; mass and volume within 0.04 %. The reconstruction is the best of 8 inversion iterations (0.026 m); it did not reach its 0.01 m tolerance |
| Phobos | 30 Myr ago → today → +30 Myr | Surface landslides from 27.25 Myr ahead; largest displacement 246 m at 30 Myr; no interior yield (computed before the runout routing was added) |
| Deimos | 30 Myr ago → +30 Myr | No yield; orbital period within −0.005 % |

Material and orbital constants come from the literature where a value exists (sources in the comments of each `dist/.stage/assets/projects/*.toml`). The others, such as the grain size, the contact force, the tidal back-stress factor, the active-layer thickness of Phobos and Deimos, and the numerical parameters (γ̇_ε, p_min, trust radius, Δt_max), are listed in the report or in the project files.

### What these results rest on

- **The densities are not uniform.** Each body matches its measured mass. Itokawa has a denser head than body, fitted to the mass and the 9.7 m centre-of-mass offset of Kanamaru et al. (2019), consistent with Lowry et al. (2014). Phobos has its fit to the measured C20. Ryugu and Deimos, whose interior profiles are not measured, carry an assumed central compaction profile (density 1.2 times higher at the centre than at the mean radius); the assumption and its reasons are in the project files. The table below was computed with uniform Ryugu and Deimos densities and has not been recomputed.
- **In these runs slope relief, not creep, moved the surface.** The yield test is a stress-ratio test: a point yields when equilibrium needs more than 95 % of the local strength (mathstrict §2.8 B2). The Stokes solve resolves only the long wavelengths of the body and decides deep-seated and whole-body failure; it does not resolve the 1–5 m loose layer. In all three runs no surface vertex ever leaves the rigid core and Mode C never erodes: deep creep never started, and every change of shape comes from slope promotion, the subgrid closure of shallow failure: the loose-layer vertices of faces that fail the infinite-slope balance at depth h_s (the angle-of-repose proxy) are freed for the cone program and relieved (mathstrict §2.8c, hypothesis A-C4).
- **Slide timing is partly numerical.** The relief is instantaneous on the geological scale. The onset of a slide (the step its slope first exceeds the proxy) is physical; its duration and the displacement per step are set by the trust radius (`geometry.trust0`) and by `step.dt_max`. On Phobos the largest displacement per step equals the trust radius.
- **These results were computed at a coarser resolution than the projects now ship.** The runs above solved the Stokes problem on a harmonic surface of degree 4 with a velocity basis of degree 2 (64 unknowns) on 96 quadrature points (rotational residual 0.25–0.62). The bundled projects now use velocity degree 5, pressure degrees 7 (angular) and 5 (radial), and a 28 × 56 × 8 volume rule (12 544 points): one step past the rule at which the Ryugu yield fraction stops changing (rotational residual 0.38). At this resolution, with the uniform density of these runs, Ryugu at the 3.40 h start spin yields 44 % of its mass; with the present compaction profile, about 15 %. Deep creep is then active, and the creep step is limited by its own CFL bound to about a year (the surface erosion time, about an hour, no longer limits it under the time split). Phobos and Deimos do not yield at either resolution. Phobos now ships one degree higher (velocity degree 6, pressure degree 8, a 32 × 64 × 9 rule), where its failure orbit has converged: the orbit at which its certified limit load reaches 1 moves by 1.0 % from degree 5 to 6 and by 0.12 % from 6 to 7, and by 0.1 % from icosphere level 5 to 6. The results in the table have not been recomputed at the new resolution.

## Architecture

```
src/
  rtfp/      Tier 1  RTFP gravity: ray integrals, Metal hardware ray tracing (software BVH on WebGPU)
  rheo/      μ(I) rheology and the friction-angle chain (GPU, double-single)
  scf/       Tier 2  quasi-static Stokes / Picard, yield classification, slope promotion and SOCP relief, Mode C,
             rotating-frame runout of released regolith (Coriolis, μ(I), ballistic hops), inversion;
             scf/evolving/ the optional state of mathstrict Chapter 11 (spin feedback, thermal fatigue,
             damage, dilatancy)
  shsmooth/  spherical-harmonic mesh smoothing
  gpu/       wgpu compute executor (buffer pool, WESL shaders compiled by src/build.rs)
  app/       Tier 3  Bevy view: playback, shaders, JSON bridge to the panels (bridge.rs), cached runs (cache.rs)
  web/       browser side: page, evolution worker, GPU worker (SharedArrayBuffer)
  bin/       command-line tools scf-cli, rtfp-cli, rheo-cli, sh-cli
src-tauri/   desktop shell: Tauri 2 window, Bevy renders natively under a transparent webview
ui/          React + TypeScript control panels (plain HTML/CSS/SVG, no 3D library)
dist/.stage/ projects, models, scenario catalog scenarios.json, reconstruction snapshots, cache/ (cached runs)
tests/       all tests (src holds no test code)
```

Bevy draws all 3D. React draws only the panels around it: body and scenario choice, parameter sliders, timeline, telemetry and view toggles. The panels drive Bevy with JSON commands, and Bevy sends back a status object about ten times a second. A run request names its project and snapshot relative to the data root; absolute paths and paths that climb out of it are refused.

## Requirements

- Rust stable (edition 2024; tested with 1.99)
- [bun](https://bun.sh) for the panels
- Desktop: macOS (Metal). Windows and Linux packages are built by CI but not yet verified on hardware.
- Browser: [Trunk](https://trunkrs.dev) 0.22, nightly Rust with `rust-src` (`rustup component add rust-src --toolchain nightly`), and a WebGPU browser (desktop Chrome). The page computes on several threads (wasm threads), which only nightly can build; `src/web/rust-toolchain.toml` and `src/web/.cargo/config.toml` select it for builds run from `src/web`, so everything else stays on stable

## Build and run

Panels (needed once for both the desktop app and the page):

```bash
cd ui && bun install && bun run build
```

Desktop, development mode (Vite hot reload; Rust changes recompile):

```bash
ui/node_modules/.bin/tauri dev
```

Desktop, release bundle (whole-program LTO, stripped, an `.app` of about 41 MB):

```bash
ui/node_modules/.bin/tauri build --bundles app -- --profile dist
```

Browser (needs COOP/COEP headers, already set in `src/web/Trunk.toml`):

```bash
cd src/web && trunk serve
```

Then open <http://127.0.0.1:8080>. Release build: `trunk build --release --cargo-profile dist`, written to `target/web-dist`. A static host must also send `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp`.

`?timing=1` on the page URL records the time of every module of the run; the totals are in `asteroid.timing` (`[calls, seconds]` per module) in the browser console. The CLI does the same with `SCF_TIMING=1` (written to `timing.json` in the output directory).

The cone program of each geometry step is solved by an interior-point method ([Clarabel](https://github.com/oxfordcontrol/Clarabel.rs)) in double precision on the CPU.

The desktop app and the page each load the GPU fully; do not run both at once.

### Controls

| Input | Action |
|---|---|
| Space | Pause / resume |
| 1–5 | Layer: yield / slope / regolith / combined / principal tension |
| I, [ ] | Interior density cutaway, move the cut plane |
| W | Wireframe |
| R | Reset the view |
| P | Screenshot (desktop: `Pictures/Asteroid Evolution`; browser: download) |
| Mouse | Right drag orbits, left drag pans, wheel zooms |

### Cached runs

A cached run (`.aecache`) is the stored timeline of a run: at most 480 frames, quantized to 16 bits and deflated. Playing one does not compute anything. The two import/export buttons at the right of the transport bar are:

- **Export cache** saves the timeline of the current run. The desktop writes it to `dist/.stage/cache/` from the source tree, or to the app data folder for a bundle. The browser downloads it.
- **Import cache** plays a file. The run it holds replaces the current one, and its project and spin come with it.

The page looks for `cache/<project>-<key>.aecache` before it computes a run, and plays it if found. `<key>` is a hash of the project after the slider overrides, the spin schedule, the step limit and the start snapshot, so an exported file is found again when it is copied to `dist/.stage/cache/` and the page is rebuilt. `?live` on the page URL always computes. The desktop always computes; it plays a cache only on import.

### Chapter 11 options

The project sections `[spin_law]`, `[thermal]`, `[markers]`, `[damage]` and `[dilatancy]` switch on the evolving-state hypotheses of mathstrict Chapter 11. They are A-L (spin from the angular momentum), A-Θ (diurnal thermal fatigue and the radiative torque), A-ϖ (damage that weakens cohesion) and A-Ξ (dilatancy and compaction). Damage and dilatancy need `[markers]`; the radiative torque needs `[thermal]`. A project without these sections runs exactly as before.

### A decaying satellite that loses mass

With an `orbit:` spin schedule, the section

```toml
[orbit]
primary_radius = 3.3895e6   # m; the run stops when the orbit reaches it
limit_every = 10            # steps between certified limit-load brackets (0: none)
```

switches on hypothesis A-O (mathstrict §10.4.6).
- The decay rate follows the satellite's mass.
- Mass leaves through the two necks of its Hill region (L1 toward the primary, L2 away from it): runout packets that pass a neck plane, a loose layer pulled outward harder than its cohesion holds, and movable material carried beyond a neck.
- Every step logs the semi-major axis, the mass lost by each route, the Hill radius and the lobe margin (positive when the body overflows).
- Every `limit_every` steps the run brackets the limit load L*. A certified L* > 1 ends the run as a limit-load collapse.

### Eccentricity, libration and the tidal cycle

With `[orbit]`, the section `[cycle]` switches on hypothesis A-Λ (mathstrict §10.4.7). A real satellite's orbit is eccentric, so the tide changes strength and direction once per orbit and the spin librates. A body that is only rigid–viscoplastic cannot feel that cycle, so A-Λ adds an elastic one.

```toml
[cycle]
eccentricity = 0.0151     # at the orbit where the run starts
primary_spin = 7.0882e-5  # rad/s, for the signs of the primary's tidal lags
rigidity = 1.69e8         # Pa, shear modulus of the elastic cycle and of the satellite's tide
q_satellite = 100.0
phases = 16               # samples of the cycle
psi_max = 0.2             # rad; a larger libration ends the run (the lock may fail)
cycles_max = 30
tol = 1.0e-3
tol_cycle = 1.0e-3
# [cycle.fatigue] intercept = 0.9455, slope = 0.0278, limit = 0.7   (needs [damage])
```

- **Eccentricity.** Both tides damp it at constant Q, as a power of the semi-major axis (25/8 from Mars' tide for Phobos, plus Phobos' own). Phobos' e falls about sixfold by its failure orbit.
- **Libration.** The forced libration comes from the body's own inertia. For Phobos it gives 1.1°, against the measured 1.14°. Resonance at 3(B−A)/C = 1 ends the run.
- **Elastic cycle.** Two incompressible elastic solves per step, at the unit-viscosity stiffness the solver already holds. The stress does not depend on the modulus.
- **Brackets.** Every `limit_every` steps the run brackets the peak limit load and the shakedown limit (Koiter and Melan). A certified peak above 1 ends the run.
- **Ratchet.** Where shakedown fails, the steady elastic–perfectly plastic cycle gives a ratchet velocity, whose strain per cycle scales as load over modulus. That velocity moves the whole body. The alternating plastic shear feeds the damage of A-ϖ.
- **Fatigue.** With `[cycle.fatigue]`, cycles that stay elastic break bonds by the rock S–N line, with its 0.7 endurance limit.
- **Settling and escape.** Settling tests repose at the steepest phase. Packets fly in the field of their launch phase, and the Hill necks move with the phase.

`failure-orbit` and `breakup` then need `--a-ref` and `--adot-ref`, the orbit where the eccentricity holds and the tidal rate there. They report the shakedown orbit, the ratchet orbit and the peak collapse.

A-Θ can include a satellite's primary: `[thermal.primary]` with `temperature` and `albedo` adds the primary's eclipses and its heat and reflected light (mathstrict §11.6).

### Breakup, fragments and the debris ring

`scf-cli breakup` follows a satellite past its failure (mathstrict Chapter 12). It needs a project with the `[orbit]` and `[breakup]` sections; `dist/.stage/assets/projects/phobos_breakup.toml` is Phobos with both.

1. **Failure orbit.** It finds the orbit where the certified limit load first exceeds 1, or takes `--a`.
2. **Cut.** It cuts the body into the rigid blocks of its collapse mechanism. Each block becomes a body with the parent's density, and the total mass is kept.
3. **Fragments.** It follows the fragments about the primary (point mass and J₂) as rigid bodies under their mutual gravity: RTFP close up, the quadrupole expansion far apart. The integrator conserves momentum and angular momentum exactly.
   - Fragments that touch while approaching merge.
   - A fragment whose own limit load exceeds 1 breaks again; if its collapse is no set of rigid blocks, it disperses.
   - Pieces below `debris_diameter`, impacts and escapes leave the dynamics.
4. **Ring.** It spreads the debris as a viscous ring (the self-gravity-wake viscosity of Daisaka et al. 2001 and Salmon et al. 2010) between Mars and the Roche limit. Matter crossing the inner edge falls on Mars; matter crossing the Roche limit forms moonlets.

```bash
target/release/scf-cli breakup --project dist/.stage/assets/projects/phobos_breakup.toml --gm 4.282837e13 --a-max 6.95e6 --a-min 6.8e6 --out-dir output/phobos_breakup
```

It writes:
- `summary.json`: the failure orbit, the cut, every event, and the final ledgers;
- `frames.jsonl`: fragments, ledgers, momentum, angular momentum and energy over time;
- `ring.csv`: the final surface density;
- `fragment_<id>.obj`: the final fragments, in the frame of the primary.

The viewers show one body; these outputs are for the command line.

## Command line

```bash
cargo build --release --features tier2 --bins
target/release/scf-cli --help
```

`scf-cli` subcommands: `validate`, `spin`, `relax`, `realign`, `run-sim`, `inspect`, `failure-orbit`, `breakup`, `export`, `history` (inversion). `run-sim --no-lag` evaluates the deep state on every step instead of reusing it while the surface has barely moved (for audits). `failure-orbit` finds, for a frozen body, the orbit below which it can no longer stand: it scans the semi-major axis downward and bisects on the certified bracket of L* (mathstrict §2.17; under `[cycle]`, on the peak and shakedown brackets of §10.4.7), for example

```bash
target/release/scf-cli failure-orbit --project dist/.stage/assets/projects/phobos.toml --gm 4.282837e13 --a-max 9.376e6 --a-min 6.0e6
```

with `--strength static` for the onset of yield at rest and `--davis` for the reduced strength of the non-associated bound. To reproduce the three runs of the report:

```bash
cargo build --release --features tier2 --bin scf-cli
sh dist/.stage/output/run_all.sh          # resumable; delete a body's directory to recompute it
python3 dist/.stage/output/evaluate.py    # compare with measurements, write evaluation.json
```

## Tests

```bash
cargo test --features tier3
```

178 tests: 166 with `--features tier2` (unit, GPU double-single, RTFP, rheology, spherical harmonics, regolith runout, slope promotion, Tier 2 SCF) and 12 more for the Tier 3 view and bridge. A working GPU is required. Run them one at a time (`-- --test-threads=1`) on a small machine: the Tier 2 runs each use the GPU and every core.

## Cargo features

| Feature | Contents |
|---|---|
| `tier1`, `rheo`, `cli` (default) | gravity, rheology, command-line tools |
| `tier2` | SCF evolution |
| `tier3` | Bevy view (no window backend) |
| `desktop` | `tier3` plus the multi-threaded executor, for `src-tauri` |
| `bevy-web` | `tier3` plus the winit canvas and WebGPU, for the browser |

`[profile.dist]` is for shipped builds; day-to-day builds use `release` (thin LTO).

## Releases

`.github/workflows/release.yml` builds the macOS (arm64, x86_64), Windows and Linux packages and the web bundle. Pushing a `v*` tag creates a draft release; a manual run only uploads the build artifacts.
