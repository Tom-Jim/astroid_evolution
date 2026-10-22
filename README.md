# Asteroid Evolution

Long-term shape evolution of rubble-pile bodies (Ryugu, Phobos, Deimos) under spin change (YORP) and tides. The body is a quasi-static viscoplastic Stokes flow with Mode C regolith transport; gravity, rheology and geometry all run on the GPU in double-single arithmetic. The desktop app is built with Tauri 2, the browser version runs on WebGPU, and both share one set of React control panels.

The mathematical model and every acceptance criterion are in [`docs/mathstrict_EN.md`](docs/mathstrict_EN.md). The reconstruction, forward check and forecast for the three bodies are in [`docs/evolution_report.md`](docs/evolution_report.md).

## Results at a glance

| Body | Span | Main result |
|---|---|---|
| Ryugu | 9.41 Myr ago (spin 3.40 h, the fastest spin with the whole surface bound) → today (7.6326 h) → +5 Myr | Forward run vs. measured shape: RMS 0.026 m, max 1.9 m; mass and volume within 0.04 % |
| Phobos | 30 Myr ago → today → +30 Myr | Surface landslides from 27.25 Myr ahead; largest displacement 246 m at 30 Myr; no interior yield |
| Deimos | 30 Myr ago → +30 Myr | No yield; orbital period within −0.005 % |

Every material and orbital constant comes from the literature (sources in the comments of each `dist/.stage/assets/projects/*.toml`); the quantities without a published value are listed separately in the report.

## Architecture

```
src/
  rtfp/      Tier 1  RTFP gravity: ray integrals, Metal hardware ray tracing (software BVH on WebGPU)
  rheo/      μ(I) rheology and the friction-angle chain (GPU, double-single)
  scf/       Tier 2  quasi-static Stokes / Picard, yield classification, SOCP slope relief, Mode C,
             rotating-frame runout of released regolith (Coriolis, μ(I), ballistic hops), inversion
  shsmooth/  spherical-harmonic mesh smoothing
  gpu/       wgpu compute executor (buffer pool, WESL shaders compiled by src/build.rs)
  app/       Tier 3  Bevy view: playback, shaders, JSON bridge to the panels (bridge.rs)
  web/       browser side: page, evolution worker, GPU worker (SharedArrayBuffer)
  bin/       command-line tools scf-cli, rtfp-cli, rheo-cli, sh-cli
src-tauri/   desktop shell: Tauri 2 window, Bevy renders natively under a transparent webview
ui/          React + TypeScript control panels (plain HTML/CSS/SVG, no 3D library)
dist/.stage/ projects, models, scenario catalog scenarios.json, reconstruction snapshots
tests/       all tests (src holds no test code)
```

Bevy draws all 3D. React draws only the panels around it: body and scenario choice, parameter sliders, timeline, telemetry and view toggles. The panels drive Bevy with JSON commands, and Bevy sends back a status object about ten times a second.

## Requirements

- Rust stable (edition 2024; tested with 1.99)
- [bun](https://bun.sh) for the panels
- Desktop: macOS (Metal). Windows and Linux packages are built by CI but not yet verified on hardware.
- Browser: [Trunk](https://trunkrs.dev) 0.22, `rustup target add wasm32-unknown-unknown`, and a WebGPU browser (desktop Chrome)

## Build and run

Panels (needed once for both the desktop app and the page):

```bash
cd ui && bun install && bun run build
```

Desktop, development mode (Vite hot reload; Rust changes recompile):

```bash
ui/node_modules/.bin/tauri dev
```

Desktop, release bundle (whole-program LTO, stripped, an `.app` of about 60 MB):

```bash
ui/node_modules/.bin/tauri build --bundles app -- --profile dist
```

Browser (needs COOP/COEP headers, already set in `src/web/Trunk.toml`):

```bash
cd src/web && trunk serve
```

Then open <http://127.0.0.1:8080>. Release build: `trunk build --release --cargo-profile dist`, written to `target/web-dist`. A static host must also send `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp`.

The desktop app and the page each load the GPU fully; do not run both at once.

### Controls

| Input | Action |
|---|---|
| Space | Pause / resume |
| 1–4 | Layer: yield / slope / regolith / combined |
| I, [ ] | Interior density cutaway, move the cut plane |
| W | Wireframe |
| R | Reset the view |
| P | Screenshot (desktop: `Pictures/Asteroid Evolution`; browser: download) |
| Mouse | Right drag orbits, left drag pans, wheel zooms |

## Command line

```bash
cargo build --release --features tier2 --bins
target/release/scf-cli --help
```

`scf-cli` subcommands: `validate`, `spin`, `relax`, `realign`, `run-sim`, `inspect`, `export`, `history` (inversion). To reproduce the three runs of the report:

```bash
cargo build --release --features tier2 --bin scf-cli
sh dist/.stage/output/run_all.sh          # resumable; delete a body's directory to recompute it
python3 dist/.stage/output/evaluate.py    # compare with measurements, write evaluation.json
```

## Tests

```bash
cargo test --features tier3
```

156 tests (unit, GPU double-single, RTFP, rheology, spherical harmonics, regolith runout, Tier 2 SCF, Tier 3 view and bridge). A working GPU is required.

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
