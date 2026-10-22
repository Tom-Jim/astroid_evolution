# Ryugu, Phobos, Deimos: reconstruction → forward check → forecast

Every run was computed on the macOS desktop (Apple M3, Metal). To reproduce, run `dist/.stage/output/run_all.sh`, then `evaluate.py` for the errors. Results are written to `dist/.stage/output/<body>/{history,forward,future}`; the error summary is `dist/.stage/output/evaluation.json`.

## 1. Model and constants (all from the literature or measurement)

| Item | Ryugu | Phobos | Deimos |
|---|---|---|---|
| Shape | SHAPE model `ryugu.glb`, volume 0.377 km³ (Watanabe 2019) | `Phobos.glb` scaled to 5742 km³ (Willner 2014); the file is 5 % too large | `Deimos.glb` scaled to 1033 km³ (Ernst 2023); the file is 17 % too small |
| Discretization | Level-5 icosphere ray resampling (`mesh_rays`), 10242 nodes, spacing ~15 m | Same, spacing ~384 m | Same, spacing ~217 m |
| Mass / density | GM = 30.0 m³/s² → uniform 1193 kg/m³ | GM = 7.0765×10⁵ m³/s²; uniform matrix plus a central core, fitted to C20 = −0.103 | M = 1.51×10¹⁵ kg (Jacobson 2010) → 1464 kg/m³ |
| Interior cohesion c_b | 4 Pa (Hirabayashi 2019, lower end of 4–10 Pa) | 25 Pa (Sánchez & Scheeres 2014) | 25 Pa |
| Surface cohesion c_s | 0.5 Pa (Arakawa 2020; Sugimura) | 0.5 Pa (by analogy with Ryugu) | 0.5 Pa |
| Transition depth h_c | 1 m (SCI ejecta came from a ~1 m subsurface layer, Wada 2021) | 100 m (regolith up to ~100 m deep) | 10 m (regolith 5–10 m) |
| Friction | 35°, μ₂ − μ_s = 0.26, I₀ = 0.279 (Jop 2006) | Same | Same |
| Regolith grains | CI chondrite: ρ_p = 2420 kg/m³, porosity 0.46 (Yada 2022 / Nakamura 2022) | Same (by analogy) | Same (by analogy) |
| Spin / orbit | YORP spin-down −4.1×10⁻⁷ °/day² (Cho 2021), from 3.40 h to 7.6326 h in 9.41 Myr. 3.40 h lies inside Hirabayashi's 3.0–3.5 h deformation window and is the fastest spin at which the whole surface stays bound (below) | Synchronous spin. Constant-Q tidal law ȧ ∝ a^(−11/2), Mars k₂/Q = 1.816×10⁻³ (Brozović 2025); today ȧ = −3.75 cm/yr | Synchronous spin, same law, ȧ = +0.003 cm/yr (receding) |

Notes:

- **Phobos ȧ correction.** The often quoted 1.8 cm/yr mistakes the tidal acceleration coefficient s for ṅ. The k₂/Q formula and the measured acceleration s = 1.258×10⁻³ °/yr² each give about 3.8 cm/yr independently, consistent with the published conclusion that Phobos falls within 30–50 Myr.
- **Phobos C20.** The measured central value −0.1378 needs an outer layer denser than the grains, which is impossible, so the model uses −0.103 at the edge of the 1σ interval.
- **Quantities still assumed** (no direct measurement): the surface cohesion and grain density of Phobos and Deimos, and their active-layer thickness h_s = 5 m.

## 2. New physics and numerics (also in docs/mathstrict_EN.md)

- **§4.3a two-layer strength (assumption A-C2):** c(x) = c_b + (c_s − c_b)·e^(−d/h_c), with d the depth below the free surface. ∇η gains the term ∂η/∂c·∇c; the analytic ∇d is checked against finite differences in a unit test. With c_s = c_b the model reduces to the single-layer theory.
- **§2.13 scale of the exact penalty:** ϱ ≥ 10·h̄². This also fixed a bug: the code passed vertex areas (m²) as the SOCP fidelity weights w, while the solver expects dimensionless weights of order one. The bug disabled the penalty on Phobos: slopes were supercritical yet did not move at all.
- **§10.4.3:** if the cone program still fails at the smallest trust radius, the step keeps its geometry and is recorded as `cone_unresolved`. A geometry rejection is still an error.
- **Inversion:** damping β = 0.5, and the iterate with the smallest misfit is returned. Slope relief is a many-to-one projection, so the fixed-point iteration need not contract.
- **§2.8b rotating-frame runout (hypothesis A-C3):** material set loose on the surface runs out over the rotating body before it is deposited: depth-averaged μ(I) sliding with the Coriolis acceleration, ballistic hops where the normal reaction vanishes, inelastic landings. It applies to the mass eroded by Mode C and to the mass the slope relief removes from supercritical faces (§10.4.3: the cone is solved, the removed mass is routed and deposited where it lands, and the cone is solved again). Mass is conserved exactly. This is the mid-latitude-to-equator migration of Watanabe et al. (2019). At geological creep rates Mode C does not erode on these bodies, so the relief routing is the part that acts.
- **Surface binding and the start spin of Ryugu.** `scf-cli inspect` reports the smallest inward effective gravity −g_eff·n over all faces. On Ryugu the equatorial ridge (r ≈ 526 m) lies just outside the synchronous radius at fast spin: at 3.30 h 786 faces are unbound (margin −12 %), at 3.35 h about 100, at 3.40 h none (margin +0.5 %), and from 3.42 h on no face is supercritical at all. The start spin is therefore 3.40 h, the fastest spin without equatorial shedding. Ridge failure and equatorial unbinding come together on Ryugu: with the literature friction and cohesion, a spin that keeps the surface bound leaves only small landslides.

## 3. Results

### Ryugu: 9.41 Myr ago (spin 3.40 h) → today (7.6326 h) → 5 Myr ahead

| Quantity | Result |
|---|---|
| Terrain motion in the forward run | Landslides during the first 0.095 Myr: 11 geometry steps, largest single-step displacement 1.6 m. Interior yield at most 0.6 %, no instability |
| Routed relief (A-C3) | 1.6×10⁶ kg in 89 packets, mass-weighted mean runout 56 m (largest 247 m), mean shift 0.86° toward the equator; 55 packets hopped (highest 48 m), 39 escaped and were returned to their source; the two cone solves differ by at most 3.5 % in removed mass; the cone failed on 2 steps, which kept their geometry |
| Reconstructed past shape vs. today | RMS 0.04 m, maximum 2.81 m; 4 nodes move more than 1 m |
| Forward run to today vs. measured shape | RMS 0.026 m, maximum 1.94 m; one node is off by more than 1 m and none by more than 2 m |
| Mass | 4.4949×10¹¹ kg; measured 4.4949×10¹¹ kg (relative error 1.3×10⁻⁵, 0.001σ) |
| Volume | 0.37687 km³; measured 0.377 ± 0.005 km³ (−0.035 %, 0.03σ) |
| Spin | 7.6326 h (the YORP rate is calibrated to it, so this is not an independent check) |
| Next 5 Myr | Period grows from 7.63 h to 22.5 h; no supercritical slope, the shape does not change |

Comparison with the earlier run from 3.3 h without routing (forward misfit RMS 1.08 m, maximum 18.3 m, 382 equatorial nodes off by more than 2 m): at 3.3 h the equator was unbound, landslides ran for 0.87 Myr and lowered the ridge beyond what the inversion could undo. At 3.40 h the surface is bound, the slides are small and short-lived, and the inversion reproduces today's shape almost exactly. Most of the improvement comes from the bound start spin; the routed mass is small. The model does not reproduce the building of Ryujin Dorsum: with the measured friction and cohesion, Ryugu's ridge fails only when its surface is nearly unbound, so most of the material that would migrate at a faster spin is shed instead. This is consistent with Hirabayashi et al. (2019), who place Ryugu's structural failure in the 3.0–3.5 h range.

### Phobos: 30 Myr ago (a = 10245 km, period 8.75 h) → today → 30 Myr ahead (a = 7430 km, period 5.40 h)

The Phobos and Deimos runs below were computed before the runout of §2.8b was added. Deimos does not move, so it is unaffected; the late landslides of the Phobos forecast would now be routed.

| Quantity | Result |
|---|---|
| Past 30 Myr | Tides were weaker than today and nothing yields; the reconstructed past shape is today's shape |
| Forward run vs. measured shape | Identical by construction. Mass error < 10⁻⁵; volume 5734.3 km³ vs. measured 5742 ± 35 km³ (−0.13 %, 0.22σ); C20 −0.103 vs. −0.1378 ± 0.0348 (1.0σ); synchronous orbital period 7.6566 h vs. 7.6538 h (+0.04 %) |
| Next 30 Myr | Surface landslides start at 27.25 Myr. By 30 Myr, 336 nodes have moved more than 1 m, maximum 246 m, RMS 11.9 m; interior yield stays at 0, no disruption |
| Onset of disruption | After about 32 Myr (a ≈ 7.2 Mm) 32 % of the interior yields, a global instability; the forecast window therefore stops at 30 Myr |

### Deimos: 30 Myr back and 30 Myr ahead

- The semi-major axis changes by only about 1 km; period 30.3105 h vs. measured 30.312 h (−0.005 %).
- Nothing yields or slides; the past, present and future shapes are the same.
- Volume 1031.5 km³ vs. measured 1033 ± 19 km³ (−0.15 %).

## 4. Performance

| Item | Before | After |
|---|---|---|
| Ryugu full 9.92 Myr forward run (1044 steps) | 320 s | 83 s (3.9×), bit-identical results |
| Steps where only the spin changes | 300 ms/step | 50 ms/step |
| Full test suite | ~5 min | ~40 s |

How: Stokes operators are cached by a hash of geometry and density; the velocity basis on the classification grid stays resident on the GPU, so reusing a geometry costs one GEMV. The runout traces only the released packets, in parallel, and stops a flight as soon as its escape is certain; the second cone solve of the relief routing is warm-started from the first. The whole Ryugu forward run from 3.40 h (990 steps, 11 with geometry) takes 47 s.

## 5. Viewing (desktop or browser, one at a time)

The desktop app is built with Tauri 2: Bevy renders the body natively in the window, and the React control panels (plain HTML/CSS, a black telemetry-HUD style) sit on top. The browser version shows the same panels over a WebGPU canvas. Both read the scenario catalog `dist/.stage/assets/scenarios.json`: pick a body (Ryugu / Phobos / Deimos) and a scenario (reconstruction → today, today → future), adjust the sliders, then press LAUNCH.

```bash
cd ui && bun install && bun run build && cd ..
ui/node_modules/.bin/tauri build --bundles app -- --profile dist   # macOS: target/dist/bundle/macos/Asteroid Evolution.app (~60 MB)
cd src/web && trunk serve                                          # browser: http://127.0.0.1:8080
```

Keys: Space pause/resume, 1–4 layers, I interior density cutaway, [ ] move the cut plane, W wireframe, R reset the view, P screenshot (desktop: `Pictures/Asteroid Evolution`; browser: download).

Windows and Linux packages are built on GitHub Actions by `.github/workflows/release.yml`.
