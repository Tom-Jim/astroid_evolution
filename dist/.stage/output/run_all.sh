#!/bin/sh
# Reconstruction, forward check and prediction for Ryugu, Phobos and Deimos.
# Schedules (derivation in docs/evolution_report.md):
#  Ryugu  YORP  -4.1e-7 deg/day^2 (Cho et al. 2021): 3.40 h -> 7.6326 h in 9.41 Myr; +5 Myr.
#         3.40 h is the fastest spin at which the whole surface stays bound (scf-cli inspect
#         surface_binding); at 3.35 h about 100 equatorial faces would shed material.
#  Phobos tide  k2/Q = 1.816e-3 (Brozovic et al. 2025), constant-Q law: 30 Myr back, 30 Myr ahead.
#  Deimos tide  same law (receding orbit): 30 Myr back, 30 Myr ahead.
set -e
cd "$(dirname "$0")/../../.."
CLI=./target/release/scf-cli
P=dist/.stage/assets/projects
O=dist/.stage/output
# A stage whose result exists is skipped, so an interrupted run resumes; delete a
# body's directory to recompute it.
run() { # body spin_past spin_future
  b=$1
  mkdir -p $O/$b
  [ -f $O/$b/history/history.json ] || $CLI history --project $P/$b.toml --spin "$2" --iterations 8 --tol 0.01 --damping 0.5 --out-dir $O/$b/history > $O/$b/history.log 2>&1
  [ -f $O/$b/forward/final.json ] || $CLI run-sim --project $P/$b.toml --spin "$2" --resume $O/$b/history/initial.json --out-dir $O/$b/forward --quiet > $O/$b/forward.log 2>&1
  [ -f $O/$b/future/final.json ] || $CLI run-sim --project $P/$b.toml --spin "$3" --resume $O/$b/history/present.json --out-dir $O/$b/future --quiet > $O/$b/future.log 2>&1
  echo "$b done"
}
run ryugu  "linear:5.1333213e-4,-9.585919e-19,2.969610e14" "linear:2.286677e-4,-9.585919e-19,1.577880e14"
run phobos "orbit:4.282837e13,1.024543e7,-7.293772e-10,9.467280e14" "orbit:4.282837e13,9.376e6,-1.187900e-9,9.467280e14"
run deimos "orbit:4.282837e13,2.346217e7,1.089952e-12,9.467280e14" "orbit:4.282837e13,2.34632e7,1.089952e-12,9.467280e14"
