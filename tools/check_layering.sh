#!/bin/sh
# Reject filesystem access and upward layer imports inside Tier 1 and common.
set -eu
root=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
fail=0
if grep -R -n --include='*.rs' 'std::fs' "$root/src/common" | grep -v 'native_fs.rs'; then
  echo "std::fs is only allowed in src/common/io/native_fs.rs"
  fail=1
fi
if grep -R -n --include='*.rs' 'std::fs' "$root/src/rtfp_solver"; then
  echo "rtfp_solver must not use std::fs"
  fail=1
fi
if grep -R -n --include='*.rs' -E 'crate::(scf_simulator|scf_calculation|model_calculation|realtime_display|rheology)' "$root/src/rtfp_solver"; then
  echo "rtfp_solver must not import an upper tier"
  fail=1
fi
exit "$fail"
