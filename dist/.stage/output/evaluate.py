#!/usr/bin/env python3
"""Errors of the forward runs against the observed bodies, and terrain change of every run.

Reads the snapshots written by run_all.sh and `scf-cli inspect` reports (inspect_*.json).
"""
import json
import math
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent
OBS = {
    # mass kg (GM/G), volume m^3, rotation period h, C20 at r_ref (None: not measured)
    "ryugu": dict(mass=30.0 / 6.6743e-11, mass_err=0.4 / 6.6743e-11, volume=0.377e9,
                  volume_err=0.005e9, period=7.6326, c20=None, r_ref=None),
    "phobos": dict(mass=7.0765e5 / 6.6743e-11, mass_err=0.0075e5 / 6.6743e-11,
                   volume=5742e9, volume_err=35e9, period=7.65384, c20=-0.1378,
                   c20_err=0.0348, r_ref=11100.0),
    "deimos": dict(mass=1.51e15, mass_err=0.04e15, volume=1033e9, volume_err=19e9,
                   period=30.312, c20=None, r_ref=None),
}


def snap(path):
    return json.loads(path.read_text())


def radial(a, b):
    d = [x - y for x, y in zip(a["radius"], b["radius"])]
    rms = math.sqrt(sum(v * v for v in d) / len(d))
    return rms, max(abs(v) for v in d), sum(abs(v) > 1.0 for v in d)


def main():
    report = {}
    for body, obs in OBS.items():
        base = OUT / body
        if not (base / "forward" / "final.json").exists():
            continue
        present = snap(base / "history" / "present.json")
        initial = snap(base / "history" / "initial.json")
        final = snap(base / "forward" / "final.json")
        future = snap(base / "future" / "final.json")
        hist = json.loads((base / "history" / "history.json").read_text())
        r = {"nodes": len(present["radius"])}
        rms, mx, n = radial(initial, present)
        r["reconstruction_vs_present"] = dict(rms_m=rms, max_m=mx, nodes_moved_over_1m=n)
        rms, mx, n = radial(final, present)
        r["forward_vs_observed_shape"] = dict(rms_m=rms, max_m=mx, nodes_off_over_1m=n)
        rms, mx, n = radial(future, present)
        r["future_vs_present"] = dict(rms_m=rms, max_m=mx, nodes_moved_over_1m=n)
        r["forward_end"] = dict(t_s=final["time"], omega=final["omega"],
                                period_h=2 * math.pi / final["omega"] / 3600 if final["omega"] else None)
        r["future_end"] = dict(t_s=future["time"], omega=future["omega"],
                               period_h=2 * math.pi / future["omega"] / 3600 if future["omega"] else None)
        r["inversion"] = hist["iterations"]
        r["converged"] = hist["converged"]
        ins = base / "inspect_forward.json"
        if ins.exists():
            d = json.loads(ins.read_text())
            r["forward_mass"] = dict(model=d["mass"], observed=obs["mass"],
                                     rel=(d["mass"] - obs["mass"]) / obs["mass"],
                                     sigma=(d["mass"] - obs["mass"]) / obs["mass_err"])
            r["forward_volume"] = dict(model=d["volume"], observed=obs["volume"],
                                       rel=(d["volume"] - obs["volume"]) / obs["volume"],
                                       sigma=(d["volume"] - obs["volume"]) / obs["volume_err"])
            if obs["c20"] is not None and d.get("degree2"):
                c = d["degree2"]["c20"]
                r["forward_c20"] = dict(model=c, observed=obs["c20"],
                                        sigma=(c - obs["c20"]) / obs["c20_err"])
        if r["forward_end"]["period_h"]:
            p = r["forward_end"]["period_h"]
            r["forward_period"] = dict(model_h=p, observed_h=obs["period"],
                                       rel=(p - obs["period"]) / obs["period"])
        report[body] = r
    text = json.dumps(report, indent=2)
    (OUT / "evaluation.json").write_text(text)
    print(text)


if __name__ == "__main__":
    sys.exit(main())
