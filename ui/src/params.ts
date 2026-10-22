// Project keys the sliders edit. Each slider works in display units; `toKeys`
// turns the display value back into project values (`section.key`).

export interface ParamSpec {
  id: string;
  /** HUD tag. */
  en: string;
  /** Symbol of the quantity in mathstrict_EN.md. */
  symbol: string;
  hint: string;
  unit: string;
  min: number;
  max: number;
  /** Logarithmic track (strengths, depths and time steps span decades). */
  log?: boolean;
  integer?: boolean;
  step?: number;
  read(params: Record<string, number>): number;
  toKeys(value: number, params: Record<string, number>): Record<string, number>;
}

const DEG = Math.PI / 180;

const direct = (key: string) => ({
  read: (p: Record<string, number>) => p[key],
  toKeys: (value: number) => ({ [key]: value }),
});

export const PARAMS: ParamSpec[] = [
  {
    id: "phi",
    en: "FRICTION ANGLE",
    symbol: "φ",
    hint: "μ_s = tan φ; μ₂ moves with it, keeping μ₂ − μ_s fixed (μ(I) law)",
    unit: "°",
    min: 20,
    max: 45,
    step: 0.5,
    read: (p) => Math.atan(p["material.mu_s"]) / DEG,
    toKeys: (value, p) => {
      const mu = Math.tan(value * DEG);
      return {
        "material.mu_s": mu,
        "material.mu_2": mu + (p["material.mu_2"] - p["material.mu_s"]),
      };
    },
  },
  {
    id: "cb",
    en: "BULK COHESION",
    symbol: "c_b",
    hint: "Strength of the rubble pile at depth (§4.3a)",
    unit: "Pa",
    min: 0.1,
    max: 200,
    log: true,
    ...direct("material.cohesion"),
  },
  {
    id: "cs",
    en: "SURFACE COHESION",
    symbol: "c_s",
    hint: "Strength of the regolith at the surface, the limit d → 0",
    unit: "Pa",
    min: 0.01,
    max: 20,
    log: true,
    ...direct("material.cohesion_surface"),
  },
  {
    id: "hc",
    en: "STRENGTH DEPTH",
    symbol: "h_c",
    hint: "c(d) = c_b + (c_s − c_b)·e^(−d/h_c)",
    unit: "m",
    min: 0.1,
    max: 500,
    log: true,
    ...direct("material.cohesion_depth"),
  },
  {
    id: "hs",
    en: "ACTIVE LAYER",
    symbol: "h_s",
    hint: "Thickness of the Mode C loose (landslide) layer",
    unit: "m",
    min: 0.1,
    max: 50,
    log: true,
    ...direct("material.h_s"),
  },
  {
    id: "level",
    en: "MESH LEVEL",
    symbol: "L",
    hint: "Icosphere ray resampling: 10·4ᴸ + 2 nodes",
    unit: "",
    min: 3,
    max: 6,
    integer: true,
    step: 1,
    ...direct("body.level"),
  },
  {
    id: "dt",
    en: "MAX TIME STEP",
    symbol: "Δt_max",
    hint: "The CFL bound and the spin-rate change can still shorten the step",
    unit: "s",
    min: 1e9,
    max: 1e13,
    log: true,
    ...direct("step.dt_max"),
  },
];

/** Overrides that turn `base` into the draft values. */
export function overrides(
  draft: Record<string, number>,
  base: Record<string, number>,
): Record<string, number> {
  const out: Record<string, number> = {};
  for (const spec of PARAMS) {
    const value = draft[spec.id];
    if (value === undefined || same(value, spec.read(base))) {
      continue;
    }
    Object.assign(out, spec.toKeys(value, base));
  }
  return out;
}

export function displayValues(params: Record<string, number>): Record<string, number> {
  const out: Record<string, number> = {};
  for (const spec of PARAMS) {
    const value = spec.read(params);
    if (Number.isFinite(value)) {
      out[spec.id] = value;
    }
  }
  return out;
}

export function same(a: number, b: number): boolean {
  return Math.abs(a - b) <= 1e-9 * Math.max(1, Math.abs(a), Math.abs(b));
}

/** Compact engineering-style number: 3 significant digits, exponent outside 1e-3..1e5. */
export function fmt(value: number | null | undefined, digits = 3): string {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return "—";
  }
  if (value === 0) {
    return "0";
  }
  const abs = Math.abs(value);
  if (abs >= 1e5 || abs < 1e-3) {
    const [mantissa, exponent] = value.toExponential(digits - 1).split("e");
    return `${mantissa}×10${superscript(Number(exponent))}`;
  }
  return Number(value.toPrecision(digits)).toLocaleString("en-US", {
    maximumFractionDigits: 6,
  });
}

function superscript(n: number): string {
  const map: Record<string, string> = {
    "-": "⁻",
    "0": "⁰",
    "1": "¹",
    "2": "²",
    "3": "³",
    "4": "⁴",
    "5": "⁵",
    "6": "⁶",
    "7": "⁷",
    "8": "⁸",
    "9": "⁹",
  };
  return String(n)
    .split("")
    .map((c) => map[c] ?? c)
    .join("");
}
