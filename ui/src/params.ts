// Project keys the sliders edit. Each slider works in display units; `toKeys`
// turns the display value back into project values (`section.key`).

export interface ParamSpec {
  id: string;
  /** HUD tag. */
  en: string;
  label: string;
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
    label: "内摩擦角",
    hint: "μ_s = tan φ；μ₂ 随之平移，保持 μ₂ − μ_s 不变（μ(I) 律）",
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
    label: "内部黏聚力",
    hint: "深部碎石堆强度（§4.3a）",
    unit: "Pa",
    min: 0.1,
    max: 200,
    log: true,
    ...direct("material.cohesion"),
  },
  {
    id: "cs",
    en: "SURFACE COHESION",
    label: "表层黏聚力",
    hint: "风化层表面强度，d → 0 时的极限",
    unit: "Pa",
    min: 0.01,
    max: 20,
    log: true,
    ...direct("material.cohesion_surface"),
  },
  {
    id: "hc",
    en: "STRENGTH DEPTH",
    label: "过渡深度",
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
    label: "活动层",
    hint: "Mode C 松散层（滑坡层）厚度",
    unit: "m",
    min: 0.1,
    max: 50,
    log: true,
    ...direct("material.h_s"),
  },
  {
    id: "level",
    en: "MESH LEVEL",
    label: "网格级别",
    hint: "icosphere 射线重采样：10·4ᴸ + 2 个节点",
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
    label: "最大步长",
    hint: "CFL 与自转变化率仍会进一步限制步长",
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
