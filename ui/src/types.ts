// Mirrors of `src/app/bridge.rs` and of `dist/.stage/assets/scenarios.json`.

export type Phase = "idle" | "preparing" | "running" | "paused" | "done" | "failed";

export interface FrameStats {
  step: number;
  time: number;
  omega: number;
  mass: number;
  radius_min: number;
  radius_max: number;
  yielded_fraction: number;
  max_slack: number;
  rho_s_total: number;
  max_displacement: number;
  eps_momentum: number;
  eps_lambda: number;
  eps_picard: number;
  picard_iterations: number;
  tensile_fraction: number;
  inverse_ekman: number;
  mass_residual: number;
  mobile_area_fraction: number;
  dt: number;
  dt_cfl: number;
  dt_bound: string;
  tension_max: number;
  runout_traced: number;
  runout_distance: number;
  runout_dlat: number;
  runout_hops: number;
}

export interface Status {
  phase: Phase;
  status: string;
  notice: string | null;
  project: string;
  mesh: string;
  mesh_unit: string;
  spin: string;
  snapshot: string | null;
  params: Record<string, number>;
  stats: FrameStats | null;
  time_myr: number | null;
  t_end_myr: number | null;
  period_h: number | null;
  orbit: { a_km: number; period_h: number } | null;
  frames: number;
  view: number | null;
  live_step: number | null;
  live_time_myr: number | null;
  layer: number;
  interior: boolean;
  cut: number;
  wireframe: boolean;
  spin_axis: boolean;
  rho_range: [number, number];
  view_m: number;
}

export interface RunRequest {
  project: string;
  spin: string;
  overrides: Record<string, number>;
  snapshot: string | null;
  max_steps?: number | null;
}

export type Command =
  | { cmd: "pause"; paused: boolean }
  | ({ cmd: "restart" } & RunRequest)
  | { cmd: "layer"; mode: number }
  | { cmd: "interior"; on: boolean; offset: number }
  | { cmd: "wireframe"; on: boolean }
  | { cmd: "spin_axis"; on: boolean }
  | { cmd: "reset_view" }
  | { cmd: "screenshot" }
  | { cmd: "seek"; index: number | null }
  | { cmd: "pointer"; x: number; y: number; button?: number; down?: boolean }
  | { cmd: "wheel"; dy: number };

export interface Scenario {
  id: string;
  label: string;
  span: string;
  spin: string;
  snapshot: string | null;
}

export interface BodyEntry {
  id: string;
  name: string;
  designation: string;
  project: string;
  note: string;
  scenarios: Scenario[];
}
