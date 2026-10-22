// The scenario catalog and the shipped project files, read at build time from
// `dist/.stage/assets` (one copy of the data for the page, the shell and the panels).

import scenarios from "../../dist/.stage/assets/scenarios.json";
import type { BodyEntry, Scenario } from "./types";

export const BODIES: BodyEntry[] = scenarios.bodies;

const projectFiles = import.meta.glob("../../dist/.stage/assets/projects/*.toml", {
  query: "?raw",
  import: "default",
  eager: true,
}) as Record<string, string>;

/** Numeric `section.key` values of a project file (enough TOML for the sliders). */
function numbers(text: string): Record<string, number> {
  const out: Record<string, number> = {};
  let section = "";
  for (const raw of text.split("\n")) {
    const line = raw.replace(/#.*/, "").trim();
    const header = line.match(/^\[([\w.]+)\]$/);
    if (header) {
      section = header[1];
      continue;
    }
    const pair = line.match(/^(\w+)\s*=\s*([-+0-9.eE]+)$/);
    if (pair && section) {
      out[`${section}.${pair[1]}`] = Number(pair[2]);
    }
  }
  return out;
}

/** Parameters of the project as shipped, keyed by its catalog path. */
export function baseParams(project: string): Record<string, number> {
  const name = project.split("/").pop();
  const hit = Object.entries(projectFiles).find(([path]) => path.endsWith(`/${name}`));
  return hit ? numbers(hit[1]) : {};
}

/** The catalog entry of a running project, matched by path suffix. */
export function findRun(
  project: string,
  spin: string,
  snapshot: string | null,
): { body: BodyEntry; scenario: Scenario | null } | null {
  const body = BODIES.find((b) => project.endsWith(b.project));
  if (!body) {
    return null;
  }
  const scenario =
    body.scenarios.find(
      (s) =>
        s.spin === spin &&
        (s.snapshot === null ? snapshot === null : (snapshot ?? "").endsWith(s.snapshot)),
    ) ?? null;
  return { body, scenario };
}
