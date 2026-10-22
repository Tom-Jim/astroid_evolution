// Plain HTML/SVG controls in one style: bracketed panel, slider, tabs, toggle,
// telemetry row and ring gauge.

import { useId, type CSSProperties, type ReactNode } from "react";
import { fmt, type ParamSpec } from "./params";

/** Tag in tracked capitals, with an optional detail (symbol, description) beside it. */
export function Label(props: { en: string; sub?: string }) {
  return (
    <span className="label">
      <b>{props.en}</b>
      {props.sub && <i>{props.sub}</i>}
    </span>
  );
}

export function Panel(props: {
  en: string;
  sub?: string;
  className?: string;
  aside?: ReactNode;
  children: ReactNode;
}) {
  return (
    <section className={`panel ${props.className ?? ""}`}>
      <header className="panel-head">
        <Label en={props.en} sub={props.sub} />
        {props.aside}
      </header>
      {props.children}
    </section>
  );
}

/** A slider in display units; log specs move along log10 of the value. */
export function ParamSlider(props: {
  spec: ParamSpec;
  value: number;
  changed: boolean;
  onChange: (value: number) => void;
}) {
  const { spec, value } = props;
  const id = useId();
  const toTrack = (v: number) => (spec.log ? Math.log10(v) : v);
  const fromTrack = (t: number) => {
    const v = spec.log ? 10 ** t : t;
    return spec.integer ? Math.round(v) : v;
  };
  const lo = toTrack(spec.min);
  const hi = toTrack(spec.max);
  const track = Math.min(hi, Math.max(lo, toTrack(value)));
  const fill = ((track - lo) / (hi - lo)) * 100;
  return (
    <div className={`slider ${props.changed ? "changed" : ""}`} title={spec.hint}>
      <label htmlFor={id}>
        <Label en={spec.en} sub={spec.symbol} />
        <output>
          {spec.integer ? value.toFixed(0) : fmt(value)}
          {spec.unit && <small>{spec.unit}</small>}
        </output>
      </label>
      <input
        id={id}
        type="range"
        min={lo}
        max={hi}
        step={spec.log ? (hi - lo) / 400 : (spec.step ?? (hi - lo) / 200)}
        value={track}
        style={{ "--fill": `${fill}%` } as CSSProperties}
        onChange={(e) => props.onChange(fromTrack(Number(e.target.value)))}
      />
    </div>
  );
}

export function Tabs<T extends string | number>(props: {
  options: { value: T; label: ReactNode; title?: string }[];
  value: T;
  onChange: (value: T) => void;
  small?: boolean;
}) {
  return (
    <div className={`tabs ${props.small ? "small" : ""}`} role="radiogroup">
      {props.options.map((o) => (
        <button
          key={String(o.value)}
          role="radio"
          aria-checked={o.value === props.value}
          className={o.value === props.value ? "on" : ""}
          title={o.title}
          onClick={() => props.onChange(o.value)}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function Toggle(props: {
  on: boolean;
  label: ReactNode;
  title?: string;
  onChange: (on: boolean) => void;
}) {
  return (
    <button
      className={`chip ${props.on ? "on" : ""}`}
      aria-pressed={props.on}
      title={props.title}
      onClick={() => props.onChange(!props.on)}
    >
      <i className="led" aria-hidden />
      {props.label}
    </button>
  );
}

/** One telemetry line: tag on the left, value on the right. */
export function Readout(props: { en: string; sub?: string; value: ReactNode; unit?: string }) {
  return (
    <div className="readout">
      <Label en={props.en} sub={props.sub} />
      <strong title={typeof props.value === "string" ? props.value : undefined}>
        {props.value}
        {props.unit && <small>{props.unit}</small>}
      </strong>
    </div>
  );
}

/** A ring that fills with a fraction in [0, 1]. */
export function Gauge(props: { en: string; sub?: string; fraction: number | null; text: string }) {
  const r = 30;
  const c = 2 * Math.PI * r;
  const f = Math.max(0, Math.min(1, props.fraction ?? 0));
  return (
    <figure className="gauge">
      <div className="dial">
        <svg viewBox="0 0 76 76" aria-hidden>
          <circle cx="38" cy="38" r={r + 5} className="ticks" />
          <circle cx="38" cy="38" r={r} className="rail" />
          <circle
            cx="38"
            cy="38"
            r={r}
            className="arc"
            strokeDasharray={`${c * f} ${c}`}
            transform="rotate(-90 38 38)"
          />
        </svg>
        <strong>{props.text}</strong>
      </div>
      <figcaption>
        <Label en={props.en} sub={props.sub} />
      </figcaption>
    </figure>
  );
}

export function Icon(props: { name: "play" | "pause" | "reset" | "camera" | "restart" }) {
  const paths: Record<string, string> = {
    play: "M8 5v14l11-7z",
    pause: "M7 5h3.5v14H7zM13.5 5H17v14h-3.5z",
    reset: "M12 5a7 7 0 1 1-6.6 4.7M5 4v5h5",
    camera: "M4 8h3l2-3h6l2 3h3v11H4zM12 16.5a3.5 3.5 0 1 0 0-7 3.5 3.5 0 0 0 0 7z",
    restart: "M5 12a7 7 0 1 0 2-4.9M5 4v4h4",
  };
  const filled = props.name === "play" || props.name === "pause";
  return (
    <svg viewBox="0 0 24 24" width="16" height="16" aria-hidden>
      <path
        d={paths[props.name]}
        fill={filled ? "currentColor" : "none"}
        stroke={filled ? "none" : "currentColor"}
        strokeWidth="1.5"
        strokeLinecap="square"
      />
    </svg>
  );
}

/** Orbit mark of the header. */
export function Mark() {
  return (
    <svg className="mark" viewBox="0 0 40 40" aria-hidden>
      <ellipse cx="20" cy="20" rx="18" ry="7" transform="rotate(-24 20 20)" />
      <circle cx="20" cy="20" r="6.5" />
      <circle cx="36.4" cy="12.7" r="1.6" className="moon" />
    </svg>
  );
}
