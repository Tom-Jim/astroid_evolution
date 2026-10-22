import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
  type PointerEvent,
} from "react";
import { BODIES, baseParams, findRun } from "./catalog";
import { Gauge, Icon, Label, Mark, Panel, ParamSlider, Readout, Tabs, Toggle } from "./controls";
import type { Host } from "./host";
import { PARAMS, displayValues, fmt, overrides, same } from "./params";
import type { Phase, Status } from "./types";

const PHASE: Record<Phase, [string, string]> = {
  idle: ["STANDBY", "未启动"],
  preparing: ["INITIALIZING", "准备中"],
  running: ["NOMINAL", "演化中"],
  paused: ["HOLD", "已暂停"],
  done: ["COMPLETE", "完成"],
  failed: ["FAULT", "失败"],
};

const SCENARIO_TAG: Record<string, string> = {
  past: "RECONSTRUCTION",
  future: "FORECAST",
  measured: "MEASURED SHAPE",
};

const LAYERS = [
  { value: 1, en: "YIELD", zh: "屈服", legend: "橙色：已屈服区域，按 F_δ / p95 着色" },
  { value: 2, en: "SLOPE", zh: "坡度", legend: "闪烁黄：坡面松弛量超过容差" },
  { value: 3, en: "REGOLITH", zh: "风化层", legend: "棕褐：松散层面密度 ρ_s，移动条纹为流向" },
  { value: 4, en: "COMBINED", zh: "综合", legend: "屈服 + 坡度 + 风化层叠加" },
];

export function App({ host }: { host: Host }) {
  const [status, setStatus] = useState<Status | null>(null);
  const [bodyId, setBodyId] = useState(BODIES[0].id);
  const [scenarioId, setScenarioId] = useState(BODIES[0].scenarios[0].id);
  const [draft, setDraft] = useState<Record<string, number>>({});
  const [spin, setSpin] = useState(BODIES[0].scenarios[0].spin);
  const [collapsed, setCollapsed] = useState(false);
  const [busy, setBusy] = useState(false);
  const adopted = useRef(false);

  useEffect(() => host.onStatus(setStatus), [host]);

  const body = BODIES.find((b) => b.id === bodyId) ?? BODIES[0];
  const scenario = body.scenarios.find((s) => s.id === scenarioId) ?? body.scenarios[0];
  const base = useMemo(() => baseParams(body.project), [body.project]);

  const running = status ? findRun(status.project, status.spin, status.snapshot) : null;
  const future = running?.scenario?.id === "future";

  // The first status tells the panels which run the viewer opened with.
  useEffect(() => {
    if (!status || adopted.current) {
      return;
    }
    adopted.current = true;
    if (running) {
      setBodyId(running.body.id);
      setScenarioId((running.scenario ?? running.body.scenarios[0]).id);
      setSpin(status.spin);
    }
    setDraft(displayValues(status.params));
  }, [status, running]);

  const chooseBody = (id: string) => {
    const next = BODIES.find((b) => b.id === id)!;
    setBodyId(id);
    setScenarioId(next.scenarios[0].id);
    setSpin(next.scenarios[0].spin);
    setDraft(displayValues(baseParams(next.project)));
  };
  const chooseScenario = (id: string) => {
    setScenarioId(id);
    setSpin(body.scenarios.find((s) => s.id === id)!.spin);
  };

  const start = useCallback(async () => {
    setBusy(true);
    try {
      await host.restart({
        project: body.project,
        spin: spin.trim(),
        overrides: overrides(draft, base),
        snapshot: scenario.snapshot,
      });
    } catch (err) {
      console.error(err);
    } finally {
      setBusy(false);
    }
  }, [host, body, spin, draft, base, scenario]);

  const send = host.send.bind(host);
  const paused = status?.phase === "paused";
  const layer = status?.layer ?? 4;
  const interior = status?.interior ?? false;
  const cut = status?.cut ?? 0;

  useKeys(host, status);

  const edited = PARAMS.some((p) => draft[p.id] !== undefined && !same(draft[p.id], p.read(base)));
  const runningHere = running?.body.id === body.id && running?.scenario?.id === scenario.id;

  return (
    <div className={`shell ${host.forwardsPointer ? "forward" : ""}`}>
      {host.forwardsPointer && <Viewport host={host} />}

      <aside className={`column left ${collapsed ? "collapsed" : ""}`}>
        <div className="brand">
          <Mark />
          <div>
            <h1>ASTEROID EVOLUTION</h1>
            <p>小行星形貌演化 · STOKES / MODE C / YORP / TIDES</p>
          </div>
          <button
            className="collapse"
            title={collapsed ? "展开面板" : "收起面板"}
            onClick={() => setCollapsed(!collapsed)}
          >
            {collapsed ? "+" : "−"}
          </button>
        </div>

        {!collapsed && (
          <>
            <Panel en="MISSION" zh="天体与情景">
              <Tabs
                options={BODIES.map((b) => {
                  const [zh, en] = b.name.split(" ");
                  return {
                    value: b.id,
                    title: b.name,
                    label: (
                      <>
                        <b>{en.toUpperCase()}</b>
                        <i>{zh}</i>
                      </>
                    ),
                  };
                })}
                value={body.id}
                onChange={chooseBody}
              />
              <p className="note">{body.note}</p>
              <div className="scenarios">
                {body.scenarios.map((s) => (
                  <button
                    key={s.id}
                    className={`scenario ${s.id === scenario.id ? "on" : ""}`}
                    onClick={() => chooseScenario(s.id)}
                  >
                    <Label en={SCENARIO_TAG[s.id] ?? s.id.toUpperCase()} zh={s.label} />
                    <small>{s.span}</small>
                    {running?.body.id === body.id && running?.scenario?.id === s.id && (
                      <i className="live" title="正在运行" />
                    )}
                  </button>
                ))}
              </div>
              <details className="spin">
                <summary>
                  <Label en="SPIN SCHEDULE" zh="自转 / 轨道时间表" />
                </summary>
                <input
                  value={spin}
                  spellCheck={false}
                  onChange={(e) => setSpin(e.target.value)}
                  aria-label="spin schedule"
                />
              </details>
            </Panel>

            <Panel
              en="PARAMETERS"
              zh="物理参数"
              aside={edited ? <span className="badge">MODIFIED</span> : null}
            >
              <div className="sliders">
                {PARAMS.map((spec) => (
                  <ParamSlider
                    key={spec.id}
                    spec={spec}
                    value={draft[spec.id] ?? spec.read(base)}
                    changed={draft[spec.id] !== undefined && !same(draft[spec.id], spec.read(base))}
                    onChange={(v) => setDraft({ ...draft, [spec.id]: v })}
                  />
                ))}
              </div>
              <div className="actions">
                <button className="ghost" onClick={() => setDraft(displayValues(base))}>
                  RESET <i>文献值</i>
                </button>
                <button className="primary" disabled={busy} onClick={start}>
                  <Icon name="restart" />
                  {runningHere ? "RESTART" : "LAUNCH"} <i>{runningHere ? "重新开始" : "开始演化"}</i>
                </button>
              </div>
            </Panel>
          </>
        )}
      </aside>

      <aside className="column right">
        <StatusPanel status={status} future={future} />
      </aside>

      <footer className="transport">
        <Timeline status={status} future={future} />
        <div className="tools">
          <button
            className="round"
            title="暂停 / 继续（空格）"
            disabled={!status || !["running", "paused"].includes(status.phase)}
            onClick={() => send({ cmd: "pause", paused: !paused })}
          >
            <Icon name={paused ? "play" : "pause"} />
          </button>
          <Tabs
            small
            options={LAYERS.map((l, i) => ({
              value: l.value,
              label: (
                <>
                  <b>{l.en}</b>
                  <i>{l.zh}</i>
                </>
              ),
              title: `${l.legend}（${i + 1}）`,
            }))}
            value={layer}
            onChange={(mode) => send({ cmd: "layer", mode })}
          />
          <span className="sep" />
          <Toggle
            on={interior}
            label={<Label en="INTERIOR" zh="内部密度" />}
            title="过球心、朝向相机的剖面（I）"
            onChange={(on) => send({ cmd: "interior", on, offset: 0 })}
          />
          {interior && (
            <input
              className="cut"
              type="range"
              min={-0.9}
              max={0.9}
              step={0.01}
              value={cut}
              title="剖面偏移（[ ]）"
              style={{ "--fill": `${((cut + 0.9) / 1.8) * 100}%` } as CSSProperties}
              onChange={(e) => send({ cmd: "interior", on: true, offset: Number(e.target.value) })}
            />
          )}
          <Toggle
            on={status?.wireframe ?? false}
            label={<Label en="MESH" zh="网格" />}
            title="线框（W）"
            onChange={(on) => send({ cmd: "wireframe", on })}
          />
          <span className="grow" />
          <button className="icon" title="重置视角（R）" onClick={() => send({ cmd: "reset_view" })}>
            <Icon name="reset" />
          </button>
          <button className="icon" title="截图（P）" onClick={() => send({ cmd: "screenshot" })}>
            <Icon name="camera" />
          </button>
        </div>
        <div className="legend">
          <span>
            {interior && status
              ? `内部密度 ${fmt(status.rho_range[0])} → ${fmt(status.rho_range[1])} kg/m³（靛蓝 → 黄，每 ${fmt((status.rho_range[1] - status.rho_range[0]) / 10)} 一条等值线）`
              : LAYERS.find((l) => l.value === layer)?.legend}
          </span>
          <span className="mouse">RMB 旋转 · LMB 平移 · WHEEL 缩放</span>
        </div>
      </footer>

      {status?.notice && (
        <div className="toast">
          <Label en="FAULT" zh="命令未执行" />
          <p>{status.notice}</p>
        </div>
      )}
    </div>
  );
}

/** Signed epoch relative to today: history runs end today, the future run starts today. */
function epoch(status: Status, future: boolean): number | null {
  if (status.time_myr === null) {
    return null;
  }
  if (future) {
    return status.time_myr;
  }
  return status.t_end_myr === null ? status.time_myr : status.time_myr - status.t_end_myr;
}

function Timeline({ status, future }: { status: Status | null; future: boolean }) {
  const t = status?.time_myr ?? 0;
  const end = status?.t_end_myr ?? null;
  const progress = end ? Math.min(1, t / end) : 0;
  const when = status ? epoch(status, future) : null;
  const sign = when === null ? "" : when < 0 ? "−" : "+";
  return (
    <div className="timeline">
      <div className="epoch">
        <strong>
          T{sign}
          {when === null ? "--.---" : Math.abs(when).toFixed(3).padStart(6, "0")}
        </strong>
        <Label
          en={when === null ? "AWAITING STEP" : when < 0 ? "MYR BEFORE PRESENT" : "MYR AFTER PRESENT"}
          zh={when === null ? "等待第一步" : when < 0 ? "距今" : "未来"}
        />
      </div>
      <div className="track" aria-label="schedule progress">
        <div className="bar" style={{ width: `${progress * 100}%` }} />
        <div className="head" style={{ left: `${progress * 100}%` }} />
      </div>
      <div className="span">
        <strong>{fmt(t)}</strong>
        <small> / {fmt(end)} MYR</small>
      </div>
    </div>
  );
}

function StatusPanel({ status, future }: { status: Status | null; future: boolean }) {
  const s = status?.stats;
  const phase = status ? PHASE[status.phase] : null;
  const progress = status?.time_myr != null && status.t_end_myr ? status.time_myr / status.t_end_myr : null;
  return (
    <Panel
      en="TELEMETRY"
      zh="演化状态"
      aside={
        phase && (
          <span className={`phase ${status!.phase}`}>
            <i className="led" />
            {phase[0]} <em>{phase[1]}</em>
          </span>
        )
      }
    >
      <p className="status-line">{status?.status ?? "connecting…"}</p>
      <div className="gauges">
        <Gauge en="YIELD" zh="屈服" fraction={s?.yielded_fraction ?? null} text={s ? `${fmt(100 * s.yielded_fraction, 2)}%` : "—"} />
        <Gauge
          en="MOBILE"
          zh="可动面积"
          fraction={s?.mobile_area_fraction ?? null}
          text={s ? `${fmt(100 * s.mobile_area_fraction, 2)}%` : "—"}
        />
        <Gauge en="SCHEDULE" zh="进度" fraction={progress} text={progress === null ? "—" : `${(100 * progress).toFixed(1)}%`} />
      </div>
      <div className="readouts">
        <Readout en="STEP" zh="步数" value={s ? s.step.toLocaleString() : "—"} />
        <Readout en={future ? "ELAPSED" : "RUN TIME"} zh={future ? "今后" : "演化时间"} value={s ? fmt(status?.time_myr) : "—"} unit="Myr" />
        <Readout en="SPIN PERIOD" zh="自转周期" value={fmt(status?.period_h, 5)} unit="h" />
        {status?.orbit && <Readout en="SEMI-MAJOR AXIS" zh="轨道半长轴" value={fmt(status.orbit.a_km, 5)} unit="km" />}
        <Readout en="MASS" zh="质量" value={fmt(s?.mass, 5)} unit="kg" />
        <Readout en="RADIUS" zh="半径范围" value={s ? `${fmt(s.radius_min, 4)} – ${fmt(s.radius_max, 4)}` : "—"} unit="m" />
        <Readout en="MAX |ΔR|" zh="单步最大位移" value={fmt(s?.max_displacement)} unit="m" />
        <Readout en="REGOLITH" zh="松散层质量" value={fmt(s?.rho_s_total)} unit="kg" />
        <Readout en="TIME STEP" zh="时间步" value={fmt(s?.dt)} unit="s" />
        <Readout en="LIMITED BY" zh="步长约束" value={s?.dt_bound ?? "—"} />
      </div>
      <details className="diagnostics">
        <summary>
          <Label en="SOLVER" zh="求解器诊断" />
        </summary>
        <div className="readouts">
          <Readout en="ε MOMENTUM" zh="动量残差" value={fmt(s?.eps_momentum)} />
          <Readout en="ε MULTIPLIER" zh="乘子残差" value={fmt(s?.eps_lambda)} />
          <Readout en="PICARD" zh="迭代" value={s ? `${s.picard_iterations} · ${fmt(s.eps_picard)}` : "—"} />
          <Readout en="TENSILE" zh="张拉比例" value={fmt(s?.tensile_fraction)} />
          <Readout en="EK⁻¹" zh="Ekman 逆数" value={fmt(s?.inverse_ekman)} />
          <Readout en="MASS RESIDUAL" zh="质量残差" value={fmt(s?.mass_residual)} />
          <Readout en="MAX SLACK" zh="坡面松弛" value={fmt(s?.max_slack)} />
          <Readout en="CFL STEP" zh="CFL 步长" value={fmt(s?.dt_cfl)} unit="s" />
        </div>
        {status && (
          <dl className="files">
            <dt>PROJECT</dt>
            <dd>{status.project}</dd>
            <dt>MESH</dt>
            <dd>
              {status.mesh} ({status.mesh_unit})
            </dd>
            {status.snapshot && (
              <>
                <dt>START</dt>
                <dd>{status.snapshot}</dd>
              </>
            )}
            <dt>SCALE</dt>
            <dd>1 unit = {fmt(status.view_m)} m</dd>
          </dl>
        )}
      </details>
    </Panel>
  );
}

/** Keyboard shortcuts of the old in-scene HUD, now handled by the panels. */
function useKeys(host: Host, status: Status | null) {
  const ref = useRef(status);
  ref.current = status;
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement;
      if (target.tagName === "INPUT" && (target as HTMLInputElement).type === "text") {
        return;
      }
      const s = ref.current;
      if (!s || e.metaKey || e.ctrlKey || e.altKey) {
        return;
      }
      const key = e.key.toLowerCase();
      if (key === " ") {
        e.preventDefault();
        host.send({ cmd: "pause", paused: s.phase !== "paused" });
      } else if (key >= "1" && key <= "4") {
        host.send({ cmd: "layer", mode: Number(key) });
      } else if (key === "w") {
        host.send({ cmd: "wireframe", on: !s.wireframe });
      } else if (key === "r") {
        host.send({ cmd: "reset_view" });
      } else if (key === "p") {
        host.send({ cmd: "screenshot" });
      } else if (key === "i") {
        host.send({ cmd: "interior", on: !s.interior, offset: 0 });
      } else if ((key === "[" || key === "]") && s.interior) {
        const offset = Math.max(-0.9, Math.min(0.9, s.cut + (key === "]" ? 0.05 : -0.05)));
        host.send({ cmd: "interior", on: true, offset });
      }
    };
    addEventListener("keydown", onKey);
    return () => removeEventListener("keydown", onKey);
  }, [host]);
}

/**
 * The Tauri webview covers the Bevy surface. This layer sits under the panels and
 * forwards the pointer (coalesced to one move per animation frame).
 */
function Viewport({ host }: { host: Host }) {
  const pending = useRef<{ x: number; y: number } | null>(null);
  const wheel = useRef(0);
  const frame = useRef(0);

  const flush = useCallback(() => {
    frame.current = 0;
    if (pending.current) {
      host.send({ cmd: "pointer", ...pending.current });
      pending.current = null;
    }
    if (wheel.current) {
      host.send({ cmd: "wheel", dy: wheel.current });
      wheel.current = 0;
    }
  }, [host]);
  const schedule = () => {
    if (!frame.current) {
      frame.current = requestAnimationFrame(flush);
    }
  };
  const button = (e: PointerEvent, down: boolean) => {
    pending.current = null;
    host.send({ cmd: "pointer", x: e.clientX, y: e.clientY, button: e.button, down });
  };

  return (
    <div
      className="viewport"
      onContextMenu={(e) => e.preventDefault()}
      onPointerMove={(e) => {
        pending.current = { x: e.clientX, y: e.clientY };
        schedule();
      }}
      onPointerDown={(e) => {
        e.currentTarget.setPointerCapture(e.pointerId);
        button(e, true);
      }}
      onPointerUp={(e) => button(e, false)}
      onWheel={(e) => {
        wheel.current += e.deltaY;
        schedule();
      }}
    />
  );
}
