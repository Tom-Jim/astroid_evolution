// The viewer behind the panels: the Tauri shell (IPC) or the browser page (wasm
// bindings that boot.js publishes on `globalThis.asteroid`). Both take the same JSON
// commands (`src/app/bridge.rs`) and send back the same status object.

import type { Command, RunRequest, Status } from "./types";

export interface Host {
  /** The Tauri webview covers the window, so pointer input is forwarded to Bevy. */
  readonly forwardsPointer: boolean;
  send(command: Command): void;
  /** Fetch whatever the run needs (the page installs files in memory), then restart. */
  restart(request: RunRequest): Promise<void>;
  onStatus(listener: (status: Status) => void): () => void;
}

interface PageBridge {
  command(json: string): void;
  restart(request: RunRequest): Promise<void>;
  last?: string;
}

declare global {
  interface Window {
    __TAURI_INTERNALS__?: unknown;
    asteroid?: Partial<PageBridge>;
  }
}

async function tauriHost(): Promise<Host> {
  const { invoke } = await import("@tauri-apps/api/core");
  const { listen } = await import("@tauri-apps/api/event");
  const send = (command: Command) => {
    invoke("ui_command", { json: JSON.stringify(command) }).catch((err) =>
      console.error("ui_command", err),
    );
  };
  return {
    forwardsPointer: true,
    send,
    async restart(request) {
      send({ cmd: "restart", ...request });
    },
    onStatus(listener) {
      const stop = listen<string>("status", (event) => listener(JSON.parse(event.payload)));
      return () => {
        stop.then((unlisten) => unlisten());
      };
    },
  };
}

function pageHost(): Host {
  const bridge = () => window.asteroid as PageBridge;
  return {
    forwardsPointer: false,
    send(command) {
      try {
        bridge().command(JSON.stringify(command));
      } catch (err) {
        console.error("ui_command", err);
      }
    },
    restart(request) {
      return bridge().restart(request);
    },
    onStatus(listener) {
      const handler = (event: Event) => listener(JSON.parse((event as CustomEvent<string>).detail));
      addEventListener("asteroid-status", handler);
      const last = window.asteroid?.last;
      if (last) {
        listener(JSON.parse(last));
      }
      return () => removeEventListener("asteroid-status", handler);
    },
  };
}

export async function connect(): Promise<Host> {
  if (window.__TAURI_INTERNALS__) {
    return tauriHost();
  }
  // The page publishes its bridge once the wasm bindings are loaded.
  if (!window.asteroid?.command) {
    await new Promise((resolve) => addEventListener("asteroid-ready", resolve, { once: true }));
  }
  return pageHost();
}
