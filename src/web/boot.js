const status = document.getElementById("status");

function fail(error) {
  const text = String(error && error.stack ? error.stack : error);
  status.textContent = text;
  console.error(text);
}

function normalize(path) {
  const out = [];
  for (const part of path.split("/")) {
    if (part === "" || part === ".") {
      continue;
    }
    if (part === "..") {
      out.pop();
    } else {
      out.push(part);
    }
  }
  return out.join("/");
}

function filesFor(project, text, spin) {
  const files = [project];
  const mesh = text.match(/^\s*mesh\s*=\s*"([^"]+)"/m);
  if (mesh) {
    const base = project.split("/").slice(0, -1).join("/");
    files.push(normalize(`${base}/${mesh[1]}`));
  }
  const prefix = "table-file:";
  if (spin.startsWith(prefix)) {
    files.push(spin.slice(prefix.length).trim());
  }
  return files;
}

function trunkAsset(pattern, scriptPattern) {
  const hits = performance
    .getEntriesByType("resource")
    .map((entry) => entry.name)
    .filter((name) => pattern.test(name));
  if (hits.length > 0) {
    return hits.at(-1);
  }
  for (const script of document.scripts) {
    const match = script.textContent.match(scriptPattern);
    if (match) {
      return new URL(match[1], document.baseURI).href;
    }
  }
  return "";
}

function viewerModuleUrl() {
  const url = trunkAsset(
    /\/astroid_evolution[^/]*\.js(?:\?|$)/,
    /from '([^']*astroid_evolution[^']*\.js)'/,
  );
  if (!url) {
    throw new Error("Trunk did not load the viewer module");
  }
  return url;
}

function viewerWasmUrl() {
  const url = trunkAsset(
    /\/astroid_evolution[^/]*_bg\.wasm(?:\?|$)/,
    /module_or_path:\s*'([^']*_bg\.wasm)'/,
  );
  if (!url) {
    throw new Error("Trunk did not load the viewer wasm");
  }
  return url;
}

addEventListener("TrunkApplicationStarted", () => {
  start().catch(fail);
});

async function start() {
  if (!navigator.gpu) {
    fail("This page needs WebGPU. Use a current desktop Chrome or Safari Technology Preview.");
    return;
  }
  if (typeof SharedArrayBuffer === "undefined") {
    fail("This page needs SharedArrayBuffer. Trunk must send COOP and COEP.");
    return;
  }
  const bindings = globalThis.wasmBindings;
  if (!bindings) {
    fail("Trunk did not publish the viewer bindings.");
    return;
  }
  const { boot, install_asset, deliver, prepare_viewer, viewer_defaults, ui_command } = bindings;
  const query = new URLSearchParams(location.search);
  // Files installed in this page's memory table; every new worker gets a copy.
  const installed = new Map();
  // The current run: its shared buffer and its two workers. A restart replaces it.
  let run = null;

  async function fetchBytes(path) {
    const response = await fetch(new URL(path, document.baseURI));
    if (!response.ok) {
      throw new Error(path + " " + response.status);
    }
    return new Uint8Array(await response.arrayBuffer());
  }

  async function install(path) {
    if (!installed.has(path)) {
      const bytes = await fetchBytes(path);
      install_asset(path, bytes);
      installed.set(path, bytes);
    }
    return installed.get(path);
  }

  /** The project, its mesh, a table spin file and the start snapshot. */
  async function ensure(project, spin, snapshot) {
    const text = new TextDecoder().decode(await install(project));
    const paths = filesFor(project, text, spin);
    if (snapshot) {
      paths.push(snapshot);
    }
    await Promise.all(paths.map(install));
  }

  function onWorker(event) {
    const data = event.data;
    if (!data || !run || (event.target !== run.worker && event.target !== run.gpu)) {
      return;
    }
    if (data.kind === "status" || data.kind === "frame" || data.kind === "finished" || data.kind === "failed") {
      deliver(data.kind, data.payload || "");
    }
  }

  globalThis.asteroid = {
    gpuCall() {
      throw new Error("compute runs on the evolution worker");
    },
    emit() {},
    waitGate() {
      return 0;
    },
    setFlag(index, value) {
      if (run) {
        Atomics.store(run.header, index, value);
        Atomics.notify(run.header, index);
      }
    },
    note(text) {
      status.textContent = String(text);
    },
    // Status JSON for the panels (ui/src/host.ts).
    last: "",
    status(json) {
      globalThis.asteroid.last = json;
      dispatchEvent(new CustomEvent("asteroid-status", { detail: json }));
    },
    command(json) {
      ui_command(json);
    },
    async restart(request) {
      await ensure(request.project, request.spin, request.snapshot);
      ui_command(JSON.stringify({ cmd: "restart", ...request }));
    },
    startWorker(config) {
      // A new run gets new workers and a new buffer; the old ones stop mid-step.
      if (run) {
        run.worker.terminate();
        run.gpu.terminate();
      }
      // Compute jobs go from the evolution worker straight to the GPU worker,
      // which owns its own WebGPU device; this thread only renders.
      const sab = new SharedArrayBuffer(256 + 96 * 1024 * 1024);
      const channel = new MessageChannel();
      const module = viewerModuleUrl();
      const wasm = viewerWasmUrl();
      const gpu = new Worker(new URL("gpu_worker.js", document.baseURI), { type: "module" });
      const worker = new Worker(new URL("worker.js", document.baseURI), { type: "module" });
      run = { sab, header: new Int32Array(sab, 0, 16), gpu, worker };
      gpu.onmessage = onWorker;
      worker.onmessage = onWorker;
      gpu.postMessage({ kind: "start", sab, module, wasm, port: channel.port2 }, [channel.port2]);
      const files = [...installed].map(([path, bytes]) => ({ path, bytes: bytes.slice().buffer }));
      worker.postMessage(
        { kind: "start", config, sab, module, wasm, files, port: channel.port1 },
        [channel.port1, ...files.map((file) => file.bytes)],
      );
    },
  };

  // Without a query the page opens the run the desktop shell opens.
  const defaults = JSON.parse(viewer_defaults());
  const project = query.get("project") || defaults.project;
  const spin = query.get("spin") || defaults.spin;
  const snapshot = query.get("snapshot");
  const maxSteps = query.get("max_steps");
  await ensure(project, spin, snapshot);
  const config = JSON.stringify({
    project,
    spin,
    snapshot,
    max_steps: maxSteps ? Number(maxSteps) : null,
  });
  status.textContent = "Decoding the mesh on the GPU…";
  await prepare_viewer(config);
  status.textContent = "";
  dispatchEvent(new Event("asteroid-ready"));
  boot(config);
}
