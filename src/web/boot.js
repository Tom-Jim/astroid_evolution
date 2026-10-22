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
  const { boot, install_asset, deliver, execute_gpu, prepare_viewer, viewer_defaults } = bindings;
  const query = new URLSearchParams(location.search);
  const sab = new SharedArrayBuffer(256 + 96 * 1024 * 1024);
  const header = new Int32Array(sab, 0, 16);
  let worker = null;
  const pending = [];

  globalThis.asteroid = {
    gpuCall() {
      throw new Error("compute runs on the evolution worker");
    },
    emit() {},
    waitGate() {
      return 0;
    },
    setFlag(index, value) {
      Atomics.store(header, index, value);
      Atomics.notify(header, index);
    },
    note(text) {
      status.textContent = String(text);
    },
    startWorker(config) {
      worker = new Worker(new URL("worker.js", document.baseURI), { type: "module" });
      worker.onmessage = onWorker;
      worker.postMessage(
        { kind: "start", config, sab, module: viewerModuleUrl(), wasm: viewerWasmUrl(), files: pending },
        pending.map((file) => file.bytes),
      );
    },
  };

  async function onWorker(event) {
    const data = event.data;
    if (!data) {
      return;
    }
    if (data.kind === "gpu") {
      const len = Atomics.load(header, 1);
      const dataBytes = new Uint8Array(sab, 256);
      const request = dataBytes.slice(0, len);
      try {
        const out = new Uint8Array(await execute_gpu(request));
        if (out.length > dataBytes.length) {
          throw new Error("gpu response does not fit (" + out.length + " bytes)");
        }
        dataBytes.set(out);
        Atomics.store(header, 1, out.length);
        Atomics.store(header, 0, 2);
      } catch (error) {
        const text = String(error && error.stack ? error.stack : error);
        console.error(text);
        const msg = new TextEncoder().encode(text);
        dataBytes.set(msg);
        Atomics.store(header, 1, msg.length);
        Atomics.store(header, 0, 3);
      }
      Atomics.notify(header, 0);
      return;
    }
    if (data.kind === "status" || data.kind === "frame" || data.kind === "finished" || data.kind === "failed") {
      deliver(data.kind, data.payload || "");
      if (data.kind === "failed") {
        const text = data.payload || "evolution failed";
        const prior = status.dataset.fail || "";
        status.dataset.fail = prior && !text.includes(prior) ? prior + "\n" + text : text;
        status.textContent = status.dataset.fail;
      }
    }
  }

  // Without a query the page opens what `asteroid-app view` opens.
  const defaults = JSON.parse(viewer_defaults());
  const project = query.get("project") || defaults.project;
  const spin = query.get("spin") || defaults.spin;
  const maxSteps = query.get("max_steps");
  const projectResponse = await fetch(new URL(project, document.baseURI));
  if (!projectResponse.ok) {
    throw new Error(project + " " + projectResponse.status);
  }
  const projectBytes = new Uint8Array(await projectResponse.arrayBuffer());
  const projectText = new TextDecoder().decode(projectBytes);
  const paths = filesFor(project, projectText, spin);
  for (const path of paths) {
    const bytes = path === project
      ? projectBytes
      : await (async () => {
          const response = await fetch(new URL(path, document.baseURI));
          if (!response.ok) {
            throw new Error(path + " " + response.status);
          }
          return new Uint8Array(await response.arrayBuffer());
        })();
    install_asset(path, bytes);
    pending.push({ path, bytes: bytes.buffer });
  }
  const config = JSON.stringify({
    project,
    spin,
    max_steps: maxSteps ? Number(maxSteps) : null,
  });
  status.textContent = "Decoding the mesh on the GPU…";
  await prepare_viewer(config);
  status.textContent = "";
  boot(config);
}
