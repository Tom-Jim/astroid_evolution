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

function jetTables() {
  const count = 35;
  const mono = [];
  const deg = [];
  for (let degree = 0; degree <= 4; degree += 1) {
    for (let a = degree; a >= 0; a -= 1) {
      for (let b = degree - a; b >= 0; b -= 1) {
        mono.push([a, b, degree - a - b]);
        deg.push(degree);
      }
    }
  }
  const indexOf = (term) => {
    for (let i = 0; i < count; i += 1) {
      if (mono[i][0] === term[0] && mono[i][1] === term[1] && mono[i][2] === term[2]) {
        return i;
      }
    }
    return count;
  };
  const up = [[], [], []];
  for (let axis = 0; axis < 3; axis += 1) {
    for (let i = 0; i < count; i += 1) {
      const term = [mono[i][0], mono[i][1], mono[i][2]];
      term[axis] += 1;
      up[axis][i] = term[0] + term[1] + term[2] <= 4 ? indexOf(term) : count;
    }
  }
  const words = [];
  const mulOff = [];
  const mulN = [];
  for (let ord = 0; ord <= 4; ord += 1) {
    mulOff[ord] = words.length;
    let n = 0;
    for (let i = 0; i < count; i += 1) {
      for (let j = 0; j < count; j += 1) {
        if (deg[i] + deg[j] <= ord) {
          const term = [
            mono[i][0] + mono[j][0],
            mono[i][1] + mono[j][1],
            mono[i][2] + mono[j][2],
          ];
          words.push((i | (j << 8) | (indexOf(term) << 16)) >>> 0);
          n += 1;
        }
      }
    }
    mulN[ord] = n;
  }
  const upOff = words.length;
  for (let axis = 0; axis < 3; axis += 1) {
    for (let i = 0; i < count; i += 1) {
      words.push(up[axis][i]);
    }
  }
  const monoOff = words.length;
  for (let i = 0; i < count; i += 1) {
    words.push(mono[i][0], mono[i][1], mono[i][2]);
  }
  const degOff = words.length;
  for (let i = 0; i < count; i += 1) {
    words.push(deg[i]);
  }
  const pairOff = words.length;
  for (let b = 0; b < 3; b += 1) {
    for (let c = 0; c < 3; c += 1) {
      const term = [0, 0, 0];
      term[b] += 1;
      term[c] += 1;
      words.push(indexOf(term));
    }
  }
  return { words, mul_off: mulOff, mul_n: mulN, up_off: upOff, mono_off: monoOff, deg_off: degOff, pair_off: pairOff };
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
    /\/asteroid-viewer[^/]*\.js(?:\?|$)/,
    /from '([^']*asteroid-viewer[^']*\.js)'/,
  );
  if (!url) {
    throw new Error("Trunk did not load the viewer module");
  }
  return url;
}

function viewerWasmUrl() {
  const url = trunkAsset(
    /\/asteroid-viewer[^/]*_bg\.wasm(?:\?|$)/,
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
  const { boot, install_asset, deliver, execute_gpu, prepare_viewer } = bindings;
  const query = new URLSearchParams(location.search);
  const probe = query.get("probe");
  if (probe) {
    status.style.whiteSpace = "pre-wrap";
    const u32 = (n) => {
      const bytes = new Uint8Array(4);
      new DataView(bytes.buffer).setUint32(0, n >>> 0, true);
      return bytes;
    };
    const concat = (parts) => {
      const payload = new Uint8Array(parts.reduce((sum, part) => sum + part.length, 0));
      let offset = 0;
      for (const part of parts) {
        payload.set(part, offset);
        offset += part.length;
      }
      return payload;
    };
    const text = (value) => {
      const raw = new TextEncoder().encode(value);
      const pad = (4 - (raw.length % 4)) % 4;
      const out = new Uint8Array(4 + raw.length + pad);
      out.set(u32(raw.length), 0);
      out.set(raw, 4);
      return out;
    };
    const rawBytes = (raw) => {
      const pad = (4 - (raw.length % 4)) % 4;
      const out = new Uint8Array(4 + raw.length + pad);
      out.set(u32(raw.length), 0);
      out.set(raw, 4);
      return out;
    };
    const parts = [u32(0x31475341), u32(1), text("basis"), text(probe), u32(1), u32(1), u32(1)];
    if (query.get("run") === "1") {
      const tables = jetTables();
      const params = new Uint8Array(136);
      const view = new DataView(params.buffer);
      const words = [
        1, 1, 1, 3, 4, 4, 0, 0, 30, 7, 6, 1, 1, 1, 1, 0,
        tables.mul_n[0], tables.mul_n[1], tables.mul_n[2], tables.mul_n[3], tables.mul_n[4],
        tables.mul_off[0], tables.mul_off[1], tables.mul_off[2], tables.mul_off[3], tables.mul_off[4],
        tables.up_off, tables.mono_off, tables.deg_off, tables.pair_off,
      ];
      words.forEach((word, index) => view.setUint32(index * 4, word, true));
      view.setFloat32(120, Math.SQRT2, true);
      view.setFloat32(128, Math.sqrt(0.25 / Math.PI), true);
      const tableBytes = new Uint8Array(tables.words.length * 4);
      const tableView = new DataView(tableBytes.buffer);
      tables.words.forEach((word, index) => tableView.setUint32(index * 4, word, true));
      const points = new Uint8Array(24);
      new DataView(points.buffer).setFloat32(0, 100, true);
      parts.push(
        u32(6),
        u32(0), u32(0), rawBytes(points),
        u32(1), u32(0), rawBytes(params),
        u32(2), u32(0), rawBytes(new Uint8Array(200)),
        u32(3), u32(0), rawBytes(new Uint8Array(896)),
        u32(4), u32(0), rawBytes(tableBytes),
        u32(5), u32(3), u32(28800),
      );
    } else {
      parts.push(u32(0));
    }
    status.textContent = (query.get("run") === "1" ? "running " : "compiling ") + probe;
    try {
      const out = await execute_gpu(concat(parts));
      status.textContent = (query.get("run") === "1" ? "ran " : "compiled ") + probe + " " + out.length;
    } catch (error) {
      fail(error);
    }
    return;
  }
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

  // Same inputs as:
  // cargo run --release --features tier3 --bin asteroid-app -- view
  //   --project output/project_cauchy.toml --omega 4.363323e-4
  const project = query.get("project") || "output/project_cauchy.toml";
  const spin = query.get("spin") || "const:4.363323e-4@3.15576e13";
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
