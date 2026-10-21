let sab = null;

const reportError = console.error.bind(console);
console.error = (...args) => {
  const text = args.map((item) => String(item && item.stack ? item.stack : item)).join("\n");
  try {
    self.postMessage({ kind: "failed", payload: text });
  } catch (_err) {
    // The page may already be gone.
  }
  reportError(...args);
};

function header() {
  return new Int32Array(sab, 0, 16);
}

function dataBytes() {
  return new Uint8Array(sab, 256);
}

function installBridge() {
  globalThis.asteroid = {
    gpuCall(payload) {
      const src = payload instanceof Uint8Array ? payload : new Uint8Array(payload);
      const dst = dataBytes();
      if (src.length > dst.length) {
        throw new Error("gpu request does not fit (" + src.length + " bytes)");
      }
      dst.set(src);
      const words = header();
      Atomics.store(words, 1, src.length);
      Atomics.store(words, 0, 1);
      self.postMessage({ kind: "gpu" });
      while (Atomics.load(words, 0) === 1) {
        Atomics.wait(words, 0, 1);
      }
      const phase = Atomics.load(words, 0);
      const len = Atomics.load(words, 1);
      const out = dst.slice(0, len);
      Atomics.store(words, 0, 0);
      if (phase === 3) {
        throw new Error(new TextDecoder().decode(out));
      }
      return out;
    },
    emit(kind, payload) {
      self.postMessage({ kind, payload });
    },
    startWorker() {},
    setFlag() {},
    waitGate() {
      const words = header();
      while (Atomics.load(words, 2) === 1 && Atomics.load(words, 3) === 0) {
        Atomics.wait(words, 2, 1, 250);
      }
      return Atomics.load(words, 3) === 0 ? 0 : 1;
    },
  };
}

self.onmessage = async (event) => {
  if (!event.data || event.data.kind !== "start") {
    return;
  }
  sab = event.data.sab;
  installBridge();
  try {
    const { default: init, install_asset, worker_main } = await import(event.data.module);
    await init(event.data.wasm);
    for (const file of event.data.files) {
      install_asset(file.path, new Uint8Array(file.bytes));
    }
    worker_main(event.data.config);
  } catch (error) {
    self.postMessage({
      kind: "failed",
      payload: String(error && error.stack ? error.stack : error),
    });
  }
};
