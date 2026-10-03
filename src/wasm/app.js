// Asteroid Evolution - WebGPU / WASM 前端交互脚本
// 严格对照 ryugu_cauchy_gravity 的前端控制逻辑

import init, { WasmSession } from './pkg/astroid_evolution.js';

let session = null;
let isRunning = false;
let animationFrameId = null;

async function bootstrap() {
  console.log('[AsteroidEvolution] 正在初始化 WebGPU / WASM 运行时...');
  
  // 1. 探测 WebGPU 支持
  const hasWebGpu = !!navigator.gpu;
  const badge = document.getElementById('backend-badge');
  const badgeText = document.getElementById('backend-text');
  
  if (!hasWebGpu) {
    console.warn('[AsteroidEvolution] 当前浏览器未检测到 navigator.gpu，已停用 GPU 求解路径');
    if (badge) {
      badge.className = 'badge cpu';
      badgeText.textContent = 'WebGPU 不可用';
    }
  } else {
    console.log('[AsteroidEvolution] 检测到原生 WebGPU 支持');
    if (badge) {
      badge.className = 'badge gpu';
      badgeText.textContent = 'WebGPU 硬件加速就绪';
    }
  }

  // 2. 加载并实例化 WASM 模块 (若尚未编译 pkg 则优雅回退)
  try {
    await init();
    session = new WasmSession();
    console.log('[AsteroidEvolution] WASM 会话控制器创建成功:', session.getCapabilityJson());
  } catch (err) {
    console.warn('[AsteroidEvolution] WASM 模块不可用，已禁用求解器控制:', err);
  }

  // 3. 绑定 UI 事件
  setupUiListeners();
}

function setupUiListeners() {
  const btnRun = document.getElementById('btn-run');
  const btnStep = document.getElementById('btn-step');
  const btnReset = document.getElementById('btn-reset');

  btnRun?.addEventListener('click', () => {
    if (!session) return;
    isRunning = !isRunning;
    if (isRunning) {
      btnRun.textContent = '暂停 (Pause)';
      btnRun.style.background = '#facc15';
      session.run();
      startLoop();
    } else {
      btnRun.textContent = '运行 (Run)';
      btnRun.style.background = 'var(--cyan)';
      session.pause();
      stopLoop();
    }
  });

  btnStep?.addEventListener('click', () => {
    if (!session) return;
    stepSimulation();
  });

  btnReset?.addEventListener('click', () => {
    isRunning = false;
    stopLoop();
    btnRun.textContent = '运行 (Run)';
    btnRun.style.background = 'var(--cyan)';
    if (session) session.reset();
    resetDiagnostics();
  });
}

function stepSimulation() {
  if (!session) return;
  session.step();
  updateDiagnostics();
}

function updateDiagnostics() {
  if (!session) return;
  document.getElementById('diag-step').textContent = session.currentStep();
  // 当前 WASM 会话只提供生命周期控制，不提供 SCF 场和残差数据。
  // 数据缺失时保持未知，避免把占位值当作物理结果。
  for (const id of ['diag-geom', 'diag-mass', 'diag-slope', 'diag-trans', 'diag-kkt']) {
    document.getElementById(id).textContent = '—';
  }
  const statusElem = document.getElementById('diag-status');
  statusElem.textContent = session ? '等待求解器数据' : 'WASM 求解器不可用';
  statusElem.style.color = '#facc15';
}

function resetDiagnostics() {
  document.getElementById('diag-step').textContent = '0';
  document.getElementById('diag-geom').textContent = '—';
  document.getElementById('diag-mass').textContent = '—';
  document.getElementById('diag-slope').textContent = '—';
  document.getElementById('diag-trans').textContent = '—';
  document.getElementById('diag-kkt').textContent = '—';
  const statusElem = document.getElementById('diag-status');
  statusElem.textContent = session ? '等待求解器数据' : 'WASM 求解器不可用';
  statusElem.style.color = '#facc15';
}

function startLoop() {
  let lastTime = performance.now();
  function loop(time) {
    if (!isRunning) return;
    if (time - lastTime > 100) { // 限制迭代频率
      stepSimulation();
      lastTime = time;
    }
    animationFrameId = requestAnimationFrame(loop);
  }
  animationFrameId = requestAnimationFrame(loop);
}

function stopLoop() {
  if (animationFrameId) {
    cancelAnimationFrame(animationFrameId);
    animationFrameId = null;
  }
}

window.addEventListener('DOMContentLoaded', bootstrap);
