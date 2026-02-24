/**
 * Neural Network Design: The Gradient Puzzle
 *
 * FINAL STABLE VERSION
 * - Uses tf.topk for sorting (gradient-friendly)
 * - Full UI support (log, architecture label)
 * - Tuned loss weights
 */

// ==========================================
// 1. Global Config
// ==========================================
const CONFIG = {
  inputShape: [1, 16, 16, 1],
  learningRate: 0.01,
  autoTrainSpeed: 50,
};

let state = {
  step: 0,
  isAutoTraining: false,
  xInput: null,
  baselineModel: null,
  studentModel: null,
  baselineOpt: null,
  studentOpt: null,
};

// ==========================================
// 2. Basic Losses
// ==========================================

function mse(a, b) {
  return tf.losses.meanSquaredError(a, b);
}

// Smoothness (Total Variation)
function smoothness(img) {
  const dx = img.slice([0, 0, 0, 0], [-1, -1, 15, -1])
    .sub(img.slice([0, 0, 1, 0], [-1, -1, 15, -1]));

  const dy = img.slice([0, 0, 0, 0], [-1, 15, -1, -1])
    .sub(img.slice([0, 1, 0, 0], [-1, 15, -1, -1]));

  return tf.mean(tf.square(dx)).add(tf.mean(tf.square(dy)));
}

// Direction constraint
function directionX(img) {
  const mask = tf.linspace(-1, 1, 16).reshape([1, 1, 16, 1]);
  return tf.mean(img.mul(mask)).mul(-1);
}

// Histogram matching using topk (gradient-friendly)
function sortedMSE(a, b) {
  const flatA = a.flatten();
  const flatB = b.flatten();
  const k = flatA.shape[0];

  const { values: sortedA } = tf.topk(flatA, k);
  const { values: sortedB } = tf.topk(flatB, k);

  return mse(sortedA, sortedB);
}

// ==========================================
// 3. Models
// ==========================================

function createBaseline() {
  const m = tf.sequential();
  m.add(tf.layers.flatten({ inputShape: [16, 16, 1] }));
  m.add(tf.layers.dense({ units: 64, activation: "relu" }));
  m.add(tf.layers.dense({ units: 256, activation: "sigmoid" }));
  m.add(tf.layers.reshape({ targetShape: [16, 16, 1] }));
  return m;
}

function createStudent(type) {
  const m = tf.sequential();
  m.add(tf.layers.flatten({ inputShape: [16, 16, 1] }));

  if (type === "compression") {
    m.add(tf.layers.dense({ units: 64, activation: "relu" }));
  } else if (type === "transformation") {
    m.add(tf.layers.dense({ units: 256, activation: "relu" }));
  } else if (type === "expansion") {
    m.add(tf.layers.dense({ units: 512, activation: "relu" }));
  }

  m.add(tf.layers.dense({ units: 256, activation: "sigmoid" }));
  m.add(tf.layers.reshape({ targetShape: [16, 16, 1] }));

  return m;
}

// ==========================================
// 4. Student Custom Loss
// ==========================================

function studentLoss(input, output) {
  return tf.tidy(() => {
    const lSorted = sortedMSE(input, output);
    const lSmooth = smoothness(output).mul(0.1);
    const lDir = directionX(output).mul(0.2);
    return lSorted.add(lSmooth).add(lDir);
  });
}

// ==========================================
// 5. Training
// ==========================================

async function trainStep() {
  state.step++;

  // Baseline update
  const baseLoss = state.baselineOpt.minimize(
    () => {
      const pred = state.baselineModel.predict(state.xInput);
      return mse(state.xInput, pred);
    },
    true,
    state.baselineModel.trainableVariables
  );

  const baseVal = baseLoss.dataSync()[0];
  baseLoss.dispose();

  // Student update
  const studLoss = state.studentOpt.minimize(
    () => {
      const pred = state.studentModel.predict(state.xInput);
      return studentLoss(state.xInput, pred);
    },
    true,
    state.studentModel.trainableVariables
  );

  const studVal = studLoss.dataSync()[0];
  studLoss.dispose();

  // Log to console and UI
  const msg = `Step ${state.step}: Base=${baseVal.toFixed(4)} | Student=${studVal.toFixed(4)}`;
  console.log(msg);
  log(msg);

  if (state.step % 5 === 0) {
    await render();
  }

  document.getElementById("loss-baseline").innerText = `Loss: ${baseVal.toFixed(5)}`;
  document.getElementById("loss-student").innerText = `Loss: ${studVal.toFixed(5)}`;
}

// ==========================================
// 6. Rendering
// ==========================================

async function render() {
  const base = state.baselineModel.predict(state.xInput);
  const stud = state.studentModel.predict(state.xInput);

  await tf.browser.toPixels(
    base.squeeze(),
    document.getElementById("canvas-baseline")
  );
  await tf.browser.toPixels(
    stud.squeeze(),
    document.getElementById("canvas-student")
  );

  base.dispose();
  stud.dispose();
}

// ==========================================
// 7. Init & UI Helpers
// ==========================================

function log(msg, isError = false) {
  const el = document.getElementById("log-area");
  if (!el) return;
  const span = document.createElement("div");
  span.innerText = `> ${msg}`;
  if (isError) span.classList.add("error");
  el.prepend(span);
}

function resetModels(type = null) {
  if (!type) {
    const checked = document.querySelector('input[name="arch"]:checked');
    type = checked ? checked.value : "compression";
  }

  if (state.isAutoTraining) {
    toggleAutoTrain(); // stop auto-train
  }

  if (state.baselineModel) state.baselineModel.dispose();
  if (state.studentModel) state.studentModel.dispose();

  state.baselineModel = createBaseline();
  state.studentModel = createStudent(type);

  state.baselineOpt = tf.train.adam(CONFIG.learningRate);
  state.studentOpt = tf.train.adam(CONFIG.learningRate);

  state.step = 0;

  // Update UI label
  const label = document.getElementById("student-arch-label");
  if (label) label.innerText = type.charAt(0).toUpperCase() + type.slice(1);

  log(`Models reset. Student Arch: ${type}`);
  render();
}

function toggleAutoTrain() {
  const btn = document.getElementById("btn-auto");
  if (!btn) return;

  state.isAutoTraining = !state.isAutoTraining;
  btn.innerText = state.isAutoTraining ? "Auto Train (Stop)" : "Auto Train (Start)";
  btn.classList.toggle("btn-stop", state.isAutoTraining);
  btn.classList.toggle("btn-auto", !state.isAutoTraining);

  if (state.isAutoTraining) loop();
}

function loop() {
  if (!state.isAutoTraining) return;
  trainStep();
  setTimeout(loop, CONFIG.autoTrainSpeed);
}

function init() {
  state.xInput = tf.randomUniform(CONFIG.inputShape);

  resetModels();

  tf.browser.toPixels(
    state.xInput.squeeze(),
    document.getElementById("canvas-input")
  );

  document.getElementById("btn-train").addEventListener("click", trainStep);
  document.getElementById("btn-auto").addEventListener("click", toggleAutoTrain);
  document.getElementById("btn-reset").addEventListener("click", () => resetModels());

  document.querySelectorAll('input[name="arch"]').forEach((radio) => {
    radio.addEventListener("change", (e) => resetModels(e.target.value));
  });

  log("Initialized. Ready to train.");
}

// Start
init();
