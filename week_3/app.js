/**
 * Neural Network Design: The Gradient Puzzle
 *
 * FINAL STABLE VERSION
 * - Proper optimizer usage (no variableGrads)
 * - Sorted MSE (histogram preservation)
 * - Smoothness (Total Variation)
 * - Direction constraint (Left dark → Right bright)
 */

// ==========================================
// 1. Global State & Config
// ==========================================
const CONFIG = {
  inputShapeModel: [16, 16, 1],
  inputShapeData: [1, 16, 16, 1],
  learningRate: 0.01,
  autoTrainSpeed: 50,
};

let state = {
  step: 0,
  isAutoTraining: false,
  xInput: null,
  baselineModel: null,
  studentModel: null,
  baselineOptimizer: null,
  studentOptimizer: null,
};

// ==========================================
// 2. Helper Functions (Loss Components)
// ==========================================

// Standard pixel-wise MSE
function mse(yTrue, yPred) {
  return tf.losses.meanSquaredError(yTrue, yPred);
}

// Smoothness (Total Variation loss)
// Penalizes differences between adjacent pixels
function smoothness(yPred) {
  const diffX = yPred
    .slice([0, 0, 0, 0], [-1, -1, 15, -1])
    .sub(yPred.slice([0, 0, 1, 0], [-1, -1, 15, -1]));

  const diffY = yPred
    .slice([0, 0, 0, 0], [-1, 15, -1, -1])
    .sub(yPred.slice([0, 1, 0, 0], [-1, 15, -1, -1]));

  return tf.mean(tf.square(diffX)).add(tf.mean(tf.square(diffY)));
}

// Horizontal direction constraint
// Encourages left side dark and right side bright
function directionX(yPred) {
  const width = 16;
  const mask = tf.linspace(-1, 1, width).reshape([1, 1, width, 1]);
  return tf.mean(yPred.mul(mask)).mul(-1);
}

// ==========================================
// 3. Model Architectures
// ==========================================

// Baseline: Undercomplete autoencoder
function createBaselineModel() {
  const model = tf.sequential();
  model.add(tf.layers.flatten({ inputShape: CONFIG.inputShapeModel }));
  model.add(tf.layers.dense({ units: 64, activation: "relu" }));
  model.add(tf.layers.dense({ units: 256, activation: "sigmoid" }));
  model.add(tf.layers.reshape({ targetShape: [16, 16, 1] }));
  return model;
}

// Student model with selectable projection type
function createStudentModel(archType) {
  const model = tf.sequential();
  model.add(tf.layers.flatten({ inputShape: CONFIG.inputShapeModel }));

  if (archType === "compression") {
    model.add(tf.layers.dense({ units: 64, activation: "relu" }));
    model.add(tf.layers.dense({ units: 256, activation: "sigmoid" }));

  } else if (archType === "transformation") {
    model.add(tf.layers.dense({ units: 256, activation: "relu" }));
    model.add(tf.layers.dense({ units: 256, activation: "sigmoid" }));

  } else if (archType === "expansion") {
    model.add(tf.layers.dense({ units: 512, activation: "relu" }));
    model.add(tf.layers.dense({ units: 256, activation: "sigmoid" }));

  } else {
    throw new Error(`Unknown architecture type: ${archType}`);
  }

  model.add(tf.layers.reshape({ targetShape: [16, 16, 1] }));
  return model;
}

// ==========================================
// 4. Custom Student Loss
// ==========================================

function studentLoss(yTrue, yPred) {
  return tf.tidy(() => {

    // ----- Sorted MSE (Histogram Preservation) -----

    const flatTrue = yTrue.reshape([256]);
    const flatPred = yPred.reshape([256]);

    const sortedTrue = tf.sort(flatTrue);
    const sortedPred = tf.sort(flatPred);

    const lossSorted = mse(sortedTrue, sortedPred);

    // ----- Smoothness -----
    const lossSmooth = smoothness(yPred).mul(0.1);

    // ----- Direction -----
    const lossDir = directionX(yPred).mul(0.2);

    return lossSorted
      .add(lossSmooth)
      .add(lossDir);
  });
}

// ==========================================
// 5. Training Step
// ==========================================

async function trainStep() {
  state.step++;

  // ---- Baseline Training ----
  const baselineLossTensor = state.baselineOptimizer.minimize(() => {
    return tf.tidy(() => {
      const yPred = state.baselineModel.predict(state.xInput);
      return mse(state.xInput, yPred);
    });
  }, true, state.baselineModel.trainableVariables);

  const baselineLoss = baselineLossTensor.dataSync()[0];
  baselineLossTensor.dispose();

  // ---- Student Training ----
  const studentLossTensor = state.studentOptimizer.minimize(() => {
    return tf.tidy(() => {
      const yPred = state.studentModel.predict(state.xInput);
      return studentLoss(state.xInput, yPred);
    });
  }, true, state.studentModel.trainableVariables);

  const studentLossValue = studentLossTensor.dataSync()[0];
  studentLossTensor.dispose();

  log(
    `Step ${state.step}: Base=${baselineLoss.toFixed(4)} | Student=${studentLossValue.toFixed(4)}`
  );

  if (state.step % 5 === 0 || !state.isAutoTraining) {
    await render();
    updateLossDisplay(baselineLoss, studentLossValue);
  }
}

// ==========================================
// 6. UI & Initialization
// ==========================================

function init() {
  state.xInput = tf.randomUniform(CONFIG.inputShapeData);

  resetModels();

  tf.browser.toPixels(
    state.xInput.squeeze(),
    document.getElementById("canvas-input"),
  );

  document
    .getElementById("btn-train")
    .addEventListener("click", () => trainStep());

  document
    .getElementById("btn-auto")
    .addEventListener("click", toggleAutoTrain);

  document
    .getElementById("btn-reset")
    .addEventListener("click", resetModels);

  document.querySelectorAll('input[name="arch"]').forEach((radio) => {
    radio.addEventListener("change", (e) => {
      resetModels(e.target.value);
      document.getElementById("student-arch-label").innerText =
        e.target.value.charAt(0).toUpperCase() + e.target.value.slice(1);
    });
  });

  log("Initialized. Ready to train.");
}

function resetModels(archType = null) {
  if (typeof archType !== "string") {
    const checked = document.querySelector('input[name="arch"]:checked');
    archType = checked ? checked.value : "compression";
  }

  if (state.isAutoTraining) stopAutoTrain();

  if (state.baselineModel) state.baselineModel.dispose();
  if (state.studentModel) state.studentModel.dispose();

  state.baselineModel = createBaselineModel();
  state.studentModel = createStudentModel(archType);

  state.baselineOptimizer = tf.train.adam(CONFIG.learningRate);
  state.studentOptimizer = tf.train.adam(CONFIG.learningRate);

  state.step = 0;

  log(`Models reset. Student Arch: ${archType}`);
  render();
}

async function render() {
  const basePred = state.baselineModel.predict(state.xInput);
  const studPred = state.studentModel.predict(state.xInput);

  await tf.browser.toPixels(
    basePred.squeeze(),
    document.getElementById("canvas-baseline"),
  );

  await tf.browser.toPixels(
    studPred.squeeze(),
    document.getElementById("canvas-student"),
  );

  basePred.dispose();
  studPred.dispose();
}

function updateLossDisplay(base, stud) {
  document.getElementById("loss-baseline").innerText =
    `Loss: ${base.toFixed(5)}`;
  document.getElementById("loss-student").innerText =
    `Loss: ${stud.toFixed(5)}`;
}

function log(msg, isError = false) {
  const el = document.getElementById("log-area");
  const span = document.createElement("div");
  span.innerText = `> ${msg}`;
  if (isError) span.classList.add("error");
  el.prepend(span);
}

function toggleAutoTrain() {
  const btn = document.getElementById("btn-auto");

  if (state.isAutoTraining) {
    stopAutoTrain();
  } else {
    state.isAutoTraining = true;
    btn.innerText = "Auto Train (Stop)";
    btn.classList.add("btn-stop");
    btn.classList.remove("btn-auto");
    loop();
  }
}

function stopAutoTrain() {
  state.isAutoTraining = false;
  const btn = document.getElementById("btn-auto");
  btn.innerText = "Auto Train (Start)";
  btn.classList.add("btn-auto");
  btn.classList.remove("btn-stop");
}

function loop() {
  if (state.isAutoTraining) {
    trainStep();
    setTimeout(loop, CONFIG.autoTrainSpeed);
  }
}

// Start app
init();
