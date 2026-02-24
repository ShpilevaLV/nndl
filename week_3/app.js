/**
 * Neural Network Design: The Gradient Puzzle
 *
 * Final Student Version:
 * - Transformation + Expansion architectures implemented
 * - Sorted MSE (Histogram preservation)
 * - Smoothness (Total Variation)
 * - Direction constraint (Left dark -> Right bright)
 */

// ==========================================
// 1. Global State & Config
// ==========================================
const CONFIG = {
  inputShapeModel: [16, 16, 1],
  inputShapeData: [1, 16, 16, 1],
  learningRate: 0.05,
  autoTrainSpeed: 50,
};

let state = {
  step: 0,
  isAutoTraining: false,
  autoTrainInterval: null,
  xInput: null,
  baselineModel: null,
  studentModel: null,
  optimizer: null,
};

// ==========================================
// 2. Helper Functions (Loss Components)
// ==========================================

// Standard pixel-wise MSE
function mse(yTrue, yPred) {
  return tf.losses.meanSquaredError(yTrue, yPred);
}

// Total Variation style smoothness loss
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

// Direction constraint (horizontal gradient)
// Encourages left side dark and right side bright
function directionX(yPred) {
  const width = 16;
  const mask = tf.linspace(-1, 1, width).reshape([1, 1, width, 1]);
  return tf.mean(yPred.mul(mask)).mul(-1);
}

// ==========================================
// 3. Model Architecture
// ==========================================

// Baseline model: fixed compression autoencoder
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
    // Undercomplete bottleneck
    model.add(tf.layers.dense({ units: 64, activation: "relu" }));
    model.add(tf.layers.dense({ units: 256, activation: "sigmoid" }));

  } else if (archType === "transformation") {
    // 1:1 mapping (same dimensionality)
    model.add(tf.layers.dense({ units: 256, activation: "relu" }));
    model.add(tf.layers.dense({ units: 256, activation: "sigmoid" }));

  } else if (archType === "expansion") {
    // Overcomplete representation
    model.add(tf.layers.dense({ units: 512, activation: "relu" }));
    model.add(tf.layers.dense({ units: 256, activation: "sigmoid" }));

  } else {
    throw new Error(Unknown architecture type: ${archType});
  }

  model.add(tf.layers.reshape({ targetShape: [16, 16, 1] }));
  return model;
}

// ==========================================
// 4. Custom Student Loss
// ==========================================

function studentLoss(yTrue, yPred) {
  return tf.tidy(() => {

    // -----------------------------------
    // 1. Sorted MSE (Histogram Preservation)
    // -----------------------------------

    const flatTrue = yTrue.reshape([256]);
    const flatPred = yPred.reshape([256]);

    const sortedTrue = tf.sort(flatTrue);
    const sortedPred = tf.sort(flatPred);

    const lossSorted = mse(sortedTrue, sortedPred);

    // -----------------------------------
    // 2. Smoothness (Local consistency)
    // -----------------------------------
    const lossSmooth = smoothness(yPred).mul(0.05);

    // -----------------------------------
    // 3. Direction (Global gradient)
    // -----------------------------------
    const lossDir = directionX(yPred).mul(0.1);
    // -----------------------------------
    // Total Combined Loss
    // -----------------------------------
    return lossSorted
      .add(lossSmooth)
      .add(lossDir);
  });
}

// ==========================================
// 5. Training Loop
// ==========================================

async function trainStep() {
  state.step++;

  if (!state.studentModel) {
    log("Error: Student model not initialized.", true);
    stopAutoTrain();
    return;
  }

  // Baseline training (MSE only)
  const baselineLossVal = tf.tidy(() => {
    const { value, grads } = tf.variableGrads(() => {
      const yPred = state.baselineModel.predict(state.xInput);
      return mse(state.xInput, yPred);
    }, state.baselineModel.getWeights());

    state.optimizer.applyGradients(grads);
    return value.dataSync()[0];
  });

  // Student training (Custom loss)
  const studentLossVal = tf.tidy(() => {
    const { value, grads } = tf.variableGrads(() => {
      const yPred = state.studentModel.predict(state.xInput);
      return studentLoss(state.xInput, yPred);
    }, state.studentModel.getWeights());

    state.optimizer.applyGradients(grads);
    return value.dataSync()[0];
  });

  log(
    Step ${state.step}: Base=${baselineLossVal.toFixed(4)} | Student=${studentLossVal.toFixed(4)}
  );

  if (state.step % 5 === 0 || !state.isAutoTraining) {
    await render();
    updateLossDisplay(baselineLossVal, studentLossVal);
  }
}

// ==========================================
// 6. Initialization & UI
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
  if (typeof archType !== "string") archType = null;

  if (state.isAutoTraining) stopAutoTrain();

  if (!archType) {
    const checked = document.querySelector('input[name="arch"]:checked');
    archType = checked ? checked.value : "compression";
  }

  if (state.baselineModel) state.baselineModel.dispose();
  if (state.studentModel) state.studentModel.dispose();
  if (state.optimizer) state.optimizer.dispose();

  state.baselineModel = createBaselineModel();
  state.studentModel = createStudentModel(archType);
  state.optimizer = tf.train.adam(CONFIG.learningRate);
  state.step = 0;

  log(Models reset. Student Arch: ${archType});
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
    Loss: ${base.toFixed(5)};
  document.getElementById("loss-student").innerText =
    Loss: ${stud.toFixed(5)};
}

function log(msg, isError = false) {
  const el = document.getElementById("log-area");
  const span = document.createElement("div");
  span.innerText = > ${msg};
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
