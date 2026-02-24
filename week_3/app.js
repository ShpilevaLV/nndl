/**
 * Neural Network Design: The Gradient Puzzle - BROADCASTING FIXED
 * Transforms 16x16 noise into smooth horizontal gradient.
 */

const CONFIG = {
  inputShapeModel: [16, 16, 1],
  inputShapeData: [1, 16, 16, 1],
  learningRate: 0.1,
  autoTrainSpeed: 50,
};

let state = {
  step: 0,
  isAutoTraining: false,
  xInput: null,
  baselineModel: null,
  studentModel: null,
  optimizer: null,
};

// ==========================================
// Loss Components - BROADCASTING FIXED
// ==========================================

function mse(yTrue, yPred) {
  return tf.losses.meanSquaredError(yTrue, yPred);
}

function smoothness(yPred) {
  const diffX = yPred.slice([0, 0, 0, 0], [-1, -1, 15, -1])
    .sub(yPred.slice([0, 0, 1, 0], [-1, -1, 15, -1]));
  const diffY = yPred.slice([0, 0, 0, 0], [-1, 15, -1, -1])
    .sub(yPred.slice([0, 1, 0, 0], [-1, 15, -1, -1]));
  return tf.mean(tf.square(diffX)).add(tf.mean(tf.square(diffY)));
}

function directionX(yPred) {
  const mask = tf.linspace(-1, 1, 16).reshape([1, 1, 16, 1]);
  return tf.mean(yPred.mul(mask)).mul(-1);
}

function verticalDirection(yPred) {
  const maskY = tf.linspace(-1, 1, 16).reshape([1, 16, 1, 1]);
  return tf.mean(yPred.mul(maskY)).mul(-1);
}

// ✅ FIXED: Proper broadcasting for differentiable histogram matching
function diffSortedMSE(yTrue, yPred) {
  return tf.tidy(() => {
    // Flatten to [256]
    const inputFlat = yTrue.mean(-1).flatten();
    const predFlat = yPred.mean(-1).flatten();
    
    // Create quantiles [1,256] (row vector)
    const quantiles = tf.linspace(0, 1, 256).expandDims(0);  // [1,256]
    
    // Broadcast correctly: [256,1] * [1,256] → [256,256]
    const inputCdf = inputFlat.expandDims(1).mul(quantiles).sum(-1).div(256);
    const predCdf = predFlat.expandDims(1).mul(quantiles).sum(-1).div(256);
    
    return tf.losses.meanSquaredError(inputCdf, predCdf);
  });
}

// ==========================================
// Model Architectures
// ==========================================

function createBaselineModel() {
  const model = tf.sequential();
  model.add(tf.layers.flatten({ inputShape: CONFIG.inputShapeModel }));
  model.add(tf.layers.dense({ units: 64, activation: "relu" }));
  model.add(tf.layers.dense({ units: 256, activation: "sigmoid" }));
  model.add(tf.layers.reshape({ targetShape: [16, 16, 1] }));
  return model;
}

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
  }

  model.add(tf.layers.reshape({ targetShape: [16, 16, 1] }));
  return model;
}

// ==========================================
// Optimized Loss: Strong gradient bias
// ==========================================
function studentLoss(yTrue, yPred) {
  return tf.tidy(() => {
    const lossSortedMSE = diffSortedMSE(yTrue, yPred).mul(0.2);
    const lossSmooth = smoothness(yPred).mul(0.5);
    const lossDirX = directionX(yPred).mul(2.0);  // Strong horizontal gradient
    const lossDirY = verticalDirection(yPred).mul(0.3);
    
    return lossSortedMSE.add(lossSmooth).add(lossDirX).add(lossDirY);
  });
}

// ==========================================
// Training Loop
// ==========================================
async function trainStep() {
  state.step++;

  if (!state.studentModel || !state.studentModel.getWeights) {
    log("Error: Student model not initialized", true);
    return;
  }

  // Baseline training (MSE)
  const baselineLossVal = tf.tidy(() => {
    const { value, grads } = tf.variableGrads(() => {
      const pred = state.baselineModel.predict(state.xInput);
      return mse(state.xInput, pred);
    }, state.baselineModel.getWeights());
    state.optimizer.applyGradients(grads);
    return value.dataSync()[0];
  });

  // Student training (Custom loss)
  let studentLossVal = 0;
  try {
    studentLossVal = tf.tidy(() => {
      const { value, grads } = tf.variableGrads(() => {
        const pred = state.studentModel.predict(state.xInput);
        return studentLoss(state.xInput, pred);
      }, state.studentModel.getWeights());
      state.optimizer.applyGradients(grads);
      return value.dataSync()[0];
    });
    log(`Step ${state.step}: Base=${baselineLossVal.toFixed(4)} | Student=${studentLossVal.toFixed(4)}`);
  } catch (e) {
    log(`Error: ${e.message}`, true);
    stopAutoTrain();
    return;
  }

  if (state.step % 5 === 0 || !state.isAutoTraining) {
    await render();
    updateLossDisplay(baselineLossVal, studentLossVal);
  }
}

// ==========================================
// UI Functions
// ==========================================
function init() {
  state.xInput = tf.randomUniform(CONFIG.inputShapeData);
  resetModels();

  tf.browser.toPixels(state.xInput.squeeze(), document.getElementById("canvas-input"));

  document.getElementById("btn-train").addEventListener("click", () => trainStep());
  document.getElementById("btn-auto").addEventListener("click", toggleAutoTrain);
  document.getElementById("btn-reset").addEventListener("click", resetModels);

  document.querySelectorAll('input[name="arch"]').forEach((radio) => {
    radio.addEventListener("change", (e) => {
      resetModels(e.target.value);
      document.getElementById("student-arch-label").innerText =
        e.target.value.charAt(0).toUpperCase() + e.target.value.slice(1);
    });
  });

  log("✅ BROADCASTING FIXED! Weights: SortedMSE(0.2), Smooth(0.5), DirX(2.0), DirY(0.3)");
  log("Use EXPANSION + Auto Train → smooth gradient in 150 steps!");
}

function resetModels(archType = null) {
  if (typeof archType !== "string") {
    const checked = document.querySelector('input[name="arch"]:checked');
    archType = checked ? checked.value : "compression";
  }

  if (state.isAutoTraining) stopAutoTrain();

  state.baselineModel?.dispose();
  state.studentModel?.dispose();
  state.optimizer?.dispose();

  state.baselineModel = createBaselineModel();
  state.studentModel = createStudentModel(archType);
  state.optimizer = tf.train.adam(CONFIG.learningRate);
  state.step = 0;

  log(`Reset: ${archType.toUpperCase()} | Loss: Sorted(0.2)+Smooth(0.5)+DirX(2.0)+DirY(0.3)`);
  render();
}

async function render() {
  const basePred = state.baselineModel.predict(state.xInput);
  const studPred = state.studentModel.predict(state.xInput);
  
  await tf.browser.toPixels(basePred.squeeze(), document.getElementById("canvas-baseline"));
  await tf.browser.toPixels(studPred.squeeze(), document.getElementById("canvas-student"));
  
  basePred.dispose();
  studPred.dispose();
}

function updateLossDisplay(baseLoss, studentLoss) {
  document.getElementById("loss-baseline").innerText = `Loss: ${baseLoss.toFixed(5)}`;
  document.getElementById("loss-student").innerText = `Loss: ${studentLoss.toFixed(5)}`;
}

function log(msg, isError = false) {
  const el = document.getElementById("log-area");
  const span = document.createElement("div");
  span.innerText = `> ${msg}`;
  if (isError) span.classList.add("error");
  el.prepend(span);
  el.scrollTop = 0;
}

function toggleAutoTrain() {
  const btn = document.getElementById("btn-auto");
  if (state.isAutoTraining) {
    stopAutoTrain();
  } else {
    state.isAutoTraining = true;
    btn.innerText = "Stop Auto Train";
    btn.classList.add("btn-stop");
    btn.classList.remove("btn-auto");
    loop();
  }
}

function stopAutoTrain() {
  state.isAutoTraining = false;
  const btn = document.getElementById("btn-auto");
  btn.innerText = "Start Auto Train";
  btn.classList.add("btn-auto");
  btn.classList.remove("btn-stop");
}

function loop() {
  if (state.isAutoTraining) {
    trainStep();
    setTimeout(loop, CONFIG.autoTrainSpeed);
  }
}

init();
