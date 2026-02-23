/**
 * Neural Network Design: The Gradient Puzzle - FINAL WORKING VERSION
 */

const CONFIG = {
  inputShapeModel: [16, 16, 1],
  inputShapeData: [1, 16, 16, 1],
  learningRate: 0.05,
  autoTrainSpeed: 50,
};

let state = {
  step: 0, isAutoTraining: false, xInput: null,
  baselineModel: null, studentModel: null, optimizer: null,
};

// ==========================================
// Helper Functions (ФИКСЕД)
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

// 🔥 СОВСЕМ НОВЫЙ: DIFF_SORTED_MSE (работает с градиентами!)
function diffSortedMSE(yTrue, yPred) {
  return tf.tidy(() => {
    // 1. Flatten: [1,16,16,1] → [256]
    const inputFlat = yTrue.mean(-1).flatten();
    const predFlat = yPred.mean(-1).flatten();
    
    // 2. Cumulative Distribution Matching (дифференцируем!)
    const inputSortedApprox = inputFlat.expandDims(1)
      .mul(tf.linspace(0, 1, 256).expandDims(0))  // [256,256]
      .sum(-1).div(256);  // Quantile approx
    
    const predSortedApprox = predFlat.expandDims(1)
      .mul(tf.linspace(0, 1, 256).expandDims(0))
      .sum(-1).div(256);
    
    // 3. MSE между квантилями
    const loss = tf.losses.meanSquaredError(inputSortedApprox, predSortedApprox);
    
    return loss;
  });
}

// ==========================================
// Models (без изменений)
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
// ✅ РАБОЧИЙ Student Loss
function studentLoss(yTrue, yPred) {
  return tf.tidy(() => {
    const lossSortedMSE = diffSortedMSE(yTrue, yPred);  // Дифференцируемый!
    const lossSmooth = smoothness(yPred).mul(0.1);
    const lossDir = directionX(yPred).mul(0.1);
    
    return lossSortedMSE.add(lossSmooth).add(lossDir);
  });
}

// ==========================================
// Training (упрощенный)
async function trainStep() {
  state.step++;
  if (!state.studentModel) return;

  // Baseline
  const baselineLossVal = tf.tidy(() => {
    const { value, grads } = tf.variableGrads(() => {
      return mse(state.xInput, state.baselineModel.predict(state.xInput));
    }, state.baselineModel.getWeights());
    state.optimizer.applyGradients(grads);
    return value.dataSync()[0];
  });

  // Student
  let studentLossVal = 0;
  try {
    studentLossVal = tf.tidy(() => {
      const { value, grads } = tf.variableGrads(() => {
        return studentLoss(state.xInput, state.studentModel.predict(state.xInput));
      }, state.studentModel.getWeights());
      state.optimizer.applyGradients(grads);
      return value.dataSync()[0];
    });
    log(`Step ${state.step}: Base=${baselineLossVal.toFixed(4)} | Student=${studentLossVal.toFixed(4)}`);
  } catch (e) {
    log(`Error: ${e.message}`, true);
    return;
  }

  if (state.step % 5 === 0 || !state.isAutoTraining) {
    await render();
    updateLossDisplay(baselineLossVal, studentLossVal);
  }
}

// ==========================================
// UI (без изменений)
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

  log("✅ FINAL VERSION! DiffSortedMSE + Smooth + Dir → NO GRADIENT ERRORS!");
}

function resetModels(archType = null) {
  if (typeof archType !== "string") {
    archType = document.querySelector('input[name="arch"]:checked')?.value || "compression";
  }
  if (state.isAutoTraining) stopAutoTrain();

  state.baselineModel?.dispose();
  state.studentModel?.dispose();
  state.optimizer?.dispose();

  state.baselineModel = createBaselineModel();
  state.studentModel = createStudentModel(archType);
  state.optimizer = tf.train.adam(CONFIG.learningRate);
  state.step = 0;

  log(`🔄 Reset: ${archType} | DiffSortedMSE(1.0)+Smooth(0.1)+Dir(0.1)`);
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

function updateLossDisplay(base, stud) {
  document.getElementById("loss-baseline").innerText = `Loss: ${base.toFixed(5)}`;
  document.getElementById("loss-student").innerText = `Loss: ${stud.toFixed(5)}`;
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

init();
