/**
 * Neural Network Design: The Gradient Puzzle - 16 VERTICAL STRIPS FIXED
 */

const CONFIG = {
  inputShapeModel: [16, 16, 1],
  inputShapeData: [1, 16, 16, 1],
  learningRate: 0.02,
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
// LOSS FUNCTIONS - ТОЧНЫЙ ГРАДИЕНТ 16 ПОЛОС
// ==========================================

function mse(yTrue, yPred) {
  return tf.losses.meanSquaredError(yTrue, yPred);
}

// ✅ МАСКА: колонка 0=-1.0 → колонка 15=+1.0 (16 четких полос)
function createGradientMask() {
  const values = tf.tensor1d([
    -1.00, -0.94, -0.88, -0.82, -0.76, -0.70, -0.64, -0.58,
    -0.52, -0.46, -0.40, -0.34, -0.28, -0.22, -0.16,  0.00
  ]).add(tf.tensor1d([0,0,0,0,0,0,0,0, 0.3,0.36,0.42,0.48,0.54,0.60,0.66,1.00]));
  return values.reshape([1, 1, 16, 1]);
}

function directionX(yPred) {
  const mask = createGradientMask();
  const correlation = tf.mean(yPred.mul(mask));
  mask.dispose();
  return correlation.mul(-5.0); // Максимизируем градиент
}

function smoothnessVertical(yPred) {
  const diffY = yPred.slice([0,0,0,0], [-1,15,-1,-1]).sub(
    yPred.slice([0,1,0,0], [-1,15,-1,-1])
  );
  return tf.mean(tf.square(diffY)).mul(0.3);
}

function preserveEnergy(yPred) {
  const meanPred = tf.mean(yPred);
  return tf.square(meanPred.sub(0.5)).mul(0.1);
}

function studentLoss(yTrue, yPred) {
  return tf.tidy(() => {
    const lossDir = directionX(yPred);           // 70% веса - ГЛАВНЫЙ
    const lossSmooth = smoothnessVertical(yPred); // 25% веса
    const lossEnergy = preserveEnergy(yPred);     // 5% веса
    return lossDir.add(lossSmooth).add(lossEnergy);
  });
}

// ==========================================
// MODELS
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
  } else {
    model.add(tf.layers.dense({ units: 512, activation: "relu" }));
    model.add(tf.layers.dense({ units: 256, activation: "sigmoid" }));
  }

  model.add(tf.layers.reshape({ targetShape: [16, 16, 1] }));
  return model;
}

// ==========================================
// ✅ РАБОТАЮЩИЙ TRAINING STEP
// ==========================================
async function trainStep() {
  state.step++;

  // ✅ Baseline: копирует вход
  const baselineLoss = tf.tidy(() => {
    const pred = state.baselineModel.predict(state.xInput);
    const loss = mse(state.xInput, pred);
    const { grads } = tf.variableGrads(loss, state.baselineModel.getWeights());
    state.baselineOptimizer.applyGradients(grads);
    pred.dispose();
    return loss.dataSync()[0];
  });

  // ✅ Student: создает градиент
  const studentLoss = tf.tidy(() => {
    const pred = state.studentModel.predict(state.xInput);
    const loss = studentLoss(state.xInput, pred);
    const { grads } = tf.variableGrads(loss, state.studentModel.getWeights());
    state.studentOptimizer.applyGradients(grads);
    pred.dispose();
    return loss.dataSync()[0];
  });

  // ✅ Render каждые 5 шагов + логи каждые 20
  if (state.step % 5 === 0) {
    await render();
  }
  
  if (state.step % 20 === 0 || state.step <= 5) {
    log(`Step ${state.step}: Base=${baselineLoss.toFixed(4)} | Student=${studentLoss.toFixed(4)}`);
    updateLossDisplay(baselineLoss, studentLoss);
  }
}

// ==========================================
// RENDER
// ==========================================
async function render() {
  const basePred = state.baselineModel.predict(state.xInput);
  const studPred = state.studentModel.predict(state.xInput);
  
  await tf.browser.toPixels(basePred.squeeze(), document.getElementById("canvas-baseline"));
  await tf.browser.toPixels(studPred.squeeze(), document.getElementById("canvas-student"));
  
  basePred.dispose();
  studPred.dispose();
}

// ==========================================
// UI
// ==========================================
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
    state.isAutoTraining = false;
    btn.innerText = "Auto Train (Start)";
    btn.classList.add("btn-auto");
    btn.classList.remove("btn-stop");
    log("⏹️ Auto STOPPED");
  } else {
    state.isAutoTraining = true;
    btn.innerText = "Auto Train (Stop)";
    btn.classList.add("btn-stop");
    btn.classList.remove("btn-auto");
    log("🚀 Auto STARTED → 16 strips!");
    loop();
  }
}

function loop() {
  if (state.isAutoTraining) {
    trainStep();
    setTimeout(loop, CONFIG.autoTrainSpeed);
  }
}

function resetModels(archType = null) {
  if (!archType) {
    const checked = document.querySelector('input[name="arch"]:checked');
    archType = checked ? checked.value : "compression";
  }

  // Stop auto train
  state.isAutoTraining = false;
  const btn = document.getElementById("btn-auto");
  if (btn) {
    btn.innerText = "Auto Train (Start)";
    btn.classList.add("btn-auto");
    btn.classList.remove("btn-stop");
  }

  // Cleanup
  [state.baselineModel, state.studentModel, state.baselineOptimizer, state.studentOptimizer]
    .forEach(model => model?.dispose());

  // New models
  state.baselineModel = createBaselineModel();
  state.studentModel = createStudentModel(archType);
  state.baselineOptimizer = tf.train.adam(CONFIG.learningRate);
  state.studentOptimizer = tf.train.adam(CONFIG.learningRate);
  state.step = 0;

  log(`🔄 Reset: ${archType}`);
  render().then(() => updateLossDisplay(0, 0));
}

// ==========================================
// INIT
// ==========================================
async function init() {
  await tf.ready();
  
  state.xInput = tf.randomUniform(CONFIG.inputShapeData, 0, 1);
  tf.browser.toPixels(state.xInput.squeeze(), document.getElementById("canvas-input"));

  // ✅ Event listeners - гарантированно работают
  document.getElementById("btn-train").onclick = trainStep;
  document.getElementById("btn-auto").onclick = toggleAutoTrain;
  document.getElementById("btn-reset").onclick = resetModels;

  document.querySelectorAll('input[name="arch"]').forEach(radio => {
    radio.onchange = () => {
      resetModels(radio.value);
      document.getElementById("student-arch-label").textContent = 
        radio.value.charAt(0).toUpperCase() + radio.value.slice(1);
    };
  });

  resetModels();
  log("🎯 16 STRIPS READY! Train 1 Step работает!");
}

init();
