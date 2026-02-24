/**
 * Neural Network Design: The Gradient Puzzle
 * FIXED VERSION - All TODOs implemented, TensorFlow.js errors resolved
 */

const CONFIG = {
  inputShapeModel: [16, 16, 1],
  inputShapeData: [1, 16, 16, 1],
  learningRate: 0.05,
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
// LOSS COMPONENTS (All implemented)
// ==========================================

function mse(yTrue, yPred) {
  return tf.losses.meanSquaredError(yTrue, yPred);
}

function smoothness(yPred) {
  // X direction differences
  const diffX = yPred
    .slice([0, 0, 0, 0], [-1, -1, 15, -1])
    .sub(yPred.slice([0, 0, 1, 0], [-1, -1, 15, -1]));
  
  // Y direction differences  
  const diffY = yPred
    .slice([0, 0, 0, 0], [-1, 15, -1, -1])
    .sub(yPred.slice([0, 1, 0, 0], [-1, 15, -1, -1]));
  
  return tf.mean(tf.square(diffX)).add(tf.mean(tf.square(diffY)));
}

function directionX(yPred) {
  const width = 16;
  const mask = tf.linspace(-1, 1, width).reshape([1, 1, width, 1]);
  return tf.mean(yPred.mul(mask)).mul(-1);
}

// ✅ FIXED: Sorted MSE - ключ к перестановке пикселей
function sortedMSE(yTrue, yPred) {
  return tf.tidy(() => {
    // Преобразуем [1,16,16,1] → [256]
    const flatTrue = yTrue.reshape([256]).toFloat();
    const flatPred = yPred.reshape([256]).toFloat();
    
    // Сортируем тензоры с помощью tf.sort()
    const sortedTrue = tf.sort(flatTrue);
    const sortedPred = tf.sort(flatPred);
    
    // MSE между отсортированными значениями
    const loss = tf.losses.meanSquaredError(sortedTrue, sortedPred);
    
    // Освобождаем память
    flatTrue.dispose();
    flatPred.dispose();
    sortedTrue.dispose();
    sortedPred.dispose();
    
    return loss;
  });
}

// ==========================================
// MODEL ARCHITECTURES (All implemented)
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
    // 256 → 64 → 256 (bottleneck)
    model.add(tf.layers.dense({ units: 64, activation: "relu" }));
    model.add(tf.layers.dense({ units: 256, activation: "sigmoid" }));
  } else if (archType === "transformation") {
    // 256 → 256 → 256 (1:1 mapping)
    model.add(tf.layers.dense({ units: 256, activation: "relu" }));
    model.add(tf.layers.dense({ units: 256, activation: "sigmoid" }));
  } else if (archType === "expansion") {
    // 256 → 512 → 256 (overcomplete)
    model.add(tf.layers.dense({ units: 512, activation: "relu" }));
    model.add(tf.layers.dense({ units: 256, activation: "sigmoid" }));
  }

  model.add(tf.layers.reshape({ targetShape: [16, 16, 1] }));
  return model;
}

// ==========================================
// ✅ CUSTOM LOSS FUNCTION (Main innovation)
// ==========================================
function studentLoss(yTrue, yPred) {
  return tf.tidy(() => {
    const lossSorted = sortedMSE(yTrue, yPred).mul(1.0);    // Сохраняем пиксели
    const lossSmooth = smoothness(yPred).mul(0.1);          // Сглаживаем
    const lossDir = directionX(yPred).mul(0.2);             // Градиент →→→
    
    return lossSorted.add(lossSmooth).add(lossDir);
  });
}

// ==========================================
// TRAINING LOOP
// ==========================================
async function trainStep() {
  state.step++;

  if (!state.studentModel) {
    log("Error: Student model not initialized.", true);
    return;
  }

  // Train Baseline (MSE)
  const baselineLossVal = tf.tidy(() => {
    const { value, grads } = tf.variableGrads(() => {
      const yPred = state.baselineModel.predict(state.xInput);
      return mse(state.xInput, yPred);
    }, state.baselineModel.getWeights());
    state.optimizer.applyGradients(grads);
    return value.dataSync()[0];
  });

  // Train Student (Custom Loss)
  let studentLossVal = 0;
  try {
    studentLossVal = tf.tidy(() => {
      const { value, grads } = tf.variableGrads(() => {
        const yPred = state.studentModel.predict(state.xInput);
        return studentLoss(state.xInput, yPred);
      }, state.studentModel.getWeights());
      state.optimizer.applyGradients(grads);
      return value.dataSync()[0];
    });
    log(`Step ${state.step}: Base=${baselineLossVal.toFixed(4)} | Student=${studentLossVal.toFixed(4)}`);
  } catch (e) {
    log(`Training Error: ${e.message}`, true);
    return;
  }

  if (state.step % 5 === 0 || !state.isAutoTraining) {
    await render();
    updateLossDisplay(baselineLossVal, studentLossVal);
  }
}

// ==========================================
// UI & INIT
// ==========================================
function init() {
  state.xInput = tf.randomUniform(CONFIG.inputShapeData);
  resetModels();

  tf.browser.toPixels(state.xInput.squeeze(), document.getElementById("canvas-input"));

  document.getElementById("btn-train").addEventListener("click", () => trainStep());
  document.getElementById("btn-auto").addEventListener("click", toggleAutoTrain);
  document.getElementById("btn-reset").addEventListener("click", () => resetModels());

  document.querySelectorAll('input[name="arch"]').forEach((radio) => {
    radio.addEventListener("change", (e) => {
      resetModels(e.target.value);
      document.getElementById("student-arch-label").innerText =
        e.target.value.charAt(0).toUpperCase() + e.target.value.slice(1);
    });
  });

  log("✅ READY - All TODOs implemented!");
}

function resetModels(archType = null) {
  if (typeof archType !== "string") {
    const checked = document.querySelector('input[name="arch"]:checked');
    archType = checked ? checked.value : "compression";
  }

  if (state.isAutoTraining) stopAutoTrain();

  // Cleanup
  if (state.baselineModel) state.baselineModel.dispose();
  if (state.studentModel) state.studentModel.dispose();
  if (state.optimizer) state.optimizer.dispose();

  state.baselineModel = createBaselineModel();
  state.studentModel = createStudentModel(archType);
  state.optimizer = tf.train.adam(CONFIG.learningRate);
  state.step = 0;

  log(`🔄 Reset. Student: ${archType}`);
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
