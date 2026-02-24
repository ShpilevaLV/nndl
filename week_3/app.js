/**
 * Neural Network Design: The Gradient Puzzle - 16 VERTICAL STRIPS PERFECT
 */

const CONFIG = {
  inputShapeModel: [16, 16, 1],
  inputShapeData: [1, 16, 16, 1],
  learningRate: 0.01,  // Медленнее = точнее
  autoTrainSpeed: 30,
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
// LOSS FUNCTIONS - ПЕРЕСТАНОВКА ПИКСЕЛЕЙ
// ==========================================

function mse(yTrue, yPred) {
  return tf.losses.meanSquaredError(yTrue, yPred);
}

// ✅ ТОЧНЫЙ ГРАДИЕНТ: 16 вертикальных полос [-1.0, -0.8, ..., 1.0]
function directionX(yPred) {
  // Создаём маску: колонка 0 = -1.0, колонка 1 = -0.8, ..., колонка 15 = 1.0
  const gradientMask = tf.tidy(() => {
    const values = [-1.0, -0.875, -0.75, -0.625, -0.5, -0.375, -0.25, -0.125,
                    0.125,  0.25,  0.375,  0.5,   0.625,  0.75,  0.875, 1.0];
    return tf.tensor1d(values).reshape([1, 1, 16, 1]);  // [1,1,16,1]
  });
  // Максимизируем корреляцию: -mean(yPred * mask)
  return tf.mean(yPred.mul(gradientMask)).mul(-2.0);  // Усилено
}

// ✅ СГЛАЖИВАНИЕ внутри колонок (вертикальное)
function smoothnessVertical(yPred) {
  // Только вертикальные различия между строками в одной колонке
  const diffY = yPred.slice([0,0,0,0], [-1,15,-1,-1]).sub(
    yPred.slice([0,1,0,0], [-1,15,-1,-1])
  );
  return tf.mean(tf.square(diffY)).mul(0.5);
}

// ✅ СЛАБЫЙ histogram loss - только для сохранения яркости
function histogramLoss(yTrue, yPred) {
  return tf.tidy(() => {
    const meanTrue = tf.mean(yTrue);
    const meanPred = tf.mean(yPred);
    return tf.square(meanTrue.sub(meanPred)).mul(0.5);  // Только среднее
  });
}

function studentLoss(yTrue, yPred) {
  return tf.tidy(() => {
    const lossHist = histogramLoss(yTrue, yPred).mul(0.3);      // Слабое сохранение яркости
    const lossDir = directionX(yPred).mul(3.0);                 // ГЛАВНЫЙ: точный градиент
    const lossSmooth = smoothnessVertical(yPred).mul(0.8);      // Только вертикальное сглаживание
    return lossHist.add(lossSmooth).add(lossDir);
  });
}

// ==========================================
// MODELS - БОЛЬШЕ ЕМКОСТИ ДЛЯ ПЕРЕСТАНОВКИ
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
    // БОЛЬШЕ СЛОЁВ для точной перестановки
    model.add(tf.layers.dense({ units: 256, activation: "relu" }));
    model.add(tf.layers.dense({ units: 256, activation: "relu" }));
    model.add(tf.layers.dense({ units: 256, activation: "sigmoid" }));
  } else {
    // ЕЩЁ БОЛЬШЕ ЕМКОСТИ
    model.add(tf.layers.dense({ units: 512, activation: "relu" }));
    model.add(tf.layers.dense({ units: 512, activation: "relu" }));
    model.add(tf.layers.dense({ units: 256, activation: "sigmoid" }));
  }

  model.add(tf.layers.reshape({ targetShape: [16, 16, 1] }));
  return model;
}

// ==========================================
// TRAINING & UI (упрощённые логи)
// ==========================================
async function trainStep() {
  state.step++;

  // Baseline
  const baselineLossVal = tf.tidy(() => {
    const { value, grads } = tf.variableGrads(() => {
      return mse(state.xInput, state.baselineModel.predict(state.xInput));
    }, state.baselineModel.getWeights());
    state.baselineOptimizer.applyGradients(grads);
    return value.dataSync()[0];
  });

  // Student
  let studentLossVal = 0;
  try {
    studentLossVal = tf.tidy(() => {
      const { value, grads } = tf.variableGrads(() => {
        return studentLoss(state.xInput, state.studentModel.predict(state.xInput));
      }, state.studentModel.getWeights());
      state.studentOptimizer.applyGradients(grads);
      return value.dataSync()[0];
    });
  } catch (e) {
    console.error(e);
    return;
  }

  // Логи только каждые 20 шагов
  if (state.step % 20 === 0) {
    console.log(`Step ${state.step}: Base=${baselineLossVal.toFixed(4)} | Student=${studentLossVal.toFixed(4)}`);
  }
  
  if (state.step % 10 === 0) {
    await render();
    updateLossDisplay(baselineLossVal, studentLossVal);
  }
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

// Остальные функции UI без изменений...
function toggleAutoTrain() {
  const btn = document.getElementById("btn-auto");
  if (state.isAutoTraining) {
    state.isAutoTraining = false;
    btn.innerText = "Auto Train (Start)";
    btn.classList.add("btn-auto");
    btn.classList.remove("btn-stop");
    log("⏹️ Auto training STOPPED");
  } else {
    state.isAutoTraining = true;
    btn.innerText = "Auto Train (Stop)";
    btn.classList.add("btn-stop");
    btn.classList.remove("btn-auto");
    log("🚀 Auto training STARTED - 16 strips gradient!");
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
  if (typeof archType !== "string") {
    const checked = document.querySelector('input[name="arch"]:checked');
    archType = checked ? checked.value : "compression";
  }

  if (state.isAutoTraining) {
    state.isAutoTraining = false;
    document.getElementById("btn-auto").innerText = "Auto Train (Start)";
  }

  if (state.baselineModel) state.baselineModel.dispose();
  if (state.studentModel) state.studentModel.dispose();
  if (state.baselineOptimizer) state.baselineOptimizer.dispose();
  if (state.studentOptimizer) state.studentOptimizer.dispose();

  state.baselineModel = createBaselineModel();
  state.studentModel = createStudentModel(archType);
  state.baselineOptimizer = tf.train.adam(CONFIG.learningRate);
  state.studentOptimizer = tf.train.adam(CONFIG.learningRate);
  state.step = 0;

  log(`🔄 Reset. Arch: ${archType} → 16 strips ready`);
  render();
}

async function init() {
  await tf.ready();
  state.xInput = tf.randomUniform(CONFIG.inputShapeData);
  resetModels();

  tf.browser.toPixels(state.xInput.squeeze(), document.getElementById("canvas-input"));

  // Event listeners
  document.getElementById("btn-train").addEventListener("click", trainStep);
  document.getElementById("btn-auto").addEventListener("click", toggleAutoTrain);
  document.getElementById("btn-reset").addEventListener("click", resetModels);

  document.querySelectorAll('input[name="arch"]').forEach((radio) => {
    radio.addEventListener("change", (e) => {
      resetModels(e.target.value);
      document.getElementById("student-arch-label").innerText =
        e.target.value.charAt(0).toUpperCase() + e.target.value.slice(1);
    });
  });

  log("🎯 16 VERTICAL STRIPS GRADIENT READY!");
}

init();
