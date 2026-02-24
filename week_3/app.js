/**
 * Neural Network Design: The Gradient Puzzle - DEBUG VERSION
 */

const CONFIG = {
  inputShapeModel: [16, 16, 1],
  inputShapeData: [1, 16, 16, 1],
  learningRate: 0.03,
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
// LOSS FUNCTIONS
// ==========================================
function mse(yTrue, yPred) {
  return tf.losses.meanSquaredError(yTrue, yPred);
}

function smoothness(yPred) {
  const diffX = yPred.slice([0,0,0,0], [-1,-1,15,-1]).sub(yPred.slice([0,0,1,0], [-1,-1,15,-1]));
  const diffY = yPred.slice([0,0,0,0], [-1,15,-1,-1]).sub(yPred.slice([0,1,0,0], [-1,15,-1,-1]));
  return tf.mean(tf.square(diffX)).add(tf.mean(tf.square(diffY)));
}

function directionX(yPred) {
  const mask = tf.linspace(0, 1, 16).reshape([1, 1, 16, 1]);
  return tf.mean(yPred.mul(mask)).mul(-1);
}

function histogramLoss(yTrue, yPred) {
  return tf.tidy(() => {
    const flatTrue = yTrue.reshape([256]);
    const flatPred = yPred.reshape([256]);
    const meanTrue = tf.mean(flatTrue);
    const meanPred = tf.mean(flatPred);
    const lossMean = tf.square(meanTrue.sub(meanPred));
    const varTrue = tf.mean(tf.square(flatTrue.sub(meanTrue)));
    const varPred = tf.mean(tf.square(flatPred.sub(meanPred)));
    const lossVar = tf.square(varTrue.sub(varPred));
    const loss = lossMean.add(lossVar.mul(0.5));
    
    flatTrue.dispose();
    flatPred.dispose();
    meanTrue.dispose();
    meanPred.dispose();
    varTrue.dispose();
    varPred.dispose();
    return loss;
  });
}

function studentLoss(yTrue, yPred) {
  return tf.tidy(() => {
    const lossHist = histogramLoss(yTrue, yPred).mul(2.0);
    const lossSmooth = smoothness(yPred).mul(0.1);
    const lossDir = directionX(yPred).mul(1.0);
    return lossHist.add(lossSmooth).add(lossDir);
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
// TRAINING STEP
// ==========================================
async function trainStep() {
  log(`🔄 Training step ${state.step + 1}...`);
  
  state.step++;

  // Baseline training
  const baselineLossVal = tf.tidy(() => {
    const { value, grads } = tf.variableGrads(() => {
      return mse(state.xInput, state.baselineModel.predict(state.xInput));
    }, state.baselineModel.getWeights());
    state.baselineOptimizer.applyGradients(grads);
    return value.dataSync()[0];
  });

  // Student training
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
    log(`❌ Student Error: ${e.message}`, true);
    return;
  }

  log(`✅ Step ${state.step}: Base=${baselineLossVal.toFixed(4)} | Student=${studentLossVal.toFixed(4)}`);
  
  await render();
  updateLossDisplay(baselineLossVal, studentLossVal);
}

// ==========================================
// RENDER
// ==========================================
async function render() {
  log("🎨 Rendering...");
  const basePred = state.baselineModel.predict(state.xInput);
  const studPred = state.studentModel.predict(state.xInput);

  await tf.browser.toPixels(basePred.squeeze(), document.getElementById("canvas-baseline"));
  await tf.browser.toPixels(studPred.squeeze(), document.getElementById("canvas-student"));

  basePred.dispose();
  studPred.dispose();
  log("✅ Render complete");
}

// ==========================================
// UI UPDATES
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

// ==========================================
// UI EVENT HANDLERS
// ==========================================
function toggleAutoTrain() {
  const btn = document.getElementById("btn-auto");
  if (state.isAutoTraining) {
    stopAutoTrain();
  } else {
    state.isAutoTraining = true;
    btn.innerText = "Auto Train (Stop)";
    btn.classList.add("btn-stop");
    btn.classList.remove("btn-auto");
    log("🚀 Auto training STARTED");
    loop();
  }
}

function stopAutoTrain() {
  state.isAutoTraining = false;
  const btn = document.getElementById("btn-auto");
  btn.innerText = "Auto Train (Start)";
  btn.classList.add("btn-auto");
  btn.classList.remove("btn-stop");
  log("⏹️ Auto training STOPPED");
}

function loop() {
  if (state.isAutoTraining) {
    trainStep();
    setTimeout(loop, CONFIG.autoTrainSpeed);
  }
}

// ==========================================
// RESET MODELS
// ==========================================
function resetModels(archType = null) {
  log("🔄 Resetting models...");
  
  if (typeof archType !== "string") {
    const checked = document.querySelector('input[name="arch"]:checked');
    archType = checked ? checked.value : "compression";
  }

  if (state.isAutoTraining) stopAutoTrain();

  // Cleanup
  if (state.baselineModel) state.baselineModel.dispose();
  if (state.studentModel) state.studentModel.dispose();
  if (state.baselineOptimizer) state.baselineOptimizer.dispose();
  if (state.studentOptimizer) state.studentOptimizer.dispose();

  // Create new models
  state.baselineModel = createBaselineModel();
  state.studentModel = createStudentModel(archType);
  state.baselineOptimizer = tf.train.adam(CONFIG.learningRate);
  state.studentOptimizer = tf.train.adam(CONFIG.learningRate);
  state.step = 0;

  log(`✅ Models reset. Arch: ${archType}`);
  render();
}

// ==========================================
// MAIN INIT
// ==========================================
async function init() {
  log("🚀 Initializing TensorFlow.js...");
  
  // Wait for TF.js to be ready
  await tf.ready();
  log("✅ TensorFlow.js ready!");
  
  // Create input
  state.xInput = tf.randomUniform(CONFIG.inputShapeData);
  log("✅ Input noise created");
  
  // Initial models
  resetModels();
  
  // Render input
  tf.browser.toPixels(state.xInput.squeeze(), document.getElementById("canvas-input"));
  log("✅ Input rendered");

  // ✅ DEBUG: Check if DOM elements exist
  const btnTrain = document.getElementById("btn-train");
  const btnAuto = document.getElementById("btn-auto");
  const btnReset = document.getElementById("btn-reset");
  
  log(`DOM check: train=${!!btnTrain}, auto=${!!btnAuto}, reset=${!!btnReset}`);

  // Event listeners
  if (btnTrain) {
    btnTrain.addEventListener("click", () => {
      log("🔘 Train button clicked!");
      trainStep();
    });
    log("✅ Train button bound");
  }

  if (btnAuto) {
    btnAuto.addEventListener("click", toggleAutoTrain);
    log("✅ Auto button bound");
  }

  if (btnReset) {
    btnReset.addEventListener("click", () => {
      log("🔄 Reset button clicked!");
      resetModels();
    });
    log("✅ Reset button bound");
  }

  // Architecture selector
  document.querySelectorAll('input[name="arch"]').forEach((radio) => {
    radio.addEventListener("change", (e) => {
      log(`📡 Arch changed to: ${e.target.value}`);
      resetModels(e.target.value);
      document.getElementById("student-arch-label").innerText =
        e.target.value.charAt(0).toUpperCase() + e.target.value.slice(1);
    });
  });
  log("✅ Arch selector bound");

  log("🎉 READY TO TRAIN! Click 'Train 1 Step' or 'Auto Train'");
}

// Start when page loads
init();
