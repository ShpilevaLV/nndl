/**
 * Neural Network Design: The Gradient Puzzle
 *
 * Objective:
 * Modify the Student Model architecture and loss function to transform
 * random noise input into a smooth, directional gradient output.
 *
 * Student tasks (already implemented below):
 * - TODO-A: Implement 'transformation' and 'expansion' architectures.
 * - TODO-B: Implement custom loss combining sorted MSE, smoothness, and direction.
 * - TODO-C: Observe visual difference between baseline (MSE only) and student.
 */

// ==========================================
// 1. Global State & Config
// ==========================================
const CONFIG = {
  inputShapeModel: [16, 16, 1],          // without batch
  inputShapeData: [1, 16, 16, 1],        // with batch
  learningRate: 0.05,
  autoTrainSpeed: 50,                    // ms per step
  // Loss weights (adjustable)
  smoothWeight: 0.1,
  dirWeight: 0.1,
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

// Standard MSE (pixel-wise)
function mse(yTrue, yPred) {
  return tf.losses.meanSquaredError(yTrue, yPred);
}

// Sorted MSE: compares sorted pixel values – allows rearrangement but preserves histogram
function sortedMSE(yTrue, yPred) {
  return tf.tidy(() => {
    // Flatten to 1D
    const yTrueFlat = yTrue.flatten();
    const yPredFlat = yPred.flatten();
    // Sort both in descending order (using topk). Order doesn't matter for MSE as long as consistent.
    const yTrueSorted = tf.topk(yTrueFlat, yTrueFlat.size).values;
    const yPredSorted = tf.topk(yPredFlat, yPredFlat.size).values;
    return tf.losses.meanSquaredError(yTrueSorted, yPredSorted);
  });
}

// Smoothness (Total Variation) – penalizes large differences between adjacent pixels
function smoothness(yPred) {
  return tf.tidy(() => {
    // Difference in X direction: pixel[i, j] - pixel[i, j+1]
    const diffX = yPred
      .slice([0, 0, 0, 0], [-1, -1, 15, -1])
      .sub(yPred.slice([0, 0, 1, 0], [-1, -1, 15, -1]));

    // Difference in Y direction: pixel[i, j] - pixel[i+1, j]
    const diffY = yPred
      .slice([0, 0, 0, 0], [-1, 15, -1, -1])
      .sub(yPred.slice([0, 1, 0, 0], [-1, 15, -1, -1]));

    // Return mean of squared differences
    return tf.mean(tf.square(diffX)).add(tf.mean(tf.square(diffY)));
  });
}

// Directionality – encourage right side to be brighter than left
function directionX(yPred) {
  return tf.tidy(() => {
    const width = 16;
    // Mask increasing from -1 (left) to +1 (right)
    const mask = tf.linspace(-1, 1, width).reshape([1, 1, width, 1]);
    // We want yPred to correlate with mask => maximize mean(yPred * mask)
    // Minimize the negative of that product.
    return tf.mean(yPred.mul(mask)).mul(-1);
  });
}

// ==========================================
// 3. Model Architecture
// ==========================================

// Baseline Model: Fixed Compression (Undercomplete AE)
function createBaselineModel() {
  const model = tf.sequential();
  model.add(tf.layers.flatten({ inputShape: CONFIG.inputShapeModel }));
  model.add(tf.layers.dense({ units: 64, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 256, activation: 'sigmoid' }));
  model.add(tf.layers.reshape({ targetShape: [16, 16, 1] }));
  return model;
}

// ------------------------------------------------------------------
// [TODO-A] STUDENT ARCHITECTURE DESIGN – now fully implemented
// ------------------------------------------------------------------
function createStudentModel(archType) {
  const model = tf.sequential();
  model.add(tf.layers.flatten({ inputShape: CONFIG.inputShapeModel }));

  if (archType === 'compression') {
    // Bottleneck: compress information
    model.add(tf.layers.dense({ units: 64, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 256, activation: 'sigmoid' }));
  } else if (archType === 'transformation') {
    // Transformation: keep dimension roughly constant (1:1 mapping)
    model.add(tf.layers.dense({ units: 256, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 256, activation: 'sigmoid' }));
  } else if (archType === 'expansion') {
    // Expansion: overcomplete representation
    model.add(tf.layers.dense({ units: 512, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 256, activation: 'sigmoid' }));
  } else {
    throw new Error(`Unknown architecture type: ${archType}`);
  }

  model.add(tf.layers.reshape({ targetShape: [16, 16, 1] }));
  return model;
}

// ==========================================
// 4. Custom Loss Function
// ==========================================

// ------------------------------------------------------------------
// [TODO-B] STUDENT LOSS DESIGN – now combines sorted MSE, smoothness, direction
// ------------------------------------------------------------------
function studentLoss(yTrue, yPred) {
  return tf.tidy(() => {
    // 1. Sorted MSE – allows pixel rearrangement, conserves color histogram
    const lossSorted = sortedMSE(yTrue, yPred);

    // 2. Smoothness – encourages local consistency
    const lossSmooth = smoothness(yPred).mul(CONFIG.smoothWeight);

    // 3. Direction – makes right side brighter
    const lossDir = directionX(yPred).mul(CONFIG.dirWeight);

    // Total loss
    return lossSorted.add(lossSmooth).add(lossDir);
  });
}

// ==========================================
// 5. Training Loop
// ==========================================

async function trainStep() {
  state.step++;

  if (!state.studentModel || !state.studentModel.getWeights) {
    log('Error: Student model not initialized properly.', true);
    stopAutoTrain();
    return;
  }

  // Train Baseline (always uses standard MSE)
  const baselineLossVal = tf.tidy(() => {
    const { value, grads } = tf.variableGrads(() => {
      const yPred = state.baselineModel.predict(state.xInput);
      return mse(state.xInput, yPred);
    }, state.baselineModel.getWeights());

    state.optimizer.applyGradients(grads);
    return value.dataSync()[0];
  });

  // Train Student (custom loss)
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
    log(`Step ${state.step}: Base Loss=${baselineLossVal.toFixed(4)} | Student Loss=${studentLossVal.toFixed(4)}`);
  } catch (e) {
    log(`Error in Student Training: ${e.message}`, true);
    stopAutoTrain();
    return;
  }

  // Update visuals every 5 steps or when manual step is used
  if (state.step % 5 === 0 || !state.isAutoTraining) {
    await render();
    updateLossDisplay(baselineLossVal, studentLossVal);
  }
}

// ==========================================
// 6. UI & Initialization logic
// ==========================================

function init() {
  // 1. Generate fixed noise (batch size included)
  state.xInput = tf.randomUniform(CONFIG.inputShapeData);

  // 2. Initialize Models
  resetModels();

  // 3. Render Input
  tf.browser.toPixels(state.xInput.squeeze(), document.getElementById('canvas-input'));

  // 4. Bind Events
  document.getElementById('btn-train').addEventListener('click', () => trainStep());
  document.getElementById('btn-auto').addEventListener('click', toggleAutoTrain);
  document.getElementById('btn-reset').addEventListener('click', resetModels);

  document.querySelectorAll('input[name="arch"]').forEach((radio) => {
    radio.addEventListener('change', (e) => {
      resetModels(e.target.value);
      document.getElementById('student-arch-label').innerText =
        e.target.value.charAt(0).toUpperCase() + e.target.value.slice(1);
    });
  });

  log('Initialized. Ready to train.');
}

function resetModels(archType = null) {
  // Called by event or programmatically
  if (typeof archType !== 'string') archType = null;

  if (state.isAutoTraining) stopAutoTrain();

  if (!archType) {
    const checked = document.querySelector('input[name="arch"]:checked');
    archType = checked ? checked.value : 'compression';
  }

  // Dispose old resources
  if (state.baselineModel) state.baselineModel.dispose();
  if (state.studentModel) state.studentModel.dispose();
  if (state.optimizer) state.optimizer.dispose();

  // Create new models
  state.baselineModel = createBaselineModel();
  try {
    state.studentModel = createStudentModel(archType);
  } catch (e) {
    log(`Error creating model: ${e.message}`, true);
    state.studentModel = createBaselineModel(); // fallback
  }

  state.optimizer = tf.train.adam(CONFIG.learningRate);
  state.step = 0;

  log(`Models reset. Student Arch: ${archType}`);
  render();
}

async function render() {
  const basePred = state.baselineModel.predict(state.xInput);
  const studPred = state.studentModel.predict(state.xInput);

  await tf.browser.toPixels(basePred.squeeze(), document.getElementById('canvas-baseline'));
  await tf.browser.toPixels(studPred.squeeze(), document.getElementById('canvas-student'));

  basePred.dispose();
  studPred.dispose();
}

function updateLossDisplay(base, stud) {
  document.getElementById('loss-baseline').innerText = `Loss: ${base.toFixed(5)}`;
  document.getElementById('loss-student').innerText = `Loss: ${stud.toFixed(5)}`;
}

function log(msg, isError = false) {
  const el = document.getElementById('log-area');
  const span = document.createElement('div');
  span.innerText = `> ${msg}`;
  if (isError) span.classList.add('error');
  el.prepend(span);
}

function toggleAutoTrain() {
  const btn = document.getElementById('btn-auto');
  if (state.isAutoTraining) {
    stopAutoTrain();
  } else {
    state.isAutoTraining = true;
    btn.innerText = 'Auto Train (Stop)';
    btn.classList.add('btn-stop');
    btn.classList.remove('btn-auto');
    loop();
  }
}

function stopAutoTrain() {
  state.isAutoTraining = false;
  const btn = document.getElementById('btn-auto');
  btn.innerText = 'Auto Train (Start)';
  btn.classList.add('btn-auto');
  btn.classList.remove('btn-stop');
}

function loop() {
  if (state.isAutoTraining) {
    trainStep();
    setTimeout(loop, CONFIG.autoTrainSpeed);
  }
}

// Start everything
init();
