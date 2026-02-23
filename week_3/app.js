/**
 * Neural Network Design: The Gradient Puzzle
 *
 * Objective:
 * Transform random noise into a smooth directional gradient.
 * Student model uses a custom loss: MSE + Smoothness + Direction.
 * Architectures: Compression, Transformation, Expansion.
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

function mse(yTrue, yPred) {
  return tf.losses.meanSquaredError(yTrue, yPred);
}

/**
 * Smoothness (Total Variation) Loss.
 * Penalizes large differences between neighboring pixels.
 */
function smoothness(yPred) {
  // yPred shape: [1, 16, 16, 1]
  // Compute differences in x direction (between columns)
  const left = yPred.slice([0, 0, 0, 0], [1, 16, 15, 1]);     // first 15 columns
  const right = yPred.slice([0, 0, 1, 0], [1, 16, 15, 1]);    // last 15 columns
  const diffX = tf.sub(left, right);

  // Differences in y direction (between rows)
  const top = yPred.slice([0, 0, 0, 0], [1, 15, 16, 1]);      // first 15 rows
  const bottom = yPred.slice([0, 1, 0, 0], [1, 15, 16, 1]);   // last 15 rows
  const diffY = tf.sub(top, bottom);

  // Return mean of squared differences
  return tf.mean(tf.square(diffX)).add(tf.mean(tf.square(diffY)));
}

/**
 * Directionality (Gradient) Loss.
 * Encourages pixels on the right to be brighter than those on the left.
 */
function directionX(yPred) {
  const width = 16;
  // Mask: linear increase from -1 (left) to +1 (right)
  const mask = tf.linspace(-1, 1, width).reshape([1, 1, width, 1]);
  // We want yPred to be positively correlated with the mask.
  // Minimizing -mean(yPred * mask) pushes bright pixels to the right.
  return tf.mean(yPred.mul(mask)).mul(-1);
}

// ==========================================
// 3. Model Architecture
// ==========================================

function createBaselineModel() {
  const model = tf.sequential();
  model.add(tf.layers.flatten({ inputShape: CONFIG.inputShapeModel }));
  model.add(tf.layers.dense({ units: 64, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 256, activation: 'sigmoid' }));
  model.add(tf.layers.reshape({ targetShape: [16, 16, 1] }));
  return model;
}

/**
 * Creates the student model with the selected architecture.
 * @param {string} archType - 'compression', 'transformation', or 'expansion'
 */
function createStudentModel(archType) {
  const model = tf.sequential();
  model.add(tf.layers.flatten({ inputShape: CONFIG.inputShapeModel }));

  if (archType === 'compression') {
    // Bottleneck: compresses information
    model.add(tf.layers.dense({ units: 64, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 256, activation: 'sigmoid' }));
  } else if (archType === 'transformation') {
    // Transformation: same dimension as input (256)
    model.add(tf.layers.dense({ units: 256, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 256, activation: 'sigmoid' }));
  } else if (archType === 'expansion') {
    // Expansion: overcomplete representation (512)
    model.add(tf.layers.dense({ units: 512, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 256, activation: 'sigmoid' }));
  } else {
    throw new Error(`Unknown architecture type: ${archType}`);
  }

  model.add(tf.layers.reshape({ targetShape: [16, 16, 1] }));
  return model;
}

// ==========================================
// 4. Custom Loss Function (The Heart of the Puzzle)
// ==========================================

/**
 * Student loss: combination of MSE, smoothness, and direction.
 * Adjust the lambdas to achieve a clean gradient.
 */
function studentLoss(yTrue, yPred) {
  return tf.tidy(() => {
    // Standard MSE – keeps the output from collapsing
    const lossMSE = mse(yTrue, yPred);

    // Smoothness – encourages local consistency
    const lambdaSmooth = 0.8;   // try 0.5–1.0
    const lossSmooth = smoothness(yPred).mul(lambdaSmooth);

    // Direction – encourages bright‑right gradient
    const lambdaDir = 0.5;       // try 0.3–0.7
    const lossDir = directionX(yPred).mul(lambdaDir);

    return lossMSE.add(lossSmooth).add(lossDir);
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

  // Train Baseline (MSE only)
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
    log(`Step ${state.step}: Base Loss=${baselineLossVal.toFixed(4)} | Student Loss=${studentLossVal.toFixed(4)}`);
  } catch (e) {
    log(`Error in Student Training: ${e.message}`, true);
    stopAutoTrain();
    return;
  }

  // Visualize every 5 steps or after a single step
  if (state.step % 5 === 0 || !state.isAutoTraining) {
    await render();
    updateLossDisplay(baselineLossVal, studentLossVal);
  }
}

// ==========================================
// 6. UI & Initialization logic
// ==========================================

function init() {
  state.xInput = tf.randomUniform(CONFIG.inputShapeData);
  resetModels();

  tf.browser.toPixels(state.xInput.squeeze(), document.getElementById('canvas-input'));

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
  if (typeof archType !== 'string') {
    archType = null;
  }

  if (state.isAutoTraining) {
    stopAutoTrain();
  }

  if (!archType) {
    const checked = document.querySelector('input[name="arch"]:checked');
    archType = checked ? checked.value : 'compression';
  }

  // Dispose old resources
  if (state.baselineModel) {
    state.baselineModel.dispose();
    state.baselineModel = null;
  }
  if (state.studentModel) {
    state.studentModel.dispose();
    state.studentModel = null;
  }
  if (state.optimizer) {
    state.optimizer.dispose();
    state.optimizer = null;
  }

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

// Start
init();
