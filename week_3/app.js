/**
 * Neural Network Design: The Gradient Puzzle
 *
 * Objective:
 * Modify the Student Model architecture and loss function to transform
 * random noise input into a smooth, directional gradient output.
 *
 * Levels:
 * 1. The Trap of Standard Reconstruction (MSE) – currently commented out.
 * 2. The "Distribution" Constraint (Sorted MSE) – implemented.
 * 3. Shaping the Geometry (Smoothness & Direction) – implemented with tunable weights.
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
 * Smoothness (Total Variation) Loss
 * Penalizes large differences between neighboring pixels.
 */
function smoothness(yPred) {
  // Difference in X direction: pixel[i, j] - pixel[i, j+1]
  const diffX = yPred
    .slice([0, 0, 0, 0], [-1, -1, 15, -1])
    .sub(yPred.slice([0, 0, 1, 0], [-1, -1, 15, -1]));

  // Difference in Y direction: pixel[i, j] - pixel[i+1, j]
  const diffY = yPred
    .slice([0, 0, 0, 0], [-1, 15, -1, -1])
    .sub(yPred.slice([0, 1, 0, 0], [-1, 15, -1, -1]));

  return tf.mean(tf.square(diffX)).add(tf.mean(tf.square(diffY)));
}

/**
 * Directionality (Gradient) Loss
 * Encourages pixels on the right to be brighter than pixels on the left.
 */
function directionX(yPred) {
  const width = 16;
  // Mask increases linearly from -1 (left) to +1 (right)
  const mask = tf.linspace(-1, 1, width).reshape([1, 1, width, 1]);
  // We want yPred to be positively correlated with the mask.
  // Minimizing -mean(yPred * mask) pushes bright values to the right.
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

// ------------------------------------------------------------------
// [STUDENT TODO-A]: STUDENT ARCHITECTURE DESIGN
// Implement 'transformation' and 'expansion' architectures.
// ------------------------------------------------------------------
function createStudentModel(archType) {
  const model = tf.sequential();
  model.add(tf.layers.flatten({ inputShape: CONFIG.inputShapeModel }));

  if (archType === 'compression') {
    // Bottleneck: compresses information
    model.add(tf.layers.dense({ units: 64, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 256, activation: 'sigmoid' }));
  } else if (archType === 'transformation') {
    // [IMPLEMENTED] Transformation: same dimension as input (256)
    // This acts like a rotation or a 1x1 mapping.
    model.add(tf.layers.dense({ units: 256, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 256, activation: 'sigmoid' }));
  } else if (archType === 'expansion') {
    // [IMPLEMENTED] Expansion: overcomplete representation
    // Increases dimension to capture more complex features.
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

// ------------------------------------------------------------------
// [STUDENT TODO-B]: STUDENT LOSS DESIGN
// Combine Sorted MSE, Smoothness, and Direction.
// Tune the lambda coefficients to achieve a clean gradient.
// ------------------------------------------------------------------
function studentLoss(yTrue, yPred) {
  return tf.tidy(() => {
    // --- Level 2: "Distribution" Constraint (Sorted MSE) ---
    // Flatten, sort, and reshape back to original shape.
    const yTrueSorted = tf.topk(tf.reshape(yTrue, [-1]), 256).values.reshape(yTrue.shape);
    const yPredSorted = tf.topk(tf.reshape(yPred, [-1]), 256).values.reshape(yPred.shape);
    const lossSortedMSE = mse(yTrueSorted, yPredSorted);

    // --- Level 3: Shaping the Geometry ---
    // Experiment with these weights!
    const lambdaSmooth = 0.1;   // Weight for smoothness (start small)
    const lambdaDir = 0.05;      // Weight for direction (start small)

    const lossSmooth = smoothness(yPred).mul(lambdaSmooth);
    const lossDir = directionX(yPred).mul(lambdaDir);

    // Total loss
    return lossSortedMSE.add(lossSmooth).add(lossDir);

    // --- Level 1 (The Trap) - kept for reference ---
    // return mse(yTrue, yPred); // This would just copy the input.
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
  // 1. Generate fixed noise
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
  // Ensure archType is a string, not an event object
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

// Start everything
init();
