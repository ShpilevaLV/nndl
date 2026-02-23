/**
 * Neural Network Design: The Gradient Puzzle
 *
 * Objective:
 * Modify the Student Model architecture and loss function to transform
 * random noise input into a smooth, directional gradient output.
 */

// ==========================================
// 1. Global State & Config
// ==========================================
const CONFIG = {
  // Model definition shape (no batch dim) - used for layer creation
  inputShapeModel: [16, 16, 1],
  // Data tensor shape (includes batch dim) - used for input tensor creation
  inputShapeData: [1, 16, 16, 1],
  learningRate: 0.05,
  autoTrainSpeed: 50, // ms delay between steps (lower is faster)
};

let state = {
  step: 0,
  isAutoTraining: false,
  autoTrainInterval: null,
  xInput: null, // The fixed noise input
  baselineModel: null,
  studentModel: null,
  optimizer: null,
};

// ==========================================
// 2. Helper Functions (Loss Components)
// ==========================================

// Standard MSE: Mean Squared Error
function mse(yTrue, yPred) {
  return tf.losses.meanSquaredError(yTrue, yPred);
}

// Smoothness (Total Variation) - Penalize differences between adjacent pixels
function smoothness(yPred) {
  // Difference in X direction: pixel[i, j] - pixel[i, j+1]
  const diffX = yPred
    .slice([0, 0, 0, 0], [-1, -1, 15, -1])
    .sub(yPred.slice([0, 0, 1, 0], [-1, -1, 15, -1]));

  // Difference in Y direction: pixel[i, j] - pixel[i+1, j]
  const diffY = yPred
    .slice([0, 0, 0, 0], [-1, 15, -1, -1])
    .sub(yPred.slice([0, 1, 0, 0], [-1, 15, -1, -1]));

  // Return sum of squares
  return tf.mean(tf.square(diffX)).add(tf.mean(tf.square(diffY)));
}

// Directionality (Gradient) - Encourage pixels on the right to be brighter
function directionX(yPred) {
  // Create a weight mask that increases from left (-1) to right (+1)
  const width = 16;
  const mask = tf.linspace(-1, 1, width).reshape([1, 1, width, 1]); // [1, 1, 16, 1]

  // We want yPred to correlate with mask.
  // Maximize (yPred * mask) => Minimize -(yPred * mask)
  return tf.mean(yPred.mul(mask)).mul(-1);
}

// ==========================================
// 3. Model Architecture
// ==========================================

// Baseline Model: Fixed Compression (Undercomplete AE)
// 16x16 -> 64 -> 16x16
function createBaselineModel() {
  const model = tf.sequential();
  model.add(tf.layers.flatten({ inputShape: CONFIG.inputShapeModel }));
  model.add(tf.layers.dense({ units: 64, activation: "relu" })); // Bottleneck
  model.add(tf.layers.dense({ units: 256, activation: "sigmoid" })); // Output 0-1
  model.add(tf.layers.reshape({ targetShape: [16, 16, 1] }));
  return model;
}

// [TODO-A IMPLEMENTED]: STUDENT ARCHITECTURE DESIGN
// Three projection types: compression / transformation / expansion
function createStudentModel(archType) {
  const model = tf.sequential();
  model.add(tf.layers.flatten({ inputShape: CONFIG.inputShapeModel }));

  if (archType === "compression") {
    // Bottleneck: Compress information (undercomplete)
    model.add(tf.layers.dense({ units: 64, activation: "relu" }));
    model.add(tf.layers.dense({ units: 256, activation: "sigmoid" }));
  } else if (archType === "transformation") {
    // 1:1 mapping - transformation without compression/expansion
    model.add(tf.layers.dense({ units: 256, activation: "relu" }));
    model.add(tf.layers.dense({ units: 256, activation: "sigmoid" }));
  } else if (archType === "expansion") {
    // Overcomplete: More parameters for flexibility
    model.add(tf.layers.dense({ units: 512, activation: "relu" }));
    model.add(tf.layers.dense({ units: 256, activation: "sigmoid" }));
  } else {
    throw new Error(`Unknown architecture type: ${archType}`);
  }

  model.add(tf.layers.reshape({ targetShape: [16, 16, 1] }));
  return model;
}

// ==========================================
// 4. Custom Loss Function
// ==========================================

// [TODO-B IMPLEMENTED]: STUDENT LOSS DESIGN
// Level 3: Sorted MSE + Smoothness + Direction = Smooth Gradient!
function studentLoss(yTrue, yPred) {
  return tf.tidy(() => {
    // Level 2: Sorted MSE - Conserve pixel values (allow rearrangement)
    const inputFlat = tf.mean(yTrue, -1).flatten();  // [256]
    const predFlat = tf.mean(yPred, -1).flatten();   // [256]
    const inputSorted = inputFlat.expandDims(0).sort({axis: 1}).squeeze();
    const predSorted = predFlat.expandDims(0).sort({axis: 1}).squeeze();
    const lossSortedMSE = tf.losses.meanSquaredError(inputSorted, predSorted);

    // Level 3: Smoothness - Be locally consistent (λ1 = 0.1)
    const lossSmooth = smoothness(yPred).mul(0.1);

    // Level 3: Direction - Bright on the right (λ2 = 0.1)
    const lossDir = directionX(yPred).mul(0.1);

    // L_total = L_sortedMSE + 0.1*L_smooth + 0.1*L_dir
    return lossSortedMSE.add(lossSmooth).add(lossDir);
  });
}

// ==========================================
// 5. Training Loop
// ==========================================

async function trainStep() {
  state.step++;

  // Safety check: Ensure models are initialized
  if (!state.studentModel || !state.studentModel.getWeights) {
    log("Error: Student model not initialized properly.", true);
    stopAutoTrain();
    return;
  }

  // Train Baseline (MSE Only)
  const baselineLossVal = tf.tidy(() => {
    const { value, grads } = tf.variableGrads(() => {
      const yPred = state.baselineModel.predict(state.xInput);
      return mse(state.xInput, yPred);
    }, state.baselineModel.getWeights());

    state.optimizer.applyGradients(grads);
    return value.dataSync()[0];
  });

  // Train Student (Custom Loss: Sorted MSE + Smooth + Direction)
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
    log(
      `Step ${state.step}: Base=${baselineLossVal.toFixed(4)} | Student=${studentLossVal.toFixed(4)} (SortedMSE+Smooth+Dir)`,
    );
  } catch (e) {
    log(`Error in Student Training: ${e.message}`, true);
    stopAutoTrain();
    return;
  }

  // Visualize every 5 steps or manual mode
  if (state.step % 5 === 0 || !state.isAutoTraining) {
    await render();
    updateLossDisplay(baselineLossVal, studentLossVal);
  }
}

// ==========================================
// 6. UI & Initialization logic
// ==========================================

function init() {
  // 1. Generate fixed noise (Batch size included: [1, 16, 16, 1])
  state.xInput = tf.randomUniform(CONFIG.inputShapeData);

  // 2. Initialize Models
  resetModels();

  // 3. Render Initial Input
  tf.browser.toPixels(
    state.xInput.squeeze(),
    document.getElementById("canvas-input"),
  );

  // 4. Bind Events
  document
    .getElementById("btn-train")
    .addEventListener("click", () => trainStep());
  document
    .getElementById("btn-auto")
    .addEventListener("click", toggleAutoTrain);
  document.getElementById("btn-reset").addEventListener("click", resetModels);

  document.querySelectorAll('input[name="arch"]').forEach((radio) => {
    radio.addEventListener("change", (e) => {
      resetModels(e.target.value);
      document.getElementById("student-arch-label").innerText =
        e.target.value.charAt(0).toUpperCase() + e.target.value.slice(1);
    });
  });

  log("✅ Initialized! TODOs completed: Custom architectures + SortedMSE+Smooth+Direction loss.");
  log("Press 'Auto Train' → watch noise → smooth gradient (right side bright)!");
}

function resetModels(archType = null) {
  if (typeof archType !== "string") {
    archType = null;
  }

  if (state.isAutoTraining) {
    stopAutoTrain();
  }

  if (!archType) {
    const checked = document.querySelector('input[name="arch"]:checked');
    archType = checked ? checked.value : "compression";
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

  // Create New Models
  state.baselineModel = createBaselineModel();
  state.studentModel = createStudentModel(archType);
  state.optimizer = tf.train.adam(CONFIG.learningRate);
  state.step = 0;

  log(`🔄 Reset. Student: ${archType} | Loss: SortedMSE(1.0)+Smooth(0.1)+Dir(0.1)`);
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

// UI Helpers
function updateLossDisplay(base, stud) {
  document.getElementById("loss-baseline").innerText =
    `Loss: ${base.toFixed(5)}`;
  document.getElementById("loss-student").innerText =
    `Loss: ${stud.toFixed(5)}`;
}

function log(msg, isError = false) {
  const el = document.getElementById("log-area");
  const span = document.createElement("div");
  span.innerText = `> ${msg}`;
  if (isError) span.classList.add("error");
  el.prepend(span);
  el.scrollTop = 0;
}

// Auto Train Logic
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

// Start
init();
