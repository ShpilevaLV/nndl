/**
 * Neural Network Design: The Gradient Puzzle
 *
 * FULLY STABLE VERSION
 */

// ==========================================
// 1. Global Config
// ==========================================
const CONFIG = {
  inputShape: [1, 16, 16, 1],
  learningRate: 0.01,
  autoTrainSpeed: 50,
};

let state = {
  step: 0,
  isAutoTraining: false,
  xInput: null,
  baselineModel: null,
  studentModel: null,
  baselineOpt: null,
  studentOpt: null,
};

// ==========================================
// 2. Basic Losses
// ==========================================

function mse(a, b) {
  return tf.losses.meanSquaredError(a, b);
}

// Smoothness (Total Variation)
function smoothness(img) {
  const dx = img.slice([0,0,0,0], [-1,-1,15,-1])
    .sub(img.slice([0,0,1,0], [-1,-1,15,-1]));

  const dy = img.slice([0,0,0,0], [-1,15,-1,-1])
    .sub(img.slice([0,1,0,0], [-1,15,-1,-1]));

  return tf.mean(tf.square(dx)).add(tf.mean(tf.square(dy)));
}

// Direction constraint
function directionX(img) {
  const mask = tf.linspace(-1,1,16).reshape([1,1,16,1]);
  return tf.mean(img.mul(mask)).mul(-1);
}

// Histogram matching using sort
function sortedMSE(a, b) {
  const flatA = a.flatten();
  const flatB = b.flatten();

  const sortA = tf.sort(flatA);
  const sortB = tf.sort(flatB);

  return mse(sortA, sortB);
}

// ==========================================
// 3. Models
// ==========================================

function createBaseline() {
  const m = tf.sequential();
  m.add(tf.layers.flatten({inputShape:[16,16,1]}));
  m.add(tf.layers.dense({units:64, activation:"relu"}));
  m.add(tf.layers.dense({units:256, activation:"sigmoid"}));
  m.add(tf.layers.reshape({targetShape:[16,16,1]}));
  return m;
}

function createStudent(type) {
  const m = tf.sequential();
  m.add(tf.layers.flatten({inputShape:[16,16,1]}));

  if (type === "compression") {
    m.add(tf.layers.dense({units:64, activation:"relu"}));
  }
  else if (type === "transformation") {
    m.add(tf.layers.dense({units:256, activation:"relu"}));
  }
  else if (type === "expansion") {
    m.add(tf.layers.dense({units:512, activation:"relu"}));
  }

  m.add(tf.layers.dense({units:256, activation:"sigmoid"}));
  m.add(tf.layers.reshape({targetShape:[16,16,1]}));

  return m;
}

// ==========================================
// 4. Student Custom Loss
// ==========================================

function studentLoss(input, output) {

  const lSorted = sortedMSE(input, output);
  const lSmooth = smoothness(output).mul(0.1);
  const lDir = directionX(output).mul(0.2);

  return lSorted.add(lSmooth).add(lDir);
}

// ==========================================
// 5. Training
// ==========================================

async function trainStep() {

  state.step++;

  // Baseline update
  const baseLoss = state.baselineOpt.minimize(() => {
    const pred = state.baselineModel.predict(state.xInput);
    return mse(state.xInput, pred);
  }, true);

  const baseVal = baseLoss.dataSync()[0];
  baseLoss.dispose();

  // Student update
  const studLoss = state.studentOpt.minimize(() => {
    const pred = state.studentModel.predict(state.xInput);
    return studentLoss(state.xInput, pred);
  }, true);

  const studVal = studLoss.dataSync()[0];
  studLoss.dispose();

  console.log("Step", state.step, baseVal, studVal);

  if (state.step % 5 === 0) {
    await render();
  }

  document.getElementById("loss-baseline").innerText =
    `Loss: ${baseVal.toFixed(4)}`;

  document.getElementById("loss-student").innerText =
    `Loss: ${studVal.toFixed(4)}`;
}

// ==========================================
// 6. Rendering
// ==========================================

async function render() {
  const base = state.baselineModel.predict(state.xInput);
  const stud = state.studentModel.predict(state.xInput);

  await tf.browser.toPixels(base.squeeze(),
    document.getElementById("canvas-baseline"));

  await tf.browser.toPixels(stud.squeeze(),
    document.getElementById("canvas-student"));

  base.dispose();
  stud.dispose();
}

// ==========================================
// 7. Init
// ==========================================

function resetModels(type=null) {

  if (!type) {
    const checked = document.querySelector('input[name="arch"]:checked');
    type = checked ? checked.value : "compression";
  }

  if (state.baselineModel) state.baselineModel.dispose();
  if (state.studentModel) state.studentModel.dispose();

  state.baselineModel = createBaseline();
  state.studentModel = createStudent(type);

  state.baselineOpt = tf.train.adam(CONFIG.learningRate);
  state.studentOpt = tf.train.adam(CONFIG.learningRate);

  state.step = 0;

  render();
}

function init() {

  state.xInput = tf.randomUniform(CONFIG.inputShape);

  resetModels();

  tf.browser.toPixels(state.xInput.squeeze(),
    document.getElementById("canvas-input"));

  document.getElementById("btn-train")
    .addEventListener("click", trainStep);

  document.getElementById("btn-auto")
    .addEventListener("click", () => {
      state.isAutoTraining = !state.isAutoTraining;
      loop();
    });

  document.getElementById("btn-reset")
    .addEventListener("click", resetModels);

  document.querySelectorAll('input[name="arch"]')
    .forEach(r =>
      r.addEventListener("change", e => resetModels(e.target.value))
    );
}

function loop() {
  if (!state.isAutoTraining) return;
  trainStep();
  setTimeout(loop, CONFIG.autoTrainSpeed);
}

init();
