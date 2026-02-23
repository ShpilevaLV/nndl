## Prompt Example

**Role:**  
You are an expert educational software engineer and machine learning engineer. You build minimal, readable browser demos using TensorFlow.js, with clean code and strong inline TODO guidance for students.

**Context:**  
I teach a university lecture titled “Neural Network Design: Projections and Custom Loss” for Management Information Systems students. The key lesson is: neural networks are projection machines (compression / transformation / expansion), and the loss function is the “rule of the game.”  
I want a hands-on demo called **“The Gradient Puzzle”** where students start from a baseline MSE-only training setup and then **modify source code** (not a textarea editor) to implement custom losses (e.g., smoothness + direction) and compare results.

**Task:**  
Generate two files for a browser-only TensorFlow.js app: `index.html` and `app.js`.  
The default app must run out-of-the-box and show a baseline model trained with **MSE only**, plus a “student model” that is initially identical but is designed to be modified by students via TODO blocks in code.

**Instruction:**

1. **Project structure**

- Output exactly two files: `index.html` and `app.js`.
- No build tools (no Vite/Webpack). Use TF.js via CDN in `index.html`.
- Put all CSS inside `index.html` (dark theme is fine).

2. **UI requirements (`index.html`)**

- Title: “Neural Network Design: The Gradient Puzzle”
- A control panel with buttons:
  - `Train 1 Step`
  - `Auto Train (Start/Stop)`
  - `Reset Weights`
- A simple architecture selector UI (radio buttons):
  - Compression (implemented)
  - Transformation (NOT implemented in baseline; TODO for students)
  - Expansion (NOT implemented in baseline; TODO for students)  
    This selector should apply to the **student model only**.
- A 3-column canvas grid (pixelated rendering):
  1. Input (fixed noise, 16×16)
  2. Baseline Output (MSE)
  3. Student Output (initially MSE, but intended to be customized)
- A small “Status / Log” area that prints: step count, baseline loss, student loss, and any errors.

3. **Core logic (`app.js`)**

- Generate a fixed noise tensor `xInput` of shape `[1, 16, 16, 1]` at initialization.
- Create **two separate models**:
  - `baselineModel`: fixed architecture, fixed MSE loss
  - `studentModel`: starts identical but is meant to be edited (custom loss + selectable architecture)
- Implement a **custom training loop** (do NOT rely solely on `model.fit`) using an optimizer (Adam). The training step should compute predictions and update weights with backprop.
- Provide helper functions (implemented, tested, and used in comments):
  - `mse(yTrue, yPred)`
  - `smoothness(yPred)` (total variation style: squared neighbor differences)
  - `directionX(yPred)` (encourage left-dark / right-bright; define a coordinate-based penalty)
- Provide a clear TODO section for students:
  - TODO-A (Architecture): implement `createStudentModel(archType)` with three projection types: compression / transformation / expansion. In baseline code, only compression is implemented; transformation and expansion throw a clear “Not implemented” error.
  - TODO-B (Custom loss): implement `studentLoss(yTrue, yPred)` by starting from MSE and then adding smoothness + direction terms with tunable coefficients.
  - TODO-C (Comparison): print and compare baseline loss vs student loss and show visual difference on canvases.

4. **Baseline design choices**

- Baseline training objective: MSE only.
- Student model starts as MSE only, so initially baseline/student outputs look similar; the educational point is that students will change the student loss and observe emergent gradient structure.

5. **Performance & stability**

- Use `tf.tidy` to avoid memory leaks.
- Limit auto-train speed (e.g., a small number of steps per animation frame).
- Ensure the app runs smoothly on a typical laptop in Chrome.

**Format:**

- Output **two separate code blocks**:
  - First block labeled `index.html` in an `html` fenced code block
  - Second block labeled `app.js` in a `javascript` fenced code block
- Include concise comments and explicit TODO markers in `app.js`.
- Do not include extra files, screenshots, or explanations outside the two code blocks.
