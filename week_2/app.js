// Global variables
let trainData = null;
let testData = null;
let preprocessedTrainData = null;
let preprocessedTestData = null;
let model = null;
let trainingHistory = null;
let validationData = null;
let validationLabels = null;
let validationPredictions = null;
let testPredictions = null;

// Schema configuration - change these for different datasets
const TARGET_FEATURE = 'Survived'; // Binary classification target
const ID_FEATURE = 'PassengerId'; // Identifier to exclude from features
const NUMERICAL_FEATURES = ['Age', 'Fare', 'SibSp', 'Parch']; // Numerical features
const CATEGORICAL_FEATURES = ['Pclass', 'Sex', 'Embarked']; // Categorical features

// Load data from uploaded CSV files
async function loadData() {
    const trainFile = document.getElementById('train-file').files[0];
    const testFile = document.getElementById('test-file').files[0];
    
    if (!trainFile || !testFile) {
        alert('Please upload both training and test CSV files.');
        return;
    }
    
    const statusDiv = document.getElementById('data-status');
    statusDiv.innerHTML = 'Loading data...';
    
    try {
        // Load training data
        const trainText = await readFile(trainFile);
        trainData = parseCSV(trainText);
        
        // Load test data
        const testText = await readFile(testFile);
        testData = parseCSV(testText);
        
        statusDiv.innerHTML = `Data loaded successfully! Training: ${trainData.length} samples, Test: ${testData.length} samples`;
        
        // Enable the inspect button
        document.getElementById('inspect-btn').disabled = false;
    } catch (error) {
        statusDiv.innerHTML = `Error loading data: ${error.message}`;
        console.error(error);
    }
}

// Read file as text
function readFile(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = e => resolve(e.target.result);
        reader.onerror = e => reject(new Error('Failed to read file'));
        reader.readAsText(file);
    });
}

// Parse CSV text to array of objects with proper quote handling
function parseCSV(csvText) {
    const lines = csvText.split('\n').filter(line => line.trim() !== '');
    if (lines.length === 0) return [];
    
    const results = [];
    const headers = parseCSVLine(lines[0]);
    
    for (let i = 1; i < lines.length; i++) {
        const values = parseCSVLine(lines[i]);
        if (values.length !== headers.length) {
            // If line has fewer values than headers, pad with null
            while (values.length < headers.length) {
                values.push(null);
            }
            console.warn(`Line ${i+1} has ${values.length} values, expected ${headers.length}.`);
        }
        
        const obj = {};
        headers.forEach((header, index) => {
            let value = index < values.length ? values[index] : null;
            
            // Convert to null for empty strings
            if (value === '' || value === null || value === undefined) {
                obj[header] = null;
            } 
            // Try to convert to number if it looks like a number
            else if (!isNaN(value) && value !== '') {
                const num = parseFloat(value);
                if (!isNaN(num) && isFinite(num)) {
                    obj[header] = num;
                } else {
                    obj[header] = value;
                }
            } else {
                obj[header] = value;
            }
        });
        results.push(obj);
    }
    
    return results;
}

// Parse a single CSV line, handling quoted fields with commas
function parseCSVLine(line) {
    const values = [];
    let currentValue = '';
    let insideQuotes = false;
    
    for (let i = 0; i < line.length; i++) {
        const char = line[i];
        
        if (char === '"') {
            if (insideQuotes && i + 1 < line.length && line[i + 1] === '"') {
                // Escaped quote inside quotes
                currentValue += '"';
                i++;
            } else {
                // Toggle quote state
                insideQuotes = !insideQuotes;
            }
        } else if (char === ',' && !insideQuotes) {
            // End of field
            values.push(currentValue);
            currentValue = '';
        } else {
            currentValue += char;
        }
    }
    
    // Add the last field
    values.push(currentValue);
    
    // Clean up quotes and trim
    return values.map(value => {
        if (value.startsWith('"') && value.endsWith('"')) {
            value = value.substring(1, value.length - 1);
        }
        // Replace escaped quotes
        value = value.replace(/""/g, '"');
        return value.trim();
    });
}

// Create a preview table from data
function createPreviewTable(data) {
    const table = document.createElement('table');
    table.className = 'data-table';
    
    // Create header row
    const headerRow = document.createElement('tr');
    Object.keys(data[0]).forEach(key => {
        const th = document.createElement('th');
        th.textContent = key;
        headerRow.appendChild(th);
    });
    table.appendChild(headerRow);
    
    // Create data rows
    data.forEach(row => {
        const tr = document.createElement('tr');
        Object.values(row).forEach(value => {
            const td = document.createElement('td');
            if (value === null || value === undefined) {
                td.textContent = 'NULL';
                td.style.color = '#999';
            } else {
                td.textContent = typeof value === 'number' ? 
                    (Number.isInteger(value) ? value : value.toFixed(2)) : 
                    String(value);
            }
            tr.appendChild(td);
        });
        table.appendChild(tr);
    });
    
    return table;
}

// Calculate median of an array
function calculateMedian(values) {
    const filtered = values.filter(v => v !== null && v !== undefined);
    if (filtered.length === 0) return 0;
    
    filtered.sort((a, b) => a - b);
    const half = Math.floor(filtered.length / 2);
    
    if (filtered.length % 2 === 0) {
        return (filtered[half - 1] + filtered[half]) / 2;
    }
    
    return filtered[half];
}

// Calculate mode of an array
function calculateMode(values) {
    const filtered = values.filter(v => v !== null && v !== undefined);
    if (filtered.length === 0) return 'S'; // Default for Embarked
    
    const frequency = {};
    let maxCount = 0;
    let mode = filtered[0];
    
    filtered.forEach(value => {
        frequency[value] = (frequency[value] || 0) + 1;
        if (frequency[value] > maxCount) {
            maxCount = frequency[value];
            mode = value;
        }
    });
    
    return mode;
}

// Calculate standard deviation of an array
function calculateStdDev(values) {
    const filtered = values.filter(v => v !== null && v !== undefined);
    if (filtered.length === 0) return 1;
    
    const mean = filtered.reduce((sum, val) => sum + val, 0) / filtered.length;
    const squaredDiffs = filtered.map(value => Math.pow(value - mean, 2));
    const variance = squaredDiffs.reduce((sum, val) => sum + val, 0) / filtered.length;
    return Math.sqrt(variance);
}

// One-hot encode a value
function oneHotEncode(value, categories) {
    const encoding = new Array(categories.length).fill(0);
    const index = categories.indexOf(value);
    if (index !== -1) {
        encoding[index] = 1;
    }
    return encoding;
}

// Extract features from a row with imputation and normalization
function extractFeatures(row, ageMedian, fareMedian, embarkedMode) {
    // Impute missing values
    const age = row.Age !== null && row.Age !== undefined ? row.Age : ageMedian;
    const fare = row.Fare !== null && row.Fare !== undefined ? row.Fare : fareMedian;
    const embarked = row.Embarked !== null && row.Embarked !== undefined ? row.Embarked : embarkedMode;
    
    // Get standard deviation for normalization
    const ageStd = calculateStdDev(trainData.map(r => r.Age).filter(a => a !== null && a !== undefined));
    const fareStd = calculateStdDev(trainData.map(r => r.Fare).filter(f => f !== null && f !== undefined));
    
    // Standardize numerical features (handle division by zero)
    const standardizedAge = ageStd !== 0 ? (age - ageMedian) / ageStd : 0;
    const standardizedFare = fareStd !== 0 ? (fare - fareMedian) / fareStd : 0;
    
    // One-hot encode categorical features
    const pclassOneHot = oneHotEncode(row.Pclass, [1, 2, 3]);
    const sexOneHot = oneHotEncode(row.Sex, ['male', 'female']);
    const embarkedOneHot = oneHotEncode(embarked, ['C', 'Q', 'S']);
    
    // Start with numerical features
    let features = [
        standardizedAge,
        standardizedFare,
        row.SibSp || 0,
        row.Parch || 0
    ];
    
    // Add one-hot encoded features
    features = features.concat(pclassOneHot, sexOneHot, embarkedOneHot);
    
    // Add optional family features if enabled
    if (document.getElementById('add-family-features').checked) {
        const familySize = (row.SibSp || 0) + (row.Parch || 0) + 1;
        const isAlone = familySize === 1 ? 1 : 0;
        features.push(familySize, isAlone);
    }
    
    return features;
}

// Create the model
function createModel() {
    if (!preprocessedTrainData) {
        alert('Please preprocess data first.');
        return;
    }
    
    const inputShape = preprocessedTrainData.features.shape[1];
    
    // Create a sequential model
    model = tf.sequential();
    
    // Add layers
    model.add(tf.layers.dense({
        units: 16,
        activation: 'relu',
        inputShape: [inputShape],
        kernelInitializer: 'glorotNormal',
        name: 'hidden_layer'
    }));
    
    model.add(tf.layers.dense({
        units: 1,
        activation: 'sigmoid',
        name: 'output_layer'
    }));
    
    // Compile the model
    model.compile({
        optimizer: tf.train.adam(0.001),
        loss: 'binaryCrossentropy',
        metrics: ['accuracy']
    });
    
    // Display model summary
    const summaryDiv = document.getElementById('model-summary');
    summaryDiv.innerHTML = '<h3>Model Summary</h3>';
    
    // Create a simple summary table
    let summaryHTML = `
        <table class="model-table">
            <tr>
                <th>Layer (type)</th>
                <th>Output Shape</th>
                <th>Parameters</th>
            </tr>
    `;
    
    let totalParams = 0;
    model.layers.forEach((layer, i) => {
        const outputShape = JSON.stringify(layer.outputShape).replace(/"/g, '');
        const params = layer.countParams();
        totalParams += params;
        summaryHTML += `
            <tr>
                <td>${layer.name}</td>
                <td>${outputShape}</td>
                <td>${params.toLocaleString()}</td>
            </tr>
        `;
    });
    
    summaryHTML += `
        <tr class="total-row">
            <td colspan="2"><strong>Total Parameters</strong></td>
            <td><strong>${totalParams.toLocaleString()}</strong></td>
        </tr>
    </table>
    
    <p><strong>Model Architecture:</strong> Input(${inputShape}) → Dense(16, ReLU) → Dense(1, Sigmoid)</p>
    <p><strong>Parameters:</strong> (${inputShape} × 16 + 16) + (16 × 1 + 1) = ${totalParams}</p>
    `;
    
    summaryDiv.innerHTML += summaryHTML;
    
    // Enable the train button
    document.getElementById('train-btn').disabled = false;
}

// Train the model
async function trainModel() {
    if (!model || !preprocessedTrainData) {
        alert('Please create model first.');
        return;
    }
    
    const statusDiv = document.getElementById('training-status');
    statusDiv.innerHTML = 'Training model...';
    
    try {
        // Split training data into train and validation sets (80/20)
        const splitIndex = Math.floor(preprocessedTrainData.features.shape[0] * 0.8);
        
        const trainFeatures = preprocessedTrainData.features.slice(0, splitIndex);
        const trainLabels = preprocessedTrainData.labels.slice(0, splitIndex);
        
        const valFeatures = preprocessedTrainData.features.slice(splitIndex);
        const valLabels = preprocessedTrainData.labels.slice(splitIndex);
        
        // Store validation data for later evaluation
        validationData = valFeatures;
        validationLabels = valLabels;
        
        // Prepare callbacks for training visualization
        const fitCallbacks = {
            onEpochEnd: async (epoch, logs) => {
                // Update status
                statusDiv.innerHTML = `Epoch ${epoch + 1}/50 - Loss: ${logs.loss.toFixed(4)}, Acc: ${logs.acc.toFixed(4)}, Val Loss: ${logs.val_loss.toFixed(4)}, Val Acc: ${logs.val_acc.toFixed(4)}`;
                
                // Update tfjs-vis charts
                tfvis.show.history({name: 'Training History', tab: 'Training'}, [logs], ['loss', 'acc', 'val_loss', 'val_acc'], {
                    xLabel: 'Epoch',
                    yLabel: 'Value',
                    height: 300
                });
            }
        };
        
        // Train the model
        trainingHistory = await model.fit(trainFeatures, trainLabels, {
            epochs: 50,
            batchSize: 32,
            validationData: [valFeatures, valLabels],
            callbacks: fitCallbacks,
            verbose: 0
        });
        
        statusDiv.innerHTML += '<p style="color: green; font-weight: bold;">Training completed successfully!</p>';
        
        // Make predictions on validation set for evaluation
        validationPredictions = model.predict(validationData);
        
        // Enable the threshold slider and evaluation
        document.getElementById('threshold-slider').disabled = false;
        document.getElementById('threshold-slider').addEventListener('input', updateMetrics);
        
        // Enable the predict button
        document.getElementById('predict-btn').disabled = false;
        
        // Enable the importance button
        document.getElementById('importance-btn').disabled = false;
        
        // Calculate initial metrics
        await updateMetrics();
        
    } catch (error) {
        statusDiv.innerHTML = `<p style="color: red;">Error during training: ${error.message}</p>`;
        console.error('Training error:', error);
    }
}

// Update metrics based on threshold
async function updateMetrics() {
    if (!validationPredictions || !validationLabels) {
        console.log('Validation data not available');
        return;
    }
    
    const threshold = parseFloat(document.getElementById('threshold-slider').value);
    document.getElementById('threshold-value').textContent = threshold.toFixed(2);
    
    try {
        // Get predictions and true values
        const predVals = await validationPredictions.array();
        const trueVals = await validationLabels.array();
        
        // Flatten arrays if needed
        const predictions = predVals.flat();
        const actuals = trueVals.flat();
        
        // Calculate confusion matrix
        let tp = 0, tn = 0, fp = 0, fn = 0;
        
        for (let i = 0; i < predictions.length; i++) {
            const prediction = predictions[i] >= threshold ? 1 : 0;
            const actual = actuals[i];
            
            if (prediction === 1 && actual === 1) tp++;
            else if (prediction === 0 && actual === 0) tn++;
            else if (prediction === 1 && actual === 0) fp++;
            else if (prediction === 0 && actual === 1) fn++;
        }
        
        // Update confusion matrix display
        const cmDiv = document.getElementById('confusion-matrix');
        cmDiv.innerHTML = `
            <table class="confusion-table">
                <tr>
                    <th></th>
                    <th>Predicted Positive</th>
                    <th>Predicted Negative</th>
                </tr>
                <tr>
                    <th>Actual Positive</th>
                    <td class="true-positive">${tp}</td>
                    <td class="false-negative">${fn}</td>
                </tr>
                <tr>
                    <th>Actual Negative</th>
                    <td class="false-positive">${fp}</td>
                    <td class="true-negative">${tn}</td>
                </tr>
            </table>
        `;
        
        // Calculate performance metrics
        const accuracy = (tp + tn) / (tp + tn + fp + fn) || 0;
        const precision = tp / (tp + fp) || 0;
        const recall = tp / (tp + fn) || 0;
        const f1 = 2 * (precision * recall) / (precision + recall) || 0;
        
        // Update performance metrics display
        const metricsDiv = document.getElementById('performance-metrics');
        metricsDiv.innerHTML = `
            <div class="metric-item">
                <span class="metric-label">Accuracy:</span>
                <span class="metric-value">${(accuracy * 100).toFixed(2)}%</span>
            </div>
            <div class="metric-item">
                <span class="metric-label">Precision:</span>
                <span class="metric-value">${precision.toFixed(4)}</span>
            </div>
            <div class="metric-item">
                <span class="metric-label">Recall:</span>
                <span class="metric-value">${recall.toFixed(4)}</span>
            </div>
            <div class="metric-item">
                <span class="metric-label">F1 Score:</span>
                <span class="metric-value">${f1.toFixed(4)}</span>
            </div>
        `;
        
        // Plot ROC curve
        await plotROC(actuals, predictions);
        
    } catch (error) {
        console.error('Error in updateMetrics:', error);
    }
}

// Plot ROC curve
async function plotROC(trueLabels, predictions) {
    // Calculate TPR and FPR for different thresholds
    const thresholds = Array.from({ length: 101 }, (_, i) => i / 100);
    const rocData = [];
    
    thresholds.forEach(threshold => {
        let tp = 0, fn = 0, fp = 0, tn = 0;
        
        for (let i = 0; i < predictions.length; i++) {
            const prediction = predictions[i] >= threshold ? 1 : 0;
            const actual = trueLabels[i];
            
            if (actual === 1) {
                if (prediction === 1) tp++;
                else fn++;
            } else {
                if (prediction === 1) fp++;
                else tn++;
            }
        }
        
        const tpr = tp / (tp + fn) || 0; // True Positive Rate (Recall)
        const fpr = fp / (fp + tn) || 0; // False Positive Rate
        
        rocData.push({ threshold, fpr, tpr });
    });
    
    // Calculate AUC (approximate using trapezoidal rule)
    let auc = 0;
    for (let i = 1; i < rocData.length; i++) {
        auc += (rocData[i].fpr - rocData[i-1].fpr) * (rocData[i].tpr + rocData[i-1].tpr) / 2;
    }
    
    // Update metrics with AUC
    const metricsDiv = document.getElementById('performance-metrics');
    metricsDiv.innerHTML += `
        <div class="metric-item">
            <span class="metric-label">AUC:</span>
            <span class="metric-value">${auc.toFixed(4)}</span>
        </div>
    `;
    
    // Plot ROC curve using tfvis
    const rocValues = rocData.map(d => ({ x: d.fpr, y: d.tpr }));
    
    // Clear previous chart
    const rocContainer = document.getElementById('roc-chart');
    if (rocContainer) {
        rocContainer.innerHTML = '';
    }
    
    // Plot using tfvis
    const surface = { name: 'ROC Curve', tab: 'Evaluation' };
    const data = { values: rocValues };
    const options = {
        xLabel: 'False Positive Rate',
        yLabel: 'True Positive Rate',
        height: 300,
        width: 400
    };
    
    // We'll use a simple canvas-based approach for better compatibility
    createSimpleROCCurve(rocData, auc);
}

// Create a simple ROC curve without tfvis
function createSimpleROCCurve(rocData, auc) {
    const container = document.getElementById('roc-chart');
    if (!container) return;
    
    container.innerHTML = `
        <div style="position: relative; width: 400px; height: 300px;">
            <canvas id="roc-canvas" width="400" height="300" style="border: 1px solid #ddd;"></canvas>
            <div style="position: absolute; top: 10px; left: 10px; background: rgba(255,255,255,0.8); padding: 5px;">
                AUC = ${auc.toFixed(4)}
            </div>
        </div>
    `;
    
    const canvas = document.getElementById('roc-canvas');
    const ctx = canvas.getContext('2d');
    
    // Clear canvas
    ctx.clearRect(0, 0, 400, 300);
    
    // Draw axes
    ctx.beginPath();
    ctx.moveTo(50, 250);
    ctx.lineTo(350, 250);
    ctx.moveTo(50, 250);
    ctx.lineTo(50, 50);
    ctx.strokeStyle = '#000';
    ctx.stroke();
    
    // Draw labels
    ctx.fillText('False Positive Rate', 180, 280);
    ctx.save();
    ctx.translate(20, 150);
    ctx.rotate(-Math.PI/2);
    ctx.fillText('True Positive Rate', 0, 0);
    ctx.restore();
    
    // Draw diagonal reference line
    ctx.beginPath();
    ctx.moveTo(50, 250);
    ctx.lineTo(350, 50);
    ctx.strokeStyle = '#ccc';
    ctx.stroke();
    
    // Draw ROC curve
    ctx.beginPath();
    rocData.forEach((point, i) => {
        const x = 50 + point.fpr * 300;
        const y = 250 - point.tpr * 200;
        
        if (i === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
    });
    ctx.strokeStyle = '#1a73e8';
    ctx.lineWidth = 2;
    ctx.stroke();
    
    // Draw current threshold point
    const threshold = parseFloat(document.getElementById('threshold-slider').value);
    const thresholdPoint = rocData.find(d => Math.abs(d.threshold - threshold) < 0.01) || rocData[50];
    if (thresholdPoint) {
        const x = 50 + thresholdPoint.fpr * 300;
        const y = 250 - thresholdPoint.tpr * 200;
        
        ctx.beginPath();
        ctx.arc(x, y, 5, 0, Math.PI * 2);
        ctx.fillStyle = '#ff0000';
        ctx.fill();
    }
}

// Main functions (same as before, just adding the missing ones)

// Inspect the loaded data
function inspectData() {
    if (!trainData || trainData.length === 0) {
        alert('Please load data first.');
        return;
    }
    
    // Show data preview
    const previewDiv = document.getElementById('data-preview');
    previewDiv.innerHTML = '<h3>Data Preview (First 10 Rows)</h3>';
    previewDiv.appendChild(createPreviewTable(trainData.slice(0, 10)));
    
    // Calculate and show data statistics
    const statsDiv = document.getElementById('data-stats');
    statsDiv.innerHTML = '<h3>Data Statistics</h3>';
    
    const shapeInfo = `Dataset shape: ${trainData.length} rows × ${Object.keys(trainData[0]).length} columns`;
    const survivalCount = trainData.filter(row => row[TARGET_FEATURE] === 1).length;
    const survivalRate = (survivalCount / trainData.length * 100).toFixed(2);
    const targetInfo = `Survival rate: ${survivalCount}/${trainData.length} (${survivalRate}%)`;
    
    // Calculate missing values percentage for each feature
    let missingInfo = '<h4>Missing Values Percentage:</h4><ul>';
    Object.keys(trainData[0]).forEach(feature => {
        const missingCount = trainData.filter(row => row[feature] === null || row[feature] === undefined).length;
        const missingPercent = (missingCount / trainData.length * 100).toFixed(2);
        missingInfo += `<li>${feature}: ${missingPercent}%</li>`;
    });
    missingInfo += '</ul>';
    
    statsDiv.innerHTML += `<p>${shapeInfo}</p><p>${targetInfo}</p>${missingInfo}`;
    
    // Create visualizations
    createVisualizations();
    
    // Enable the preprocess button
    document.getElementById('preprocess-btn').disabled = false;
}

// Create visualizations
function createVisualizations() {
    const chartsDiv = document.getElementById('charts');
    chartsDiv.innerHTML = '<h3>Data Visualizations</h3>';
    
    // Create simple charts using div elements instead of tfvis
    createSimpleCharts();
}

function createSimpleCharts() {
    const chartsDiv = document.getElementById('charts');
    
    // Survival by Sex
    const survivalBySex = {};
    trainData.forEach(row => {
        if (row.Sex && row.Survived !== undefined) {
            if (!survivalBySex[row.Sex]) {
                survivalBySex[row.Sex] = { survived: 0, total: 0 };
            }
            survivalBySex[row.Sex].total++;
            if (row.Survived === 1) {
                survivalBySex[row.Sex].survived++;
            }
        }
    });
    
    let sexChartHTML = '<h4>Survival Rate by Sex</h4><div class="chart-container">';
    Object.entries(survivalBySex).forEach(([sex, stats]) => {
        const rate = (stats.survived / stats.total) * 100;
        sexChartHTML += `
            <div class="chart-bar">
                <div class="bar-label">${sex}</div>
                <div class="bar-bg">
                    <div class="bar-fill" style="width: ${rate}%"></div>
                </div>
                <div class="bar-value">${rate.toFixed(1)}%</div>
            </div>
        `;
    });
    sexChartHTML += '</div>';
    
    chartsDiv.innerHTML += sexChartHTML;
}

// Preprocess the data
function preprocessData() {
    if (!trainData || !testData) {
        alert('Please load data first.');
        return;
    }
    
    const outputDiv = document.getElementById('preprocessing-output');
    outputDiv.innerHTML = 'Preprocessing data...';
    
    try {
        // Calculate imputation values from training data
        const ageMedian = calculateMedian(trainData.map(row => row.Age));
        const fareMedian = calculateMedian(trainData.map(row => row.Fare));
        const embarkedMode = calculateMode(trainData.map(row => row.Embarked));
        
        console.log('Imputation values:', { ageMedian, fareMedian, embarkedMode });
        
        // Preprocess training data
        preprocessedTrainData = {
            features: [],
            labels: []
        };
        
        trainData.forEach(row => {
            const features = extractFeatures(row, ageMedian, fareMedian, embarkedMode);
            preprocessedTrainData.features.push(features);
            preprocessedTrainData.labels.push(row[TARGET_FEATURE]);
        });
        
        // Preprocess test data
        preprocessedTestData = {
            features: [],
            passengerIds: []
        };
        
        testData.forEach(row => {
            const features = extractFeatures(row, ageMedian, fareMedian, embarkedMode);
            preprocessedTestData.features.push(features);
            preprocessedTestData.passengerIds.push(row[ID_FEATURE]);
        });
        
        // Convert to tensors
        preprocessedTrainData.features = tf.tensor2d(preprocessedTrainData.features);
        preprocessedTrainData.labels = tf.tensor1d(preprocessedTrainData.labels);
        
        const featureCount = preprocessedTestData.features[0] ? preprocessedTestData.features[0].length : 0;
        
        outputDiv.innerHTML = `
            <div class="success-message">
                <p>✅ Preprocessing completed successfully!</p>
                <p><strong>Training features shape:</strong> [${preprocessedTrainData.features.shape}]</p>
                <p><strong>Training labels shape:</strong> [${preprocessedTrainData.labels.shape}]</p>
                <p><strong>Test samples:</strong> ${preprocessedTestData.features.length}</p>
                <p><strong>Number of features:</strong> ${featureCount}</p>
                <p><strong>Imputation values used:</strong></p>
                <ul>
                    <li>Age median: ${ageMedian.toFixed(2)}</li>
                    <li>Fare median: ${fareMedian.toFixed(2)}</li>
                    <li>Embarked mode: ${embarkedMode}</li>
                </ul>
            </div>
        `;
        
        // Enable the create model button
        document.getElementById('create-model-btn').disabled = false;
    } catch (error) {
        outputDiv.innerHTML = `<div class="error-message">Error during preprocessing: ${error.message}</div>`;
        console.error('Preprocessing error:', error);
    }
}

// Predict on test data
async function predict() {
    if (!model || !preprocessedTestData) {
        alert('Please train model first.');
        return;
    }
    
    const outputDiv = document.getElementById('prediction-output');
    outputDiv.innerHTML = 'Making predictions...';
    
    try {
        // Convert test features to tensor
        const testFeatures = tf.tensor2d(preprocessedTestData.features);
        
        // Make predictions
        testPredictions = model.predict(testFeatures);
        const predValues = await testPredictions.array();
        
        // Create prediction results
        const results = preprocessedTestData.passengerIds.map((id, i) => ({
            PassengerId: id,
            Survived: predValues[i] >= 0.5 ? 1 : 0,
            Probability: predValues[i][0] || predValues[i]
        }));
        
        // Show first 10 predictions
        outputDiv.innerHTML = '<h3>Prediction Results (First 10 Rows)</h3>';
        outputDiv.appendChild(createPredictionTable(results.slice(0, 10)));
        
        outputDiv.innerHTML += `<p>Predictions completed! Total: ${results.length} samples</p>`;
        
        // Enable the export button
        document.getElementById('export-btn').disabled = false;
    } catch (error) {
        outputDiv.innerHTML = `<div class="error-message">Error during prediction: ${error.message}</div>`;
        console.error('Prediction error:', error);
    }
}

// Create prediction table
function createPredictionTable(data) {
    const table = document.createElement('table');
    table.className = 'prediction-table';
    
    // Create header row
    const headerRow = document.createElement('tr');
    ['PassengerId', 'Survived', 'Probability'].forEach(header => {
        const th = document.createElement('th');
        th.textContent = header;
        headerRow.appendChild(th);
    });
    table.appendChild(headerRow);
    
    // Create data rows
    data.forEach(row => {
        const tr = document.createElement('tr');
        
        // PassengerId
        const idCell = document.createElement('td');
        idCell.textContent = row.PassengerId;
        tr.appendChild(idCell);
        
        // Survived
        const survivedCell = document.createElement('td');
        survivedCell.textContent = row.Survived;
        survivedCell.style.fontWeight = 'bold';
        survivedCell.style.color = row.Survived === 1 ? 'green' : 'red';
        tr.appendChild(survivedCell);
        
        // Probability
        const probCell = document.createElement('td');
        probCell.textContent = row.Probability.toFixed(4);
        // Color based on probability
        if (row.Probability >= 0.7) probCell.style.color = 'green';
        else if (row.Probability <= 0.3) probCell.style.color = 'red';
        tr.appendChild(probCell);
        
        table.appendChild(tr);
    });
    
    return table;
}

// Export results
async function exportResults() {
    if (!testPredictions || !preprocessedTestData) {
        alert('Please make predictions first.');
        return;
    }
    
    const statusDiv = document.getElementById('export-status');
    statusDiv.innerHTML = 'Exporting results...';
    
    try {
        // Get predictions
        const predValues = await testPredictions.array();
        
        // Create submission CSV (PassengerId, Survived)
        let submissionCSV = 'PassengerId,Survived\n';
        preprocessedTestData.passengerIds.forEach((id, i) => {
            const survived = predValues[i] >= 0.5 ? 1 : 0;
            submissionCSV += `${id},${survived}\n`;
        });
        
        // Create probabilities CSV (PassengerId, Probability)
        let probabilitiesCSV = 'PassengerId,Probability\n';
        preprocessedTestData.passengerIds.forEach((id, i) => {
            probabilitiesCSV += `${id},${predValues[i].toFixed(6)}\n`;
        });
        
        // Create download links
        const submissionLink = document.createElement('a');
        submissionLink.href = URL.createObjectURL(new Blob([submissionCSV], { type: 'text/csv' }));
        submissionLink.download = 'titanic_submission.csv';
        submissionLink.textContent = 'Download submission.csv';
        
        const probabilitiesLink = document.createElement('a');
        probabilitiesLink.href = URL.createObjectURL(new Blob([probabilitiesCSV], { type: 'text/csv' }));
        probabilitiesLink.download = 'titanic_probabilities.csv';
        probabilitiesLink.textContent = 'Download probabilities.csv';
        
        statusDiv.innerHTML = `
            <div class="success-message">
                <p>✅ Export completed!</p>
                <p>Click to download:</p>
                <div style="display: flex; gap: 10px; margin: 10px 0;">
                    ${submissionLink.outerHTML}
                    ${probabilitiesLink.download = 'titanic_probabilities.csv'; probabilitiesLink.outerHTML}
                </div>
            </div>
        `;
        
        // Trigger clicks programmatically
        setTimeout(() => {
            submissionLink.click();
            probabilitiesLink.click();
        }, 100);
        
    } catch (error) {
        statusDiv.innerHTML = `<div class="error-message">Error during export: ${error.message}</div>`;
        console.error('Export error:', error);
    }
}

// Sigmoid visualization
function visualizeSigmoid() {
    const sigmoidDiv = document.getElementById('sigmoid-vis');
    sigmoidDiv.innerHTML = `
        <h3>Sigmoid Activation Function</h3>
        <p>The sigmoid function converts any real number to a value between 0 and 1:</p>
        <p><strong>σ(z) = 1 / (1 + e^(-z))</strong></p>
        <div style="position: relative; width: 400px; height: 300px;">
            <canvas id="sigmoid-canvas" width="400" height="300" style="border: 1px solid #ddd;"></canvas>
        </div>
        <p>Properties:</p>
        <ul>
            <li>Output range: (0, 1) - perfect for probability</li>
            <li>S-shaped curve (sigmoid means "S-shaped")</li>
            <li>Used as the final activation in binary classification</li>
            <li>Derivative: σ'(z) = σ(z) × (1 - σ(z)) - easy to compute</li>
        </ul>
    `;
    
    // Draw sigmoid function
    setTimeout(() => {
        const canvas = document.getElementById('sigmoid-canvas');
        const ctx = canvas.getContext('2d');
        
        // Clear canvas
        ctx.clearRect(0, 0, 400, 300);
        
        // Draw axes
        ctx.beginPath();
        ctx.moveTo(50, 150);
        ctx.lineTo(350, 150);
        ctx.moveTo(200, 50);
        ctx.lineTo(200, 250);
        ctx.strokeStyle = '#000';
        ctx.stroke();
        
        // Draw labels
        ctx.fillText('z (input)', 180, 280);
        ctx.save();
        ctx.translate(10, 150);
        ctx.rotate(-Math.PI/2);
        ctx.fillText('σ(z) (output)', 0, 0);
        ctx.restore();
        
        // Draw sigmoid curve
        ctx.beginPath();
        for (let x = -6; x <= 6; x += 0.1) {
            const y = 1 / (1 + Math.exp(-x));
            const canvasX = 200 + x * 25;
            const canvasY = 150 - y * 100;
            
            if (x === -6) {
                ctx.moveTo(canvasX, canvasY);
            } else {
                ctx.lineTo(canvasX, canvasY);
            }
        }
        ctx.strokeStyle = '#1a73e8';
        ctx.lineWidth = 2;
        ctx.stroke();
        
        // Mark key points
        [-2, 0, 2].forEach(x => {
            const y = 1 / (1 + Math.exp(-x));
            const canvasX = 200 + x * 25;
            const canvasY = 150 - y * 100;
            
            ctx.beginPath();
            ctx.arc(canvasX, canvasY, 3, 0, Math.PI * 2);
            ctx.fillStyle = '#ff0000';
            ctx.fill();
            
            ctx.fillText(`σ(${x}) = ${y.toFixed(3)}`, canvasX + 5, canvasY - 10);
        });
    }, 100);
}

// Analyze feature importance
async function analyzeFeatureImportance() {
    if (!model || !validationData) {
        alert('Please train model first.');
        return;
    }
    
    const statusDiv = document.getElementById('importance-status');
    statusDiv.innerHTML = 'Analyzing feature importance...';
    
    try {
        // Get feature names
        const featureNames = getFeatureNames();
        
        // Compute simple feature importance using permutation
        const importanceScores = await computePermutationImportance(featureNames);
        
        // Prepare data for visualization
        const importanceData = featureNames.map((name, index) => ({
            feature: name,
            importance: importanceScores[index]
        }));
        
        // Sort by absolute importance (descending)
        importanceData.sort((a, b) => Math.abs(b.importance) - Math.abs(a.importance));
        
        // Display results
        statusDiv.innerHTML = '<h3>Feature Importance Analysis</h3>';
        
        // Create table
        const table = document.createElement('table');
        table.className = 'importance-table';
        
        // Header
        const headerRow = document.createElement('tr');
        ['Rank', 'Feature', 'Importance Score', 'Direction'].forEach(header => {
            const th = document.createElement('th');
            th.textContent = header;
            headerRow.appendChild(th);
        });
        table.appendChild(headerRow);
        
        // Data rows (top 10)
        importanceData.slice(0, 10).forEach((item, index) => {
            const row = document.createElement('tr');
            
            // Rank
            const rankCell = document.createElement('td');
            rankCell.textContent = index + 1;
            row.appendChild(rankCell);
            
            // Feature
            const featureCell = document.createElement('td');
            featureCell.textContent = item.feature;
            row.appendChild(featureCell);
            
            // Importance Score
            const importanceCell = document.createElement('td');
            importanceCell.textContent = item.importance.toFixed(4);
            importanceCell.style.color = item.importance > 0 ? 'green' : 'red';
            row.appendChild(importanceCell);
            
            // Direction
            const directionCell = document.createElement('td');
            directionCell.textContent = item.importance > 0 ? 'Positive' : 'Negative';
            directionCell.style.color = item.importance > 0 ? 'green' : 'red';
            row.appendChild(directionCell);
            
            table.appendChild(row);
        });
        
        statusDiv.appendChild(table);
        
        // Explanation
        statusDiv.innerHTML += `
            <div style="margin-top: 20px; padding: 15px; background: #f8f9fa; border-radius: 5px;">
                <h4>How Feature Importance is Calculated:</h4>
                <p>1. For each feature, we randomly shuffle its values in the validation set</p>
                <p>2. We measure how much the model's accuracy decreases</p>
                <p>3. Greater decrease = more important feature</p>
                <p><strong>Positive importance</strong>: Feature helps predict survival</p>
                <p><strong>Negative importance</strong>: Feature helps predict non-survival</p>
            </div>
        `;
        
    } catch (error) {
        statusDiv.innerHTML = `<div class="error-message">Error analyzing feature importance: ${error.message}</div>`;
        console.error('Feature importance error:', error);
    }
}

// Get feature names
function getFeatureNames() {
    const featureNames = [];
    
    // Numerical features
    featureNames.push('Age (std)', 'Fare (std)', 'SibSp', 'Parch');
    
    // One-hot encoded features
    featureNames.push('Pclass_1', 'Pclass_2', 'Pclass_3');
    featureNames.push('Sex_male', 'Sex_female');
    featureNames.push('Embarked_C', 'Embarked_Q', 'Embarked_S');
    
    // Optional family features
    if (document.getElementById('add-family-features').checked) {
        featureNames.push('FamilySize', 'IsAlone');
    }
    
    return featureNames;
}

// Compute permutation importance
async function computePermutationImportance(featureNames) {
    // Get baseline accuracy
    const baselinePred = model.predict(validationData);
    const baselineProbs = await baselinePred.array();
    const baselineLabels = await validationLabels.array();
    
    let baselineAcc = 0;
    for (let i = 0; i < baselineProbs.length; i++) {
        const pred = baselineProbs[i] >= 0.5 ? 1 : 0;
        if (pred === baselineLabels[i]) baselineAcc++;
    }
    baselineAcc /= baselineProbs.length;
    
    // Compute importance for each feature
    const importanceScores = new Array(featureNames.length).fill(0);
    
    // For each feature, shuffle and measure accuracy drop
    for (let featIdx = 0; featIdx < featureNames.length; featIdx++) {
        // Create shuffled validation data
        const shuffledData = validationData.arraySync();
        const shuffledCol = shuffledData.map(row => row[featIdx]);
        
        // Shuffle the column
        for (let i = shuffledCol.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [shuffledCol[i], shuffledCol[j]] = [shuffledCol[j], shuffledCol[i]];
        }
        
        // Apply shuffle
        shuffledData.forEach((row, i) => {
            row[featIdx] = shuffledCol[i];
        });
        
        // Predict with shuffled data
        const shuffledTensor = tf.tensor2d(shuffledData);
        const shuffledPred = model.predict(shuffledTensor);
        const shuffledProbs = await shuffledPred.array();
        
        // Calculate accuracy
        let shuffledAcc = 0;
        for (let i = 0; i < shuffledProbs.length; i++) {
            const pred = shuffledProbs[i] >= 0.5 ? 1 : 0;
            if (pred === baselineLabels[i]) shuffledAcc++;
        }
        shuffledAcc /= shuffledProbs.length;
        
        // Importance = decrease in accuracy (can be negative if shuffling improves accuracy)
        importanceScores[featIdx] = baselineAcc - shuffledAcc;
        
        // Clean up
        shuffledTensor.dispose();
        shuffledPred.dispose();
    }
    
    // Clean up
    baselinePred.dispose();
    
    return importanceScores;
}
