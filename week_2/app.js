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

// Parse CSV text to array of objects
function parseCSV(csvText) {
    const lines = csvText.split('\n').filter(line => line.trim() !== '');
    if (lines.length === 0) return [];
    
    const headers = parseCSVLine(lines[0]);
    const results = [];
    
    for (let i = 1; i < lines.length; i++) {
        const values = parseCSVLine(lines[i]);
        const obj = {};
        
        headers.forEach((header, index) => {
            let value = index < values.length ? values[index] : null;
            
            // Handle empty strings
            if (value === '' || value === null || value === undefined) {
                obj[header] = null;
            } 
            // Convert numeric values
            else if (!isNaN(value) && value.trim() !== '') {
                const num = parseFloat(value);
                obj[header] = isNaN(num) ? value : num;
            } 
            // Keep string values
            else {
                obj[header] = value;
            }
        });
        
        results.push(obj);
    }
    
    return results;
}

// Parse a single CSV line, properly handling quoted fields with commas
function parseCSVLine(line) {
    const result = [];
    let current = '';
    let inQuotes = false;
    let i = 0;
    
    while (i < line.length) {
        const char = line[i];
        
        if (char === '"') {
            // Check if this is an escaped quote (two quotes in a row)
            if (inQuotes && i + 1 < line.length && line[i + 1] === '"') {
                current += '"';
                i += 2;
                continue;
            } else {
                inQuotes = !inQuotes;
                i++;
                continue;
            }
        }
        
        if (char === ',' && !inQuotes) {
            result.push(current);
            current = '';
            i++;
            continue;
        }
        
        current += char;
        i++;
    }
    
    // Don't forget the last field
    result.push(current);
    
    return result.map(field => field.trim());
}

// Create a preview table from data
function createPreviewTable(data) {
    const table = document.createElement('table');
    table.className = 'data-table';
    
    if (data.length === 0) return table;
    
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
            if (value === null || value === undefined || value === '') {
                td.textContent = 'NULL';
                td.style.color = '#999';
                td.style.fontStyle = 'italic';
            } else {
                td.textContent = typeof value === 'number' ? 
                    (Number.isInteger(value) ? value : value.toFixed(2)) : 
                    String(value);
                
                // Truncate long strings
                if (String(value).length > 50) {
                    td.textContent = String(value).substring(0, 50) + '...';
                    td.title = value;
                }
            }
            tr.appendChild(td);
        });
        table.appendChild(tr);
    });
    
    return table;
}

// Inspect the loaded data
function inspectData() {
    if (!trainData || trainData.length === 0) {
        alert('Please load data first.');
        return;
    }
    
    // Show data preview
    const previewDiv = document.getElementById('data-preview');
    previewDiv.innerHTML = '<h3>Data Preview (First 10 Rows)</h3>';
    
    // Debug: Show what data looks like
    console.log('First row of trainData:', trainData[0]);
    console.log('All keys:', Object.keys(trainData[0]));
    
    // Create table with all available columns
    previewDiv.appendChild(createPreviewTable(trainData.slice(0, 10)));
    
    // Calculate and show data statistics
    const statsDiv = document.getElementById('data-stats');
    statsDiv.innerHTML = '<h3>Data Statistics</h3>';
    
    // Basic statistics
    const columns = Object.keys(trainData[0]);
    const shapeInfo = `Dataset shape: ${trainData.length} rows × ${columns.length} columns`;
    const survivalCount = trainData.filter(row => row[TARGET_FEATURE] === 1).length;
    const survivalRate = (survivalCount / trainData.length * 100).toFixed(2);
    const targetInfo = `Survival rate: ${survivalCount}/${trainData.length} (${survivalRate}%)`;
    
    // Calculate missing values
    let missingInfo = '<h4>Missing Values Percentage:</h4><ul>';
    columns.forEach(feature => {
        const missingCount = trainData.filter(row => 
            row[feature] === null || 
            row[feature] === undefined || 
            row[feature] === '' ||
            (typeof row[feature] === 'number' && isNaN(row[feature]))
        ).length;
        const missingPercent = (missingCount / trainData.length * 100).toFixed(2);
        missingInfo += `<li><strong>${feature}:</strong> ${missingPercent}% (${missingCount} missing)</li>`;
    });
    missingInfo += '</ul>';
    
    // Feature types
    let typeInfo = '<h4>Feature Types:</h4><ul>';
    columns.forEach(feature => {
        const firstValue = trainData[0][feature];
        const type = firstValue === null ? 'null' : typeof firstValue;
        typeInfo += `<li><strong>${feature}:</strong> ${type}</li>`;
    });
    typeInfo += '</ul>';
    
    statsDiv.innerHTML += `
        <div style="margin: 15px 0;">
            <p><strong>${shapeInfo}</strong></p>
            <p><strong>${targetInfo}</strong></p>
        </div>
        ${missingInfo}
        ${typeInfo}
    `;
    
    // Create simple visualizations
    createSimpleCharts();
    
    // Enable the preprocess button
    document.getElementById('preprocess-btn').disabled = false;
}

// Create simple charts using HTML/CSS
function createSimpleCharts() {
    const chartsDiv = document.getElementById('charts');
    chartsDiv.innerHTML = '<h3>Data Visualizations</h3>';
    
    // Survival by Sex
    const survivalBySex = {};
    trainData.forEach(row => {
        if (row.Sex !== null && row.Sex !== undefined && row[TARGET_FEATURE] !== null) {
            const sex = String(row.Sex).trim();
            if (!survivalBySex[sex]) {
                survivalBySex[sex] = { survived: 0, total: 0 };
            }
            survivalBySex[sex].total++;
            if (row[TARGET_FEATURE] === 1) {
                survivalBySex[sex].survived++;
            }
        }
    });
    
    let sexChartHTML = '<div class="chart-container"><h4>Survival Rate by Sex</h4>';
    Object.entries(survivalBySex).forEach(([sex, stats]) => {
        if (stats.total > 0) {
            const rate = (stats.survived / stats.total) * 100;
            sexChartHTML += `
                <div class="chart-bar">
                    <div class="bar-label">${sex}</div>
                    <div class="bar-bg">
                        <div class="bar-fill" style="width: ${rate}%; background-color: ${sex === 'female' ? '#ff6b6b' : '#4ecdc4'};"></div>
                    </div>
                    <div class="bar-value">${rate.toFixed(1)}% (${stats.survived}/${stats.total})</div>
                </div>
            `;
        }
    });
    sexChartHTML += '</div>';
    
    // Survival by Pclass
    const survivalByPclass = {};
    trainData.forEach(row => {
        if (row.Pclass !== null && row.Pclass !== undefined && row[TARGET_FEATURE] !== null) {
            const pclass = String(row.Pclass);
            if (!survivalByPclass[pclass]) {
                survivalByPclass[pclass] = { survived: 0, total: 0 };
            }
            survivalByPclass[pclass].total++;
            if (row[TARGET_FEATURE] === 1) {
                survivalByPclass[pclass].survived++;
            }
        }
    });
    
    let pclassChartHTML = '<div class="chart-container"><h4>Survival Rate by Passenger Class</h4>';
    Object.entries(survivalByPclass).forEach(([pclass, stats]) => {
        if (stats.total > 0) {
            const rate = (stats.survived / stats.total) * 100;
            pclassChartHTML += `
                <div class="chart-bar">
                    <div class="bar-label">Class ${pclass}</div>
                    <div class="bar-bg">
                        <div class="bar-fill" style="width: ${rate}%; background-color: ${pclass === '1' ? '#1a73e8' : pclass === '2' ? '#34a853' : '#ea4335'};"></div>
                    </div>
                    <div class="bar-value">${rate.toFixed(1)}% (${stats.survived}/${stats.total})</div>
                </div>
            `;
        }
    });
    pclassChartHTML += '</div>';
    
    chartsDiv.innerHTML += sexChartHTML + pclassChartHTML;
}

// Calculate median of an array
function calculateMedian(values) {
    const filtered = values.filter(v => v !== null && v !== undefined && !isNaN(v));
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
        const val = String(value);
        frequency[val] = (frequency[val] || 0) + 1;
        if (frequency[val] > maxCount) {
            maxCount = frequency[val];
            mode = val;
        }
    });
    
    return mode;
}

// Calculate standard deviation of an array
function calculateStdDev(values) {
    const filtered = values.filter(v => v !== null && v !== undefined && !isNaN(v));
    if (filtered.length <= 1) return 1;
    
    const mean = filtered.reduce((sum, val) => sum + val, 0) / filtered.length;
    const squaredDiffs = filtered.map(value => Math.pow(value - mean, 2));
    const variance = squaredDiffs.reduce((sum, val) => sum + val, 0) / filtered.length;
    return Math.sqrt(variance);
}

// One-hot encode a value
function oneHotEncode(value, categories) {
    const encoding = new Array(categories.length).fill(0);
    const index = categories.indexOf(String(value));
    if (index !== -1) {
        encoding[index] = 1;
    }
    return encoding;
}

// Extract features from a row with imputation and normalization
function extractFeatures(row, ageMedian, fareMedian, embarkedMode) {
    // Impute missing values
    const age = (row.Age !== null && row.Age !== undefined && !isNaN(row.Age)) ? row.Age : ageMedian;
    const fare = (row.Fare !== null && row.Fare !== undefined && !isNaN(row.Fare)) ? row.Fare : fareMedian;
    const embarked = (row.Embarked !== null && row.Embarked !== undefined && row.Embarked !== '') ? 
        String(row.Embarked).trim() : embarkedMode;
    
    // Get standard deviation for normalization (only from training data)
    const ageValues = trainData.map(r => r.Age).filter(a => a !== null && a !== undefined && !isNaN(a));
    const fareValues = trainData.map(r => r.Fare).filter(f => f !== null && f !== undefined && !isNaN(f));
    
    const ageStd = calculateStdDev(ageValues);
    const fareStd = calculateStdDev(fareValues);
    
    // Standardize numerical features (handle division by zero)
    const standardizedAge = ageStd > 0 ? (age - ageMedian) / ageStd : 0;
    const standardizedFare = fareStd > 0 ? (fare - fareMedian) / fareStd : 0;
    
    // One-hot encode categorical features
    const pclassOneHot = oneHotEncode(row.Pclass, ['1', '2', '3']);
    const sexOneHot = oneHotEncode(row.Sex, ['male', 'female']);
    const embarkedOneHot = oneHotEncode(embarked, ['C', 'Q', 'S']);
    
    // Start with numerical features
    let features = [
        standardizedAge,
        standardizedFare,
        (row.SibSp || 0),
        (row.Parch || 0)
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

// Preprocess the data
function preprocessData() {
    if (!trainData || !testData) {
        alert('Please load data first.');
        return;
    }
    
    const outputDiv = document.getElementById('preprocessing-output');
    outputDiv.innerHTML = '<div class="processing">Preprocessing data...</div>';
    
    try {
        // Calculate imputation values from training data
        const ageValues = trainData.map(row => row.Age).filter(a => a !== null && a !== undefined && !isNaN(a));
        const fareValues = trainData.map(row => row.Fare).filter(f => f !== null && f !== undefined && !isNaN(f));
        const embarkedValues = trainData.map(row => row.Embarked).filter(e => e !== null && e !== undefined && e !== '');
        
        const ageMedian = calculateMedian(ageValues);
        const fareMedian = calculateMedian(fareValues);
        const embarkedMode = calculateMode(embarkedValues);
        
        console.log('Imputation values:', { 
            ageMedian, 
            fareMedian, 
            embarkedMode,
            ageValuesLength: ageValues.length,
            fareValuesLength: fareValues.length,
            embarkedValuesLength: embarkedValues.length
        });
        
        // Preprocess training data
        preprocessedTrainData = {
            features: [],
            labels: []
        };
        
        trainData.forEach(row => {
            const features = extractFeatures(row, ageMedian, fareMedian, embarkedMode);
            preprocessedTrainData.features.push(features);
            // Ensure label is 0 or 1
            const label = row[TARGET_FEATURE] === 1 ? 1 : 0;
            preprocessedTrainData.labels.push(label);
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
                <h4>✅ Preprocessing completed successfully!</h4>
                <p><strong>Training samples:</strong> ${preprocessedTrainData.features.shape[0]}</p>
                <p><strong>Number of features:</strong> ${featureCount}</p>
                <p><strong>Training features shape:</strong> [${preprocessedTrainData.features.shape}]</p>
                <p><strong>Training labels shape:</strong> [${preprocessedTrainData.labels.shape}]</p>
                <p><strong>Test samples:</strong> ${preprocessedTestData.features.length}</p>
                <p><strong>Imputation values used:</strong></p>
                <ul>
                    <li>Age median: ${ageMedian.toFixed(2)}</li>
                    <li>Fare median: ${fareMedian.toFixed(2)}</li>
                    <li>Embarked mode: "${embarkedMode}"</li>
                </ul>
                <p><strong>Features included:</strong> Age, Fare, SibSp, Parch, Pclass (one-hot), Sex (one-hot), Embarked (one-hot)
                ${document.getElementById('add-family-features').checked ? ', FamilySize, IsAlone' : ''}</p>
            </div>
        `;
        
        // Enable the create model button
        document.getElementById('create-model-btn').disabled = false;
        
    } catch (error) {
        outputDiv.innerHTML = `<div class="error-message">Error during preprocessing: ${error.message}</div>`;
        console.error('Preprocessing error:', error, error.stack);
    }
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
    
    <div class="note">
        <p><strong>Model Architecture:</strong> Input(${inputShape}) → Dense(16, ReLU) → Dense(1, Sigmoid)</p>
        <p><strong>Parameters calculation:</strong> (${inputShape} × 16 + 16) + (16 × 1 + 1) = ${totalParams}</p>
        <p><strong>Purpose:</strong> Binary classification (survived vs not survived)</p>
        <p><strong>Sigmoid activation:</strong> Converts output to probability between 0 and 1</p>
    </div>
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
    statusDiv.innerHTML = '<div class="processing">Training model...</div>';
    
    try {
        // Split training data into train and validation sets (80/20)
        const splitIndex = Math.floor(preprocessedTrainData.features.shape[0] * 0.8);
        
        const trainFeatures = preprocessedTrainData.features.slice(0, splitIndex);
        const trainLabels = preprocessedTrainData.labels.slice(0, splitIndex);
        
        const valFeatures = preprocessedTrainData.features.slice(splitIndex);
        const valLabels = preprocessedTrainData.labels.slice(splitIndex);
        
        console.log('Data split:', {
            total: preprocessedTrainData.features.shape[0],
            train: splitIndex,
            validation: preprocessedTrainData.features.shape[0] - splitIndex
        });
        
        // Store validation data for later evaluation
        validationData = valFeatures;
        validationLabels = valLabels;
        
        // Train the model
        trainingHistory = await model.fit(trainFeatures, trainLabels, {
            epochs: 50,
            batchSize: 32,
            validationData: [valFeatures, valLabels],
            callbacks: {
                onEpochEnd: (epoch, logs) => {
                    // Update status
                    statusDiv.innerHTML = `
                        <div class="processing">
                            <p>Epoch ${epoch + 1}/50</p>
                            <p>Loss: ${logs.loss.toFixed(4)} | Accuracy: ${(logs.acc * 100).toFixed(2)}%</p>
                            <p>Val Loss: ${logs.val_loss.toFixed(4)} | Val Accuracy: ${(logs.val_acc * 100).toFixed(2)}%</p>
                        </div>
                    `;
                    
                    // Update training chart if tfvis is available
                    if (typeof tfvis !== 'undefined') {
                        try {
                            const history = [{epoch: epoch, loss: logs.loss, acc: logs.acc, val_loss: logs.val_loss, val_acc: logs.val_acc}];
                            tfvis.show.history({name: 'Training History', tab: 'Training'}, history, ['loss', 'acc', 'val_loss', 'val_acc']);
                        } catch (e) {
                            console.log('tfvis not available for charting');
                        }
                    }
                }
            },
            verbose: 0
        });
        
        statusDiv.innerHTML += '<div class="success-message"><p>✅ Training completed successfully!</p></div>';
        
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
        statusDiv.innerHTML = `<div class="error-message">Error during training: ${error.message}</div>`;
        console.error('Training error:', error, error.stack);
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
        const predictions = Array.isArray(predVals[0]) ? predVals.flat() : predVals;
        const actuals = Array.isArray(trueVals[0]) ? trueVals.flat() : trueVals;
        
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
                    <th>Predicted Survived</th>
                    <th>Predicted Not Survived</th>
                </tr>
                <tr>
                    <th>Actual Survived</th>
                    <td class="true-positive">${tp} (True Positive)</td>
                    <td class="false-negative">${fn} (False Negative)</td>
                </tr>
                <tr>
                    <th>Actual Not Survived</th>
                    <td class="false-positive">${fp} (False Positive)</td>
                    <td class="true-negative">${tn} (True Negative)</td>
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
                <span class="metric-label">Recall (Sensitivity):</span>
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
        const metricsDiv = document.getElementById('performance-metrics');
        metricsDiv.innerHTML = `<div class="error-message">Error calculating metrics: ${error.message}</div>`;
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
            <span class="metric-label">AUC (Area Under ROC):</span>
            <span class="metric-value">${auc.toFixed(4)}</span>
        </div>
    `;
    
    // Create ROC curve visualization
    createSimpleROCCurve(rocData, auc);
}

// Create a simple ROC curve without tfvis
function createSimpleROCCurve(rocData, auc) {
    const container = document.getElementById('roc-chart');
    if (!container) return;
    
    container.innerHTML = `
        <h4>ROC Curve (AUC = ${auc.toFixed(4)})</h4>
        <div style="position: relative; width: 100%; max-width: 500px; margin: 0 auto;">
            <canvas id="roc-canvas" width="500" height="400" style="border: 1px solid #ddd; background: white;"></canvas>
            <div style="position: absolute; top: 10px; left: 10px; background: rgba(255,255,255,0.9); padding: 5px 10px; border-radius: 3px; font-size: 12px;">
                AUC = ${auc.toFixed(4)}
            </div>
        </div>
        <p style="text-align: center; font-size: 0.9em; color: #666;">
            ROC curve shows trade-off between True Positive Rate and False Positive Rate at different classification thresholds.
        </p>
    `;
    
    setTimeout(() => {
        const canvas = document.getElementById('roc-canvas');
        if (!canvas) return;
        
        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;
        const padding = 50;
        
        // Clear canvas
        ctx.clearRect(0, 0, width, height);
        
        // Draw axes
        ctx.beginPath();
        ctx.moveTo(padding, height - padding);
        ctx.lineTo(width - padding, height - padding);
        ctx.moveTo(padding, height - padding);
        ctx.lineTo(padding, padding);
        ctx.strokeStyle = '#333';
        ctx.lineWidth = 1;
        ctx.stroke();
        
        // Draw axis labels
        ctx.font = '12px Arial';
        ctx.fillStyle = '#333';
        ctx.fillText('False Positive Rate', width / 2 - 40, height - 10);
        ctx.save();
        ctx.translate(15, height / 2);
        ctx.rotate(-Math.PI/2);
        ctx.fillText('True Positive Rate', 0, 0);
        ctx.restore();
        
        // Draw grid
        ctx.strokeStyle = '#eee';
        ctx.lineWidth = 0.5;
        for (let i = 0; i <= 1; i += 0.2) {
            // Horizontal grid lines
            const y = padding + (1 - i) * (height - 2 * padding);
            ctx.beginPath();
            ctx.moveTo(padding, y);
            ctx.lineTo(width - padding, y);
            ctx.stroke();
            
            // Vertical grid lines
            const x = padding + i * (width - 2 * padding);
            ctx.beginPath();
            ctx.moveTo(x, padding);
            ctx.lineTo(x, height - padding);
            ctx.stroke();
            
            // Axis ticks
            ctx.fillStyle = '#666';
            ctx.fillText(i.toFixed(1), x - 5, height - padding + 15);
            ctx.fillText(i.toFixed(1), padding - 25, y + 4);
        }
        
        // Draw diagonal reference line
        ctx.beginPath();
        ctx.moveTo(padding, height - padding);
        ctx.lineTo(width - padding, padding);
        ctx.strokeStyle = '#999';
        ctx.lineWidth = 1;
        ctx.setLineDash([5, 5]);
        ctx.stroke();
        ctx.setLineDash([]);
        
        // Draw ROC curve
        ctx.beginPath();
        rocData.forEach((point, i) => {
            const x = padding + point.fpr * (width - 2 * padding);
            const y = padding + (1 - point.tpr) * (height - 2 * padding);
            
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
        const thresholdPoint = rocData.find(d => Math.abs(d.threshold - threshold) < 0.01) || rocData[Math.round(threshold * 100)];
        if (thresholdPoint) {
            const x = padding + thresholdPoint.fpr * (width - 2 * padding);
            const y = padding + (1 - thresholdPoint.tpr) * (height - 2 * padding);
            
            ctx.beginPath();
            ctx.arc(x, y, 6, 0, Math.PI * 2);
            ctx.fillStyle = '#ff0000';
            ctx.fill();
            ctx.strokeStyle = '#fff';
            ctx.lineWidth = 2;
            ctx.stroke();
            
            // Label
            ctx.fillStyle = '#ff0000';
            ctx.fillText(`Threshold: ${threshold.toFixed(2)}`, x + 10, y - 10);
        }
        
    }, 100);
}

// Predict on test data
async function predict() {
    if (!model || !preprocessedTestData) {
        alert('Please train model first.');
        return;
    }
    
    const outputDiv = document.getElementById('prediction-output');
    outputDiv.innerHTML = '<div class="processing">Making predictions...</div>';
    
    try {
        // Convert test features to tensor
        const testFeatures = tf.tensor2d(preprocessedTestData.features);
        
        // Make predictions
        testPredictions = model.predict(testFeatures);
        const predValues = await testPredictions.array();
        
        // Create prediction results
        const results = preprocessedTestData.passengerIds.map((id, i) => {
            const probability = Array.isArray(predValues[i]) ? predValues[i][0] : predValues[i];
            return {
                PassengerId: id,
                Survived: probability >= 0.5 ? 1 : 0,
                Probability: probability
            };
        });
        
        // Show first 10 predictions
        outputDiv.innerHTML = '<h3>Prediction Results (First 10 Rows)</h3>';
        outputDiv.appendChild(createPredictionTable(results.slice(0, 10)));
        
        // Calculate summary statistics
        const survivedCount = results.filter(r => r.Survived === 1).length;
        const avgProbability = results.reduce((sum, r) => sum + r.Probability, 0) / results.length;
        
        outputDiv.innerHTML += `
            <div class="success-message" style="margin-top: 20px;">
                <p>✅ Predictions completed!</p>
                <p><strong>Total predictions:</strong> ${results.length} samples</p>
                <p><strong>Predicted to survive:</strong> ${survivedCount} (${(survivedCount/results.length*100).toFixed(1)}%)</p>
                <p><strong>Average probability:</strong> ${avgProbability.toFixed(4)}</p>
                <p><strong>Threshold:</strong> 0.5 (can be adjusted in Evaluation section)</p>
            </div>
        `;
        
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
        survivedCell.textContent = row.Survived === 1 ? 'Yes ✅' : 'No ❌';
        survivedCell.style.fontWeight = 'bold';
        survivedCell.style.color = row.Survived === 1 ? 'green' : '#d32f2f';
        tr.appendChild(survivedCell);
        
        // Probability
        const probCell = document.createElement('td');
        probCell.textContent = row.Probability.toFixed(4);
        // Color based on probability
        if (row.Probability >= 0.7) {
            probCell.style.color = 'green';
            probCell.style.fontWeight = 'bold';
        } else if (row.Probability <= 0.3) {
            probCell.style.color = '#d32f2f';
            probCell.style.fontWeight = 'bold';
        } else {
            probCell.style.color = '#666';
        }
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
    statusDiv.innerHTML = '<div class="processing">Exporting results...</div>';
    
    try {
        // Get predictions
        const predValues = await testPredictions.array();
        
        // Create submission CSV (PassengerId, Survived)
        let submissionCSV = 'PassengerId,Survived\n';
        preprocessedTestData.passengerIds.forEach((id, i) => {
            const probability = Array.isArray(predValues[i]) ? predValues[i][0] : predValues[i];
            const survived = probability >= 0.5 ? 1 : 0;
            submissionCSV += `${id},${survived}\n`;
        });
        
        // Create probabilities CSV (PassengerId, Probability)
        let probabilitiesCSV = 'PassengerId,Probability\n';
        preprocessedTestData.passengerIds.forEach((id, i) => {
            const probability = Array.isArray(predValues[i]) ? predValues[i][0] : predValues[i];
            probabilitiesCSV += `${id},${probability.toFixed(6)}\n`;
        });
        
        // Create download links
        const submissionLink = document.createElement('a');
        submissionLink.href = URL.createObjectURL(new Blob([submissionCSV], { type: 'text/csv;charset=utf-8;' }));
        submissionLink.download = 'titanic_submission.csv';
        submissionLink.innerHTML = '<button style="margin: 5px;">📥 Download submission.csv</button>';
        
        const probabilitiesLink = document.createElement('a');
        probabilitiesLink.href = URL.createObjectURL(new Blob([probabilitiesCSV], { type: 'text/csv;charset=utf-8;' }));
        probabilitiesLink.download = 'titanic_probabilities.csv';
        probabilitiesLink.innerHTML = '<button style="margin: 5px;">📊 Download probabilities.csv</button>';
        
        statusDiv.innerHTML = `
            <div class="success-message">
                <h4>✅ Export completed!</h4>
                <p>Files ready for download:</p>
                <div style="margin: 15px 0;">
                    ${submissionLink.outerHTML}
                    ${probabilitiesLink.outerHTML}
                </div>
                <p><strong>submission.csv</strong> - Kaggle submission format (PassengerId, Survived)</p>
                <p><strong>probabilities.csv</strong> - Raw prediction probabilities for analysis</p>
                <p style="margin-top: 15px; font-size: 0.9em; color: #666;">
                    Note: Files will download automatically. If not, click the buttons above.
                </p>
            </div>
        `;
        
        // Trigger downloads automatically
        setTimeout(() => {
            submissionLink.click();
            setTimeout(() => probabilitiesLink.click(), 100);
        }, 500);
        
    } catch (error) {
        statusDiv.innerHTML = `<div class="error-message">Error during export: ${error.message}</div>`;
        console.error('Export error:', error);
    }
}

// Sigmoid visualization
function visualizeSigmoid() {
    const sigmoidDiv = document.getElementById('sigmoid-vis');
    sigmoidDiv.innerHTML = `
        <div class="success-message">
            <h3>Sigmoid Activation Function</h3>
            <p>The sigmoid function converts any real number to a value between 0 and 1:</p>
            <p style="text-align: center; font-size: 1.2em;"><strong>σ(z) = 1 / (1 + e^(-z))</strong></p>
            <div style="position: relative; width: 100%; max-width: 500px; margin: 20px auto;">
                <canvas id="sigmoid-canvas" width="500" height="300" style="border: 1px solid #ddd; background: white;"></canvas>
            </div>
            <div style="margin-top: 20px;">
                <h4>Properties:</h4>
                <ul>
                    <li><strong>Output range:</strong> (0, 1) - perfect for probability</li>
                    <li><strong>S-shaped curve</strong> (sigmoid means "S-shaped")</li>
                    <li><strong>Derivative:</strong> σ'(z) = σ(z) × (1 - σ(z)) - easy to compute for backpropagation</li>
                    <li><strong>Used as</strong> the final activation in binary classification</li>
                    <li><strong>Interpretation:</strong> Output > 0.5 → Class 1, Output < 0.5 → Class 0</li>
                </ul>
                <p>In our Titanic model, the sigmoid takes the weighted sum of the hidden layer outputs and produces a survival probability.</p>
            </div>
        </div>
    `;
    
    // Draw sigmoid function
    setTimeout(() => {
        const canvas = document.getElementById('sigmoid-canvas');
        if (!canvas) return;
        
        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;
        const centerX = width / 2;
        const centerY = height / 2;
        
        // Clear canvas
        ctx.clearRect(0, 0, width, height);
        
        // Draw axes
        ctx.beginPath();
        ctx.moveTo(50, centerY);
        ctx.lineTo(width - 50, centerY);
        ctx.moveTo(centerX, 20);
        ctx.lineTo(centerX, height - 20);
        ctx.strokeStyle = '#333';
        ctx.lineWidth = 1;
        ctx.stroke();
        
        // Draw labels
        ctx.font = '14px Arial';
        ctx.fillStyle = '#333';
        ctx.fillText('z (weighted input)', width - 60, centerY + 20);
        ctx.save();
        ctx.translate(centerX - 30, 15);
        ctx.rotate(-Math.PI/2);
        ctx.fillText('σ(z) (output probability)', 0, 0);
        ctx.restore();
        
        // Draw sigmoid curve
        ctx.beginPath();
        for (let x = -6; x <= 6; x += 0.1) {
            const y = 1 / (1 + Math.exp(-x));
            const canvasX = centerX + x * 40;
            const canvasY = centerY - y * 100;
            
            if (x === -6) {
                ctx.moveTo(canvasX, canvasY);
            } else {
                ctx.lineTo(canvasX, canvasY);
            }
        }
        ctx.strokeStyle = '#1a73e8';
        ctx.lineWidth = 3;
        ctx.stroke();
        
        // Mark key points and decision boundary
        ctx.fillStyle = '#1a73e8';
        ctx.font = '12px Arial';
        
        // Decision boundary at z=0, σ(0)=0.5
        const zeroX = centerX;
        const zeroY = centerY - 0.5 * 100;
        ctx.beginPath();
        ctx.arc(zeroX, zeroY, 5, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillText('σ(0)=0.5 (Decision Boundary)', zeroX + 10, zeroY - 10);
        
        // Draw horizontal line at y=0.5
        ctx.beginPath();
        ctx.moveTo(50, zeroY);
        ctx.lineTo(width - 50, zeroY);
        ctx.strokeStyle = '#999';
        ctx.lineWidth = 1;
        ctx.setLineDash([5, 5]);
        ctx.stroke();
        ctx.setLineDash([]);
        
        // Draw regions
        ctx.fillStyle = 'rgba(255, 0, 0, 0.1)';
        ctx.fillRect(50, zeroY, zeroX - 50, 100);
        ctx.fillStyle = 'rgba(0, 255, 0, 0.1)';
        ctx.fillRect(zeroX, zeroY, width - 50 - zeroX, 100);
        
        ctx.fillStyle = '#333';
        ctx.fillText('Predict "Not Survived" (σ(z) < 0.5)', 100, zeroY + 60);
        ctx.fillText('Predict "Survived" (σ(z) > 0.5)', zeroX + 30, zeroY + 60);
        
    }, 100);
}

// Analyze feature importance
async function analyzeFeatureImportance() {
    if (!model || !validationData) {
        alert('Please train model first.');
        return;
    }
    
    const statusDiv = document.getElementById('importance-status');
    statusDiv.innerHTML = '<div class="processing">Analyzing feature importance...</div>';
    
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
            importanceCell.textContent = Math.abs(item.importance).toFixed(4);
            importanceCell.style.fontWeight = 'bold';
            importanceCell.style.color = item.importance > 0 ? 'green' : '#d32f2f';
            row.appendChild(importanceCell);
            
            // Direction
            const directionCell = document.createElement('td');
            directionCell.textContent = item.importance > 0 ? 'Positive' : 'Negative';
            directionCell.style.color = item.importance > 0 ? 'green' : '#d32f2f';
            row.appendChild(directionCell);
            
            table.appendChild(row);
        });
        
        statusDiv.appendChild(table);
        
        // Create feature importance chart
        createFeatureImportanceChart(importanceData.slice(0, 10));
        
        // Explanation
        statusDiv.innerHTML += `
            <div class="note" style="margin-top: 20px;">
                <h4>How Feature Importance is Calculated:</h4>
                <ol>
                    <li>For each feature, we randomly shuffle its values in the validation set</li>
                    <li>We measure how much the model's accuracy decreases after shuffling</li>
                    <li>Greater decrease = more important feature for prediction</li>
                </ol>
                <p><strong>Positive importance:</strong> Feature helps predict survival (e.g., being female)</p>
                <p><strong>Negative importance:</strong> Feature helps predict non-survival (e.g., being in 3rd class)</p>
                <p><em>Note: This is a simplified permutation importance. For exact values, use scikit-learn in Python.</em></p>
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
    
    // Numerical features (standardized)
    featureNames.push('Age (standardized)', 'Fare (standardized)', 'SibSp', 'Parch');
    
    // One-hot encoded Pclass
    featureNames.push('Pclass 1', 'Pclass 2', 'Pclass 3');
    
    // One-hot encoded Sex
    featureNames.push('Male', 'Female');
    
    // One-hot encoded Embarked
    featureNames.push('Embarked C', 'Embarked Q', 'Embarked S');
    
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
    
    let baselineCorrect = 0;
    for (let i = 0; i < baselineProbs.length; i++) {
        const pred = (Array.isArray(baselineProbs[i]) ? baselineProbs[i][0] : baselineProbs[i]) >= 0.5 ? 1 : 0;
        const actual = Array.isArray(baselineLabels[i]) ? baselineLabels[i][0] : baselineLabels[i];
        if (pred === actual) baselineCorrect++;
    }
    const baselineAcc = baselineCorrect / baselineProbs.length;
    
    // Compute importance for each feature
    const importanceScores = new Array(featureNames.length).fill(0);
    const validationDataArray = await validationData.array();
    
    // For each feature, shuffle and measure accuracy drop
    for (let featIdx = 0; featIdx < featureNames.length; featIdx++) {
        // Create shuffled validation data
        const shuffledData = JSON.parse(JSON.stringify(validationDataArray)); // Deep copy
        const shuffledCol = shuffledData.map(row => row[featIdx]);
        
        // Shuffle the column (Fisher-Yates shuffle)
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
        let shuffledCorrect = 0;
        for (let i = 0; i < shuffledProbs.length; i++) {
            const pred = (Array.isArray(shuffledProbs[i]) ? shuffledProbs[i][0] : shuffledProbs[i]) >= 0.5 ? 1 : 0;
            const actual = Array.isArray(baselineLabels[i]) ? baselineLabels[i][0] : baselineLabels[i];
            if (pred === actual) shuffledCorrect++;
        }
        const shuffledAcc = shuffledCorrect / shuffledProbs.length;
        
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

// Create feature importance chart
function createFeatureImportanceChart(importanceData) {
    const container = document.getElementById('importance-status');
    
    const chartHTML = `
        <div style="margin: 20px 0;">
            <h4>Top Features Visualization</h4>
            <div style="position: relative; width: 100%; max-width: 600px; margin: 0 auto;">
                <canvas id="importance-canvas" width="600" height="400" style="border: 1px solid #ddd; background: white;"></canvas>
            </div>
        </div>
    `;
    
    container.innerHTML += chartHTML;
    
    setTimeout(() => {
        const canvas = document.getElementById('importance-canvas');
        if (!canvas) return;
        
        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;
        const padding = { top: 40, right: 100, bottom: 60, left: 150 };
        
        // Clear canvas
        ctx.clearRect(0, 0, width, height);
        
        // Calculate bar dimensions
        const barHeight = 25;
        const maxImportance = Math.max(...importanceData.map(d => Math.abs(d.importance)));
        const scale = (width - padding.left - padding.right) / (maxImportance * 2);
        
        // Draw bars
        importanceData.forEach((item, index) => {
            const y = padding.top + index * (barHeight + 10);
            const barWidth = Math.abs(item.importance) * scale;
            const x = padding.left + (item.importance > 0 ? maxImportance * scale : (maxImportance - Math.abs(item.importance)) * scale);
            
            // Draw bar
            ctx.fillStyle = item.importance > 0 ? 'rgba(76, 175, 80, 0.8)' : 'rgba(244, 67, 54, 0.8)';
            if (item.importance > 0) {
                ctx.fillRect(x, y, barWidth, barHeight);
            } else {
                ctx.fillRect(x - barWidth, y, barWidth, barHeight);
            }
            
            // Draw feature name
            ctx.fillStyle = '#333';
            ctx.font = '12px Arial';
            ctx.textAlign = 'right';
            ctx.fillText(item.feature, padding.left - 10, y + barHeight / 2 + 4);
            
            // Draw value
            ctx.fillStyle = item.importance > 0 ? '#2e7d32' : '#c62828';
            ctx.font = 'bold 11px Arial';
            ctx.textAlign = item.importance > 0 ? 'left' : 'right';
            const valueX = item.importance > 0 ? x + barWidth + 5 : x - barWidth - 5;
            ctx.fillText(item.importance.toFixed(4), valueX, y + barHeight / 2 + 4);
        });
        
        // Draw center line
        ctx.beginPath();
        const centerX = padding.left + maxImportance * scale;
        ctx.moveTo(centerX, padding.top - 10);
        ctx.lineTo(centerX, padding.top + importanceData.length * (barHeight + 10));
        ctx.strokeStyle = '#999';
        ctx.lineWidth = 1;
        ctx.stroke();
        
        // Draw title and labels
        ctx.fillStyle = '#333';
        ctx.font = 'bold 14px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('Feature Importance (Permutation)', width / 2, 20);
        
        ctx.font = '12px Arial';
        ctx.fillText('Negative Impact (Predicts Not Survived)', centerX - 100, padding.top + importanceData.length * (barHeight + 10) + 30);
        ctx.fillText('Positive Impact (Predicts Survived)', centerX + 100, padding.top + importanceData.length * (barHeight + 10) + 30);
        
        // Draw legend
        ctx.fillStyle = 'rgba(244, 67, 54, 0.8)';
        ctx.fillRect(width - 120, 30, 15, 15);
        ctx.fillStyle = 'rgba(76, 175, 80, 0.8)';
        ctx.fillRect(width - 120, 50, 15, 15);
        
        ctx.fillStyle = '#333';
        ctx.font = '11px Arial';
        ctx.textAlign = 'left';
        ctx.fillText('Negative (Not Survived)', width - 100, 42);
        ctx.fillText('Positive (Survived)', width - 100, 62);
        
    }, 100);
}

// Initialize the application
window.addEventListener('DOMContentLoaded', function() {
    // Add any initialization code here
    console.log('Titanic Survival Classifier initialized');
});
