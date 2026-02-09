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
    const headers = parseCSVLine(lines[0]);
    
    return lines.slice(1).map(line => {
        const values = parseCSVLine(line);
        const obj = {};
        headers.forEach((header, i) => {
            // Handle missing values (empty strings)
            let value = i < values.length ? values[i] : null;
            
            if (value === '' || value === null) {
                obj[header] = null;
            } 
            // Convert numerical values to numbers if possible
            else if (!isNaN(value) && value.trim() !== '') {
                const num = parseFloat(value);
                obj[header] = isNaN(num) ? value : num;
            } else {
                obj[header] = value;
            }
        });
        return obj;
    });
}

// Parse a single CSV line, handling quoted fields with commas
function parseCSVLine(line) {
    const result = [];
    let current = '';
    let inQuotes = false;
    
    for (let i = 0; i < line.length; i++) {
        const char = line[i];
        
        if (char === '"') {
            if (inQuotes && i + 1 < line.length && line[i + 1] === '"') {
                // Double quote escape sequence
                current += '"';
                i++;
            } else {
                // Toggle quote state
                inQuotes = !inQuotes;
            }
        } else if (char === ',' && !inQuotes) {
            // End of field
            result.push(current.trim());
            current = '';
        } else {
            current += char;
        }
    }
    
    // Add the last field
    result.push(current.trim());
    
    return result;
}

// Create a preview table from data
function createPreviewTable(data) {
    const table = document.createElement('table');
    
    if (data.length === 0) {
        return table;
    }
    
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
            td.textContent = value !== null ? value : 'NULL';
            if (value === null) {
                td.style.color = '#999';
                td.style.fontStyle = 'italic';
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
    previewDiv.appendChild(createPreviewTable(trainData.slice(0, 10)));
    
    // Calculate and show data statistics
    const statsDiv = document.getElementById('data-stats');
    statsDiv.innerHTML = '<h3>Data Statistics</h3>';
    
    const shapeInfo = `Dataset shape: ${trainData.length} rows x ${Object.keys(trainData[0]).length} columns`;
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

// Create visualizations using tfjs-vis
function createVisualizations() {
    const chartsDiv = document.getElementById('charts');
    chartsDiv.innerHTML = '<h3>Data Visualizations</h3>';
    
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
    
    const sexData = Object.entries(survivalBySex).map(([sex, stats]) => ({
        sex,
        survivalRate: (stats.survived / stats.total) * 100
    }));
    
    // Check if tfvis is available
    if (typeof tfvis !== 'undefined') {
        tfvis.render.barchart(
            { name: 'Survival Rate by Sex', tab: 'Charts' },
            sexData.map(d => ({ x: d.sex, y: d.survivalRate })),
            { xLabel: 'Sex', yLabel: 'Survival Rate (%)' }
        );
    } else {
        // Fallback to simple HTML chart
        createSimpleBarChart('Survival Rate by Sex', sexData, chartsDiv);
    }
    
    // Survival by Pclass
    const survivalByPclass = {};
    trainData.forEach(row => {
        if (row.Pclass !== undefined && row.Survived !== undefined) {
            if (!survivalByPclass[row.Pclass]) {
                survivalByPclass[row.Pclass] = { survived: 0, total: 0 };
            }
            survivalByPclass[row.Pclass].total++;
            if (row.Survived === 1) {
                survivalByPclass[row.Pclass].survived++;
            }
        }
    });
    
    const pclassData = Object.entries(survivalByPclass).map(([pclass, stats]) => ({
        pclass: `Class ${pclass}`,
        survivalRate: (stats.survived / stats.total) * 100
    }));
    
    if (typeof tfvis !== 'undefined') {
        tfvis.render.barchart(
            { name: 'Survival Rate by Passenger Class', tab: 'Charts' },
            pclassData.map(d => ({ x: d.pclass, y: d.survivalRate })),
            { xLabel: 'Passenger Class', yLabel: 'Survival Rate (%)' }
        );
    } else {
        // Fallback to simple HTML chart
        createSimpleBarChart('Survival Rate by Passenger Class', pclassData, chartsDiv);
    }
    
    chartsDiv.innerHTML += '<p>Charts are displayed in the tfjs-vis visor. Click the button in the bottom right to view.</p>';
}

// Create simple bar chart as fallback
function createSimpleBarChart(title, data, container) {
    const chartDiv = document.createElement('div');
    chartDiv.innerHTML = `<h4>${title}</h4>`;
    
    data.forEach(item => {
        const barDiv = document.createElement('div');
        barDiv.style.margin = '10px 0';
        barDiv.style.display = 'flex';
        barDiv.style.alignItems = 'center';
        
        const label = document.createElement('div');
        label.textContent = item.pclass || item.sex;
        label.style.width = '100px';
        label.style.marginRight = '10px';
        
        const barContainer = document.createElement('div');
        barContainer.style.flex = '1';
        barContainer.style.height = '20px';
        barContainer.style.backgroundColor = '#f0f0f0';
        barContainer.style.borderRadius = '4px';
        barContainer.style.overflow = 'hidden';
        
        const bar = document.createElement('div');
        bar.style.height = '100%';
        bar.style.width = `${Math.min(item.survivalRate, 100)}%`;
        bar.style.backgroundColor = '#1a73e8';
        bar.style.transition = 'width 0.5s';
        
        const value = document.createElement('div');
        value.textContent = `${item.survivalRate.toFixed(1)}%`;
        value.style.marginLeft = '10px';
        value.style.width = '60px';
        value.style.textAlign = 'right';
        
        barContainer.appendChild(bar);
        barDiv.appendChild(label);
        barDiv.appendChild(barContainer);
        barDiv.appendChild(value);
        chartDiv.appendChild(barDiv);
    });
    
    container.appendChild(chartDiv);
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
        const ageMedian = calculateMedian(trainData.map(row => row.Age).filter(age => age !== null));
        const fareMedian = calculateMedian(trainData.map(row => row.Fare).filter(fare => fare !== null));
        const embarkedMode = calculateMode(trainData.map(row => row.Embarked).filter(e => e !== null));
        
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
        
        outputDiv.innerHTML = `
            <p>Preprocessing completed!</p>
            <p>Training features shape: ${preprocessedTrainData.features.shape}</p>
            <p>Training labels shape: ${preprocessedTrainData.labels.shape}</p>
            <p>Test features shape: [${preprocessedTestData.features.length}, ${preprocessedTestData.features[0] ? preprocessedTestData.features[0].length : 0}]</p>
            <p>Feature count: ${preprocessedTestData.features[0] ? preprocessedTestData.features[0].length : 0}</p>
        `;
        
        // Enable the create model button
        document.getElementById('create-model-btn').disabled = false;
    } catch (error) {
        outputDiv.innerHTML = `Error during preprocessing: ${error.message}`;
        console.error(error);
    }
}

// Extract features from a row with imputation and normalization
function extractFeatures(row, ageMedian, fareMedian, embarkedMode) {
    // Impute missing values
    const age = row.Age !== null ? row.Age : ageMedian;
    const fare = row.Fare !== null ? row.Fare : fareMedian;
    const embarked = row.Embarked !== null ? row.Embarked : embarkedMode;
    
    // Standardize numerical features
    const ageStd = calculateStdDev(trainData.map(r => r.Age).filter(a => a !== null)) || 1;
    const fareStd = calculateStdDev(trainData.map(r => r.Fare).filter(f => f !== null)) || 1;
    
    const standardizedAge = (age - ageMedian) / ageStd;
    const standardizedFare = (fare - fareMedian) / fareStd;
    
    // One-hot encode categorical features
    const pclassOneHot = oneHotEncode(row.Pclass, [1, 2, 3]); // Pclass values: 1, 2, 3
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

// Calculate median of an array
function calculateMedian(values) {
    if (values.length === 0) return 0;
    
    values.sort((a, b) => a - b);
    const half = Math.floor(values.length / 2);
    
    if (values.length % 2 === 0) {
        return (values[half - 1] + values[half]) / 2;
    }
    
    return values[half];
}

// Calculate mode of an array
function calculateMode(values) {
    if (values.length === 0) return 'S'; // Default for Embarked
    
    const frequency = {};
    let maxCount = 0;
    let mode = values[0];
    
    values.forEach(value => {
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
    if (values.length === 0) return 0;
    
    const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
    const squaredDiffs = values.map(value => Math.pow(value - mean, 2));
    const variance = squaredDiffs.reduce((sum, val) => sum + val, 0) / values.length;
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
        inputShape: [inputShape]
    }));
    
    model.add(tf.layers.dense({
        units: 1,
        activation: 'sigmoid'
    }));
    
    // Compile the model
    model.compile({
        optimizer: 'adam',
        loss: 'binaryCrossentropy',
        metrics: ['accuracy']
    });
    
    // Display model summary
    const summaryDiv = document.getElementById('model-summary');
    summaryDiv.innerHTML = '<h3>Model Summary</h3>';
    
    // Create a simple summary table
    const table = document.createElement('table');
    table.style.borderCollapse = 'collapse';
    table.style.width = '100%';
    
    // Table header
    const headerRow = document.createElement('tr');
    ['Layer (type)', 'Output Shape', 'Parameters'].forEach(header => {
        const th = document.createElement('th');
        th.textContent = header;
        th.style.border = '1px solid #ddd';
        th.style.padding = '8px';
        th.style.textAlign = 'left';
        th.style.backgroundColor = '#f2f2f2';
        headerRow.appendChild(th);
    });
    table.appendChild(headerRow);
    
    // Table rows for each layer
    let totalParams = 0;
    model.layers.forEach((layer, i) => {
        const row = document.createElement('tr');
        
        const layerCell = document.createElement('td');
        layerCell.textContent = `Layer ${i+1} (${layer.getClassName()})`;
        layerCell.style.border = '1px solid #ddd';
        layerCell.style.padding = '8px';
        row.appendChild(layerCell);
        
        const shapeCell = document.createElement('td');
        shapeCell.textContent = JSON.stringify(layer.outputShape).replace(/"/g, '');
        shapeCell.style.border = '1px solid #ddd';
        shapeCell.style.padding = '8px';
        row.appendChild(shapeCell);
        
        const paramsCell = document.createElement('td');
        const params = layer.countParams();
        totalParams += params;
        paramsCell.textContent = params.toLocaleString();
        paramsCell.style.border = '1px solid #ddd';
        paramsCell.style.padding = '8px';
        row.appendChild(paramsCell);
        
        table.appendChild(row);
    });
    
    // Total parameters row
    const totalRow = document.createElement('tr');
    totalRow.style.backgroundColor = '#f9f9f9';
    
    const totalLabelCell = document.createElement('td');
    totalLabelCell.colSpan = 2;
    totalLabelCell.textContent = 'Total Parameters';
    totalLabelCell.style.border = '1px solid #ddd';
    totalLabelCell.style.padding = '8px';
    totalLabelCell.style.fontWeight = 'bold';
    totalRow.appendChild(totalLabelCell);
    
    const totalValueCell = document.createElement('td');
    totalValueCell.textContent = totalParams.toLocaleString();
    totalValueCell.style.border = '1px solid #ddd';
    totalValueCell.style.padding = '8px';
    totalValueCell.style.fontWeight = 'bold';
    totalRow.appendChild(totalValueCell);
    
    table.appendChild(totalRow);
    
    summaryDiv.appendChild(table);
    
    // Add architecture description
    summaryDiv.innerHTML += `
        <p><strong>Model Architecture:</strong> Input(${inputShape}) → Dense(16, ReLU) → Dense(1, Sigmoid)</p>
        <p><strong>Total parameters:</strong> ${totalParams}</p>
        <p><strong>Activation functions:</strong> ReLU (hidden layer), Sigmoid (output layer)</p>
        <p><strong>Sigmoid output:</strong> Produces probability between 0 and 1 for binary classification</p>
    `;
    
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
        
        // Use tfvis callbacks for visualization
        const callbacks = [];
        if (typeof tfvis !== 'undefined') {
            callbacks.push(tfvis.show.fitCallbacks(
                { name: 'Training Performance' },
                ['loss', 'acc', 'val_loss', 'val_acc']
            ));
        }
        
        // Add a custom callback for status updates
        callbacks.push({
            onEpochEnd: (epoch, logs) => {
                statusDiv.innerHTML = `Epoch ${epoch + 1}/50 - loss: ${logs.loss.toFixed(4)}, acc: ${logs.acc.toFixed(4)}, val_loss: ${logs.val_loss.toFixed(4)}, val_acc: ${logs.val_acc.toFixed(4)}`;
            }
        });
        
        // Train the model
        trainingHistory = await model.fit(trainFeatures, trainLabels, {
            epochs: 50,
            batchSize: 32,
            validationData: [valFeatures, valLabels],
            callbacks: callbacks
        });
        
        statusDiv.innerHTML += '<p>Training completed!</p>';
        
        // Make predictions on validation set for evaluation
        validationPredictions = model.predict(validationData);
        
        // Enable the threshold slider and evaluation
        document.getElementById('threshold-slider').disabled = false;
        document.getElementById('threshold-slider').addEventListener('input', updateMetrics);
        
        // Enable the predict button
        document.getElementById('predict-btn').disabled = false;
        
        // Enable the sigmoid visualization button
        document.getElementById('sigmoid-btn').disabled = false;
        
        // Enable the feature importance button
        document.getElementById('importance-btn').disabled = false;
        
        // Calculate initial metrics
        updateMetrics();
    } catch (error) {
        statusDiv.innerHTML = `Error during training: ${error.message}`;
        console.error(error);
    }
}

// Update metrics based on threshold
async function updateMetrics() {
    if (!validationPredictions || !validationLabels) return;
    
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
            <table style="border-collapse: collapse; margin: 10px auto;">
                <tr>
                    <th style="border: 1px solid #ddd; padding: 8px;"></th>
                    <th style="border: 1px solid #ddd; padding: 8px; background: #e8f4f8;">Predicted Survived</th>
                    <th style="border: 1px solid #ddd; padding: 8px; background: #f8e8e8;">Predicted Not Survived</th>
                </tr>
                <tr>
                    <th style="border: 1px solid #ddd; padding: 8px; background: #e8f4f8;">Actual Survived</th>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: center; background: #d4edda;">${tp}<br><small>True Positive</small></td>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: center; background: #f8d7da;">${fn}<br><small>False Negative</small></td>
                </tr>
                <tr>
                    <th style="border: 1px solid #ddd; padding: 8px; background: #f8e8e8;">Actual Not Survived</th>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: center; background: #f8d7da;">${fp}<br><small>False Positive</small></td>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: center; background: #d4edda;">${tn}<br><small>True Negative</small></td>
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
            <div style="margin: 10px 0;">
                <div style="display: flex; justify-content: space-between; padding: 5px 0;">
                    <span><strong>Accuracy:</strong></span>
                    <span>${(accuracy * 100).toFixed(2)}%</span>
                </div>
                <div style="display: flex; justify-content: space-between; padding: 5px 0;">
                    <span><strong>Precision:</strong></span>
                    <span>${precision.toFixed(4)}</span>
                </div>
                <div style="display: flex; justify-content: space-between; padding: 5px 0;">
                    <span><strong>Recall:</strong></span>
                    <span>${recall.toFixed(4)}</span>
                </div>
                <div style="display: flex; justify-content: space-between; padding: 5px 0;">
                    <span><strong>F1 Score:</strong></span>
                    <span>${f1.toFixed(4)}</span>
                </div>
            </div>
        `;
        
        // Calculate and plot ROC curve
        await plotROC(actuals, predictions, threshold);
        
    } catch (error) {
        console.error('Error updating metrics:', error);
        const metricsDiv = document.getElementById('performance-metrics');
        metricsDiv.innerHTML += `<p style="color: red;">Error calculating metrics: ${error.message}</p>`;
    }
}

// Plot ROC curve
async function plotROC(trueLabels, predictions, currentThreshold) {
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
        
        const tpr = tp / (tp + fn) || 0;
        const fpr = fp / (fp + tn) || 0;
        
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
        <div style="display: flex; justify-content: space-between; padding: 5px 0; border-top: 1px solid #eee; margin-top: 10px;">
            <span><strong>AUC (Area Under ROC):</strong></span>
            <span>${auc.toFixed(4)}</span>
        </div>
    `;
    
    // Plot ROC curve
    if (typeof tfvis !== 'undefined') {
        tfvis.render.linechart(
            { name: 'ROC Curve', tab: 'Evaluation' },
            { values: rocData.map(d => ({ x: d.fpr, y: d.tpr })) },
            { 
                xLabel: 'False Positive Rate', 
                yLabel: 'True Positive Rate',
                series: ['ROC Curve'],
                width: 400,
                height: 400
            }
        );
    } else {
        // Create simple ROC curve visualization
        createSimpleROCCurve(rocData, auc, currentThreshold);
    }
}

// Create simple ROC curve visualization
function createSimpleROCCurve(rocData, auc, currentThreshold) {
    const rocContainer = document.getElementById('roc-chart');
    if (!rocContainer) return;
    
    rocContainer.innerHTML = `
        <h4>ROC Curve (AUC = ${auc.toFixed(4)})</h4>
        <div style="position: relative; width: 100%; max-width: 400px; margin: 0 auto;">
            <canvas id="roc-canvas" width="400" height="400" style="border: 1px solid #ddd;"></canvas>
        </div>
        <p style="text-align: center; font-size: 0.9em; color: #666;">
            ROC curve shows the trade-off between True Positive Rate and False Positive Rate.
            Current threshold: ${currentThreshold.toFixed(2)}
        </p>
    `;
    
    // Draw the ROC curve
    setTimeout(() => {
        const canvas = document.getElementById('roc-canvas');
        if (!canvas) return;
        
        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;
        const padding = 40;
        
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
        
        // Draw labels
        ctx.font = '12px Arial';
        ctx.fillStyle = '#333';
        ctx.fillText('False Positive Rate', width / 2 - 40, height - 10);
        ctx.save();
        ctx.translate(15, height / 2);
        ctx.rotate(-Math.PI/2);
        ctx.fillText('True Positive Rate', 0, 0);
        ctx.restore();
        
        // Draw diagonal reference line
        ctx.beginPath();
        ctx.moveTo(padding, height - padding);
        ctx.lineTo(width - padding, padding);
        ctx.strokeStyle = '#ccc';
        ctx.stroke();
        
        // Draw ROC curve
        ctx.beginPath();
        rocData.forEach((point, i) => {
            const x = padding + point.fpr * (width - 2 * padding);
            const y = height - padding - point.tpr * (height - 2 * padding);
            
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
        const thresholdPoint = rocData.find(d => Math.abs(d.threshold - currentThreshold) < 0.01) || rocData[50];
        if (thresholdPoint) {
            const x = padding + thresholdPoint.fpr * (width - 2 * padding);
            const y = height - padding - thresholdPoint.tpr * (height - 2 * padding);
            
            ctx.beginPath();
            ctx.arc(x, y, 5, 0, Math.PI * 2);
            ctx.fillStyle = '#ff0000';
            ctx.fill();
            
            // Add label
            ctx.fillStyle = '#ff0000';
            ctx.fillText(`Threshold: ${currentThreshold.toFixed(2)}`, x + 10, y - 10);
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
    outputDiv.innerHTML = 'Making predictions...';
    
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
        
        // Show summary statistics
        const survivedCount = results.filter(r => r.Survived === 1).length;
        const totalCount = results.length;
        const survivalRate = (survivedCount / totalCount * 100).toFixed(2);
        
        outputDiv.innerHTML += `
            <div style="margin-top: 20px; padding: 15px; background: #f8f9fa; border-radius: 5px;">
                <p><strong>Predictions completed!</strong></p>
                <p>Total predictions: ${totalCount}</p>
                <p>Predicted to survive: ${survivedCount} (${survivalRate}%)</p>
                <p>Threshold for survival: ≥ 0.5</p>
            </div>
        `;
        
        // Enable the export button
        document.getElementById('export-btn').disabled = false;
    } catch (error) {
        outputDiv.innerHTML = `Error during prediction: ${error.message}`;
        console.error(error);
    }
}

// Create prediction table
function createPredictionTable(data) {
    const table = document.createElement('table');
    table.style.borderCollapse = 'collapse';
    table.style.width = '100%';
    table.style.margin = '10px 0';
    
    // Create header row
    const headerRow = document.createElement('tr');
    ['PassengerId', 'Survived', 'Probability'].forEach(header => {
        const th = document.createElement('th');
        th.textContent = header;
        th.style.border = '1px solid #ddd';
        th.style.padding = '8px';
        th.style.textAlign = 'left';
        th.style.backgroundColor = '#f2f2f2';
        headerRow.appendChild(th);
    });
    table.appendChild(headerRow);
    
    // Create data rows
    data.forEach(row => {
        const tr = document.createElement('tr');
        
        // PassengerId
        const idCell = document.createElement('td');
        idCell.textContent = row.PassengerId;
        idCell.style.border = '1px solid #ddd';
        idCell.style.padding = '8px';
        tr.appendChild(idCell);
        
        // Survived
        const survivedCell = document.createElement('td');
        survivedCell.textContent = row.Survived === 1 ? 'Yes (1)' : 'No (0)';
        survivedCell.style.border = '1px solid #ddd';
        survivedCell.style.padding = '8px';
        survivedCell.style.fontWeight = 'bold';
        survivedCell.style.color = row.Survived === 1 ? 'green' : 'red';
        tr.appendChild(survivedCell);
        
        // Probability
        const probCell = document.createElement('td');
        probCell.textContent = row.Probability.toFixed(4);
        probCell.style.border = '1px solid #ddd';
        probCell.style.padding = '8px';
        
        // Color code based on probability
        if (row.Probability >= 0.7) {
            probCell.style.color = 'green';
            probCell.style.fontWeight = 'bold';
        } else if (row.Probability <= 0.3) {
            probCell.style.color = 'red';
            probCell.style.fontWeight = 'bold';
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
    statusDiv.innerHTML = 'Exporting results...';
    
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
        submissionLink.href = URL.createObjectURL(new Blob([submissionCSV], { type: 'text/csv' }));
        submissionLink.download = 'titanic_submission.csv';
        submissionLink.innerHTML = '<button style="margin: 5px;">Download submission.csv</button>';
        
        const probabilitiesLink = document.createElement('a');
        probabilitiesLink.href = URL.createObjectURL(new Blob([probabilitiesCSV], { type: 'text/csv' }));
        probabilitiesLink.download = 'titanic_probabilities.csv';
        probabilitiesLink.innerHTML = '<button style="margin: 5px;">Download probabilities.csv</button>';
        
        statusDiv.innerHTML = `
            <div style="padding: 15px; background: #d4edda; border-radius: 5px;">
                <p><strong>Export completed!</strong></p>
                <p>Download the files:</p>
                <div style="margin: 10px 0;">
                    ${submissionLink.outerHTML}
                    ${probabilitiesLink.outerHTML}
                </div>
                <p><strong>submission.csv</strong> - Kaggle submission format (PassengerId, Survived)</p>
                <p><strong>probabilities.csv</strong> - Raw prediction probabilities</p>
            </div>
        `;
        
        // Auto-download files
        setTimeout(() => {
            submissionLink.click();
            setTimeout(() => probabilitiesLink.click(), 100);
        }, 500);
        
    } catch (error) {
        statusDiv.innerHTML = `Error during export: ${error.message}`;
        console.error(error);
    }
}

// Visualize sigmoid function
function visualizeSigmoid() {
    const sigmoidDiv = document.getElementById('sigmoid-vis');
    sigmoidDiv.innerHTML = `
        <h3>Sigmoid Activation Function</h3>
        <p>The sigmoid function converts any real number to a value between 0 and 1:</p>
        <p style="text-align: center; font-size: 1.2em;"><strong>σ(z) = 1 / (1 + e^(-z))</strong></p>
        <div style="position: relative; width: 100%; max-width: 500px; margin: 20px auto;">
            <canvas id="sigmoid-canvas" width="500" height="300" style="border: 1px solid #ddd;"></canvas>
        </div>
        <div style="padding: 15px; background: #f8f9fa; border-radius: 5px; margin-top: 20px;">
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
    `;
    
    // Draw the sigmoid function
    setTimeout(() => {
        const canvas = document.getElementById('sigmoid-canvas');
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
        
        // Draw labels
        ctx.font = '14px Arial';
        ctx.fillStyle = '#333';
        ctx.fillText('z (input)', width - 60, height - padding + 20);
        ctx.save();
        ctx.translate(padding - 30, height / 2);
        ctx.rotate(-Math.PI/2);
        ctx.fillText('σ(z) (output)', 0, 0);
        ctx.restore();
        
        // Draw grid
        ctx.strokeStyle = '#eee';
        ctx.lineWidth = 0.5;
        for (let i = 0; i <= 1; i += 0.2) {
            const y = padding + (1 - i) * (height - 2 * padding);
            ctx.beginPath();
            ctx.moveTo(padding, y);
            ctx.lineTo(width - padding, y);
            ctx.stroke();
            
            // Y-axis labels
            ctx.fillStyle = '#666';
            ctx.fillText(i.toFixed(1), padding - 25, y + 4);
        }
        
        // Draw sigmoid curve
        ctx.beginPath();
        for (let x = -6; x <= 6; x += 0.1) {
            const y = 1 / (1 + Math.exp(-x));
            const canvasX = padding + (x + 6) * (width - 2 * padding) / 12;
            const canvasY = height - padding - y * (height - 2 * padding);
            
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
        const keyPoints = [-3, 0, 3];
        keyPoints.forEach(x => {
            const y = 1 / (1 + Math.exp(-x));
            const canvasX = padding + (x + 6) * (width - 2 * padding) / 12;
            const canvasY = height - padding - y * (height - 2 * padding);
            
            ctx.beginPath();
            ctx.arc(canvasX, canvasY, 5, 0, Math.PI * 2);
            ctx.fillStyle = '#ff0000';
            ctx.fill();
            
            ctx.fillStyle = '#333';
            ctx.fillText(`σ(${x}) = ${y.toFixed(3)}`, canvasX + 10, canvasY - 10);
        });
        
        // Draw decision boundary at y=0.5
        const decisionY = height - padding - 0.5 * (height - 2 * padding);
        ctx.beginPath();
        ctx.moveTo(padding, decisionY);
        ctx.lineTo(width - padding, decisionY);
        ctx.strokeStyle = '#999';
        ctx.lineWidth = 1;
        ctx.setLineDash([5, 5]);
        ctx.stroke();
        ctx.setLineDash([]);
        
        ctx.fillStyle = '#333';
        ctx.fillText('Decision boundary (0.5)', padding + 10, decisionY - 10);
        
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
        
        // Get the weights from the first layer
        const weights = model.layers[0].getWeights()[0];
        const weightArray = await weights.array();
        
        // Calculate importance as average absolute weight for each input feature
        const importance = [];
        for (let i = 0; i < weightArray.length; i++) {
            let sum = 0;
            for (let j = 0; j < weightArray[i].length; j++) {
                sum += Math.abs(weightArray[i][j]);
            }
            importance.push(sum / weightArray[i].length);
        }
        
        // Prepare data for visualization
        const importanceData = featureNames.map((name, index) => ({
            feature: name,
            importance: importance[index] || 0
        }));
        
        // Sort by importance (descending)
        importanceData.sort((a, b) => b.importance - a.importance);
        
        // Display results
        statusDiv.innerHTML = '<h3>Feature Importance Analysis</h3>';
        
        // Create table
        const table = document.createElement('table');
        table.style.borderCollapse = 'collapse';
        table.style.width = '100%';
        table.style.margin = '10px 0';
        
        // Header row
        const headerRow = document.createElement('tr');
        ['Rank', 'Feature', 'Importance Score'].forEach(header => {
            const th = document.createElement('th');
            th.textContent = header;
            th.style.border = '1px solid #ddd';
            th.style.padding = '8px';
            th.style.textAlign = 'left';
            th.style.backgroundColor = '#f2f2f2';
            headerRow.appendChild(th);
        });
        table.appendChild(headerRow);
        
        // Data rows (top 10)
        importanceData.slice(0, 10).forEach((item, index) => {
            const row = document.createElement('tr');
            
            // Rank
            const rankCell = document.createElement('td');
            rankCell.textContent = index + 1;
            rankCell.style.border = '1px solid #ddd';
            rankCell.style.padding = '8px';
            row.appendChild(rankCell);
            
            // Feature
            const featureCell = document.createElement('td');
            featureCell.textContent = item.feature;
            featureCell.style.border = '1px solid #ddd';
            featureCell.style.padding = '8px';
            row.appendChild(featureCell);
            
            // Importance Score
            const importanceCell = document.createElement('td');
            importanceCell.textContent = item.importance.toFixed(6);
            importanceCell.style.border = '1px solid #ddd';
            importanceCell.style.padding = '8px';
            
            // Color based on importance
            if (item.importance > 0.3) {
                importanceCell.style.color = 'green';
                importanceCell.style.fontWeight = 'bold';
            } else if (item.importance > 0.1) {
                importanceCell.style.color = '#ff9800';
            }
            
            row.appendChild(importanceCell);
            
            table.appendChild(row);
        });
        
        statusDiv.appendChild(table);
        
        // Create feature importance chart
        createFeatureImportanceChart(importanceData.slice(0, 10));
        
        // Explanation
        statusDiv.innerHTML += `
            <div style="padding: 15px; background: #f8f9fa; border-radius: 5px; margin-top: 20px;">
                <h4>How Feature Importance is Calculated:</h4>
                <p>Feature importance is calculated based on the weights of the neural network:</p>
                <ol>
                    <li>For each input feature, we look at all connections from that feature to the hidden layer</li>
                    <li>We take the absolute value of each weight (ignoring direction)</li>
                    <li>We average these absolute weights to get an importance score</li>
                    <li>Higher scores mean the feature has stronger connections in the network</li>
                </ol>
                <p><strong>Interpretation:</strong> Features with higher importance have more influence on the model's predictions.</p>
                <p><em>Note: This is a simplified importance measure. For production, use permutation importance or SHAP values.</em></p>
            </div>
        `;
        
    } catch (error) {
        statusDiv.innerHTML = `Error analyzing feature importance: ${error.message}`;
        console.error(error);
    }
}

// Get feature names
function getFeatureNames() {
    const featureNames = [];
    
    // Numerical features
    featureNames.push('Age (standardized)', 'Fare (standardized)', 'SibSp', 'Parch');
    
    // One-hot encoded Pclass
    featureNames.push('Pclass 1', 'Pclass 2', 'Pclass 3');
    
    // One-hot encoded Sex
    featureNames.push('Sex: Male', 'Sex: Female');
    
    // One-hot encoded Embarked
    featureNames.push('Embarked: C', 'Embarked: Q', 'Embarked: S');
    
    // Optional family features
    if (document.getElementById('add-family-features').checked) {
        featureNames.push('Family Size', 'Is Alone');
    }
    
    return featureNames;
}

// Create feature importance chart
function createFeatureImportanceChart(importanceData) {
    const container = document.getElementById('importance-status');
    
    const chartHTML = `
        <h4>Top Features Visualization</h4>
        <div style="position: relative; width: 100%; max-width: 600px; margin: 20px auto;">
            <canvas id="importance-canvas" width="600" height="400" style="border: 1px solid #ddd;"></canvas>
        </div>
    `;
    
    container.innerHTML += chartHTML;
    
    setTimeout(() => {
        const canvas = document.getElementById('importance-canvas');
        if (!canvas) return;
        
        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;
        const padding = { top: 40, right: 20, bottom: 60, left: 150 };
        
        // Clear canvas
        ctx.clearRect(0, 0, width, height);
        
        // Calculate dimensions
        const maxImportance = Math.max(...importanceData.map(d => d.importance));
        const barHeight = 30;
        const gap = 10;
        const chartWidth = width - padding.left - padding.right;
        
        // Draw bars
        importanceData.forEach((item, index) => {
            const y = padding.top + index * (barHeight + gap);
            const barWidth = (item.importance / maxImportance) * chartWidth * 0.8;
            
            // Draw bar
            ctx.fillStyle = index % 2 === 0 ? '#1a73e8' : '#4285f4';
            ctx.fillRect(padding.left, y, barWidth, barHeight);
            
            // Draw feature name
            ctx.fillStyle = '#333';
            ctx.font = '12px Arial';
            ctx.textAlign = 'right';
            ctx.fillText(item.feature, padding.left - 10, y + barHeight / 2 + 4);
            
            // Draw importance value
            ctx.fillStyle = '#333';
            ctx.font = 'bold 11px Arial';
            ctx.textAlign = 'left';
            ctx.fillText(item.importance.toFixed(4), padding.left + barWidth + 10, y + barHeight / 2 + 4);
        });
        
        // Draw title
        ctx.fillStyle = '#333';
        ctx.font = 'bold 16px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('Feature Importance', width / 2, 20);
        
        // Draw x-axis label
        ctx.font = '12px Arial';
        ctx.fillText('Importance Score', width / 2, height - 10);
        
    }, 100);
}

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    console.log('Titanic Survival Classifier initialized');
    
    // Add event listeners for new buttons
    const sigmoidBtn = document.getElementById('sigmoid-btn');
    if (sigmoidBtn) {
        sigmoidBtn.addEventListener('click', visualizeSigmoid);
    }
    
    const importanceBtn = document.getElementById('importance-btn');
    if (importanceBtn) {
        importanceBtn.addEventListener('click', analyzeFeatureImportance);
    }
});
