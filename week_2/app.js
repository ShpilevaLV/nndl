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
    statusDiv.innerHTML = '<div class="processing">Loading data...</div>';
    
    try {
        // Load training data
        const trainText = await readFile(trainFile);
        trainData = parseCSV(trainText);
        
        // Load test data
        const testText = await readFile(testFile);
        testData = parseCSV(testText);
        
        statusDiv.innerHTML = `<div class="success-message">Data loaded successfully!<br>Training: ${trainData.length} samples, Test: ${testData.length} samples</div>`;
        
        // Enable the inspect button
        document.getElementById('inspect-btn').disabled = false;
        updateStatusIndicator('inspect-btn', 'ready');
        
        // Initialize tfjs-vis
        initializeTFVis();
    } catch (error) {
        statusDiv.innerHTML = `<div class="error-message">Error loading data: ${error.message}</div>`;
        console.error(error);
        updateStatusIndicator('inspect-btn', 'not-ready');
    }
}

// Initialize tfjs-vis
function initializeTFVis() {
    if (typeof tfvis === 'undefined') {
        console.warn('tfjs-vis not available. Charts will use fallback.');
        return;
    }
    
    // Create a button to open the visor
    const chartsDiv = document.getElementById('charts');
    if (chartsDiv && !document.getElementById('open-charts-btn')) {
        const openButton = document.createElement('button');
        openButton.id = 'open-charts-btn';
        openButton.innerHTML = '<i class="fas fa-chart-bar"></i> Open Charts Panel';
        openButton.onclick = openChartsPanel;
        openButton.style.marginTop = '10px';
        chartsDiv.appendChild(openButton);
    }
}

// Open charts panel
function openChartsPanel() {
    if (typeof tfvis === 'undefined') {
        alert('tfjs-vis is not available. Please check your internet connection.');
        return;
    }
    
    const visor = tfvis.visor();
    if (!visor.isOpen()) {
        visor.open();
    }
    
    // Switch to Charts tab
    setTimeout(() => {
        const tabs = document.querySelectorAll('.tfjs-visor__tab');
        tabs.forEach(tab => {
            if (tab.textContent && tab.textContent.includes('Charts')) {
                tab.click();
            }
        });
    }, 100);
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
    
    // Parse headers first
    const headers = parseCSVLine(lines[0]);
    const results = [];
    
    // Parse each data line
    for (let i = 1; i < lines.length; i++) {
        const values = parseCSVLine(lines[i]);
        const obj = {};
        
        // Match headers with values
        for (let j = 0; j < headers.length; j++) {
            const header = headers[j];
            const value = j < values.length ? values[j] : null;
            
            // Handle empty strings
            if (value === '' || value === null) {
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
        }
        
        results.push(obj);
    }
    
    return results;
}

// Parse a single CSV line, handling quoted fields with commas
function parseCSVLine(line) {
    const result = [];
    let current = '';
    let inQuotes = false;
    
    for (let i = 0; i < line.length; i++) {
        const char = line[i];
        
        if (char === '"') {
            // Handle escaped quotes (two quotes in a row)
            if (inQuotes && i + 1 < line.length && line[i + 1] === '"') {
                current += '"';
                i++; // Skip next quote
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
                
                // Truncate very long strings
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
        const missingCount = trainData.filter(row => 
            row[feature] === null || 
            row[feature] === undefined || 
            row[feature] === ''
        ).length;
        const missingPercent = (missingCount / trainData.length * 100).toFixed(2);
        missingInfo += `<li><strong>${feature}:</strong> ${missingPercent}% (${missingCount} missing)</li>`;
    });
    missingInfo += '</ul>';
    
    statsDiv.innerHTML += `<p>${shapeInfo}</p><p>${targetInfo}</p>${missingInfo}`;
    
    // Create visualizations
    createVisualizations();
    
    // Enable the preprocess button
    document.getElementById('preprocess-btn').disabled = false;
    updateStatusIndicator('preprocess-btn', 'ready');
}

// Update status indicator
function updateStatusIndicator(buttonId, status) {
    const button = document.getElementById(buttonId);
    if (!button) return;
    
    const indicator = button.querySelector('.status-indicator');
    if (!indicator) return;
    
    indicator.classList.remove('status-ready', 'status-not-ready', 'status-processing');
    indicator.classList.add(`status-${status}`);
}

// Create visualizations using tfjs-vis
function createVisualizations() {
    const chartsDiv = document.getElementById('charts');
    chartsDiv.innerHTML = '<h3>Data Visualizations</h3>';
    
    try {
        // Survival by Sex
        const survivalBySex = {};
        trainData.forEach(row => {
            if (row.Sex && row.Survived !== undefined && row.Sex !== null) {
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
        
        // Survival by Pclass
        const survivalByPclass = {};
        trainData.forEach(row => {
            if (row.Pclass !== undefined && row.Pclass !== null && row.Survived !== undefined) {
                const pclass = `Class ${row.Pclass}`;
                if (!survivalByPclass[pclass]) {
                    survivalByPclass[pclass] = { survived: 0, total: 0 };
                }
                survivalByPclass[pclass].total++;
                if (row.Survived === 1) {
                    survivalByPclass[pclass].survived++;
                }
            }
        });
        
        const pclassData = Object.entries(survivalByPclass).map(([pclass, stats]) => ({
            pclass,
            survivalRate: (stats.survived / stats.total) * 100
        }));
        
        // Try to use tfjs-vis if available
        if (typeof tfvis !== 'undefined') {
            // Open visor if not open
            const visor = tfvis.visor();
            if (!visor.isOpen()) {
                visor.open();
            }
            
            // Create surfaces with proper styling
            const sexSurface = { 
                name: 'Survival Rate by Sex', 
                tab: 'Charts',
                styles: {
                    width: '100%',
                    height: '400px',
                    margin: '0 auto'
                }
            };
            
            const pclassSurface = { 
                name: 'Survival Rate by Passenger Class', 
                tab: 'Charts',
                styles: {
                    width: '100%',
                    height: '400px',
                    margin: '0 auto'
                }
            };
            
            // Render charts with explicit settings
            setTimeout(() => {
                try {
                    // Chart 1: Survival by Sex
                    tfvis.render.barchart(
                        sexSurface,
                        sexData.map(d => ({ x: d.sex, y: d.survivalRate })),
                        {
                            xLabel: 'Sex',
                            yLabel: 'Survival Rate (%)',
                            width: 500,
                            height: 400,
                            fontSize: 12,
                            xAxisDomain: Object.keys(survivalBySex),
                            yAxisDomain: [0, 100],
                            tickFontSize: 11,
                            labelFontSize: 13,
                            marginTop: 50,
                            marginBottom: 70,
                            marginLeft: 80,
                            marginRight: 40,
                            labelPadding: 10,
                            tickPadding: 8
                        }
                    );
                    
                    // Chart 2: Survival by Pclass
                    tfvis.render.barchart(
                        pclassSurface,
                        pclassData.map(d => ({ x: d.pclass, y: d.survivalRate })),
                        {
                            xLabel: 'Passenger Class',
                            yLabel: 'Survival Rate (%)',
                            width: 500,
                            height: 400,
                            fontSize: 12,
                            xAxisDomain: Object.keys(survivalByPclass),
                            yAxisDomain: [0, 100],
                            tickFontSize: 11,
                            labelFontSize: 13,
                            marginTop: 50,
                            marginBottom: 70,
                            marginLeft: 80,
                            marginRight: 40,
                            labelPadding: 10,
                            tickPadding: 8
                        }
                    );
                    
                    chartsDiv.innerHTML += `
                        <div class="success-message" style="margin-top: 20px;">
                            <p>✅ Charts created successfully!</p>
                            <p>Check the tfjs-vis panel in the bottom-right corner. Click the "Charts" tab to view.</p>
                        </div>
                    `;
                } catch (chartError) {
                    console.error('Error rendering charts:', chartError);
                    createEnhancedFallbackChart('Survival Rate by Sex', sexData, chartsDiv);
                    createEnhancedFallbackChart('Survival Rate by Passenger Class', pclassData, chartsDiv);
                }
            }, 500);
        } else {
            throw new Error('tfjs-vis not available');
        }
        
    } catch (error) {
        console.error('Error creating visualizations:', error);
        
        // Fallback to HTML visualization
        chartsDiv.innerHTML += '<div class="error-message">Using enhanced HTML visualizations instead of tfjs-vis.</div>';
        
        // Calculate data for fallback charts
        const survivalBySex = {};
        trainData.forEach(row => {
            if (row.Sex && row.Survived !== undefined && row.Sex !== null) {
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
            survivalRate: (stats.survived / stats.total) * 100,
            survived: stats.survived,
            total: stats.total
        }));
        
        createEnhancedFallbackChart('Survival Rate by Sex', sexData, chartsDiv);
        
        // Survival by Pclass
        const survivalByPclass = {};
        trainData.forEach(row => {
            if (row.Pclass !== undefined && row.Pclass !== null && row.Survived !== undefined) {
                const pclass = `Class ${row.Pclass}`;
                if (!survivalByPclass[pclass]) {
                    survivalByPclass[pclass] = { survived: 0, total: 0 };
                }
                survivalByPclass[pclass].total++;
                if (row.Survived === 1) {
                    survivalByPclass[pclass].survived++;
                }
            }
        });
        
        const pclassData = Object.entries(survivalByPclass).map(([pclass, stats]) => ({
            pclass,
            survivalRate: (stats.survived / stats.total) * 100,
            survived: stats.survived,
            total: stats.total
        }));
        
        createEnhancedFallbackChart('Survival Rate by Passenger Class', pclassData, chartsDiv);
    }
}

// Enhanced fallback chart with better styling
function createEnhancedFallbackChart(title, data, container) {
    const chartDiv = document.createElement('div');
    chartDiv.style.margin = '30px 0';
    chartDiv.style.padding = '20px';
    chartDiv.style.backgroundColor = '#f8f9fa';
    chartDiv.style.borderRadius = '10px';
    chartDiv.style.border = '1px solid #e0e0e0';
    
    chartDiv.innerHTML = `
        <h4 style="color: #1a73e8; margin-top: 0; padding-bottom: 10px; border-bottom: 2px solid #1a73e8;">${title}</h4>
        <div style="display: flex; flex-direction: column; gap: 15px;">
            ${data.map(item => `
                <div style="background: white; padding: 15px; border-radius: 8px; border: 1px solid #eaeaea; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                        <span style="font-weight: 600; font-size: 1.1em; color: #333; min-width: 150px;">${item.sex || item.pclass}</span>
                        <span style="font-weight: 700; font-size: 1.2em; color: #1a73e8;">${item.survivalRate.toFixed(1)}%</span>
                    </div>
                    <div style="background: #e9ecef; height: 20px; border-radius: 10px; overflow: hidden; margin-bottom: 8px;">
                        <div style="background: linear-gradient(to right, #1a73e8, #4285f4); height: 100%; width: ${Math.min(item.survivalRate, 100)}%; border-radius: 10px; transition: width 1s ease;"></div>
                    </div>
                    <div style="display: flex; justify-content: space-between; font-size: 0.9em; color: #666;">
                        <span>${item.survived} survived</span>
                        <span>${item.total} total</span>
                    </div>
                </div>
            `).join('')}
        </div>
    `;
    
    container.appendChild(chartDiv);
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
            </div>
        `;
        
        // Enable the create model button
        document.getElementById('create-model-btn').disabled = false;
        updateStatusIndicator('create-model-btn', 'ready');
    } catch (error) {
        outputDiv.innerHTML = `<div class="error-message">Error during preprocessing: ${error.message}</div>`;
        console.error('Preprocessing error:', error);
    }
}

// Extract features from a row with imputation and normalization
function extractFeatures(row, ageMedian, fareMedian, embarkedMode) {
    // Impute missing values
    const age = (row.Age !== null && row.Age !== undefined && !isNaN(row.Age)) ? row.Age : ageMedian;
    const fare = (row.Fare !== null && row.Fare !== undefined && !isNaN(row.Fare)) ? row.Fare : fareMedian;
    const embarked = (row.Embarked !== null && row.Embarked !== undefined && row.Embarked !== '') ? 
        String(row.Embarked).trim() : embarkedMode;
    
    // Calculate standard deviation for normalization
    const ageStd = calculateStdDev(trainData.map(r => r.Age).filter(a => a !== null && a !== undefined && !isNaN(a))) || 1;
    const fareStd = calculateStdDev(trainData.map(r => r.Fare).filter(f => f !== null && f !== undefined && !isNaN(f))) || 1;
    
    // Standardize numerical features
    const standardizedAge = (age - ageMedian) / ageStd;
    const standardizedFare = (fare - fareMedian) / fareStd;
    
    // One-hot encode categorical features
    const pclassOneHot = oneHotEncode(row.Pclass, ['1', '2', '3']);
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
    if (values.length <= 1) return 1;
    
    const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
    const squaredDiffs = values.map(value => Math.pow(value - mean, 2));
    const variance = squaredDiffs.reduce((sum, val) => sum + val, 0) / values.length;
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
    
    // Create a summary table
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
    updateStatusIndicator('train-btn', 'ready');
}

// Train the model
async function trainModel() {
    if (!model || !preprocessedTrainData) {
        alert('Please create model first.');
        return;
    }
    
    const statusDiv = document.getElementById('training-status');
    statusDiv.innerHTML = '<div class="processing">Training model...</div>';
    updateStatusIndicator('train-btn', 'processing');
    
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
        
        // Prepare callbacks
        const callbacks = [];
        
        if (typeof tfvis !== 'undefined') {
            callbacks.push(tfvis.show.fitCallbacks(
                { 
                    name: 'Training Performance',
                    styles: { width: '95%', height: '400px' }
                },
                ['loss', 'acc', 'val_loss', 'val_acc'],
                {
                    callbacks: ['onEpochEnd', 'onBatchEnd'],
                    height: 400,
                    width: 600,
                    fontSize: 12,
                    tickFontSize: 11
                }
            ));
        }
        
        // Add custom callback for status updates
        callbacks.push({
            onEpochEnd: (epoch, logs) => {
                statusDiv.innerHTML = `
                    <div class="processing">
                        <p><strong>Epoch ${epoch + 1}/50</strong></p>
                        <p>Loss: ${logs.loss.toFixed(4)} | Accuracy: ${(logs.acc * 100).toFixed(2)}%</p>
                        <p>Val Loss: ${logs.val_loss.toFixed(4)} | Val Accuracy: ${(logs.val_acc * 100).toFixed(2)}%</p>
                    </div>
                `;
            }
        });
        
        // Train the model
        trainingHistory = await model.fit(trainFeatures, trainLabels, {
            epochs: 50,
            batchSize: 32,
            validationData: [valFeatures, valLabels],
            callbacks: callbacks,
            verbose: 0
        });
        
        statusDiv.innerHTML += '<div class="success-message"><p>✅ Training completed successfully!</p></div>';
        updateStatusIndicator('train-btn', 'ready');
        
        // Make predictions on validation set for evaluation
        validationPredictions = model.predict(validationData);
        
        // Enable the threshold slider and evaluation
        document.getElementById('threshold-slider').disabled = false;
        document.getElementById('threshold-slider').addEventListener('input', updateMetrics);
        
        // Enable the predict button
        document.getElementById('predict-btn').disabled = false;
        updateStatusIndicator('predict-btn', 'ready');
        
        // Enable the sigmoid button
        document.getElementById('sigmoid-btn').disabled = false;
        updateStatusIndicator('sigmoid-btn', 'ready');
        
        // Enable the importance button
        document.getElementById('importance-btn').disabled = false;
        updateStatusIndicator('importance-btn', 'ready');
        
        // Calculate initial metrics
        await updateMetrics();
    } catch (error) {
        statusDiv.innerHTML = `<div class="error-message">Error during training: ${error.message}</div>`;
        console.error('Training error:', error);
        updateStatusIndicator('train-btn', 'not-ready');
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
                    <td class="true-positive">${tp}<br><small>True Positive</small></td>
                    <td class="false-negative">${fn}<br><small>False Negative</small></td>
                </tr>
                <tr>
                    <th>Actual Not Survived</th>
                    <td class="false-positive">${fp}<br><small>False Positive</small></td>
                    <td class="true-negative">${tn}<br><small>True Negative</small></td>
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
        await plotROC(actuals, predictions, threshold);
    } catch (error) {
        console.error('Error in updateMetrics:', error);
        const metricsDiv = document.getElementById('performance-metrics');
        metricsDiv.innerHTML += `<div class="error-message">Error calculating metrics: ${error.message}</div>`;
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
    
    // Create ROC curve visualization
    createROCCurve(rocData, auc, currentThreshold);
}

// Create ROC curve visualization
function createROCCurve(rocData, auc, currentThreshold) {
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
            ROC curve shows trade-off between True Positive Rate and False Positive Rate at different thresholds.
        </p>
    `;
    
    // Draw the ROC curve
    setTimeout(() => {
        const canvas = document.getElementById('roc-canvas');
        if (!canvas) return;
        
        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;
        const padding = 60;
        
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
        ctx.font = '14px Arial';
        ctx.fillStyle = '#333';
        ctx.fillText('False Positive Rate', width / 2 - 40, height - 10);
        ctx.save();
        ctx.translate(20, height / 2);
        ctx.rotate(-Math.PI/2);
        ctx.fillText('True Positive Rate', 0, 0);
        ctx.restore();
        
        // Draw diagonal reference line
        ctx.beginPath();
        ctx.moveTo(padding, height - padding);
        ctx.lineTo(width - padding, padding);
        ctx.strokeStyle = '#ccc';
        ctx.lineWidth = 1;
        ctx.setLineDash([5, 5]);
        ctx.stroke();
        ctx.setLineDash([]);
        
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
            ctx.arc(x, y, 6, 0, Math.PI * 2);
            ctx.fillStyle = '#ff0000';
            ctx.fill();
            ctx.strokeStyle = '#fff';
            ctx.lineWidth = 2;
            ctx.stroke();
            
            // Label
            ctx.fillStyle = '#ff0000';
            ctx.font = 'bold 12px Arial';
            ctx.fillText(`Threshold: ${currentThreshold.toFixed(2)}`, x + 10, y - 10);
        }
        
        // Draw grid lines
        ctx.strokeStyle = '#eee';
        ctx.lineWidth = 0.5;
        
        // Vertical grid lines
        for (let i = 0; i <= 1; i += 0.2) {
            const x = padding + i * (width - 2 * padding);
            ctx.beginPath();
            ctx.moveTo(x, padding);
            ctx.lineTo(x, height - padding);
            ctx.stroke();
            
            // Tick labels
            ctx.fillStyle = '#666';
            ctx.font = '10px Arial';
            ctx.fillText(i.toFixed(1), x - 5, height - padding + 15);
        }
        
        // Horizontal grid lines
        for (let i = 0; i <= 1; i += 0.2) {
            const y = height - padding - i * (height - 2 * padding);
            ctx.beginPath();
            ctx.moveTo(padding, y);
            ctx.lineTo(width - padding, y);
            ctx.stroke();
            
            // Tick labels
            ctx.fillStyle = '#666';
            ctx.font = '10px Arial';
            ctx.fillText(i.toFixed(1), padding - 20, y + 3);
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
    updateStatusIndicator('predict-btn', 'processing');
    
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
        updateStatusIndicator('export-btn', 'ready');
        updateStatusIndicator('predict-btn', 'ready');
    } catch (error) {
        outputDiv.innerHTML = `<div class="error-message">Error during prediction: ${error.message}</div>`;
        console.error('Prediction error:', error);
        updateStatusIndicator('predict-btn', 'not-ready');
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
    updateStatusIndicator('export-btn', 'processing');
    
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
            </div>
        `;
        
        updateStatusIndicator('export-btn', 'ready');
        
        // Trigger downloads automatically
        setTimeout(() => {
            submissionLink.click();
            setTimeout(() => probabilitiesLink.click(), 100);
        }, 500);
    } catch (error) {
        statusDiv.innerHTML = `<div class="error-message">Error during export: ${error.message}</div>`;
        console.error('Export error:', error);
        updateStatusIndicator('export-btn', 'not-ready');
    }
}

// Visualize sigmoid function
function visualizeSigmoid() {
    const sigmoidDiv = document.getElementById('sigmoid-vis');
    sigmoidDiv.innerHTML = `
        <div class="success-message">
            <h3>Sigmoid Activation Function</h3>
            <p>The sigmoid function converts any real number to a value between 0 and 1:</p>
            <p style="text-align: center; font-size: 1.2em;"><strong>σ(z) = 1 / (1 + e^(-z))</strong></p>
            <div style="position: relative; width: 100%; max-width: 600px; margin: 20px auto;">
                <canvas id="sigmoid-canvas" width="600" height="450" style="border: 1px solid #ddd; background: white;"></canvas>
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
        
        // Draw grid
        ctx.strokeStyle = '#f0f0f0';
        ctx.lineWidth = 0.5;
        
        // Vertical grid lines
        for (let x = -6; x <= 6; x += 1) {
            const canvasX = centerX + x * 40;
            ctx.beginPath();
            ctx.moveTo(canvasX, 50);
            ctx.lineTo(canvasX, height - 70);
            ctx.stroke();
        }
        
        // Horizontal grid lines
        for (let y = 0; y <= 1; y += 0.1) {
            const canvasY = centerY - y * 200;
            ctx.beginPath();
            ctx.moveTo(70, canvasY);
            ctx.lineTo(width - 70, canvasY);
            ctx.stroke();
        }
        
        // Draw axes
        ctx.beginPath();
        ctx.moveTo(70, centerY);
        ctx.lineTo(width - 70, centerY);
        ctx.moveTo(centerX, 50);
        ctx.lineTo(centerX, height - 70);
        ctx.strokeStyle = '#333';
        ctx.lineWidth = 2;
        ctx.stroke();
        
        // Draw axis labels
        ctx.font = '14px Arial';
        ctx.fillStyle = '#333';
        ctx.fillText('z (weighted input)', width - 80, centerY + 25);
        ctx.save();
        ctx.translate(centerX - 40, 40);
        ctx.rotate(-Math.PI/2);
        ctx.fillText('σ(z) (output probability)', 0, 0);
        ctx.restore();
        
        // Draw sigmoid curve
        ctx.beginPath();
        for (let x = -6; x <= 6; x += 0.1) {
            const y = 1 / (1 + Math.exp(-x));
            const canvasX = centerX + x * 40;
            const canvasY = centerY - y * 200;
            
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
        const zeroY = centerY - 0.5 * 200;
        ctx.beginPath();
        ctx.arc(zeroX, zeroY, 6, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = '#1a73e8';
        ctx.font = 'bold 12px Arial';
        ctx.fillText('σ(0)=0.5', zeroX - 30, zeroY - 15);
        ctx.fillText('(Decision Boundary)', zeroX - 50, zeroY + 15);
        
        // Draw horizontal line at y=0.5
        ctx.beginPath();
        ctx.moveTo(70, zeroY);
        ctx.lineTo(width - 70, zeroY);
        ctx.strokeStyle = '#999';
        ctx.lineWidth = 1;
        ctx.setLineDash([5, 5]);
        ctx.stroke();
        ctx.setLineDash([]);
        
        // Draw regions
        ctx.fillStyle = 'rgba(255, 0, 0, 0.1)';
        ctx.fillRect(70, zeroY, zeroX - 70, 200);
        ctx.fillStyle = 'rgba(0, 255, 0, 0.1)';
        ctx.fillRect(zeroX, zeroY, width - 70 - zeroX, 200);
        
        ctx.fillStyle = '#333';
        ctx.font = 'bold 14px Arial';
        ctx.fillText('Predict "Not Survived"', 120, zeroY + 80);
        ctx.fillText('(σ(z) < 0.5)', 140, zeroY + 100);
        ctx.fillText('Predict "Survived"', zeroX + 40, zeroY + 80);
        ctx.fillText('(σ(z) > 0.5)', zeroX + 60, zeroY + 100);
        
        // Draw tick marks and labels
        ctx.fillStyle = '#666';
        ctx.font = '11px Arial';
        
        // X-axis ticks
        for (let x = -5; x <= 5; x += 1) {
            const canvasX = centerX + x * 40;
            ctx.beginPath();
            ctx.moveTo(canvasX, centerY - 5);
            ctx.lineTo(canvasX, centerY + 5);
            ctx.strokeStyle = '#333';
            ctx.lineWidth = 1;
            ctx.stroke();
            
            ctx.fillText(x.toString(), canvasX - 3, centerY + 20);
        }
        
        // Y-axis ticks
        for (let y = 0; y <= 1; y += 0.2) {
            const canvasY = centerY - y * 200;
            ctx.beginPath();
            ctx.moveTo(centerX - 5, canvasY);
            ctx.lineTo(centerX + 5, canvasY);
            ctx.strokeStyle = '#333';
            ctx.lineWidth = 1;
            ctx.stroke();
            
            ctx.fillText(y.toFixed(1), centerX - 25, canvasY + 3);
        }
        
        // Draw formula
        ctx.fillStyle = '#1a73e8';
        ctx.font = 'bold 16px Arial';
        ctx.fillText('σ(z) = 1 / (1 + e^(-z))', centerX - 100, 35);
    }, 100);
}

// Analyze feature importance using permutation method
async function analyzeFeatureImportance() {
    if (!model || !validationData || !validationLabels) {
        alert('Please train model first.');
        return;
    }
    
    const statusDiv = document.getElementById('importance-status');
    statusDiv.innerHTML = '<div class="processing">Analyzing feature importance...<br>This may take a minute.</div>';
    updateStatusIndicator('importance-btn', 'processing');
    
    try {
        // Get feature names
        const featureNames = getFeatureNames();
        
        // Get baseline accuracy
        const baselinePred = await model.predict(validationData).array();
        const baselineLabels = await validationLabels.array();
        
        let baselineCorrect = 0;
        for (let i = 0; i < baselinePred.length; i++) {
            const pred = Array.isArray(baselinePred[i]) ? baselinePred[i][0] : baselinePred[i];
            const actual = Array.isArray(baselineLabels[i]) ? baselineLabels[i][0] : baselineLabels[i];
            if ((pred >= 0.5 ? 1 : 0) === actual) baselineCorrect++;
        }
        const baselineAccuracy = baselineCorrect / baselinePred.length;
        
        // Get validation data as array
        const validationDataArray = await validationData.array();
        const numFeatures = validationDataArray[0].length;
        const importanceScores = new Array(numFeatures).fill(0);
        
        // For each feature, shuffle and measure accuracy drop
        const numRuns = 3; // Run multiple times for more stable results
        const numSamples = Math.min(validationDataArray.length, 100); // Use subset for speed
        
        for (let run = 0; run < numRuns; run++) {
            for (let featureIdx = 0; featureIdx < numFeatures; featureIdx++) {
                // Create shuffled copy of validation data
                const shuffledData = JSON.parse(JSON.stringify(validationDataArray));
                const originalValues = shuffledData.map(row => row[featureIdx]);
                
                // Shuffle the column
                const shuffledValues = [...originalValues];
                for (let i = shuffledValues.length - 1; i > 0; i--) {
                    const j = Math.floor(Math.random() * (i + 1));
                    [shuffledValues[i], shuffledValues[j]] = [shuffledValues[j], shuffledValues[i]];
                }
                
                // Apply shuffle
                shuffledData.forEach((row, i) => {
                    row[featureIdx] = shuffledValues[i];
                });
                
                // Make predictions with shuffled data
                const shuffledTensor = tf.tensor2d(shuffledData.slice(0, numSamples));
                const shuffledPred = await model.predict(shuffledTensor).array();
                shuffledTensor.dispose();
                
                // Calculate accuracy with shuffled data
                let shuffledCorrect = 0;
                for (let i = 0; i < shuffledPred.length; i++) {
                    const pred = Array.isArray(shuffledPred[i]) ? shuffledPred[i][0] : shuffledPred[i];
                    const actual = Array.isArray(baselineLabels[i]) ? baselineLabels[i][0] : baselineLabels[i];
                    if ((pred >= 0.5 ? 1 : 0) === actual) shuffledCorrect++;
                }
                const shuffledAccuracy = shuffledCorrect / shuffledPred.length;
                
                // Importance = decrease in accuracy (can be negative if shuffling improves accuracy)
                importanceScores[featureIdx] += (baselineAccuracy - shuffledAccuracy) / numRuns;
            }
            
            // Update progress
            statusDiv.innerHTML = `<div class="processing">Analyzing feature importance...<br>Run ${run + 1}/${numRuns} completed.</div>`;
        }
        
        // Prepare data for visualization
        const importanceData = featureNames.slice(0, numFeatures).map((name, index) => ({
            feature: name,
            importance: importanceScores[index]
        }));
        
        // Sort by absolute importance (descending)
        importanceData.sort((a, b) => Math.abs(b.importance) - Math.abs(a.importance));
        
        // Display results
        statusDiv.innerHTML = '<h3>Feature Importance Analysis</h3>';
        statusDiv.innerHTML += `<p><strong>Baseline accuracy:</strong> ${(baselineAccuracy * 100).toFixed(2)}%</p>`;
        statusDiv.innerHTML += `<p><strong>Method:</strong> Permutation importance (shuffle each feature and measure accuracy drop)</p>`;
        
        // Create table
        const table = document.createElement('table');
        table.className = 'importance-table';
        
        // Header row
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
            importanceCell.style.color = item.importance > 0 ? 'green' : '#d32f2f';
            row.appendChild(importanceCell);
            
            // Direction
            const directionCell = document.createElement('td');
            if (item.importance > 0.01) {
                directionCell.textContent = 'Predicts Survival ↑';
                directionCell.style.color = 'green';
            } else if (item.importance < -0.01) {
                directionCell.textContent = 'Predicts Not Survived ↓';
                directionCell.style.color = '#d32f2f';
            } else {
                directionCell.textContent = 'Neutral';
                directionCell.style.color = '#666';
            }
            row.appendChild(directionCell);
            
            table.appendChild(row);
        });
        
        statusDiv.appendChild(table);
        
        // Create feature importance chart
        createPermutationImportanceChart(importanceData.slice(0, 10), baselineAccuracy);
        
        // Explanation
        statusDiv.innerHTML += `
            <div class="note" style="margin-top: 20px;">
                <h4>How Permutation Importance Works:</h4>
                <ol>
                    <li>Calculate baseline accuracy on validation set: <strong>${(baselineAccuracy * 100).toFixed(2)}%</strong></li>
                    <li>For each feature, randomly shuffle its values in the validation set</li>
                    <li>Measure the model's accuracy with the shuffled data</li>
                    <li>Importance = Baseline accuracy - Shuffled accuracy</li>
                    <li>Repeat 3 times and average the results</li>
                </ol>
                <p><strong>Interpretation:</strong></p>
                <ul>
                    <li><span style="color: green">Positive importance</span>: Feature helps predict survival (accuracy drops when shuffled)</li>
                    <li><span style="color: #d32f2f">Negative importance</span>: Feature helps predict non-survival</li>
                    <li><span style="color: #666">Near zero</span>: Feature has little impact on predictions</li>
                </ul>
            </div>
        `;
        
        updateStatusIndicator('importance-btn', 'ready');
    } catch (error) {
        statusDiv.innerHTML = `<div class="error-message">Error analyzing feature importance: ${error.message}</div>`;
        console.error('Feature importance error:', error);
        updateStatusIndicator('importance-btn', 'not-ready');
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
    featureNames.push('Sex: Male', 'Sex: Female');
    
    // One-hot encoded Embarked
    featureNames.push('Embarked: C', 'Embarked: Q', 'Embarked: S');
    
    // Optional family features
    if (document.getElementById('add-family-features').checked) {
        featureNames.push('Family Size', 'Is Alone');
    }
    
    return featureNames;
}

// Create permutation importance chart
function createPermutationImportanceChart(importanceData, baselineAccuracy) {
    const container = document.getElementById('importance-status');
    
    const chartHTML = `
        <h4>Feature Importance Visualization</h4>
        <div style="position: relative; width: 100%; max-width: 600px; margin: 20px auto;">
            <canvas id="permutation-canvas" width="600" height="400" style="border: 1px solid #ddd; background: white;"></canvas>
        </div>
    `;
    
    container.innerHTML += chartHTML;
    
    setTimeout(() => {
        const canvas = document.getElementById('permutation-canvas');
        if (!canvas) return;
        
        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;
        const padding = { top: 40, right: 20, bottom: 80, left: 150 };
        
        // Clear canvas
        ctx.clearRect(0, 0, width, height);
        
        // Calculate dimensions
        const maxImportance = Math.max(...importanceData.map(d => Math.abs(d.importance)));
        const barHeight = 25;
        const gap = 8;
        const chartWidth = width - padding.left - padding.right;
        
        // Draw baseline line
        ctx.beginPath();
        const baselineX = padding.left + chartWidth * 0.5;
        ctx.moveTo(baselineX, padding.top - 20);
        ctx.lineTo(baselineX, padding.top + importanceData.length * (barHeight + gap));
        ctx.strokeStyle = '#999';
        ctx.lineWidth = 1;
        ctx.setLineDash([5, 5]);
        ctx.stroke();
        ctx.setLineDash([]);
        
        ctx.fillStyle = '#666';
        ctx.font = '12px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('Baseline', baselineX, padding.top - 5);
        
        // Draw bars
        importanceData.forEach((item, index) => {
            const y = padding.top + index * (barHeight + gap);
            const barWidth = (Math.abs(item.importance) / maxImportance) * chartWidth * 0.45;
            const startX = item.importance >= 0 ? baselineX : baselineX - barWidth;
            
            // Draw bar
            ctx.fillStyle = item.importance >= 0 ? 'rgba(76, 175, 80, 0.8)' : 'rgba(244, 67, 54, 0.8)';
            ctx.fillRect(startX, y, barWidth, barHeight);
            
            // Draw feature name
            ctx.fillStyle = '#333';
            ctx.font = '11px Arial';
            ctx.textAlign = 'right';
            
            // Truncate long feature names
            let displayName = item.feature;
            if (displayName.length > 20) {
                displayName = displayName.substring(0, 20) + '...';
            }
            
            ctx.fillText(displayName, padding.left - 10, y + barHeight / 2 + 4);
            
            // Draw importance value
            ctx.fillStyle = item.importance >= 0 ? '#2e7d32' : '#c62828';
            ctx.font = 'bold 11px Arial';
            ctx.textAlign = item.importance >= 0 ? 'left' : 'right';
            const valueX = item.importance >= 0 ? baselineX + barWidth + 5 : baselineX - barWidth - 5;
            ctx.fillText(Math.abs(item.importance).toFixed(4), valueX, y + barHeight / 2 + 4);
        });
        
        // Draw title
        ctx.fillStyle = '#333';
        ctx.font = 'bold 16px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('Permutation Feature Importance', width / 2, 20);
        
        // Draw legend
        ctx.fillStyle = 'rgba(76, 175, 80, 0.8)';
        ctx.fillRect(width - 200, height - 70, 15, 15);
        ctx.fillStyle = 'rgba(244, 67, 54, 0.8)';
        ctx.fillRect(width - 200, height - 45, 15, 15);
        
        ctx.fillStyle = '#333';
        ctx.font = '12px Arial';
        ctx.textAlign = 'left';
        ctx.fillText('Predicts Survival (positive)', width - 180, height - 60);
        ctx.fillText('Predicts Not Survived (negative)', width - 180, height - 35);
        
        // Draw x-axis label
        ctx.fillText('Importance (accuracy drop when feature is shuffled)', width / 2, height - 10);
    }, 100);
}

// Initialize the application
window.addEventListener('DOMContentLoaded', function() {
    console.log('Titanic Survival Classifier initialized');
    
    // Add event listener for load data button
    document.getElementById('load-data-btn').onclick = loadData;
});
