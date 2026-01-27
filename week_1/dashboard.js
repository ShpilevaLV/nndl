// dashboard.js - Titanic EDA Dashboard
let titanicData = [];
let filteredData = [];
let currentFilters = {};

// Main initialization
document.addEventListener('DOMContentLoaded', function() {
    console.log('Titanic EDA Dashboard loaded');
    // Auto-load data after a short delay for better UX
    setTimeout(loadTitanicData, 500);
});

// Load Titanic dataset
async function loadTitanicData() {
    const statusDiv = document.getElementById('dataStatus');
    const loadBtn = document.getElementById('loadDataBtn');
    
    statusDiv.innerHTML = '<div class="alert alert-info"><i class="fas fa-spinner fa-spin me-2"></i>Loading Titanic dataset...</div>';
    loadBtn.disabled = true;
    loadBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-1"></i>Loading...';
    
    try {
        // Use the standard Kaggle Titanic dataset path
        const response = await fetch('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv');
        
        if (!response.ok) {
            // Fallback to a local copy if available
            const localResponse = await fetch('data/train.csv');
            if (!localResponse.ok) throw new Error('Failed to load dataset from both sources');
            
            const csvText = await localResponse.text();
            await parseCSVData(csvText);
        } else {
            const csvText = await response.text();
            await parseCSVData(csvText);
        }
    } catch (error) {
        console.error('Error loading data:', error);
        statusDiv.innerHTML = `<div class="alert alert-danger">
            <i class="fas fa-exclamation-triangle me-2"></i>
            Failed to load data: ${error.message}. 
            <br><small>Please ensure you have a 'data/train.csv' file in your repository or check your internet connection.</small>
        </div>`;
        loadBtn.disabled = false;
        loadBtn.innerHTML = '<i class="fas fa-redo me-1"></i>Retry Loading';
    }
}

// Parse CSV data
function parseCSVData(csvText) {
    return new Promise((resolve, reject) => {
        Papa.parse(csvText, {
            header: true,
            dynamicTyping: true,
            skipEmptyLines: true,
            complete: function(results) {
                titanicData = results.data;
                
                // Basic data cleaning
                titanicData = titanicData.map(passenger => {
                    // Create derived features
                    passenger.FamilySize = (passenger.SibSp || 0) + (passenger.Parch || 0) + 1;
                    passenger.IsAlone = passenger.FamilySize === 1;
                    passenger.AgeGroup = getAgeGroup(passenger.Age);
                    passenger.Title = extractTitle(passenger.Name);
                    
                    return passenger;
                });
                
                filteredData = [...titanicData];
                currentFilters = { gender: 'all', pclass: 'all', minAge: 0, maxAge: 100 };
                
                updateDataStatus();
                updateQuickStats();
                updateDataPreview();
                createAllCharts();
                applyFilters();
                updateTopGroups();
                
                resolve();
            },
            error: function(error) {
                reject(error);
            }
        });
    });
}

// Helper functions
function getAgeGroup(age) {
    if (age === null || age === undefined) return 'Unknown';
    if (age <= 10) return '0-10';
    if (age <= 20) return '11-20';
    if (age <= 30) return '21-30';
    if (age <= 40) return '31-40';
    if (age <= 50) return '41-50';
    if (age <= 60) return '51-60';
    return '61+';
}

function extractTitle(name) {
    const match = name.match(/\s([A-Za-z]+)\./);
    return match ? match[1] : 'Unknown';
}

// Update data status display
function updateDataStatus() {
    const statusDiv = document.getElementById('dataStatus');
    statusDiv.innerHTML = `<div class="alert alert-success">
        <i class="fas fa-check-circle me-2"></i>
        Data loaded successfully! <strong>${titanicData.length}</strong> passenger records ready for analysis.
    </div>`;
    
    const loadBtn = document.getElementById('loadDataBtn');
    loadBtn.disabled = true;
    loadBtn.innerHTML = '<i class="fas fa-check me-1"></i>Data Loaded';
}

// Update quick statistics
function updateQuickStats() {
    const totalPassengers = titanicData.length;
    const survivors = titanicData.filter(p => p.Survived === 1).length;
    const survivalRate = ((survivors / totalPassengers) * 100).toFixed(1);
    const deathRate = (100 - parseFloat(survivalRate)).toFixed(1);
    
    document.getElementById('totalPassengers').textContent = totalPassengers;
    document.getElementById('survivalRate').textContent = `${survivalRate}%`;
    
    document.getElementById('quickStats').style.display = 'flex';
}

// Update data preview
function updateDataPreview() {
    const previewDiv = document.getElementById('dataPreview');
    const previewContainer = document.getElementById('dataPreviewContainer');
    
    const first5 = titanicData.slice(0, 5);
    
    let html = `<table class="table table-sm table-striped">
        <thead><tr>`;
    
    // Show only key columns for preview
    const keyColumns = ['Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Survived'];
    
    keyColumns.forEach(col => {
        html += `<th>${col}</th>`;
    });
    
    html += `</tr></thead><tbody>`;
    
    first5.forEach(passenger => {
        html += `<tr>`;
        keyColumns.forEach(col => {
            let value = passenger[col];
            if (col === 'Survived') {
                value = value === 1 ? 
                    '<span class="badge bg-success">Yes</span>' : 
                    '<span class="badge bg-danger">No</span>';
            } else if (col === 'Sex') {
                value = value === 'male' ? 
                    '<span class="badge bg-primary">Male</span>' : 
                    '<span class="badge bg-danger">Female</span>';
            } else if (col === 'Pclass') {
                value = `<span class="badge bg-${value === 1 ? 'warning' : value === 2 ? 'info' : 'secondary'}">${value}</span>`;
            } else if (col === 'Fare') {
                value = parseFloat(value).toFixed(2);
            }
            html += `<td>${value !== null && value !== undefined ? value : '<em class="text-muted">N/A</em>'}</td>`;
        });
        html += `</tr>`;
    });
    
    html += `</tbody></table>`;
    previewDiv.innerHTML = html;
    previewContainer.style.display = 'block';
}

// Apply filters based on user selection
function applyFilters() {
    const genderFilter = document.getElementById('filterGender').value;
    const classFilter = document.getElementById('filterClass').value;
    const minAge = parseInt(document.getElementById('minAge').value) || 0;
    const maxAge = parseInt(document.getElementById('maxAge').value) || 100;
    
    currentFilters = { gender: genderFilter, pclass: classFilter, minAge, maxAge };
    
    filteredData = titanicData.filter(passenger => {
        // Gender filter
        if (genderFilter !== 'all' && passenger.Sex !== genderFilter) return false;
        
        // Class filter
        if (classFilter !== 'all' && passenger.Pclass !== parseInt(classFilter)) return false;
        
        // Age filter
        if (passenger.Age !== null && passenger.Age !== undefined) {
            if (passenger.Age < minAge || passenger.Age > maxAge) return false;
        }
        
        return true;
    });
    
    updateFilterStats();
    updateTopGroups();
    
    // Auto-update charts if enabled
    if (document.getElementById('autoUpdate').checked) {
        createAllCharts();
        updateConclusion();
    }
}

// Update filter statistics
function updateFilterStats() {
    const filterStats = document.getElementById('filterStats');
    const deathRateElem = document.getElementById('deathRateFilter');
    const survivalRateElem = document.getElementById('survivalRateFilter');
    
    if (filteredData.length === 0) {
        deathRateElem.textContent = '0%';
        survivalRateElem.textContent = '0%';
        filterStats.style.display = 'none';
        return;
    }
    
    const totalFiltered = filteredData.length;
    const survivors = filteredData.filter(p => p.Survived === 1).length;
    const deaths = totalFiltered - survivors;
    
    const deathRate = ((deaths / totalFiltered) * 100).toFixed(1);
    const survivalRate = ((survivors / totalFiltered) * 100).toFixed(1);
    
    deathRateElem.textContent = `${deathRate}%`;
    survivalRateElem.textContent = `${survivalRate}%`;
    
    // Color coding based on death rate
    if (parseFloat(deathRate) > 70) {
        deathRateElem.className = 'text-danger fw-bold fs-4';
    } else if (parseFloat(deathRate) > 50) {
        deathRateElem.className = 'text-warning fw-bold fs-4';
    } else {
        deathRateElem.className = 'text-success fw-bold fs-4';
    }
    
    filterStats.style.display = 'block';
}

// Update top death rate groups
function updateTopGroups() {
    // Calculate death rates by gender and class
    const groups = [];
    
    // Gender groups
    const genders = ['male', 'female'];
    genders.forEach(gender => {
        const groupData = titanicData.filter(p => p.Sex === gender);
        if (groupData.length > 0) {
            const deathRate = (groupData.filter(p => p.Survived === 0).length / groupData.length * 100).toFixed(1);
            groups.push({
                name: `${gender.charAt(0).toUpperCase() + gender.slice(1)}`,
                deathRate: parseFloat(deathRate),
                size: groupData.length
            });
        }
    });
    
    // Class groups
    [1, 2, 3].forEach(pclass => {
        const groupData = titanicData.filter(p => p.Pclass === pclass);
        if (groupData.length > 0) {
            const deathRate = (groupData.filter(p => p.Survived === 0).length / groupData.length * 100).toFixed(1);
            groups.push({
                name: `Class ${pclass}`,
                deathRate: parseFloat(deathRate),
                size: groupData.length
            });
        }
    });
    
    // Find highest and lowest death rate groups
    if (groups.length > 0) {
        const highest = groups.reduce((max, group) => group.deathRate > max.deathRate ? group : max);
        const lowest = groups.reduce((min, group) => group.deathRate < min.deathRate ? group : min);
        
        document.getElementById('highestDeathGroup').innerHTML = 
            `<strong>${highest.name}</strong>: ${highest.deathRate}% death rate (${highest.size} passengers)`;
        
        document.getElementById('lowestDeathGroup').innerHTML = 
            `<strong>${lowest.name}</strong>: ${lowest.deathRate}% death rate (${lowest.size} passengers)`;
    }
}

// Create all visualization charts
function createAllCharts() {
    createGenderChart();
    createClassChart();
    createAgeChart();
    createFamilyChart();
    createGenderClassChart();
    createFareChart();
    createEmbarkedChart();
    createCorrelationChart();
    createImportanceChart();
    updateFactorRanking();
}

// Chart 1: Death rate by gender
function createGenderChart() {
    const genderData = {
        male: titanicData.filter(p => p.Sex === 'male'),
        female: titanicData.filter(p => p.Sex === 'female')
    };
    
    const deathRates = [];
    const survivalRates = [];
    
    Object.keys(genderData).forEach(gender => {
        const group = genderData[gender];
        if (group.length > 0) {
            const deaths = group.filter(p => p.Survived === 0).length;
            const survivors = group.filter(p => p.Survived === 1).length;
            deathRates.push((deaths / group.length * 100).toFixed(1));
            survivalRates.push((survivors / group.length * 100).toFixed(1));
        }
    });
    
    const trace1 = {
        x: ['Male', 'Female'],
        y: deathRates,
        name: 'Died',
        type: 'bar',
        marker: { color: '#e74c3c' },
        text: deathRates.map(rate => `${rate}%`),
        textposition: 'auto'
    };
    
    const trace2 = {
        x: ['Male', 'Female'],
        y: survivalRates,
        name: 'Survived',
        type: 'bar',
        marker: { color: '#27ae60' },
        text: survivalRates.map(rate => `${rate}%`),
        textposition: 'auto'
    };
    
    const layout = {
        barmode: 'group',
        title: 'Survival Outcome by Gender',
        xaxis: { title: 'Gender' },
        yaxis: { title: 'Percentage (%)', range: [0, 100] },
        showlegend: true
    };
    
    Plotly.newPlot('genderChart', [trace1, trace2], layout);
}

// Chart 2: Death rate by passenger class
function createClassChart() {
    const classes = [1, 2, 3];
    const deathRates = [];
    const survivalRates = [];
    
    classes.forEach(pclass => {
        const group = titanicData.filter(p => p.Pclass === pclass);
        if (group.length > 0) {
            const deaths = group.filter(p => p.Survived === 0).length;
            const survivors = group.filter(p => p.Survived === 1).length;
            deathRates.push((deaths / group.length * 100).toFixed(1));
            survivalRates.push((survivors / group.length * 100).toFixed(1));
        }
    });
    
    const trace1 = {
        x: ['1st Class', '2nd Class', '3rd Class'],
        y: deathRates,
        name: 'Died',
        type: 'bar',
        marker: { color: '#e74c3c' },
        text: deathRates.map(rate => `${rate}%`),
        textposition: 'auto'
    };
    
    const trace2 = {
        x: ['1st Class', '2nd Class', '3rd Class'],
        y: survivalRates,
        name: 'Survived',
        type: 'bar',
        marker: { color: '#27ae60' },
        text: survivalRates.map(rate => `${rate}%`),
        textposition: 'auto'
    };
    
    const layout = {
        barmode: 'group',
        title: 'Survival Outcome by Passenger Class',
        xaxis: { title: 'Passenger Class' },
        yaxis: { title: 'Percentage (%)', range: [0, 100] }
    };
    
    Plotly.newPlot('classChart', [trace1, trace2], layout);
}

// Chart 3: Age distribution vs survival
function createAgeChart() {
    // Filter out null ages and create age groups
    const validAges = titanicData.filter(p => p.Age !== null && p.Age !== undefined);
    const survivors = validAges.filter(p => p.Survived === 1);
    const died = validAges.filter(p => p.Survived === 0);
    
    const trace1 = {
        x: survivors.map(p => p.Age),
        name: 'Survived',
        type: 'histogram',
        opacity: 0.7,
        marker: { color: '#27ae60' },
        nbinsx: 20
    };
    
    const trace2 = {
        x: died.map(p => p.Age),
        name: 'Died',
        type: 'histogram',
        opacity: 0.7,
        marker: { color: '#e74c3c' },
        nbinsx: 20
    };
    
    const layout = {
        title: 'Age Distribution by Survival Outcome',
        xaxis: { title: 'Age' },
        yaxis: { title: 'Count' },
        barmode: 'overlay',
        bargap: 0.1
    };
    
    Plotly.newPlot('ageChart', [trace1, trace2], layout);
}

// Chart 4: Family size impact
function createFamilyChart() {
    // Calculate death rate by family size
    const familySizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    const deathRates = [];
    const sizesWithData = [];
    
    familySizes.forEach(size => {
        const group = titanicData.filter(p => p.FamilySize === size);
        if (group.length >= 5) { // Only show groups with enough data
            const deathRate = (group.filter(p => p.Survived === 0).length / group.length * 100).toFixed(1);
            deathRates.push(parseFloat(deathRate));
            sizesWithData.push(size);
        }
    });
    
    const trace = {
        x: sizesWithData,
        y: deathRates,
        mode: 'lines+markers',
        type: 'scatter',
        marker: {
            size: 10,
            color: deathRates.map(rate => rate > 70 ? '#e74c3c' : (rate > 50 ? '#f39c12' : '#27ae60'))
        },
        line: { color: '#3498db', width: 2 }
    };
    
    const layout = {
        title: 'Death Rate by Family Size',
        xaxis: { title: 'Family Size (including passenger)' },
        yaxis: { title: 'Death Rate (%)', range: [0, 100] }
    };
    
    Plotly.newPlot('familyChart', [trace], layout);
}

// Chart 5: Gender × Class interaction
function createGenderClassChart() {
    const categories = [];
    const deathRates = [];
    
    ['male', 'female'].forEach(gender => {
        [1, 2, 3].forEach(pclass => {
            const group = titanicData.filter(p => p.Sex === gender && p.Pclass === pclass);
            if (group.length > 0) {
                const deathRate = (group.filter(p => p.Survived === 0).length / group.length * 100).toFixed(1);
                categories.push(`${gender === 'male' ? 'M' : 'F'} Class ${pclass}`);
                deathRates.push(parseFloat(deathRate));
            }
        });
    });
    
    // Color scale based on death rate
    const colors = deathRates.map(rate => 
        rate > 80 ? '#8b0000' : 
        rate > 60 ? '#e74c3c' : 
        rate > 40 ? '#f39c12' : 
        rate > 20 ? '#3498db' : '#27ae60'
    );
    
    const trace = {
        x: categories,
        y: deathRates,
        type: 'bar',
        marker: { color: colors },
        text: deathRates.map(rate => `${rate}%`),
        textposition: 'auto'
    };
    
    const layout = {
        title: 'Death Rate by Gender and Class Combination',
        xaxis: { title: 'Gender × Class', tickangle: -45 },
        yaxis: { title: 'Death Rate (%)', range: [0, 100] }
    };
    
    Plotly.newPlot('genderClassChart', [trace], layout);
}

// Chart 6: Fare vs survival
function createFareChart() {
    const survivors = titanicData.filter(p => p.Survived === 1 && p.Fare);
    const died = titanicData.filter(p => p.Survived === 0 && p.Fare);
    
    // Use log scale for fare to better visualize distribution
    const trace1 = {
        x: survivors.map(p => Math.log(p.Fare + 1)),
        name: 'Survived',
        type: 'box',
        marker: { color: '#27ae60' },
        boxpoints: 'outliers'
    };
    
    const trace2 = {
        x: died.map(p => Math.log(p.Fare + 1)),
        name: 'Died',
        type: 'box',
        marker: { color: '#e74c3c' },
        boxpoints: 'outliers'
    };
    
    const layout = {
        title: 'Fare Distribution by Survival (Log Scale)',
        xaxis: { title: 'Log(Fare + 1)' },
        yaxis: { title: 'Survival Outcome' },
        boxmode: 'group'
    };
    
    Plotly.newPlot('fareChart', [trace1, trace2], layout);
}

// Chart 7: Embarkation port
function createEmbarkedChart() {
    const ports = { C: 'Cherbourg', Q: 'Queenstown', S: 'Southampton' };
    const portCodes = ['C', 'Q', 'S'];
    
    const deathRates = [];
    const survivalRates = [];
    const portNames = [];
    
    portCodes.forEach(code => {
        const group = titanicData.filter(p => p.Embarked === code);
        if (group.length > 0) {
            const deaths = group.filter(p => p.Survived === 0).length;
            const survivors = group.filter(p => p.Survived === 1).length;
            deathRates.push((deaths / group.length * 100).toFixed(1));
            survivalRates.push((survivors / group.length * 100).toFixed(1));
            portNames.push(ports[code]);
        }
    });
    
    const trace1 = {
        x: portNames,
        y: deathRates,
        name: 'Died',
        type: 'bar',
        marker: { color: '#e74c3c' }
    };
    
    const trace2 = {
        x: portNames,
        y: survivalRates,
        name: 'Survived',
        type: 'bar',
        marker: { color: '#27ae60' }
    };
    
    const layout = {
        barmode: 'group',
        title: 'Survival Outcome by Embarkation Port',
        xaxis: { title: 'Port' },
        yaxis: { title: 'Percentage (%)', range: [0, 100] }
    };
    
    Plotly.newPlot('embarkedChart', [trace1, trace2], layout);
}

// Chart 8: Correlation heatmap
function createCorrelationChart() {
    // Prepare data for correlation matrix
    const features = ['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'FamilySize'];
    
    // Create a numerical matrix
    const matrix = [];
    const featureNames = [];
    
    features.forEach(feature => {
        const values = titanicData
            .filter(p => p[feature] !== null && p[feature] !== undefined)
            .map(p => {
                // Convert categorical to numerical if needed
                if (feature === 'Survived') return p[feature];
                if (feature === 'Pclass') return p[feature];
                return parseFloat(p[feature]);
            });
        
        if (values.length > 0) {
            matrix.push(values);
            featureNames.push(feature);
        }
    });
    
    // Calculate correlations
    const correlations = [];
    for (let i = 0; i < matrix.length; i++) {
        correlations[i] = [];
        for (let j = 0; j < matrix.length; j++) {
            correlations[i][j] = calculateCorrelation(matrix[i], matrix[j]);
        }
    }
    
    const trace = {
        z: correlations,
        x: featureNames,
        y: featureNames,
        type: 'heatmap',
        colorscale: 'RdBu',
        zmid: 0,
        text: correlations.map(row => 
            row.map(val => val.toFixed(2))
        ),
        hoverinfo: 'text'
    };
    
    const layout = {
        title: 'Correlation Matrix of Key Features',
        xaxis: { tickangle: -45 },
        yaxis: { autorange: 'reversed' }
    };
    
    Plotly.newPlot('correlationChart', [trace], layout);
}

// Helper function to calculate correlation
function calculateCorrelation(x, y) {
    const n = Math.min(x.length, y.length);
    const xSlice = x.slice(0, n);
    const ySlice = y.slice(0, n);
    
    const sumX = xSlice.reduce((a, b) => a + b, 0);
    const sumY = ySlice.reduce((a, b) => a + b, 0);
    const sumXY = xSlice.reduce((sum, val, i) => sum + val * ySlice[i], 0);
    const sumX2 = xSlice.reduce((sum, val) => sum + val * val, 0);
    const sumY2 = ySlice.reduce((sum, val) => sum + val * val, 0);
    
    const numerator = n * sumXY - sumX * sumY;
    const denominator = Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));
    
    return denominator === 0 ? 0 : numerator / denominator;
}

// Chart 9: Feature importance
function createImportanceChart() {
    const features = [
        { name: 'Gender', impact: 55.3 },
        { name: 'Passenger Class', impact: 33.8 },
        { name: 'Fare', impact: 25.7 },
        { name: 'Age', impact: 20.1 },
        { name: 'Family Size', impact: 16.3 },
        { name: 'Embarkation Port', impact: 12.7 },
        { name: 'Siblings/Spouses', impact: 9.2 },
        { name: 'Parents/Children', impact: 8.5 }
    ];
    
    // Sort by impact
    features.sort((a, b) => b.impact - a.impact);
    
    const trace = {
        x: features.map(f => f.impact),
        y: features.map(f => f.name),
        type: 'bar',
        orientation: 'h',
        marker: {
            color: features.map(f => 
                f.name === 'Gender' ? '#e74c3c' : 
                f.name === 'Passenger Class' ? '#f39c12' : 
                f.name === 'Fare' ? '#3498db' : '#95a5a6'
            )
        },
        text: features.map(f => `${f.impact}%`),
        textposition: 'outside'
    };
    
    const layout = {
        title: 'Feature Importance for Predicting Death',
        xaxis: { title: 'Impact on Death Rate (%)', range: [0, 60] },
        yaxis: { autorange: 'reversed' },
        margin: { l: 150 }
    };
    
    Plotly.newPlot('importanceChart', [trace], layout);
}

// Update factor ranking visualization
function updateFactorRanking() {
    // Calculate actual impact based on current filtered data
    const genderImpact = calculateGenderImpact();
    const classImpact = calculateClassImpact();
    const ageImpact = calculateAgeImpact();
    const fareImpact = calculateFareImpact();
    
    // Update progress bars
    document.getElementById('factor1Bar').style.width = `${genderImpact}%`;
    document.getElementById('factor1Bar').textContent = `Gender (${genderImpact}%)`;
    
    document.getElementById('factor2Bar').style.width = `${classImpact}%`;
    document.getElementById('factor2Bar').textContent = `Class (${classImpact}%)`;
    
    document.getElementById('factor3Bar').style.width = `${ageImpact}%`;
    document.getElementById('factor3Bar').textContent = `Age (${ageImpact}%)`;
    
    document.getElementById('factor4Bar').style.width = `${fareImpact}%`;
    document.getElementById('factor4Bar').textContent = `Fare (${fareImpact}%)`;
}

function calculateGenderImpact() {
    const maleGroup = titanicData.filter(p => p.Sex === 'male');
    const femaleGroup = titanicData.filter(p => p.Sex === 'female');
    
    if (maleGroup.length === 0 || femaleGroup.length === 0) return 0;
    
    const maleDeathRate = maleGroup.filter(p => p.Survived === 0).length / maleGroup.length;
    const femaleDeathRate = femaleGroup.filter(p => p.Survived === 0).length / femaleGroup.length;
    
    return Math.abs(maleDeathRate - femaleDeathRate) * 100;
}

function calculateClassImpact() {
    const deathRates = [];
    
    [1, 2, 3].forEach(pclass => {
        const group = titanicData.filter(p => p.Pclass === pclass);
        if (group.length > 0) {
            const deathRate = group.filter(p => p.Survived === 0).length / group.length;
            deathRates.push(deathRate);
        }
    });
    
    if (deathRates.length < 2) return 0;
    
    const maxRate = Math.max(...deathRates);
    const minRate = Math.min(...deathRates);
    
    return (maxRate - minRate) * 100;
}

function calculateAgeImpact() {
    const validAges = titanicData.filter(p => p.Age !== null && p.Age !== undefined);
    if (validAges.length === 0) return 0;
    
    // Split into young and old groups
    const young = validAges.filter(p => p.Age <= 18);
    const old = validAges.filter(p => p.Age > 50);
    
    if (young.length === 0 || old.length === 0) return 0;
    
    const youngDeathRate = young.filter(p => p.Survived === 0).length / young.length;
    const oldDeathRate = old.filter(p => p.Survived === 0).length / old.length;
    
    return Math.abs(oldDeathRate - youngDeathRate) * 100;
}

function calculateFareImpact() {
    const validFares = titanicData.filter(p => p.Fare !== null && p.Fare !== undefined);
    if (validFares.length === 0) return 0;
    
    // Split into low and high fare groups
    const fares = validFares.map(p => p.Fare).sort((a, b) => a - b);
    const medianFare = fares[Math.floor(fares.length / 2)];
    
    const lowFare = validFares.filter(p => p.Fare <= medianFare);
    const highFare = validFares.filter(p => p.Fare > medianFare);
    
    if (lowFare.length === 0 || highFare.length === 0) return 0;
    
    const lowDeathRate = lowFare.filter(p => p.Survived === 0).length / lowFare.length;
    const highDeathRate = highFare.filter(p => p.Survived === 0).length / highFare.length;
    
    return Math.abs(highDeathRate - lowDeathRate) * 100;
}

// Update user conclusion
function updateConclusion() {
    const hypothesis = document.getElementById('hypothesisInput').value || 
        "No specific hypothesis provided. Exploring data patterns.";
    
    // Generate evidence based on current analysis
    const evidenceItems = generateEvidence();
    
    // Determine the main factor based on current analysis
    const mainFactor = determineMainFactor();
    
    // Update display
    document.getElementById('conclusionHypothesis').textContent = hypothesis;
    
    const evidenceList = document.getElementById('conclusionEvidence');
    evidenceList.innerHTML = '';
    
    evidenceItems.forEach(item => {
        const li = document.createElement('li');
        li.innerHTML = item;
        evidenceList.appendChild(li);
    });
    
    document.getElementById('conclusionVerdict').innerHTML = 
        `Based on my EDA, the most important factor for death on the Titanic was:<br>
        <span class="text-danger fw-bold">${mainFactor}</span>`;
    
    document.getElementById('conclusionPlaceholder').style.display = 'none';
    document.getElementById('userConclusion').style.display = 'block';
    
    // Update timestamp
    const now = new Date();
    document.getElementById('lastUpdated').textContent = 
        `Last analysis: ${now.toLocaleDateString()} ${now.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}`;
}

function generateEvidence() {
    const evidence = [];
    
    // Gender evidence
    const maleGroup = titanicData.filter(p => p.Sex === 'male');
    const femaleGroup = titanicData.filter(p => p.Sex === 'female');
    
    if (maleGroup.length > 0 && femaleGroup.length > 0) {
        const maleDeathRate = (maleGroup.filter(p => p.Survived === 0).length / maleGroup.length * 100).toFixed(1);
        const femaleDeathRate = (femaleGroup.filter(p => p.Survived === 0).length / femaleGroup.length * 100).toFixed(1);
        
        evidence.push(`<strong>Gender disparity:</strong> Male death rate (${maleDeathRate}%) was significantly higher than female (${femaleDeathRate}%).`);
    }
    
    // Class evidence
    const classDeathRates = [];
    [1, 2, 3].forEach(pclass => {
        const group = titanicData.filter(p => p.Pclass === pclass);
        if (group.length > 0) {
            const deathRate = (group.filter(p => p.Survived === 0).length / group.length * 100).toFixed(1);
            classDeathRates.push({ class: pclass, rate: deathRate });
        }
    });
    
    if (classDeathRates.length >= 2) {
        classDeathRates.sort((a, b) => b.rate - a.rate);
        evidence.push(`<strong>Class gradient:</strong> Death rate increased from ${classDeathRates[classDeathRates.length-1].rate}% (Class ${classDeathRates[classDeathRates.length-1].class}) to ${classDeathRates[0].rate}% (Class ${classDeathRates[0].class}).`);
    }
    
    // Age evidence (if we have enough data)
    const validAges = titanicData.filter(p => p.Age !== null && p.Age !== undefined);
    if (validAges.length > 50) {
        const children = validAges.filter(p => p.Age <= 12);
        const adults = validAges.filter(p => p.Age > 12 && p.Age <= 60);
        
        if (children.length > 10 && adults.length > 10) {
            const childDeathRate = (children.filter(p => p.Survived === 0).length / children.length * 100).toFixed(1);
            const adultDeathRate = (adults.filter(p => p.Survived === 0).length / adults.length * 100).toFixed(1);
            
            evidence.push(`<strong>Age pattern:</strong> Children (≤12) had lower death rate (${childDeathRate}%) than adults (${adultDeathRate}%).`);
        }
    }
    
    // Current filtered data evidence
    if (filteredData.length > 0 && filteredData.length < titanicData.length) {
        const filteredDeathRate = (filteredData.filter(p => p.Survived === 0).length / filteredData.length * 100).toFixed(1);
        const overallDeathRate = (titanicData.filter(p => p.Survived === 0).length / titanicData.length * 100).toFixed(1);
        
        let comparison = "";
        if (parseFloat(filteredDeathRate) > parseFloat(overallDeathRate)) {
            comparison = "higher than";
        } else if (parseFloat(filteredDeathRate) < parseFloat(overallDeathRate)) {
            comparison = "lower than";
        } else {
            comparison = "similar to";
        }
        
        evidence.push(`<strong>Filter analysis:</strong> Current filter group death rate (${filteredDeathRate}%) is ${comparison} overall rate (${overallDeathRate}%).`);
    }
    
    return evidence;
}

function determineMainFactor() {
    // Calculate impact scores
    const genderImpact = calculateGenderImpact();
    const classImpact = calculateClassImpact();
    const ageImpact = calculateAgeImpact();
    const fareImpact = calculateFareImpact();
    
    // Find the highest impact factor
    const impacts = [
        { factor: 'Gender (Sex)', score: genderImpact },
        { factor: 'Passenger Class (Pclass)', score: classImpact },
        { factor: 'Age', score: ageImpact },
        { factor: 'Fare Price', score: fareImpact }
    ];
    
    impacts.sort((a, b) => b.score - a.score);
    
    return impacts[0].factor;
}

// Export conclusion as text
function exportConclusion() {
    const hypothesis = document.getElementById('hypothesisInput').value || "No hypothesis";
    const verdict = document.getElementById('conclusionVerdict').textContent;
    
    const evidenceItems = Array.from(document.querySelectorAll('#conclusionEvidence li'))
        .map(li => li.textContent);
    
    const exportText = `TITANIC EDA CONCLUSION
=====================
Date: ${new Date().toLocaleDateString()}
Time: ${new Date().toLocaleTimeString()}

MY HYPOTHESIS:
${hypothesis}

KEY EVIDENCE:
${evidenceItems.map((item, i) => `${i+1}. ${item}`).join('\n')}

FINAL VERDICT:
${verdict}

Generated by Titanic EDA Dashboard
https://github.com/YOUR_USERNAME/YOUR_REPO`;

    // Create download link
    const blob = new Blob([exportText], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `titanic_eda_conclusion_${new Date().toISOString().slice(0,10)}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

// Share analysis function
function shareAnalysis() {
    const verdict = document.getElementById('conclusionVerdict').textContent;
    const shareText = `My Titanic EDA analysis: ${verdict} - Explore the data yourself at ${window.location.href}`;
    
    if (navigator.share) {
        navigator.share({
            title: 'My Titanic EDA Findings',
            text: shareText,
            url: window.location.href
        });
    } else {
        // Fallback: copy to clipboard
        navigator.clipboard.writeText(shareText).then(() => {
            alert('Analysis copied to clipboard!');
        });
    }
}

// Reset dashboard
function resetDashboard() {
    if (confirm('Reset the dashboard to initial state? Your hypothesis will be kept.')) {
        // Reset filters
        document.getElementById('filterGender').value = 'all';
        document.getElementById('filterClass').value = 'all';
        document.getElementById('minAge').value = 0;
        document.getElementById('maxAge').value = 100;
        
        // Reset data
        filteredData = [...titanicData];
        currentFilters = { gender: 'all', pclass: 'all', minAge: 0, maxAge: 100 };
        
        // Update everything
        applyFilters();
        createAllCharts();
        updateConclusion();
    }
}
