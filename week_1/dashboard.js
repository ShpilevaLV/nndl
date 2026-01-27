// dashboard.js - Titanic EDA Dashboard (Optimized for nndl/week_1 structure)
let titanicData = [];
let filteredData = [];
let currentFilters = { gender: 'all', pclass: 'all', minAge: 0, maxAge: 100 };

// Main initialization
document.addEventListener('DOMContentLoaded', function() {
    console.log('Titanic EDA Dashboard initialized');
    
    // Auto-load data
    setTimeout(loadTitanicData, 300);
    
    // Add event listeners
    document.getElementById('filterGender')?.addEventListener('change', applyFilters);
    document.getElementById('filterClass')?.addEventListener('change', applyFilters);
    document.getElementById('minAge')?.addEventListener('change', applyFilters);
    document.getElementById('maxAge')?.addEventListener('change', applyFilters);
    
    // Add click event for Generate Conclusion button
    const conclusionBtn = document.getElementById('generateConclusionBtn');
    if (conclusionBtn) {
        conclusionBtn.addEventListener('click', updateConclusion);
    }
    
    // Add reset button
    const resetBtn = document.getElementById('resetBtn');
    if (resetBtn) {
        resetBtn.addEventListener('click', resetDashboard);
    }
});

// Load Titanic dataset
async function loadTitanicData() {
    const statusDiv = document.getElementById('dataStatus');
    const loadBtn = document.getElementById('loadDataBtn');
    
    if (!statusDiv || !loadBtn) {
        console.error('Required DOM elements not found');
        return;
    }
    
    statusDiv.innerHTML = '<div class="alert alert-info"><i class="fas fa-spinner fa-spin me-2"></i>Loading Titanic dataset...</div>';
    loadBtn.disabled = true;
    loadBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-1"></i>Loading...';
    
    try {
        // Try correct path for your structure: nndl/week_1/data/train.csv
        const response = await fetch('data/train.csv');
        
        if (!response.ok) {
            // Try alternative paths
            const alternativePaths = [
                'train.csv',
                './data/train.csv',
                'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
            ];
            
            let success = false;
            for (const path of alternativePaths) {
                try {
                    const altResponse = await fetch(path);
                    if (altResponse.ok) {
                        const csvText = await altResponse.text();
                        await parseCSVData(csvText);
                        console.log(`Data loaded from: ${path}`);
                        success = true;
                        break;
                    }
                } catch (e) {
                    console.log(`Failed to load from ${path}: ${e.message}`);
                }
            }
            
            if (!success) {
                throw new Error(`HTTP ${response.status}: File not found at data/train.csv`);
            }
        } else {
            const csvText = await response.text();
            await parseCSVData(csvText);
            console.log('Data loaded from: data/train.csv');
        }
    } catch (error) {
        console.error('Error loading data:', error);
        
        statusDiv.innerHTML = `<div class="alert alert-danger">
            <i class="fas fa-exclamation-triangle me-2"></i>
            <strong>Failed to load Titanic data</strong><br>
            Error: ${error.message}<br><br>
            <strong>Solution:</strong>
            <ol class="small mb-0">
                <li>Download 'train.csv' from <a href="https://www.kaggle.com/c/titanic/data" target="_blank">Kaggle Titanic</a></li>
                <li>Place it in: <code>nndl/week_1/data/train.csv</code></li>
                <li>Refresh this page</li>
            </ol>
        </div>`;
        
        loadBtn.disabled = false;
        loadBtn.innerHTML = '<i class="fas fa-redo me-1"></i>Retry Loading';
        
        // Show demo data option
        setTimeout(() => {
            if (confirm('Use sample data for demonstration?')) {
                loadSampleData();
            }
        }, 1000);
    }
}

// Parse CSV data
async function parseCSVData(csvText) {
    return new Promise((resolve, reject) => {
        Papa.parse(csvText, {
            header: true,
            dynamicTyping: true,
            skipEmptyLines: true,
            complete: function(results) {
                try {
                    titanicData = results.data.filter(p => p.PassengerId); // Filter out empty rows
                    console.log(`Parsed ${titanicData.length} passenger records`);
                    
                    // Data cleaning and feature engineering
                    titanicData = titanicData.map(passenger => {
                        // Create derived features
                        passenger.FamilySize = (passenger.SibSp || 0) + (passenger.Parch || 0) + 1;
                        passenger.IsAlone = passenger.FamilySize === 1;
                        passenger.AgeGroup = getAgeGroup(passenger.Age);
                        passenger.HasCabin = passenger.Cabin ? 1 : 0;
                        
                        // Ensure numeric fields are properly typed
                        passenger.Pclass = parseInt(passenger.Pclass) || 3;
                        passenger.Age = passenger.Age ? parseFloat(passenger.Age) : null;
                        passenger.Fare = passenger.Fare ? parseFloat(passenger.Fare) : null;
                        passenger.SibSp = parseInt(passenger.SibSp) || 0;
                        passenger.Parch = parseInt(passenger.Parch) || 0;
                        passenger.Survived = parseInt(passenger.Survived) || 0;
                        
                        return passenger;
                    });
                    
                    filteredData = [...titanicData];
                    
                    // Initialize dashboard
                    updateDataStatus();
                    updateQuickStats();
                    updateDataPreview();
                    updateTopGroups();
                    createAllCharts();
                    updateCurrentStats();
                    
                    // Auto-switch to Insights tab
                    setTimeout(() => {
                        const insightsTab = document.getElementById('insights-tab');
                        if (insightsTab) insightsTab.click();
                    }, 800);
                    
                    resolve();
                } catch (error) {
                    reject(new Error('Error processing data: ' + error.message));
                }
            },
            error: function(error) {
                reject(new Error('Error parsing CSV: ' + error.message));
            }
        });
    });
}

// Load sample data for demonstration
function loadSampleData() {
    const statusDiv = document.getElementById('dataStatus');
    const loadBtn = document.getElementById('loadDataBtn');
    
    statusDiv.innerHTML = '<div class="alert alert-warning"><i class="fas fa-spinner fa-spin me-2"></i>Loading sample data...</div>';
    
    // Create sample data
    titanicData = [
        { PassengerId: 1, Survived: 0, Pclass: 3, Name: "Braund, Mr. Owen Harris", Sex: "male", Age: 22, SibSp: 1, Parch: 0, Fare: 7.25, Embarked: "S" },
        { PassengerId: 2, Survived: 1, Pclass: 1, Name: "Cumings, Mrs. John Bradley", Sex: "female", Age: 38, SibSp: 1, Parch: 0, Fare: 71.28, Embarked: "C" },
        { PassengerId: 3, Survived: 1, Pclass: 3, Name: "Heikkinen, Miss. Laina", Sex: "female", Age: 26, SibSp: 0, Parch: 0, Fare: 7.92, Embarked: "S" },
        { PassengerId: 4, Survived: 1, Pclass: 1, Name: "Futrelle, Mrs. Jacques Heath", Sex: "female", Age: 35, SibSp: 1, Parch: 0, Fare: 53.1, Embarked: "S" },
        { PassengerId: 5, Survived: 0, Pclass: 3, Name: "Allen, Mr. William Henry", Sex: "male", Age: 35, SibSp: 0, Parch: 0, Fare: 8.05, Embarked: "S" },
        { PassengerId: 6, Survived: 0, Pclass: 3, Name: "Moran, Mr. James", Sex: "male", Age: null, SibSp: 0, Parch: 0, Fare: 8.46, Embarked: "Q" },
        { PassengerId: 7, Survived: 0, Pclass: 1, Name: "McCarthy, Mr. Timothy J", Sex: "male", Age: 54, SibSp: 0, Parch: 0, Fare: 51.86, Embarked: "S" },
        { PassengerId: 8, Survived: 0, Pclass: 3, Name: "Palsson, Master. Gosta Leonard", Sex: "male", Age: 2, SibSp: 3, Parch: 1, Fare: 21.07, Embarked: "S" },
        { PassengerId: 9, Survived: 1, Pclass: 3, Name: "Johnson, Mrs. Oscar W", Sex: "female", Age: 27, SibSp: 0, Parch: 2, Fare: 11.13, Embarked: "S" },
        { PassengerId: 10, Survived: 1, Pclass: 2, Name: "Nasser, Mrs. Nicholas", Sex: "female", Age: 14, SibSp: 1, Parch: 0, Fare: 30.07, Embarked: "C" }
    ];
    
    // Add derived features
    titanicData = titanicData.map(passenger => {
        passenger.FamilySize = (passenger.SibSp || 0) + (passenger.Parch || 0) + 1;
        passenger.IsAlone = passenger.FamilySize === 1;
        passenger.AgeGroup = getAgeGroup(passenger.Age);
        passenger.HasCabin = passenger.Cabin ? 1 : 0;
        return passenger;
    });
    
    filteredData = [...titanicData];
    
    updateDataStatus();
    updateQuickStats();
    updateDataPreview();
    updateTopGroups();
    createAllCharts();
    updateCurrentStats();
    
    statusDiv.innerHTML = `<div class="alert alert-warning">
        <i class="fas fa-exclamation-triangle me-2"></i>
        <strong>Sample data loaded (${titanicData.length} records)</strong><br>
        <small>For full analysis, download train.csv from Kaggle and place in data/ folder</small>
    </div>`;
    
    loadBtn.disabled = true;
    loadBtn.innerHTML = '<i class="fas fa-check me-1"></i>Sample Loaded';
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

function updateDataStatus() {
    const statusDiv = document.getElementById('dataStatus');
    const loadBtn = document.getElementById('loadDataBtn');
    
    if (!statusDiv || !loadBtn) return;
    
    statusDiv.innerHTML = `<div class="alert alert-success">
        <i class="fas fa-check-circle me-2"></i>
        <strong>${titanicData.length}</strong> passenger records loaded
        <br><small>Source: data/train.csv</small>
    </div>`;
    
    loadBtn.disabled = true;
    loadBtn.innerHTML = '<i class="fas fa-check me-1"></i>Data Loaded';
}

function updateQuickStats() {
    const totalPassengers = titanicData.length;
    const survivors = titanicData.filter(p => p.Survived === 1).length;
    const survivalRate = ((survivors / totalPassengers) * 100).toFixed(1);
    
    document.getElementById('totalPassengers').textContent = totalPassengers;
    document.getElementById('survivalRate').textContent = `${survivalRate}%`;
    document.getElementById('quickStats').style.display = 'flex';
}

function updateDataPreview() {
    const previewDiv = document.getElementById('dataPreview');
    const previewContainer = document.getElementById('dataPreviewContainer');
    
    if (!previewDiv || !previewContainer) return;
    
    const first5 = titanicData.slice(0, 5);
    const columns = ['Pclass', 'Name', 'Sex', 'Age', 'Fare', 'Survived'];
    
    let html = `<table class="table table-sm table-striped table-hover">
        <thead><tr>${columns.map(col => `<th>${col}</th>`).join('')}</tr></thead>
        <tbody>`;
    
    first5.forEach(passenger => {
        html += '<tr>';
        columns.forEach(col => {
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
                value = value ? parseFloat(value).toFixed(2) : 'N/A';
            } else if (col === 'Age') {
                value = value ? value.toFixed(1) : 'N/A';
            }
            html += `<td>${value || '<em class="text-muted">N/A</em>'}</td>`;
        });
        html += '</tr>';
    });
    
    html += '</tbody></table>';
    previewDiv.innerHTML = html;
    previewContainer.style.display = 'block';
}

// Apply filters
function applyFilters() {
    const genderFilter = document.getElementById('filterGender')?.value || 'all';
    const classFilter = document.getElementById('filterClass')?.value || 'all';
    const minAge = parseInt(document.getElementById('minAge')?.value) || 0;
    const maxAge = parseInt(document.getElementById('maxAge')?.value) || 100;
    
    currentFilters = { gender: genderFilter, pclass: classFilter, minAge, maxAge };
    
    filteredData = titanicData.filter(passenger => {
        if (genderFilter !== 'all' && passenger.Sex !== genderFilter) return false;
        if (classFilter !== 'all' && passenger.Pclass !== parseInt(classFilter)) return false;
        if (passenger.Age !== null) {
            if (passenger.Age < minAge || passenger.Age > maxAge) return false;
        } else if (minAge > 0 || maxAge < 100) {
            return false;
        }
        return true;
    });
    
    updateTopGroups();
    updateCurrentStats();
    createAllCharts();
    updateConclusion();
}

// Update top groups
function updateTopGroups() {
    const groups = [];
    
    // Gender groups
    ['male', 'female'].forEach(gender => {
        const group = titanicData.filter(p => p.Sex === gender);
        if (group.length > 0) {
            const deathRate = (group.filter(p => p.Survived === 0).length / group.length * 100).toFixed(1);
            groups.push({
                name: gender.charAt(0).toUpperCase() + gender.slice(1),
                deathRate: parseFloat(deathRate),
                size: group.length
            });
        }
    });
    
    // Class groups
    [1, 2, 3].forEach(pclass => {
        const group = titanicData.filter(p => p.Pclass === pclass);
        if (group.length > 0) {
            const deathRate = (group.filter(p => p.Survived === 0).length / group.length * 100).toFixed(1);
            groups.push({
                name: `Class ${pclass}`,
                deathRate: parseFloat(deathRate),
                size: group.length
            });
        }
    });
    
    if (groups.length > 0) {
        const highest = groups.reduce((max, g) => g.deathRate > max.deathRate ? g : max, groups[0]);
        const lowest = groups.reduce((min, g) => g.deathRate < min.deathRate ? g : min, groups[0]);
        
        document.getElementById('highestDeathGroup').innerHTML = 
            `<strong>${highest.name}</strong>: ${highest.deathRate}% death rate (${highest.size} passengers)`;
        
        document.getElementById('lowestDeathGroup').innerHTML = 
            `<strong>${lowest.name}</strong>: ${lowest.deathRate}% death rate (${lowest.size} passengers)`;
    }
}

// Create all charts
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

// Chart 1: Gender survival
function createGenderChart() {
    const elem = document.getElementById('genderChart');
    if (!elem || titanicData.length === 0) return;
    
    const males = titanicData.filter(p => p.Sex === 'male');
    const females = titanicData.filter(p => p.Sex === 'female');
    
    const maleDeathRate = males.length > 0 ? (males.filter(p => p.Survived === 0).length / males.length * 100).toFixed(1) : 0;
    const femaleDeathRate = females.length > 0 ? (females.filter(p => p.Survived === 0).length / females.length * 100).toFixed(1) : 0;
    const maleSurvivalRate = (100 - maleDeathRate).toFixed(1);
    const femaleSurvivalRate = (100 - femaleDeathRate).toFixed(1);
    
    const trace1 = {
        x: ['Male', 'Female'],
        y: [maleDeathRate, femaleDeathRate],
        name: 'Died',
        type: 'bar',
        marker: { color: '#e74c3c' },
        text: [`${maleDeathRate}%`, `${femaleDeathRate}%`],
        textposition: 'auto'
    };
    
    const trace2 = {
        x: ['Male', 'Female'],
        y: [maleSurvivalRate, femaleSurvivalRate],
        name: 'Survived',
        type: 'bar',
        marker: { color: '#27ae60' },
        text: [`${maleSurvivalRate}%`, `${femaleSurvivalRate}%`],
        textposition: 'auto'
    };
    
    Plotly.newPlot('genderChart', [trace1, trace2], {
        barmode: 'group',
        title: 'Survival by Gender',
        xaxis: { title: 'Gender' },
        yaxis: { title: 'Percentage (%)', range: [0, 100] },
        showlegend: true,
        margin: { t: 50, b: 50, l: 50, r: 20 }
    });
}

// Chart 2: Class survival
function createClassChart() {
    const elem = document.getElementById('classChart');
    if (!elem || titanicData.length === 0) return;
    
    const deathRates = [];
    const survivalRates = [];
    
    [1, 2, 3].forEach(pclass => {
        const group = titanicData.filter(p => p.Pclass === pclass);
        if (group.length > 0) {
            const deathRate = (group.filter(p => p.Survived === 0).length / group.length * 100).toFixed(1);
            deathRates.push(parseFloat(deathRate));
            survivalRates.push((100 - parseFloat(deathRate)).toFixed(1));
        }
    });
    
    const trace1 = {
        x: ['1st Class', '2nd Class', '3rd Class'],
        y: deathRates,
        name: 'Died',
        type: 'bar',
        marker: { color: '#e74c3c' },
        text: deathRates.map(r => `${r}%`),
        textposition: 'auto'
    };
    
    const trace2 = {
        x: ['1st Class', '2nd Class', '3rd Class'],
        y: survivalRates,
        name: 'Survived',
        type: 'bar',
        marker: { color: '#27ae60' },
        text: survivalRates.map(r => `${r}%`),
        textposition: 'auto'
    };
    
    Plotly.newPlot('classChart', [trace1, trace2], {
        barmode: 'group',
        title: 'Survival by Passenger Class',
        xaxis: { title: 'Passenger Class' },
        yaxis: { title: 'Percentage (%)', range: [0, 100] },
        margin: { t: 50, b: 50, l: 50, r: 20 }
    });
}

// Chart 3: Age distribution
function createAgeChart() {
    const elem = document.getElementById('ageChart');
    if (!elem || titanicData.length === 0) return;
    
    const validAges = titanicData.filter(p => p.Age !== null);
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
    
    Plotly.newPlot('ageChart', [trace1, trace2], {
        title: 'Age Distribution by Survival',
        xaxis: { title: 'Age' },
        yaxis: { title: 'Count' },
        barmode: 'overlay',
        margin: { t: 50, b: 50, l: 50, r: 20 }
    });
}

// Chart 4: Family size impact
function createFamilyChart() {
    const elem = document.getElementById('familyChart');
    if (!elem || titanicData.length === 0) return;
    
    const familySizes = [1, 2, 3, 4, 5, 6, 7, 8];
    const deathRates = [];
    const validSizes = [];
    
    familySizes.forEach(size => {
        const group = titanicData.filter(p => p.FamilySize === size);
        if (group.length >= 2) {
            const deathRate = (group.filter(p => p.Survived === 0).length / group.length * 100).toFixed(1);
            deathRates.push(parseFloat(deathRate));
            validSizes.push(size);
        }
    });
    
    const trace = {
        x: validSizes,
        y: deathRates,
        mode: 'lines+markers',
        type: 'scatter',
        marker: { size: 10, color: deathRates.map(r => r > 70 ? '#e74c3c' : (r > 50 ? '#f39c12' : '#27ae60')) },
        line: { color: '#3498db', width: 2 },
        text: deathRates.map(r => `${r}%`),
        textposition: 'top center'
    };
    
    Plotly.newPlot('familyChart', [trace], {
        title: 'Death Rate by Family Size',
        xaxis: { title: 'Family Size' },
        yaxis: { title: 'Death Rate (%)' },
        margin: { t: 50, b: 50, l: 50, r: 20 }
    });
}

// Chart 5: Gender × Class interaction
function createGenderClassChart() {
    const elem = document.getElementById('genderClassChart');
    if (!elem || titanicData.length === 0) return;
    
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
    
    const colors = deathRates.map(r => 
        r > 80 ? '#8b0000' : 
        r > 60 ? '#e74c3c' : 
        r > 40 ? '#f39c12' : 
        r > 20 ? '#3498db' : '#27ae60'
    );
    
    const trace = {
        x: categories,
        y: deathRates,
        type: 'bar',
        marker: { color: colors },
        text: deathRates.map(r => `${r}%`),
        textposition: 'auto'
    };
    
    Plotly.newPlot('genderClassChart', [trace], {
        title: 'Death Rate by Gender and Class',
        xaxis: { title: 'Gender × Class', tickangle: -45 },
        yaxis: { title: 'Death Rate (%)', range: [0, 100] },
        margin: { t: 50, b: 80, l: 50, r: 20 }
    });
}

// Chart 6: Fare distribution
function createFareChart() {
    const elem = document.getElementById('fareChart');
    if (!elem || titanicData.length === 0) return;
    
    const survivors = titanicData.filter(p => p.Survived === 1 && p.Fare);
    const died = titanicData.filter(p => p.Survived === 0 && p.Fare);
    
    const trace1 = {
        x: survivors.map(p => Math.log(p.Fare + 1)),
        name: 'Survived',
        type: 'box',
        marker: { color: '#27ae60' }
    };
    
    const trace2 = {
        x: died.map(p => Math.log(p.Fare + 1)),
        name: 'Died',
        type: 'box',
        marker: { color: '#e74c3c' }
    };
    
    Plotly.newPlot('fareChart', [trace1, trace2], {
        title: 'Fare Distribution (Log Scale)',
        xaxis: { title: 'Log(Fare + 1)' },
        yaxis: { title: 'Survival Outcome' },
        boxmode: 'group',
        margin: { t: 50, b: 50, l: 50, r: 20 }
    });
}

// Chart 7: Embarkation port
function createEmbarkedChart() {
    const elem = document.getElementById('embarkedChart');
    if (!elem || titanicData.length === 0) return;
    
    const ports = { C: 'Cherbourg', Q: 'Queenstown', S: 'Southampton' };
    const portCodes = ['C', 'Q', 'S'];
    
    const deathRates = [];
    const survivalRates = [];
    const portNames = [];
    
    portCodes.forEach(code => {
        const group = titanicData.filter(p => p.Embarked === code);
        if (group.length > 0) {
            const deathRate = (group.filter(p => p.Survived === 0).length / group.length * 100).toFixed(1);
            deathRates.push(parseFloat(deathRate));
            survivalRates.push((100 - parseFloat(deathRate)).toFixed(1));
            portNames.push(ports[code]);
        }
    });
    
    const trace1 = {
        x: portNames,
        y: deathRates,
        name: 'Died',
        type: 'bar',
        marker: { color: '#e74c3c' },
        text: deathRates.map(r => `${r}%`),
        textposition: 'auto'
    };
    
    const trace2 = {
        x: portNames,
        y: survivalRates,
        name: 'Survived',
        type: 'bar',
        marker: { color: '#27ae60' },
        text: survivalRates.map(r => `${r}%`),
        textposition: 'auto'
    };
    
    Plotly.newPlot('embarkedChart', [trace1, trace2], {
        barmode: 'group',
        title: 'Survival by Embarkation Port',
        xaxis: { title: 'Port' },
        yaxis: { title: 'Percentage (%)', range: [0, 100] },
        margin: { t: 50, b: 50, l: 50, r: 20 }
    });
}

// Chart 8: Correlation heatmap
function createCorrelationChart() {
    const elem = document.getElementById('correlationChart');
    if (!elem || titanicData.length === 0) return;
    
    const features = ['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare'];
    const featureNames = ['Survived', 'Class', 'Age', 'SibSp', 'Parch', 'Fare'];
    
    const validPassengers = titanicData.filter(p => 
        p.Age !== null && p.Fare !== null
    );
    
    // Create correlation matrix
    const correlations = [];
    for (let i = 0; i < features.length; i++) {
        correlations[i] = [];
        for (let j = 0; j < features.length; j++) {
            const values1 = validPassengers.map(p => p[features[i]]);
            const values2 = validPassengers.map(p => p[features[j]]);
            correlations[i][j] = calculateCorrelation(values1, values2);
        }
    }
    
    const trace = {
        z: correlations,
        x: featureNames,
        y: featureNames,
        type: 'heatmap',
        colorscale: 'RdBu',
        zmid: 0,
        text: correlations.map(row => row.map(val => val.toFixed(2))),
        hoverinfo: 'text'
    };
    
    Plotly.newPlot('correlationChart', [trace], {
        title: 'Correlation Matrix',
        xaxis: { tickangle: -45 },
        yaxis: { autorange: 'reversed' },
        margin: { t: 50, b: 80, l: 50, r: 20 }
    });
}

// Helper: Calculate correlation
function calculateCorrelation(x, y) {
    const n = x.length;
    const sumX = x.reduce((a, b) => a + b, 0);
    const sumY = y.reduce((a, b) => a + b, 0);
    const sumXY = x.reduce((sum, val, i) => sum + val * y[i], 0);
    const sumX2 = x.reduce((sum, val) => sum + val * val, 0);
    const sumY2 = y.reduce((sum, val) => sum + val * val, 0);
    
    const numerator = n * sumXY - sumX * sumY;
    const denominator = Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));
    
    return denominator === 0 ? 0 : numerator / denominator;
}

// Chart 9: Feature importance
function createImportanceChart() {
    const elem = document.getElementById('importanceChart');
    if (!elem || titanicData.length === 0) return;
    
    const features = [
        { name: 'Gender', impact: calculateGenderImpact() },
        { name: 'Passenger Class', impact: calculateClassImpact() },
        { name: 'Fare', impact: calculateFareImpact() },
        { name: 'Age', impact: calculateAgeImpact() },
        { name: 'Family Size', impact: calculateFamilyImpact() },
        { name: 'Embarkation', impact: calculateEmbarkedImpact() },
        { name: 'Siblings/Spouses', impact: calculateSibSpImpact() },
        { name: 'Parents/Children', impact: calculateParchImpact() }
    ];
    
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
                '#95a5a6'
            )
        },
        text: features.map(f => `${f.impact.toFixed(1)}%`),
        textposition: 'outside'
    };
    
    Plotly.newPlot('importanceChart', [trace], {
        title: 'Feature Importance for Death Prediction',
        xaxis: { title: 'Impact on Death Rate (%)', range: [0, 60] },
        yaxis: { autorange: 'reversed' },
        margin: { t: 50, b: 50, l: 150, r: 50 }
    });
}

// Impact calculation functions
function calculateGenderImpact() {
    const males = titanicData.filter(p => p.Sex === 'male');
    const females = titanicData.filter(p => p.Sex === 'female');
    
    if (males.length === 0 || females.length === 0) return 0;
    
    const maleDeathRate = males.filter(p => p.Survived === 0).length / males.length;
    const femaleDeathRate = females.filter(p => p.Survived === 0).length / females.length;
    
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
    return (Math.max(...deathRates) - Math.min(...deathRates)) * 100;
}

function calculateAgeImpact() {
    const validAges = titanicData.filter(p => p.Age !== null);
    if (validAges.length === 0) return 0;
    
    const young = validAges.filter(p => p.Age <= 18);
    const old = validAges.filter(p => p.Age > 50);
    
    if (young.length === 0 || old.length === 0) return 0;
    
    const youngDeathRate = young.filter(p => p.Survived === 0).length / young.length;
    const oldDeathRate = old.filter(p => p.Survived === 0).length / old.length;
    
    return Math.abs(oldDeathRate - youngDeathRate) * 100;
}

function calculateFareImpact() {
    const validFares = titanicData.filter(p => p.Fare !== null);
    if (validFares.length === 0) return 0;
    
    const fares = validFares.map(p => p.Fare).sort((a, b) => a - b);
    const medianFare = fares[Math.floor(fares.length / 2)];
    
    const lowFare = validFares.filter(p => p.Fare <= medianFare);
    const highFare = validFares.filter(p => p.Fare > medianFare);
    
    if (lowFare.length === 0 || highFare.length === 0) return 0;
    
    const lowDeathRate = lowFare.filter(p => p.Survived === 0).length / lowFare.length;
    const highDeathRate = highFare.filter(p => p.Survived === 0).length / highFare.length;
    
    return Math.abs(highDeathRate - lowDeathRate) * 100;
}

function calculateFamilyImpact() {
    const alone = titanicData.filter(p => p.IsAlone === true);
    const withFamily = titanicData.filter(p => p.IsAlone === false);
    
    if (alone.length === 0 || withFamily.length === 0) return 0;
    
    const aloneDeathRate = alone.filter(p => p.Survived === 0).length / alone.length;
    const familyDeathRate = withFamily.filter(p => p.Survived === 0).length / withFamily.length;
    
    return Math.abs(aloneDeathRate - familyDeathRate) * 100;
}

function calculateEmbarkedImpact() {
    const ports = ['C', 'Q', 'S'];
    const deathRates = [];
    
    ports.forEach(port => {
        const group = titanicData.filter(p => p.Embarked === port);
        if (group.length > 0) {
            const deathRate = group.filter(p => p.Survived === 0).length / group.length;
            deathRates.push(deathRate);
        }
    });
    
    if (deathRates.length < 2) return 0;
    return (Math.max(...deathRates) - Math.min(...deathRates)) * 100;
}

function calculateSibSpImpact() {
    const withSibSp = titanicData.filter(p => p.SibSp > 0);
    const withoutSibSp = titanicData.filter(p => p.SibSp === 0);
    
    if (withSibSp.length === 0 || withoutSibSp.length === 0) return 0;
    
    const withDeathRate = withSibSp.filter(p => p.Survived === 0).length / withSibSp.length;
    const withoutDeathRate = withoutSibSp.filter(p => p.Survived === 0).length / withoutSibSp.length;
    
    return Math.abs(withDeathRate - withoutDeathRate) * 100;
}

function calculateParchImpact() {
    const withParch = titanicData.filter(p => p.Parch > 0);
    const withoutParch = titanicData.filter(p => p.Parch === 0);
    
    if (withParch.length === 0 || withoutParch.length === 0) return 0;
    
    const withDeathRate = withParch.filter(p => p.Survived === 0).length / withParch.length;
    const withoutDeathRate = withoutParch.filter(p => p.Survived === 0).length / withoutParch.length;
    
    return Math.abs(withDeathRate - withoutDeathRate) * 100;
}

// Update factor ranking bars
function updateFactorRanking() {
    const genderImpact = calculateGenderImpact();
    const classImpact = calculateClassImpact();
    const ageImpact = calculateAgeImpact();
    const fareImpact = calculateFareImpact();
    
    const bars = [
        { id: 'factor1Bar', impact: genderImpact, text: `Gender (${genderImpact.toFixed(1)}%)` },
        { id: 'factor2Bar', impact: classImpact, text: `Class (${classImpact.toFixed(1)}%)` },
        { id: 'factor3Bar', impact: ageImpact, text: `Age (${ageImpact.toFixed(1)}%)` },
        { id: 'factor4Bar', impact: fareImpact, text: `Fare (${fareImpact.toFixed(1)}%)` }
    ];
    
    bars.forEach(bar => {
        const elem = document.getElementById(bar.id);
        if (elem) {
            elem.style.width = `${Math.min(bar.impact, 100)}%`;
            elem.innerHTML = `<span>${bar.text}</span>`;
        }
    });
}

// Update current stats
function updateCurrentStats() {
    const total = filteredData.length;
    const survivors = filteredData.filter(p => p.Survived === 1).length;
    const deaths = total - survivors;
    
    const deathRate = total > 0 ? ((deaths / total) * 100).toFixed(1) : '--';
    const survivalRate = total > 0 ? ((survivors / total) * 100).toFixed(1) : '--';
    
    document.getElementById('currentTotalPassengers').textContent = total;
    document.getElementById('currentDeathRate').textContent = `${deathRate}%`;
    document.getElementById('currentSurvivalRate').textContent = `${survivalRate}%`;
    
    const now = new Date();
    const timeString = `Last analysis: ${now.toLocaleDateString()} ${now.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}`;
    document.getElementById('lastUpdated').textContent = timeString;
    document.getElementById('lastUpdatedTime').textContent = timeString;
}

// Generate conclusion
function updateConclusion() {
    if (titanicData.length === 0) {
        alert('Please load data first!');
        return;
    }
    
    updateCurrentStats();
    
    // Generate evidence
    const evidence = generateEvidence();
    const mainFactor = determineMainFactor();
    
    // Update UI
    const evidenceList = document.getElementById('conclusionEvidence');
    const conclusionPlaceholder = document.getElementById('conclusionPlaceholder');
    const userConclusion = document.getElementById('userConclusion');
    const conclusionVerdict = document.getElementById('conclusionVerdict');
    
    if (!evidenceList || !conclusionVerdict) return;
    
    evidenceList.innerHTML = '';
    evidence.forEach(item => {
        const li = document.createElement('li');
        li.innerHTML = item;
        evidenceList.appendChild(li);
    });
    
    conclusionVerdict.innerHTML = 
        `Based on my EDA, the most important factor for death on the Titanic was:<br>
        <span class="text-danger fw-bold">${mainFactor}</span>`;
    
    if (conclusionPlaceholder) conclusionPlaceholder.style.display = 'none';
    if (userConclusion) userConclusion.style.display = 'block';
    
    // Scroll to conclusion
    const conclusionSection = document.querySelector('.conclusion-box');
    if (conclusionSection) {
        conclusionSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
}

function generateEvidence() {
    const evidence = [];
    
    // Gender evidence
    const males = titanicData.filter(p => p.Sex === 'male');
    const females = titanicData.filter(p => p.Sex === 'female');
    
    if (males.length > 0 && females.length > 0) {
        const maleDeathRate = (males.filter(p => p.Survived === 0).length / males.length * 100).toFixed(1);
        const femaleDeathRate = (females.filter(p => p.Survived === 0).length / females.length * 100).toFixed(1);
        
        evidence.push(`<strong>Gender disparity:</strong> Male death rate (${maleDeathRate}%) vs Female (${femaleDeathRate}%)`);
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
        const highest = classDeathRates[0];
        const lowest = classDeathRates[classDeathRates.length-1];
        evidence.push(`<strong>Class gradient:</strong> Death rate increased from ${lowest.rate}% (Class ${lowest.class}) to ${highest.rate}% (Class ${highest.class})`);
    }
    
    // Combined evidence
    const maleClass3 = titanicData.filter(p => p.Sex === 'male' && p.Pclass === 3);
    const femaleClass1 = titanicData.filter(p => p.Sex === 'female' && p.Pclass === 1);
    
    if (maleClass3.length > 0 && femaleClass1.length > 0) {
        const maleClass3DeathRate = (maleClass3.filter(p => p.Survived === 0).length / maleClass3.length * 100).toFixed(1);
        const femaleClass1DeathRate = (femaleClass1.filter(p => p.Survived === 0).length / femaleClass1.length * 100).toFixed(1);
        
        evidence.push(`<strong>Extreme comparison:</strong> 3rd class males (${maleClass3DeathRate}%) vs 1st class females (${femaleClass1DeathRate}%)`);
    }
    
    // Current filtered data evidence
    if (filteredData.length > 0 && filteredData.length < titanicData.length) {
        const filteredDeathRate = (filteredData.filter(p => p.Survived === 0).length / filteredData.length * 100).toFixed(1);
        const overallDeathRate = (titanicData.filter(p => p.Survived === 0).length / titanicData.length * 100).toFixed(1);
        
        let comparison = "";
        const diff = parseFloat(filteredDeathRate) - parseFloat(overallDeathRate);
        if (diff > 5) comparison = "significantly higher than";
        else if (diff < -5) comparison = "significantly lower than";
        else if (diff > 0) comparison = "slightly higher than";
        else if (diff < 0) comparison = "slightly lower than";
        else comparison = "similar to";
        
        evidence.push(`<strong>Filter analysis:</strong> Current filter group (${filteredDeathRate}%) is ${comparison} overall (${overallDeathRate}%)`);
    }
    
    return evidence;
}

function determineMainFactor() {
    const impacts = [
        { factor: 'Gender (Sex)', score: calculateGenderImpact(), description: 'Being male dramatically increased death risk' },
        { factor: 'Passenger Class', score: calculateClassImpact(), description: 'Lower class meant much higher death rate' },
        { factor: 'Fare Price', score: calculateFareImpact(), description: 'Higher fare correlated with better survival' },
        { factor: 'Age', score: calculateAgeImpact(), description: 'Children had better survival chances' }
    ];
    
    impacts.sort((a, b) => b.score - a.score);
    return `${impacts[0].factor} - ${impacts[0].description}`;
}

// Export conclusion
function exportConclusion() {
    const verdict = document.getElementById('conclusionVerdict')?.textContent || 'No conclusion available';
    const evidenceItems = Array.from(document.querySelectorAll('#conclusionEvidence li'))
        .map(li => li.textContent);
    
    const exportText = `TITANIC EDA CONCLUSION
=====================
Date: ${new Date().toLocaleDateString()}
Time: ${new Date().toLocaleTimeString()}

KEY EVIDENCE:
${evidenceItems.map((item, i) => `${i+1}. ${item}`).join('\n')}

FINAL VERDICT:
${verdict}

Dataset: ${titanicData.length} passengers
Filtered: ${filteredData.length} passengers
Generated by Titanic EDA Dashboard`;

    const blob = new Blob([exportText], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `titanic_eda_conclusion_${new Date().toISOString().slice(0,10)}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    
    // Show confirmation
    const btn = event.target;
    const originalText = btn.innerHTML;
    btn.innerHTML = '<i class="fas fa-check me-1"></i>Exported!';
    setTimeout(() => btn.innerHTML = originalText, 2000);
}

// Share analysis
function shareAnalysis() {
    const verdict = document.getElementById('conclusionVerdict')?.textContent || 'No conclusion available';
    const shareText = `My Titanic EDA analysis: ${verdict.substring(0, 100)}...`;
    
    if (navigator.share) {
        navigator.share({
            title: 'My Titanic EDA Findings',
            text: shareText,
            url: window.location.href
        });
    } else {
        navigator.clipboard.writeText(shareText).then(() => {
            alert('Analysis copied to clipboard!');
        });
    }
}

// Reset dashboard
function resetDashboard() {
    if (titanicData.length === 0) return;
    
    if (confirm('Reset all filters and charts to initial state?')) {
        // Reset filters
        document.getElementById('filterGender').value = 'all';
        document.getElementById('filterClass').value = 'all';
        document.getElementById('minAge').value = 0;
        document.getElementById('maxAge').value = 100;
        
        // Reset data
        filteredData = [...titanicData];
        currentFilters = { gender: 'all', pclass: 'all', minAge: 0, maxAge: 100 };
        
        // Update everything
        updateTopGroups();
        updateCurrentStats();
        createAllCharts();
        updateConclusion();
        
        // Switch to Insights tab
        const insightsTab = document.getElementById('insights-tab');
        if (insightsTab) insightsTab.click();
    }
}

// Responsive chart resizing
window.addEventListener('resize', function() {
    if (titanicData.length > 0) {
        setTimeout(createAllCharts, 100);
    }
});

// Auto-update conclusion when filters change
setInterval(() => {
    if (titanicData.length > 0 && Math.random() < 0.3) { // Occasional auto-update
        updateCurrentStats();
    }
}, 30000);
