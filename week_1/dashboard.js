// dashboard.js - Final Titanic EDA Dashboard with Improved Layout
let titanicData = [];
let filteredData = [];
let currentFilters = {};

// Main initialization
document.addEventListener('DOMContentLoaded', function() {
    console.log('Titanic EDA Dashboard loaded');
    
    // Initialize Bootstrap tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    const tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // Auto-load data after a short delay
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
        // Try to load from GitHub raw URL (no Kaggle API key needed)
        const response = await fetch('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv');
        
        if (!response.ok) {
            // Fallback to local copy
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
            <br><small>Using sample data for demonstration. For full analysis, add 'data/train.csv' to your repository.</small>
        </div>`;
        
        // Load sample data for demonstration
        await loadSampleData();
        
        loadBtn.disabled = false;
        loadBtn.innerHTML = '<i class="fas fa-redo me-1"></i>Retry Loading';
    }
}

// Sample data for demonstration
async function loadSampleData() {
    const sampleData = `PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
1,0,3,"Braund, Mr. Owen Harris",male,22,1,0,A/5 21171,7.25,,S
2,1,1,"Cumings, Mrs. John Bradley (Florence Briggs Thayer)",female,38,1,0,PC 17599,71.2833,C85,C
3,1,3,"Heikkinen, Miss. Laina",female,26,0,0,STON/O2. 3101282,7.925,,S
4,1,1,"Futrelle, Mrs. Jacques Heath (Lily May Peel)",female,35,1,0,113803,53.1,C123,S
5,0,3,"Allen, Mr. William Henry",male,35,0,0,373450,8.05,,S
6,0,3,"Moran, Mr. James",male,,0,0,330877,8.4583,,Q
7,0,1,"McCarthy, Mr. Timothy J",male,54,0,0,17463,51.8625,E46,S
8,0,3,"Palsson, Master. Gosta Leonard",male,2,3,1,349909,21.075,,S
9,1,3,"Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg)",female,27,0,2,347742,11.1333,,S
10,1,2,"Nasser, Mrs. Nicholas (Adele Achem)",female,14,1,0,237736,30.0708,,C`;
    
    await parseCSVData(sampleData);
    
    const statusDiv = document.getElementById('dataStatus');
    statusDiv.innerHTML = `<div class="alert alert-warning">
        <i class="fas fa-exclamation-triangle me-2"></i>
        Using sample data (10 passengers). For full analysis, download 'train.csv' from Kaggle and add to 'data/' folder.
    </div>`;
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
                
                // Data cleaning and feature engineering
                titanicData = titanicData.map(passenger => {
                    // Create derived features
                    passenger.FamilySize = (passenger.SibSp || 0) + (passenger.Parch || 0) + 1;
                    passenger.IsAlone = passenger.FamilySize === 1;
                    passenger.AgeGroup = getAgeGroup(passenger.Age);
                    
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
                currentFilters = { gender: 'all', pclass: 'all', minAge: 0, maxAge: 100 };
                
                updateDataStatus();
                updateQuickStats();
                updateDataPreview();
                createAllCharts();
                updateTopGroups();
                updateCurrentStats();
                
                // Switch to Insights tab to show the user they can start their analysis
                setTimeout(() => {
                    const insightsTab = document.getElementById('insights-tab');
                    if (insightsTab) insightsTab.click();
                }, 1000);
                
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

// Update data status display
function updateDataStatus() {
    const statusDiv = document.getElementById('dataStatus');
    const loadBtn = document.getElementById('loadDataBtn');
    
    statusDiv.innerHTML = `<div class="alert alert-success">
        <i class="fas fa-check-circle me-2"></i>
        Data loaded successfully! <strong>${titanicData.length}</strong> passenger records ready for analysis.
    </div>`;
    
    loadBtn.disabled = true;
    loadBtn.innerHTML = '<i class="fas fa-check me-1"></i>Data Loaded';
}

// Update quick statistics
function updateQuickStats() {
    const totalPassengers = titanicData.length;
    const survivors = titanicData.filter(p => p.Survived === 1).length;
    const deaths = totalPassengers - survivors;
    const survivalRate = ((survivors / totalPassengers) * 100).toFixed(1);
    const deathRate = ((deaths / totalPassengers) * 100).toFixed(1);
    
    document.getElementById('totalPassengers').textContent = totalPassengers;
    document.getElementById('survivalRate').textContent = `${survivalRate}%`;
    
    // Update overview stats
    document.getElementById('overviewPassengers').textContent = totalPassengers;
    document.getElementById('overviewSurvived').textContent = `${survivalRate}%`;
    document.getElementById('overviewDied').textContent = `${deathRate}%`;
    
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
                value = value ? parseFloat(value).toFixed(2) : 'N/A';
            } else if (col === 'Age') {
                value = value ? value.toFixed(1) : 'N/A';
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
        } else if (minAge > 0 || maxAge < 100) {
            // Exclude passengers with unknown age if age filter is active
            return false;
        }
        
        return true;
    });
    
    updateTopGroups();
    updateCurrentStats();
    
    // Auto-update charts if enabled
    createAllCharts();
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
    
    ['male', 'female'].forEach(gender => {
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
        showlegend: true,
        margin: { t: 50, b: 50, l: 50, r: 20 }
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
        yaxis: { title: 'Percentage (%)', range: [0, 100] },
        margin: { t: 50, b: 50, l: 50, r: 20 }
    };
    
    Plotly.newPlot('classChart', [trace1, trace2], layout);
}

// Chart 3: Age distribution vs survival
function createAgeChart() {
    // Filter out null ages
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
        bargap: 0.1,
        margin: { t: 50, b: 50, l: 50, r: 20 }
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
        if (group.length >= 3) { // Only show groups with enough data
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
        line: { color: '#3498db', width: 2 },
        text: deathRates.map(rate => `${rate}%`),
        textposition: 'top center'
    };
    
    const layout = {
        title: 'Death Rate by Family Size',
        xaxis: { title: 'Family Size (including passenger)' },
        yaxis: { title: 'Death Rate (%)', range: [0, Math.max(...deathRates) + 10] },
        margin: { t: 50, b: 50, l: 50, r: 20 }
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
        yaxis: { title: 'Death Rate (%)', range: [0, 100] },
        margin: { t: 50, b: 80, l: 50, r: 20 }
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
        boxmode: 'group',
        margin: { t: 50, b: 50, l: 50, r: 20 }
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
        marker: { color: '#e74c3c' },
        text: deathRates.map(rate => `${rate}%`),
        textposition: 'auto'
    };
    
    const trace2 = {
        x: portNames,
        y: survivalRates,
        name: 'Survived',
        type: 'bar',
        marker: { color: '#27ae60' },
        text: survivalRates.map(rate => `${rate}%`),
        textposition: 'auto'
    };
    
    const layout = {
        barmode: 'group',
        title: 'Survival Outcome by Embarkation Port',
        xaxis: { title: 'Port' },
        yaxis: { title: 'Percentage (%)', range: [0, 100] },
        margin: { t: 50, b: 50, l: 50, r: 20 }
    };
    
    Plotly.newPlot('embarkedChart', [trace1, trace2], layout);
}

// Chart 8: Correlation heatmap
function createCorrelationChart() {
    // Prepare data for correlation matrix
    const features = ['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'FamilySize'];
    
    // Filter out passengers with missing values for correlation
    const validPassengers = titanicData.filter(p => 
        p.Age !== null && p.Age !== undefined && 
        p.Fare !== null && p.Fare !== undefined
    );
    
    // Create a numerical matrix
    const matrix = [];
    const featureNames = [];
    
    features.forEach(feature => {
        const values = validPassengers.map(p => {
            if (feature === 'Survived') return p.Survived;
            if (feature === 'Pclass') return p.Pclass;
            if (feature === 'FamilySize') return p.FamilySize;
            return parseFloat(p[feature]);
        });
        
        matrix.push(values);
        featureNames.push(feature);
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
        hoverinfo: 'text',
        hoverlabel: { bgcolor: 'white' }
    };
    
    const layout = {
        title: 'Correlation Matrix of Key Features',
        xaxis: { tickangle: -45 },
        yaxis: { autorange: 'reversed' },
        margin: { t: 50, b: 80, l: 50, r: 20 }
    };
    
    Plotly.newPlot('correlationChart', [trace], layout);
}

// Helper function to calculate correlation
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
    // Calculate actual feature importance based on data
    const genderImpact = calculateGenderImpact();
    const classImpact = calculateClassImpact();
    const fareImpact = calculateFareImpact();
    const ageImpact = calculateAgeImpact();
    const familyImpact = calculateFamilyImpact();
    const embarkedImpact = calculateEmbarkedImpact();
    const sibSpImpact = calculateSibSpImpact();
    const parchImpact = calculateParchImpact();
    
    const features = [
        { name: 'Gender', impact: genderImpact },
        { name: 'Passenger Class', impact: classImpact },
        { name: 'Fare', impact: fareImpact },
        { name: 'Age', impact: ageImpact },
        { name: 'Family Size', impact: familyImpact },
        { name: 'Embarkation Port', impact: embarkedImpact },
        { name: 'Siblings/Spouses', impact: sibSpImpact },
        { name: 'Parents/Children', impact: parchImpact }
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
        text: features.map(f => `${f.impact.toFixed(1)}%`),
        textposition: 'outside'
    };
    
    const layout = {
        title: 'Feature Importance for Predicting Death',
        xaxis: { title: 'Impact on Death Rate (%)', range: [0, 60] },
        yaxis: { autorange: 'reversed' },
        margin: { t: 50, b: 50, l: 150, r: 50 }
    };
    
    Plotly.newPlot('importanceChart', [trace], layout);
}

// Feature importance calculation functions
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
    
    const maxRate = Math.max(...deathRates);
    const minRate = Math.min(...deathRates);
    
    return (maxRate - minRate) * 100;
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

// Update factor ranking visualization
function updateFactorRanking() {
    // Calculate actual impact based on current filtered data
    const genderImpact = calculateGenderImpact();
    const classImpact = calculateClassImpact();
    const ageImpact = calculateAgeImpact();
    const fareImpact = calculateFareImpact();
    
    // Update progress bars with text
    document.getElementById('factor1Bar').style.width = `${genderImpact}%`;
    document.getElementById('factor1Bar').innerHTML = `<span>Gender (${genderImpact.toFixed(1)}%)</span>`;
    
    document.getElementById('factor2Bar').style.width = `${classImpact}%`;
    document.getElementById('factor2Bar').innerHTML = `<span>Passenger Class (${classImpact.toFixed(1)}%)</span>`;
    
    document.getElementById('factor3Bar').style.width = `${ageImpact}%`;
    document.getElementById('factor3Bar').innerHTML = `<span>Age (${ageImpact.toFixed(1)}%)</span>`;
    
    document.getElementById('factor4Bar').style.width = `${fareImpact}%`;
    document.getElementById('factor4Bar').innerHTML = `<span>Fare (${fareImpact.toFixed(1)}%)</span>`;
}

// Update current analysis stats for Insights tab
function updateCurrentStats() {
    if (filteredData.length === 0) {
        document.getElementById('currentTotalPassengers').textContent = '--';
        document.getElementById('currentDeathRate').textContent = '--%';
        document.getElementById('currentSurvivalRate').textContent = '--%';
        return;
    }
    
    const total = filteredData.length;
    const survivors = filteredData.filter(p => p.Survived === 1).length;
    const deaths = total - survivors;
    
    const deathRate = ((deaths / total) * 100).toFixed(1);
    const survivalRate = ((survivors / total) * 100).toFixed(1);
    
    document.getElementById('currentTotalPassengers').textContent = total;
    document.getElementById('currentDeathRate').textContent = `${deathRate}%`;
    document.getElementById('currentSurvivalRate').textContent = `${survivalRate}%`;
    
    // Update timestamp
    const now = new Date();
    const timeString = `Last analysis: ${now.toLocaleDateString()} ${now.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}`;
    document.getElementById('lastUpdated').textContent = timeString;
    document.getElementById('lastUpdatedTime').textContent = timeString;
}

// Update user conclusion
function updateConclusion() {
    updateCurrentStats();
    
    // Generate evidence based on current analysis
    const evidenceItems = generateEvidence();
    
    // Determine the main factor based on current analysis
    const mainFactor = determineMainFactor();
    
    // Update display
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
        const highest = classDeathRates[0];
        const lowest = classDeathRates[classDeathRates.length-1];
        evidence.push(`<strong>Class gradient:</strong> Death rate increased from ${lowest.rate}% (Class ${lowest.class}) to ${highest.rate}% (Class ${highest.class}).`);
    }
    
    // Combined gender and class evidence
    const maleClass3 = titanicData.filter(p => p.Sex === 'male' && p.Pclass === 3);
    const femaleClass1 = titanicData.filter(p => p.Sex === 'female' && p.Pclass === 1);
    
    if (maleClass3.length > 0 && femaleClass1.length > 0) {
        const maleClass3DeathRate = (maleClass3.filter(p => p.Survived === 0).length / maleClass3.length * 100).toFixed(1);
        const femaleClass1DeathRate = (femaleClass1.filter(p => p.Survived === 0).length / femaleClass1.length * 100).toFixed(1);
        
        evidence.push(`<strong>Extreme comparison:</strong> 3rd class males had ${maleClass3DeathRate}% death rate vs 1st class females with ${femaleClass1DeathRate}%.`);
    }
    
    // Current filtered data evidence
    if (filteredData.length > 0 && filteredData.length < titanicData.length) {
        const filteredDeathRate = (filteredData.filter(p => p.Survived === 0).length / filteredData.length * 100).toFixed(1);
        const overallDeathRate = (titanicData.filter(p => p.Survived === 0).length / titanicData.length * 100).toFixed(1);
        
        let comparison = "";
        if (parseFloat(filteredDeathRate) > parseFloat(overallDeathRate) + 5) {
            comparison = "significantly higher than";
        } else if (parseFloat(filteredDeathRate) < parseFloat(overallDeathRate) - 5) {
            comparison = "significantly lower than";
        } else if (parseFloat(filteredDeathRate) > parseFloat(overallDeathRate)) {
            comparison = "slightly higher than";
        } else if (parseFloat(filteredDeathRate) < parseFloat(overallDeathRate)) {
            comparison = "slightly lower than";
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
        { factor: 'Gender (Sex)', score: genderImpact, description: 'Being male dramatically increased death risk' },
        { factor: 'Passenger Class (Pclass)', score: classImpact, description: 'Lower class meant much higher death rate' },
        { factor: 'Age', score: ageImpact, description: 'Children had better survival chances' },
        { factor: 'Fare Price', score: fareImpact, description: 'Higher fare correlated with better survival' }
    ];
    
    impacts.sort((a, b) => b.score - a.score);
    
    return `${impacts[0].factor} - ${impacts[0].description}`;
}

// Export conclusion as text
function exportConclusion() {
    const verdict = document.getElementById('conclusionVerdict').textContent;
    
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

Current Filter: ${currentFilters.gender !== 'all' ? `Gender: ${currentFilters.gender}, ` : ''}${currentFilters.pclass !== 'all' ? `Class: ${currentFilters.pclass}, ` : ''}Age: ${currentFilters.minAge}-${currentFilters.maxAge}

Generated by Titanic EDA Dashboard`;

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
    if (confirm('Reset the dashboard to initial state?')) {
        // Reset filters
        document.getElementById('filterGender').value = 'all';
        document.getElementById('filterClass').value = 'all';
        document.getElementById('minAge').value = 0;
        document.getElementById('maxAge').value = 100;
        
        // Reset data
        filteredData = [...titanicData];
        currentFilters = { gender: 'all', pclass: 'all', minAge: 0, maxAge: 100 };
        
        // Update everything
        createAllCharts();
        updateConclusion();
        
        // Switch back to Insights tab
        const insightsTab = document.getElementById('insights-tab');
        if (insightsTab) insightsTab.click();
    }
}

// Add responsive chart resizing on window resize
window.addEventListener('resize', function() {
    if (titanicData.length > 0) {
        createAllCharts();
    }
});
