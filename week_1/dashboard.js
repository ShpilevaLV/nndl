// dashboard.js - Final Titanic EDA Dashboard with Improved Data Loading
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

// Load Titanic dataset - SIMPLIFIED VERSION
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
        // First try to load from the data folder in your repository
        const localResponse = await fetch('data/train.csv');
        
        if (localResponse.ok) {
            const csvText = await localResponse.text();
            await parseCSVData(csvText);
            return;
        }
        
        // If local file not found, try GitHub raw URL as fallback
        console.log('Local file not found, trying GitHub raw URL...');
        const githubResponse = await fetch('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv');
        
        if (githubResponse.ok) {
            const csvText = await githubResponse.text();
            await parseCSVData(csvText);
        } else {
            throw new Error('Failed to load from both sources');
        }
    } catch (error) {
        console.error('Error loading data:', error);
        
        // Show user-friendly error message
        statusDiv.innerHTML = `<div class="alert alert-warning">
            <i class="fas fa-exclamation-triangle me-2"></i>
            <strong>Data Loading Issue:</strong> ${error.message}
            <br><small>To fix this:</small>
            <ol class="small mb-0">
                <li>Download 'train.csv' from <a href="https://www.kaggle.com/c/titanic/data" target="_blank">Kaggle Titanic</a></li>
                <li>Create a folder named 'data' in your repository</li>
                <li>Place 'train.csv' inside the 'data' folder</li>
                <li>Refresh this page</li>
            </ol>
        </div>`;
        
        // Load sample data for demonstration
        await loadSampleData();
        
        loadBtn.disabled = false;
        loadBtn.innerHTML = '<i class="fas fa-redo me-1"></i>Retry Loading';
    }
}

// Sample data for demonstration
async function loadSampleData() {
    console.log('Loading sample data for demonstration...');
    
    // Small sample dataset for demonstration
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
    if (statusDiv) {
        statusDiv.innerHTML = `<div class="alert alert-info">
            <i class="fas fa-info-circle me-2"></i>
            Using sample data (10 passengers) for demonstration.
            <br><small>For full analysis, download the complete dataset and place it in the 'data' folder.</small>
        </div>`;
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
                try {
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
                        if (insightsTab) {
                            insightsTab.click();
                        }
                    }, 1000);
                    
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
    
    if (!statusDiv || !loadBtn) return;
    
    statusDiv.innerHTML = `<div class="alert alert-success">
        <i class="fas fa-check-circle me-2"></i>
        Data loaded successfully! <strong>${titanicData.length}</strong> passenger records ready for analysis.
    </div>`;
    
    loadBtn.disabled = true;
    loadBtn.innerHTML = '<i class="fas fa-check me-1"></i>Data Loaded';
}

// Update quick statistics - SIMPLIFIED (removed references to removed elements)
function updateQuickStats() {
    const totalPassengers = titanicData.length;
    const survivors = titanicData.filter(p => p.Survived === 1).length;
    const survivalRate = ((survivors / totalPassengers) * 100).toFixed(1);
    
    // Update only the elements that exist
    const totalPassengersElem = document.getElementById('totalPassengers');
    const survivalRateElem = document.getElementById('survivalRate');
    const quickStatsElem = document.getElementById('quickStats');
    
    if (totalPassengersElem) {
        totalPassengersElem.textContent = totalPassengers;
    }
    
    if (survivalRateElem) {
        survivalRateElem.textContent = `${survivalRate}%`;
    }
    
    if (quickStatsElem) {
        quickStatsElem.style.display = 'flex';
    }
}

// Rest of the functions remain the same as in previous version
// (createGenderChart, createClassChart, updateCurrentStats, etc.)

// Apply filters based on user selection
function applyFilters() {
    const genderFilter = document.getElementById('filterGender')?.value || 'all';
    const classFilter = document.getElementById('filterClass')?.value || 'all';
    const minAge = parseInt(document.getElementById('minAge')?.value) || 0;
    const maxAge = parseInt(document.getElementById('maxAge')?.value) || 100;
    
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
    createAllCharts();
}

// Update data preview
function updateDataPreview() {
    const previewDiv = document.getElementById('dataPreview');
    const previewContainer = document.getElementById('dataPreviewContainer');
    
    if (!previewDiv || !previewContainer) return;
    
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
    const highestDeathGroupElem = document.getElementById('highestDeathGroup');
    const lowestDeathGroupElem = document.getElementById('lowestDeathGroup');
    
    if (groups.length > 0 && highestDeathGroupElem && lowestDeathGroupElem) {
        const highest = groups.reduce((max, group) => group.deathRate > max.deathRate ? group : max);
        const lowest = groups.reduce((min, group) => group.deathRate < min.deathRate ? group : min);
        
        highestDeathGroupElem.innerHTML = 
            `<strong>${highest.name}</strong>: ${highest.deathRate}% death rate (${highest.size} passengers)`;
        
        lowestDeathGroupElem.innerHTML = 
            `<strong>${lowest.name}</strong>: ${lowest.deathRate}% death rate (${lowest.size} passengers)`;
    }
}

// Create all visualization charts
function createAllCharts() {
    // Check if chart containers exist before creating charts
    const chartIds = ['genderChart', 'classChart', 'ageChart', 'familyChart', 
                      'genderClassChart', 'fareChart', 'embarkedChart', 
                      'correlationChart', 'importanceChart'];
    
    chartIds.forEach(id => {
        if (!document.getElementById(id)) {
            console.warn(`Chart container ${id} not found`);
        }
    });
    
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
    const genderChartElem = document.getElementById('genderChart');
    if (!genderChartElem || titanicData.length === 0) return;
    
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
    const classChartElem = document.getElementById('classChart');
    if (!classChartElem || titanicData.length === 0) return;
    
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

// Update current analysis stats for Insights tab
function updateCurrentStats() {
    if (filteredData.length === 0) {
        const currentTotalPassengers = document.getElementById('currentTotalPassengers');
        const currentDeathRate = document.getElementById('currentDeathRate');
        const currentSurvivalRate = document.getElementById('currentSurvivalRate');
        const lastUpdated = document.getElementById('lastUpdated');
        const lastUpdatedTime = document.getElementById('lastUpdatedTime');
        
        if (currentTotalPassengers) currentTotalPassengers.textContent = '--';
        if (currentDeathRate) currentDeathRate.textContent = '--%';
        if (currentSurvivalRate) currentSurvivalRate.textContent = '--%';
        if (lastUpdated) lastUpdated.textContent = 'Analysis not yet performed';
        if (lastUpdatedTime) lastUpdatedTime.textContent = 'Analysis not yet performed';
        return;
    }
    
    const total = filteredData.length;
    const survivors = filteredData.filter(p => p.Survived === 1).length;
    const deaths = total - survivors;
    
    const deathRate = ((deaths / total) * 100).toFixed(1);
    const survivalRate = ((survivors / total) * 100).toFixed(1);
    
    const currentTotalPassengers = document.getElementById('currentTotalPassengers');
    const currentDeathRate = document.getElementById('currentDeathRate');
    const currentSurvivalRate = document.getElementById('currentSurvivalRate');
    const lastUpdated = document.getElementById('lastUpdated');
    const lastUpdatedTime = document.getElementById('lastUpdatedTime');
    
    if (currentTotalPassengers) currentTotalPassengers.textContent = total;
    if (currentDeathRate) currentDeathRate.textContent = `${deathRate}%`;
    if (currentSurvivalRate) currentSurvivalRate.textContent = `${survivalRate}%`;
    
    // Update timestamp
    const now = new Date();
    const timeString = `Last analysis: ${now.toLocaleDateString()} ${now.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}`;
    if (lastUpdated) lastUpdated.textContent = timeString;
    if (lastUpdatedTime) lastUpdatedTime.textContent = timeString;
}

// [The rest of the chart functions remain the same as before]
// createAgeChart, createFamilyChart, createGenderClassChart, etc.

// For brevity, I'm including only the changed functions above.
// The rest of the chart functions (createAgeChart, createFamilyChart, etc.)
// remain exactly the same as in the previous version.

// Add responsive chart resizing on window resize
window.addEventListener('resize', function() {
    if (titanicData.length > 0) {
        createAllCharts();
    }
});
