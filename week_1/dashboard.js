// dashboard.js - Titanic EDA Dashboard (Restored with charts)
let titanicData = [];
let filteredData = [];
let currentFilters = { gender: 'all', pclass: 'all', minAge: 0, maxAge: 100 };

// Main initialization
document.addEventListener('DOMContentLoaded', function() {
    console.log('Titanic EDA Dashboard initialized');
    
    // Add event listeners for filters
    const filterGender = document.getElementById('filterGender');
    const filterClass = document.getElementById('filterClass');
    const minAge = document.getElementById('minAge');
    const maxAge = document.getElementById('maxAge');
    
    if (filterGender) filterGender.addEventListener('change', applyFilters);
    if (filterClass) filterClass.addEventListener('change', applyFilters);
    if (minAge) minAge.addEventListener('change', applyFilters);
    if (maxAge) maxAge.addEventListener('change', applyFilters);
    
    // Auto-load data
    setTimeout(loadTitanicData, 500);
});

// Load Titanic dataset
async function loadTitanicData() {
    const statusDiv = document.getElementById('dataStatus');
    const loadBtn = document.getElementById('loadDataBtn');
    
    if (!statusDiv) {
        console.error('dataStatus element not found');
        return;
    }
    
    if (loadBtn) {
        loadBtn.disabled = true;
        loadBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-1"></i>Loading...';
    }
    
    statusDiv.innerHTML = '<div class="alert alert-info"><i class="fas fa-spinner fa-spin me-2"></i>Loading Titanic dataset...</div>';
    
    try {
        // Try multiple paths
        const paths = [
            'data/train.csv',
            './data/train.csv',
            'train.csv',
            'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
        ];
        
        let response = null;
        let csvText = null;
        let loadedFrom = '';
        
        for (const path of paths) {
            try {
                console.log(`Trying to load from: ${path}`);
                response = await fetch(path);
                if (response.ok) {
                    csvText = await response.text();
                    loadedFrom = path;
                    console.log(`Successfully loaded from: ${path}`);
                    break;
                }
            } catch (e) {
                console.log(`Failed to load from ${path}: ${e.message}`);
                continue;
            }
        }
        
        if (!response || !response.ok) {
            throw new Error('Failed to load data from any source');
        }
        
        await parseCSVData(csvText, loadedFrom);
        
    } catch (error) {
        console.error('Error loading data:', error);
        
        let errorHTML = `<div class="alert alert-danger">
            <i class="fas fa-exclamation-triangle me-2"></i>
            <strong>Failed to load Titanic data</strong><br>
            Error: ${error.message}<br><br>
            <strong>To fix this:</strong>
            <ol class="small mb-0">
                <li>Download 'train.csv' from <a href="https://www.kaggle.com/c/titanic/data" target="_blank">Kaggle Titanic</a></li>
                <li>Place it in a folder named 'data' in your repository</li>
                <li>Refresh this page</li>
            </ol>
        </div>`;
        
        statusDiv.innerHTML = errorHTML;
        
        if (loadBtn) {
            loadBtn.disabled = false;
            loadBtn.innerHTML = '<i class="fas fa-redo me-1"></i>Retry Loading';
        }
        
        // Show placeholder conclusion with example data
        showPlaceholderConclusion();
    }
}

// Parse CSV data
async function parseCSVData(csvText, source) {
    return new Promise((resolve, reject) => {
        Papa.parse(csvText, {
            header: true,
            dynamicTyping: true,
            skipEmptyLines: true,
            complete: function(results) {
                try {
                    titanicData = results.data.filter(p => p.PassengerId);
                    console.log(`Successfully parsed ${titanicData.length} passenger records from ${source}`);
                    
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
                    
                    // Initialize dashboard
                    updateDataStatus(source);
                    updateQuickStats();
                    updateDataPreview();
                    updateTopGroups();
                    createAllCharts();
                    updateCurrentStats();
                    updateConclusion(); // Auto-generate conclusion
                    
                    // Auto-switch to Insights tab
                    setTimeout(() => {
                        const insightsTab = document.getElementById('insights-tab');
                        if (insightsTab) {
                            insightsTab.click();
                        }
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

function updateDataStatus(source) {
    const statusDiv = document.getElementById('dataStatus');
    const loadBtn = document.getElementById('loadDataBtn');
    
    if (!statusDiv) return;
    
    statusDiv.innerHTML = `<div class="alert alert-success">
        <i class="fas fa-check-circle me-2"></i>
        <strong>${titanicData.length}</strong> passenger records loaded
        <br><small>Source: ${source}</small>
    </div>`;
    
    if (loadBtn) {
        loadBtn.disabled = true;
        loadBtn.innerHTML = '<i class="fas fa-check me-1"></i>Data Loaded';
    }
}

function updateQuickStats() {
    const totalPassengersElem = document.getElementById('totalPassengers');
    const survivalRateElem = document.getElementById('survivalRate');
    const quickStatsElem = document.getElementById('quickStats');
    
    if (!totalPassengersElem || !survivalRateElem || !quickStatsElem) {
        console.warn('Quick stats elements not found');
        return;
    }
    
    const totalPassengers = titanicData.length;
    const survivors = titanicData.filter(p => p.Survived === 1).length;
    const survivalRate = ((survivors / totalPassengers) * 100).toFixed(1);
    
    totalPassengersElem.textContent = totalPassengers;
    survivalRateElem.textContent = `${survivalRate}%`;
    quickStatsElem.style.display = 'flex';
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
                const colors = {1: 'warning', 2: 'info', 3: 'secondary'};
                value = `<span class="badge bg-${colors[value] || 'secondary'}">${value}</span>`;
            } else if (col === 'Fare') {
                value = value ? `$${parseFloat(value).toFixed(2)}` : 'N/A';
            } else if (col === 'Age') {
                value = value ? value.toFixed(1) : 'N/A';
            } else if (col === 'Name') {
                value = value ? value.substring(0, 20) + (value.length > 20 ? '...' : '') : 'N/A';
            }
            html += `<td>${value || '<em class="text-muted">N/A</em>'}</td>`;
        });
        html += '</tr>';
    });
    
    html += '</tbody></table>';
    previewDiv.innerHTML = html;
    previewContainer.style.display = 'block';
}

// Apply filters - ONLY affects Insights tab
function applyFilters() {
    const genderFilter = document.getElementById('filterGender')?.value || 'all';
    const classFilter = document.getElementById('filterClass')?.value || 'all';
    const minAge = parseInt(document.getElementById('minAge')?.value) || 0;
    const maxAge = parseInt(document.getElementById('maxAge')?.value) || 100;
    
    currentFilters = { gender: genderFilter, pclass: classFilter, minAge, maxAge };
    
    filteredData = titanicData.filter(passenger => {
        if (genderFilter !== 'all' && passenger.Sex !== genderFilter) return false;
        if (classFilter !== 'all' && passenger.Pclass !== parseInt(classFilter)) return false;
        if (passenger.Age !== null && passenger.Age !== undefined) {
            if (passenger.Age < minAge || passenger.Age > maxAge) return false;
        } else if (minAge > 0 || maxAge < 100) {
            return false;
        }
        return true;
    });
    
    updateTopGroups();
    updateCurrentStats();
    updateConclusion(); // Only update conclusion, not charts
}

// Update top groups
function updateTopGroups() {
    const highestElem = document.getElementById('highestDeathGroup');
    const lowestElem = document.getElementById('lowestDeathGroup');
    
    if (!highestElem || !lowestElem) return;
    
    const groups = [];
    
    // Analyze gender groups
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
    
    // Analyze class groups
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
        
        highestElem.innerHTML = `<strong>${highest.name}</strong>: ${highest.deathRate}% death rate`;
        lowestElem.innerHTML = `<strong>${lowest.name}</strong>: ${lowest.deathRate}% death rate`;
    }
}

// Update current stats
function updateCurrentStats() {
    const currentTotalPassengers = document.getElementById('currentTotalPassengers');
    const currentDeathRate = document.getElementById('currentDeathRate');
    const currentSurvivalRate = document.getElementById('currentSurvivalRate');
    const lastUpdated = document.getElementById('lastUpdated');
    const lastUpdatedTime = document.getElementById('lastUpdatedTime');
    
    if (!currentTotalPassengers || !currentDeathRate || !currentSurvivalRate || !lastUpdated) {
        return;
    }
    
    if (filteredData.length === 0) {
        currentTotalPassengers.textContent = '--';
        currentDeathRate.textContent = '--%';
        currentSurvivalRate.textContent = '--%';
        lastUpdated.textContent = 'Analysis not yet performed';
        if (lastUpdatedTime) lastUpdatedTime.textContent = 'Analysis not yet performed';
        return;
    }
    
    const total = filteredData.length;
    const survivors = filteredData.filter(p => p.Survived === 1).length;
    const deaths = total - survivors;
    
    const deathRate = ((deaths / total) * 100).toFixed(1);
    const survivalRate = ((survivors / total) * 100).toFixed(1);
    
    currentTotalPassengers.textContent = total;
    currentDeathRate.textContent = `${deathRate}%`;
    currentSurvivalRate.textContent = `${survivalRate}%`;
    
    const now = new Date();
    const timeString = `Last analysis: ${now.toLocaleDateString()} ${now.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}`;
    lastUpdated.textContent = timeString;
    if (lastUpdatedTime) lastUpdatedTime.textContent = timeString;
}

// Show placeholder conclusion when data fails to load
function showPlaceholderConclusion() {
    const evidenceList = document.getElementById('conclusionEvidence');
    const conclusionPlaceholder = document.getElementById('conclusionPlaceholder');
    const userConclusion = document.getElementById('userConclusion');
    const conclusionVerdict = document.getElementById('conclusionVerdict');
    
    if (!evidenceList || !conclusionVerdict) return;
    
    // Show placeholder conclusion with example insights
    const evidence = [
        '<strong>Gender disparity:</strong> In the Titanic disaster, females had significantly higher survival rates than males',
        '<strong>Class impact:</strong> First-class passengers had much better survival chances than third-class passengers',
        '<strong>Age factor:</strong> Children and women were prioritized during evacuation'
    ];
    
    evidenceList.innerHTML = '';
    evidence.forEach(item => {
        const li = document.createElement('li');
        li.innerHTML = item;
        evidenceList.appendChild(li);
    });
    
    conclusionVerdict.innerHTML = 
        'Based on historical data analysis, the most important factor for survival on the Titanic was:<br>' +
        '<span class="text-danger fw-bold">Gender and Passenger Class combined</span>';
    
    if (conclusionPlaceholder) conclusionPlaceholder.style.display = 'none';
    if (userConclusion) userConclusion.style.display = 'block';
    
    // Update factor ranking with placeholder data
    updateFactorRankingPlaceholder();
}

// Update factor ranking with placeholder data
function updateFactorRankingPlaceholder() {
    const bars = [
        { id: 'factor1Bar', width: 45, text: 'Gender (45%)' },
        { id: 'factor2Bar', width: 35, text: 'Class (35%)' },
        { id: 'factor3Bar', width: 25, text: 'Age (25%)' },
        { id: 'factor4Bar', width: 20, text: 'Fare (20%)' }
    ];
    
    bars.forEach(bar => {
        const elem = document.getElementById(bar.id);
        if (elem) {
            elem.style.width = `${bar.width}%`;
            elem.innerHTML = `<span>${bar.text}</span>`;
        }
    });
}

// Update conclusion (always show, no button needed)
function updateConclusion() {
    if (titanicData.length === 0) {
        showPlaceholderConclusion();
        return;
    }
    
    updateCurrentStats();
    
    // Generate evidence based on data
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
        `Based on the data analysis of ${titanicData.length} passengers, the most important factor for death on the Titanic was:<br>` +
        `<span class="text-danger fw-bold">${mainFactor}</span>`;
    
    if (conclusionPlaceholder) conclusionPlaceholder.style.display = 'none';
    if (userConclusion) userConclusion.style.display = 'block';
    
    // Update factor ranking
    updateFactorRanking();
}

function generateEvidence() {
    const evidence = [];
    
    if (titanicData.length === 0) return evidence;
    
    // Gender analysis
    const males = titanicData.filter(p => p.Sex === 'male');
    const females = titanicData.filter(p => p.Sex === 'female');
    
    if (males.length > 0 && females.length > 0) {
        const maleDeathRate = (males.filter(p => p.Survived === 0).length / males.length * 100).toFixed(1);
        const femaleDeathRate = (females.filter(p => p.Survived === 0).length / females.length * 100).toFixed(1);
        
        evidence.push(`<strong>Gender disparity:</strong> Male death rate (${maleDeathRate}%) was ${parseFloat(maleDeathRate) > parseFloat(femaleDeathRate) * 2 ? 'more than double' : 'significantly higher than'} female death rate (${femaleDeathRate}%).`);
    }
    
    // Class analysis
    const classAnalysis = [];
    [1, 2, 3].forEach(pclass => {
        const group = titanicData.filter(p => p.Pclass === pclass);
        if (group.length > 0) {
            const deathRate = (group.filter(p => p.Survived === 0).length / group.length * 100).toFixed(1);
            classAnalysis.push({ class: pclass, rate: deathRate });
        }
    });
    
    if (classAnalysis.length >= 2) {
        classAnalysis.sort((a, b) => b.rate - a.rate);
        const highest = classAnalysis[0];
        const lowest = classAnalysis[classAnalysis.length - 1];
        
        evidence.push(`<strong>Class impact:</strong> Death rate increased from ${lowest.rate}% in ${lowest.class}st class to ${highest.rate}% in ${highest.class}rd class.`);
    }
    
    return evidence;
}

function determineMainFactor() {
    if (titanicData.length === 0) {
        return "Gender and Class combined (based on historical data)";
    }
    
    // Calculate impacts
    const genderImpact = calculateGenderImpact();
    const classImpact = calculateClassImpact();
    
    if (genderImpact > classImpact) {
        return "Gender - Being male dramatically increased death risk";
    } else {
        return "Passenger Class - Lower class meant much higher death rate";
    }
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
    
    // Sort by impact and assign to bars
    bars.sort((a, b) => b.impact - a.impact);
    
    bars.forEach((bar, index) => {
        const elem = document.getElementById(bar.id);
        if (elem) {
            // Normalize to percentage for display
            const maxImpact = Math.max(...bars.map(b => b.impact));
            const displayWidth = maxImpact > 0 ? (bar.impact / maxImpact) * 100 : 0;
            elem.style.width = `${Math.min(displayWidth, 100)}%`;
            elem.innerHTML = `<span>${bar.text}</span>`;
        }
    });
}

function calculateAgeImpact() {
    const validAges = titanicData.filter(p => p.Age !== null);
    if (validAges.length === 0) return 0;
    
    const children = validAges.filter(p => p.Age <= 12);
    const adults = validAges.filter(p => p.Age > 12 && p.Age <= 60);
    const elderly = validAges.filter(p => p.Age > 60);
    
    let maxDiff = 0;
    
    if (children.length > 0 && adults.length > 0) {
        const childDeathRate = children.filter(p => p.Survived === 0).length / children.length;
        const adultDeathRate = adults.filter(p => p.Survived === 0).length / adults.length;
        maxDiff = Math.max(maxDiff, Math.abs(childDeathRate - adultDeathRate));
    }
    
    if (elderly.length > 0 && adults.length > 0) {
        const elderlyDeathRate = elderly.filter(p => p.Survived === 0).length / elderly.length;
        const adultDeathRate = adults.filter(p => p.Survived === 0).length / adults.length;
        maxDiff = Math.max(maxDiff, Math.abs(elderlyDeathRate - adultDeathRate));
    }
    
    return maxDiff * 100;
}

function calculateFareImpact() {
    const validFares = titanicData.filter(p => p.Fare !== null && p.Fare > 0);
    if (validFares.length === 0) return 0;
    
    const fares = validFares.map(p => p.Fare).sort((a, b) => a - b);
    const medianIndex = Math.floor(fares.length / 2);
    const medianFare = fares[medianIndex];
    
    const lowFare = validFares.filter(p => p.Fare <= medianFare);
    const highFare = validFares.filter(p => p.Fare > medianFare);
    
    if (lowFare.length === 0 || highFare.length === 0) return 0;
    
    const lowDeathRate = lowFare.filter(p => p.Survived === 0).length / lowFare.length;
    const highDeathRate = highFare.filter(p => p.Survived === 0).length / highFare.length;
    
    return Math.abs(highDeathRate - lowDeathRate) * 100;
}

// Create all charts
function createAllCharts() {
    // Check if Plotly is available
    if (typeof Plotly === 'undefined') {
        console.error('Plotly is not loaded');
        return;
    }
    
    if (titanicData.length === 0) {
        console.warn('No data available for charts');
        return;
    }
    
    // Clear any existing charts first
    const chartIds = ['genderChart', 'classChart', 'ageChart', 'familyChart', 
                     'genderClassChart', 'fareChart', 'embarkedChart', 
                     'correlationChart', 'importanceChart'];
    
    chartIds.forEach(id => {
        const elem = document.getElementById(id);
        if (elem) {
            elem.innerHTML = '';
        }
    });
    
    // Create charts if data exists
    if (titanicData.length > 0) {
        createGenderChart();
        createClassChart();
        createAgeChart();
        createFamilyChart();
        createGenderClassChart();
        createFareChart();
        createEmbarkedChart();
        createCorrelationChart();
        createImportanceChart();
    }
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
    
    const layout = {
        barmode: 'group',
        title: 'Survival by Gender',
        xaxis: { title: 'Gender' },
        yaxis: { title: 'Percentage (%)', range: [0, 100] },
        showlegend: true,
        margin: { t: 50, b: 50, l: 50, r: 20 }
    };
    
    Plotly.newPlot(elem, [trace1, trace2], layout);
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
    
    const layout = {
        barmode: 'group',
        title: 'Survival by Passenger Class',
        xaxis: { title: 'Passenger Class' },
        yaxis: { title: 'Percentage (%)', range: [0, 100] },
        margin: { t: 50, b: 50, l: 50, r: 20 }
    };
    
    Plotly.newPlot(elem, [trace1, trace2], layout);
}

// Chart 3: Age distribution
function createAgeChart() {
    const elem = document.getElementById('ageChart');
    if (!elem || titanicData.length === 0) return;
    
    const validAges = titanicData.filter(p => p.Age !== null && p.Age !== undefined);
    if (validAges.length === 0) return;
    
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
        title: 'Age Distribution by Survival',
        xaxis: { title: 'Age' },
        yaxis: { title: 'Count' },
        barmode: 'overlay',
        margin: { t: 50, b: 50, l: 50, r: 20 }
    };
    
    Plotly.newPlot(elem, [trace1, trace2], layout);
}

// Chart 4: Family size impact
function createFamilyChart() {
    const elem = document.getElementById('familyChart');
    if (!elem || titanicData.length === 0) return;
    
    // Get unique family sizes
    const familySizes = [...new Set(titanicData.map(p => p.FamilySize))].sort((a, b) => a - b);
    const deathRates = [];
    const validSizes = [];
    
    familySizes.forEach(size => {
        const group = titanicData.filter(p => p.FamilySize === size);
        if (group.length >= 3) { // Only show groups with enough data
            const deathRate = (group.filter(p => p.Survived === 0).length / group.length * 100).toFixed(1);
            deathRates.push(parseFloat(deathRate));
            validSizes.push(size);
        }
    });
    
    if (validSizes.length === 0) return;
    
    const trace = {
        x: validSizes,
        y: deathRates,
        mode: 'lines+markers',
        type: 'scatter',
        marker: { 
            size: 10, 
            color: deathRates.map(r => r > 70 ? '#e74c3c' : (r > 50 ? '#f39c12' : '#27ae60')) 
        },
        line: { color: '#3498db', width: 2 },
        text: deathRates.map(r => `${r}%`),
        textposition: 'top center'
    };
    
    const layout = {
        title: 'Death Rate by Family Size',
        xaxis: { title: 'Family Size' },
        yaxis: { title: 'Death Rate (%)' },
        margin: { t: 50, b: 50, l: 50, r: 20 }
    };
    
    Plotly.newPlot(elem, [trace], layout);
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
    
    if (categories.length === 0) return;
    
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
    
    const layout = {
        title: 'Death Rate by Gender and Class',
        xaxis: { 
            title: 'Gender × Class', 
            tickangle: -45,
            tickmode: 'array',
            tickvals: categories
        },
        yaxis: { title: 'Death Rate (%)', range: [0, 100] },
        margin: { t: 50, b: 80, l: 50, r: 20 }
    };
    
    Plotly.newPlot(elem, [trace], layout);
}

// Chart 6: Fare distribution
function createFareChart() {
    const elem = document.getElementById('fareChart');
    if (!elem || titanicData.length === 0) return;
    
    const survivors = titanicData.filter(p => p.Survived === 1 && p.Fare);
    const died = titanicData.filter(p => p.Survived === 0 && p.Fare);
    
    if (survivors.length === 0 || died.length === 0) return;
    
    const trace1 = {
        y: survivors.map(p => Math.log(p.Fare + 1)),
        name: 'Survived',
        type: 'box',
        marker: { color: '#27ae60' },
        boxpoints: 'outliers'
    };
    
    const trace2 = {
        y: died.map(p => Math.log(p.Fare + 1)),
        name: 'Died',
        type: 'box',
        marker: { color: '#e74c3c' },
        boxpoints: 'outliers'
    };
    
    const layout = {
        title: 'Fare Distribution (Log Scale)',
        yaxis: { title: 'Log(Fare + 1)' },
        xaxis: { title: 'Survival Outcome' },
        boxmode: 'group',
        margin: { t: 50, b: 50, l: 50, r: 20 }
    };
    
    Plotly.newPlot(elem, [trace1, trace2], layout);
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
            portNames.push(ports[code] || code);
        }
    });
    
    if (portNames.length === 0) return;
    
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
    
    const layout = {
        barmode: 'group',
        title: 'Survival by Embarkation Port',
        xaxis: { title: 'Port' },
        yaxis: { title: 'Percentage (%)', range: [0, 100] },
        margin: { t: 50, b: 50, l: 50, r: 20 }
    };
    
    Plotly.newPlot(elem, [trace1, trace2], layout);
}

// Chart 8: Correlation heatmap
function createCorrelationChart() {
    const elem = document.getElementById('correlationChart');
    if (!elem || titanicData.length === 0) return;
    
    const features = ['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare'];
    const featureNames = ['Survived', 'Class', 'Age', 'Siblings', 'Parents', 'Fare'];
    
    const validPassengers = titanicData.filter(p => 
        p.Age !== null && p.Fare !== null
    );
    
    if (validPassengers.length < 10) return;
    
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
    
    const layout = {
        title: 'Correlation Matrix',
        xaxis: { tickangle: -45 },
        yaxis: { autorange: 'reversed' },
        margin: { t: 50, b: 80, l: 50, r: 20 }
    };
    
    Plotly.newPlot(elem, [trace], layout);
}

// Helper: Calculate correlation
function calculateCorrelation(x, y) {
    const n = x.length;
    if (n === 0) return 0;
    
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
        { name: 'Age', impact: calculateAgeImpact() },
        { name: 'Fare', impact: calculateFareImpact() },
        { name: 'Family Size', impact: calculateFamilyImpact() },
        { name: 'Embarkation', impact: calculateEmbarkedImpact() }
    ];
    
    // Filter out features with zero impact
    const validFeatures = features.filter(f => f.impact > 0);
    if (validFeatures.length === 0) return;
    
    validFeatures.sort((a, b) => b.impact - a.impact);
    
    const trace = {
        x: validFeatures.map(f => f.impact),
        y: validFeatures.map(f => f.name),
        type: 'bar',
        orientation: 'h',
        marker: {
            color: validFeatures.map(f => 
                f.name === 'Gender' ? '#e74c3c' : 
                f.name === 'Passenger Class' ? '#f39c12' : 
                '#95a5a6'
            )
        },
        text: validFeatures.map(f => `${f.impact.toFixed(1)}%`),
        textposition: 'outside'
    };
    
    const maxImpact = Math.max(...validFeatures.map(f => f.impact));
    const layout = {
        title: 'Feature Importance for Death Prediction',
        xaxis: { title: 'Impact on Death Rate (%)', range: [0, Math.max(maxImpact * 1.1, 10)] },
        yaxis: { autorange: 'reversed' },
        margin: { t: 50, b: 50, l: 150, r: 50 }
    };
    
    Plotly.newPlot(elem, [trace], layout);
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
Current Filters: ${JSON.stringify(currentFilters)}

Generated by Titanic EDA Dashboard`;

    const blob = new Blob([exportText], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `titanic_conclusion_${new Date().toISOString().slice(0,10)}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

// Reset dashboard
function resetDashboard() {
    if (titanicData.length === 0) {
        alert('No data loaded to reset');
        return;
    }
    
    if (confirm('Reset all filters to initial state?')) {
        // Reset filter values
        const filterGender = document.getElementById('filterGender');
        const filterClass = document.getElementById('filterClass');
        const minAge = document.getElementById('minAge');
        const maxAge = document.getElementById('maxAge');
        
        if (filterGender) filterGender.value = 'all';
        if (filterClass) filterClass.value = 'all';
        if (minAge) minAge.value = 0;
        if (maxAge) maxAge.value = 100;
        
        // Reset data
        filteredData = [...titanicData];
        currentFilters = { gender: 'all', pclass: 'all', minAge: 0, maxAge: 100 };
        
        // Update everything
        updateTopGroups();
        updateCurrentStats();
        updateConclusion();
        
        // Show confirmation
        alert('Dashboard has been reset to initial state.');
    }
}

// Responsive handling - redraw charts on resize
let resizeTimer;
window.addEventListener('resize', function() {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(function() {
        if (titanicData.length > 0) {
            createAllCharts();
        }
    }, 250);
});
