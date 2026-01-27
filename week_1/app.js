// app.js - Main logic for the Titanic EDA Dashboard
let globalData = [];

// Load the dataset from the /data folder (you need to place train.csv there)
async function loadData() {
    const statusDiv = document.getElementById('dataStatus');
    statusDiv.innerHTML = '<div class="alert alert-info">Loading data...</div>';

    try {
        // Correct path to the data file in your repository
        const response = await fetch('data/train.csv');
        if (!response.ok) throw new Error('File not found. Ensure you have a "data/train.csv" file.');

        const csvText = await response.text();
        Papa.parse(csvText, {
            header: true,
            dynamicTyping: true,
            skipEmptyLines: true,
            complete: function(results) {
                globalData = results.data;
                statusDiv.innerHTML = `<div class="alert alert-success">✅ Data loaded successfully! ${globalData.length} passenger records ready for analysis.</div>`;
                updateDataPreview();
                updateBasicStats();
                createAllCharts();
                applyFilters();
            },
            error: function(err) {
                statusDiv.innerHTML = `<div class="alert alert-danger">Failed to parse CSV: ${err.message}</div>`;
            }
        });
    } catch (error) {
        statusDiv.innerHTML = `<div class="alert alert-danger">Failed to load data: ${error.message}. Make sure the file path is correct.</div>`;
    }
}

// Show first 10 rows of data
function updateDataPreview() {
    const previewDiv = document.getElementById('dataPreview');
    const first10 = globalData.slice(0, 10);
    let html = `<table class="table table-sm table-striped"><thead><tr>`;
    if (first10.length > 0) {
        Object.keys(first10[0]).forEach(key => { html += `<th>${key}</th>`; });
        html += `</tr></thead><tbody>`;
        first10.forEach(row => {
            html += `<tr>`;
            Object.values(row).forEach(val => { html += `<td>${val === null ? '' : val}</td>`; });
            html += `</tr>`;
        });
        html += `</tbody></table>`;
    }
    previewDiv.innerHTML = html;
}

// Calculate and display basic statistics
function updateBasicStats() {
    const statsDiv = document.getElementById('basicStats');
    const total = globalData.length;
    const survived = globalData.filter(p => p.Survived === 1).length;
    const survivalRate = ((survived / total) * 100).toFixed(1);
    const maleCount = globalData.filter(p => p.Sex === 'male').length;
    const femaleCount = globalData.filter(p => p.Sex === 'female').length;

    statsDiv.innerHTML = `
        <p><strong>Dataset Overview:</strong></p>
        <ul>
            <li>Total Passengers: <strong>${total}</strong></li>
            <li>Overall Survival Rate: <strong>${survivalRate}%</strong> (${survived} survivors)</li>
            <li>Gender Distribution: <strong>${maleCount} male</strong>, <strong>${femaleCount} female</strong></li>
        </ul>
    `;
}

// Apply filters based on user selection
function applyFilters() {
    if (globalData.length === 0) return;

    const sexFilter = document.getElementById('filterSex').value;
    const classFilter = document.getElementById('filterPclass').value;

    let filteredData = globalData;

    if (sexFilter !== 'all') {
        filteredData = filteredData.filter(p => p.Sex === sexFilter);
    }
    if (classFilter !== 'all') {
        filteredData = filteredData.filter(p => p.Pclass === parseInt(classFilter));
    }

    updateFilteredStats(filteredData);
}

// Update statistics for the filtered group
function updateFilteredStats(filteredData) {
    const statsDiv = document.getElementById('filteredStats');
    const rateDiv = document.getElementById('survivalRateText');

    if (filteredData.length === 0) {
        rateDiv.innerText = 'N/A';
        statsDiv.innerHTML = '<p class="text-muted">No data for selected filters.</p>';
        return;
    }

    const totalFiltered = filteredData.length;
    const survivedFiltered = filteredData.filter(p => p.Survived === 1).length;
    const deathCount = totalFiltered - survivedFiltered;
    const survivalRateFiltered = totalFiltered > 0 ? ((survivedFiltered / totalFiltered) * 100).toFixed(1) : 0;
    const deathRate = (100 - survivalRateFiltered).toFixed(1);

    // Update the large survival rate text
    rateDiv.innerText = `${survivalRateFiltered}%`;
    // Color-code based on rate
    rateDiv.className = `text-center ${survivalRateFiltered < 30 ? 'text-danger' : (survivalRateFiltered > 60 ? 'text-success' : 'text-warning')}`;

    // Update detailed stats
    statsDiv.innerHTML = `
        <p><strong>Filtered Group Details:</strong></p>
        <ul>
            <li>Passengers in group: <strong>${totalFiltered}</strong></li>
            <li><span class="text-danger">Died: ${deathCount} (${deathRate}%)</span> | <span class="text-success">Survived: ${survivedFiltered} (${survivalRateFiltered}%)</span></li>
            <li>Average Age: <strong>${calculateAverageAge(filteredData).toFixed(1)}</strong> years</li>
        </ul>
    `;
}

function calculateAverageAge(passengers) {
    const ages = passengers.map(p => p.Age).filter(age => age !== null && !isNaN(age));
    if (ages.length === 0) return 0;
    return ages.reduce((a, b) => a + b, 0) / ages.length;
}

// Create the four main charts for EDA
function createAllCharts() {
    createChart1();
    createChart2();
    createChart3();
    createChart4();
}

function createChart1() {
    const maleData = globalData.filter(p => p.Sex === 'male');
    const femaleData = globalData.filter(p => p.Sex === 'female');

    const maleSurvived = maleData.filter(p => p.Survived === 1).length;
    const maleDied = maleData.length - maleSurvived;
    const femaleSurvived = femaleData.filter(p => p.Survived === 1).length;
    const femaleDied = femaleData.length - femaleSurvived;

    const trace1 = {
        x: ['Male', 'Female'],
        y: [maleDied, femaleDied],
        name: 'Died',
        type: 'bar',
        marker: { color: '#dc3545' }
    };
    const trace2 = {
        x: ['Male', 'Female'],
        y: [maleSurvived, femaleSurvived],
        name: 'Survived',
        type: 'bar',
        marker: { color: '#28a745' }
    };

    const layout = {
        barmode: 'group',
        title: 'Survival Counts by Gender',
        xaxis: { title: 'Gender' },
        yaxis: { title: 'Number of Passengers' }
    };

    Plotly.newPlot('chart1', [trace1, trace2], layout);
}

function createChart2() {
    const classes = [1, 2, 3];
    const diedCounts = [];
    const survivedCounts = [];

    classes.forEach(pclass => {
        const classData = globalData.filter(p => p.Pclass === pclass);
        diedCounts.push(classData.filter(p => p.Survived === 0).length);
        survivedCounts.push(classData.filter(p => p.Survived === 1).length);
    });

    const trace1 = { x: ['1st Class', '2nd Class', '3rd Class'], y: diedCounts, name: 'Died', type: 'bar', marker: { color: '#dc3545' } };
    const trace2 = { x: ['1st Class', '2nd Class', '3rd Class'], y: survivedCounts, name: 'Survived', type: 'bar', marker: { color: '#28a745' } };

    Plotly.newPlot('chart2', [trace1, trace2], {
        barmode: 'group',
        title: 'Survival Counts by Passenger Class',
        xaxis: { title: 'Passenger Class' },
        yaxis: { title: 'Number of Passengers' }
    });
}

function createChart3() {
    // Create age groups
    const ageGroups = ['0-10', '11-20', '21-30', '31-40', '41-50', '51+'];
    const ageRanges = [[0,10], [11,20], [21,30], [31,40], [41,50], [51, 100]];
    const deathRates = [];

    ageRanges.forEach(([min, max]) => {
        const group = globalData.filter(p => p.Age >= min && p.Age <= max);
        if (group.length === 0) {
            deathRates.push(0);
        } else {
            const died = group.filter(p => p.Survived === 0).length;
            deathRates.push((died / group.length * 100).toFixed(1));
        }
    });

    const trace = {
        x: ageGroups,
        y: deathRates,
        type: 'scatter',
        mode: 'lines+markers',
        line: { color: '#ff6b6b', width: 3 },
        marker: { size: 10 }
    };

    Plotly.newPlot('chart3', [trace], {
        title: 'Death Rate (%) by Age Group',
        xaxis: { title: 'Age Group' },
        yaxis: { title: 'Death Rate (%)' }
    });
}

function createChart4() {
    const ports = { C: 'Cherbourg', Q: 'Queenstown', S: 'Southampton' };
    const portCodes = Object.keys(ports);
    const deathRatesByPort = [];

    portCodes.forEach(code => {
        const portData = globalData.filter(p => p.Embarked === code);
        if (portData.length === 0) {
            deathRatesByPort.push(0);
        } else {
            const died = portData.filter(p => p.Survived === 0).length;
            deathRatesByPort.push((died / portData.length * 100).toFixed(1));
        }
    });

    const trace = {
        x: portCodes.map(code => ports[code]),
        y: deathRatesByPort,
        type: 'bar',
        marker: { color: ['#6c757d', '#adb5bd', '#495057'] }
    };

    Plotly.newPlot('chart4', [trace], {
        title: 'Death Rate (%) by Embarkation Port',
        xaxis: { title: 'Port' },
        yaxis: { title: 'Death Rate (%)' }
    });
}

// This function generates the conclusion based on user's hypothesis and data exploration
function updateFindings() {
    const hypothesis = document.getElementById('hypothesisInput').value || "No hypothesis provided.";

    // These are example insights. The user should update these based on their own filtering and chart analysis.
    const evidenceList = [
        "Filtering by <strong>Sex</strong> shows the largest disparity: ~81% of males died vs. ~26% of females.",
        "Filtering by <strong>Passenger Class</strong> shows 3rd class had the highest death rate (~75%).",
        "The <strong>Age</strong> chart indicates children (0-10) had a lower death rate than adults.",
        "Try different filter combinations to see how factors interact (e.g., 'Males in 3rd Class')."
    ];

    document.getElementById('conclusionHypothesis').innerHTML = `<em>"${hypothesis}"</em>`;
    const evidenceUl = document.getElementById('conclusionEvidence');
    evidenceUl.innerHTML = '';
    evidenceList.forEach(evidence => {
        const li = document.createElement('li');
        li.innerHTML = evidence;
        evidenceUl.appendChild(li);
    });

    document.getElementById('conclusionVerdict').innerHTML = `Based on my EDA, the strongest factor for dying on the Titanic was: <span class="text-danger">BEING MALE AND/OR A 3RD CLASS PASSENGER</span>.`;

    document.getElementById('conclusionPlaceholder').style.display = 'none';
    document.getElementById('userConclusion').style.display = 'block';
}
