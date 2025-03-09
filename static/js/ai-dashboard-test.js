/**
 * AI Dashboard Test Script
 * This script bypasses the frontend server and calls the AI server directly
 */

console.log('AI Dashboard Test Script Loaded');

// Global state
let dashboardData = null;

// Load data directly from AI server via proxy
async function loadDirectFromAIServer() {
    try {
        console.log('Fetching data from AI server...');
        
        // Update status
        const statusMessage = document.getElementById('status-message');
        if (statusMessage) {
            statusMessage.textContent = 'Fetching data...';
            statusMessage.className = 'text-primary';
        }
        
        // Use the proxy endpoint instead of direct access to avoid CORS issues
        const response = await fetch('/proxy-ai-server');
        
        if (!response.ok) {
            throw new Error(`API Error: ${response.status}`);
        }
        
        // Parse the data
        const data = await response.json();
        console.log('AI server data received:', data);
        
        // Store data
        dashboardData = data;
        
        // Update status
        if (statusMessage) {
            statusMessage.textContent = 'Data loaded successfully';
            statusMessage.className = 'text-success';
        }
        
        // Display data on the page
        displayData(data);
        
        return data;
    } catch (error) {
        console.error('Error loading data from AI server:', error);
        
        // Update status
        const statusMessage = document.getElementById('status-message');
        if (statusMessage) {
            statusMessage.textContent = `Error: ${error.message}`;
            statusMessage.className = 'text-danger';
        }
        
        document.getElementById('test-results').innerHTML = `
            <div class="alert alert-danger">
                <strong>Error:</strong> ${error.message}
            </div>
        `;
    }
}

// Test frontend dashboard connection
async function testDashboardConnection() {
    try {
        console.log('Testing frontend connection...');
        
        // Update status
        const statusMessage = document.getElementById('dashboard-status');
        if (statusMessage) {
            statusMessage.textContent = 'Testing connection...';
            statusMessage.className = 'text-primary';
        }
        
        // Use the frontend endpoint
        const response = await fetch('/api/ai-insights');
        
        if (!response.ok) {
            throw new Error(`API Error: ${response.status}`);
        }
        
        // Parse the data
        const data = await response.json();
        console.log('Frontend data received:', data);
        
        // Update status
        if (statusMessage) {
            statusMessage.textContent = 'Connection successful';
            statusMessage.className = 'text-success';
        }
        
        // Display and compare data
        const resultsContainer = document.getElementById('dashboard-test-results');
        
        if (!data) {
            resultsContainer.innerHTML = `
                <div class="alert alert-warning">
                    <strong>Warning:</strong> No data received from frontend API.
                </div>
            `;
            return;
        }
        
        // Compare data structure
        resultsContainer.innerHTML = `
            <div class="card mb-4">
                <div class="card-header bg-success text-white">
                    <h5>Frontend Connection Successful</h5>
                </div>
                <div class="card-body">
                    <p>Successfully connected to the frontend API endpoint at <code>/api/ai-insights</code>.</p>
                    <p>Data received from frontend:</p>
                    <ul class="list-group mb-3">
                        <li class="list-group-item d-flex justify-content-between align-items-center">
                            Data Type
                            <span class="badge bg-primary">${data.status || 'Unknown'}</span>
                        </li>
                        <li class="list-group-item d-flex justify-content-between align-items-center">
                            Last Updated
                            <span class="badge bg-primary">${data.lastUpdated || 'Unknown'}</span>
                        </li>
                        <li class="list-group-item d-flex justify-content-between align-items-center">
                            Indices Available
                            <span class="badge bg-primary">${data.indices ? Object.keys(data.indices).length : 0}</span>
                        </li>
                    </ul>
                    <pre class="bg-light p-3 rounded" style="max-height: 300px; overflow: auto;">${JSON.stringify(data, null, 2)}</pre>
                </div>
            </div>
        `;
        
        return data;
    } catch (error) {
        console.error('Error testing frontend connection:', error);
        
        // Update status
        const statusMessage = document.getElementById('dashboard-status');
        if (statusMessage) {
            statusMessage.textContent = `Error: ${error.message}`;
            statusMessage.className = 'text-danger';
        }
        
        document.getElementById('dashboard-test-results').innerHTML = `
            <div class="alert alert-danger">
                <strong>Error:</strong> ${error.message}
            </div>
        `;
    }
}

// Display the data
function displayData(data) {
    const resultsContainer = document.getElementById('test-results');
    
    if (!data) {
        resultsContainer.innerHTML = `
            <div class="alert alert-warning">
                <strong>Warning:</strong> No data received from AI server.
            </div>
        `;
        return;
    }
    
    // Check data structure
    const structureHtml = checkDataStructure(data);
    
    // Build HTML
    resultsContainer.innerHTML = `
        <div class="card mb-4">
            <div class="card-header">
                <h5>Data Structure Check</h5>
            </div>
            <div class="card-body">
                ${structureHtml}
            </div>
        </div>
        
        <div class="card mb-4">
            <div class="card-header">
                <h5>Market Indices</h5>
            </div>
            <div class="card-body">
                ${renderIndices(data)}
            </div>
        </div>
        
        <div class="card mb-4">
            <div class="card-header">
                <h5>Raw Data</h5>
            </div>
            <div class="card-body">
                <pre class="bg-light p-3 rounded" style="max-height: 400px; overflow: auto;">${JSON.stringify(data, null, 2)}</pre>
            </div>
        </div>
    `;
}

// Check the data structure
function checkDataStructure(data) {
    const requiredFields = [
        'indices',
        'lastUpdated',
        'insights'
    ];
    
    let html = '<ul class="list-group">';
    
    requiredFields.forEach(field => {
        const hasField = data && data[field] !== undefined;
        const status = hasField ? 'success' : 'danger';
        const icon = hasField ? 'check' : 'times';
        
        html += `
            <li class="list-group-item list-group-item-${status} d-flex justify-content-between align-items-center">
                <span>
                    <i class="fas fa-${icon} me-2"></i>
                    ${field}
                </span>
                <span class="badge bg-${status}">${hasField ? 'Present' : 'Missing'}</span>
            </li>
        `;
    });
    
    html += '</ul>';
    return html;
}

// Render market indices
function renderIndices(data) {
    if (!data.indices || Object.keys(data.indices).length === 0) {
        return `
            <div class="alert alert-warning">
                <i class="fas fa-exclamation-triangle me-2"></i>
                No market indices data found.
            </div>
        `;
    }
    
    let html = '<div class="row">';
    
    for (const symbol in data.indices) {
        if (data.indices.hasOwnProperty(symbol)) {
            const indexData = data.indices[symbol];
            const changeValue = parseFloat(indexData.change);
            const isPositive = changeValue >= 0;
            const changeColor = isPositive ? 'text-success' : 'text-danger';
            const changeIcon = isPositive ? 'fa-arrow-up' : 'fa-arrow-down';
            
            html += `
                <div class="col-md-4 mb-3">
                    <div class="card h-100">
                        <div class="card-body">
                            <h5 class="card-title">${symbol}</h5>
                            <div class="d-flex justify-content-between align-items-center mb-2">
                                <span>Price:</span>
                                <strong>${indexData.price}</strong>
                            </div>
                            <div class="d-flex justify-content-between align-items-center mb-2">
                                <span>Change:</span>
                                <strong class="${changeColor}">
                                    <i class="fas ${changeIcon}"></i>
                                    ${indexData.change} (${indexData.changePercent}%)
                                </strong>
                            </div>
                            <div class="d-flex justify-content-between align-items-center mb-2">
                                <span>High:</span>
                                <span>${indexData.high}</span>
                            </div>
                            <div class="d-flex justify-content-between align-items-center">
                                <span>Low:</span>
                                <span>${indexData.low}</span>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        }
    }
    
    html += '</div>';
    return html;
}

// Initialize when the DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    console.log('Test script DOM loaded');
    
    // Attach event listener to fetch direct button
    const fetchDirectBtn = document.getElementById('fetch-direct');
    if (fetchDirectBtn) {
        fetchDirectBtn.addEventListener('click', loadDirectFromAIServer);
    }
    
    // Attach event listener to test dashboard button
    const testDashboardBtn = document.getElementById('test-dashboard');
    if (testDashboardBtn) {
        testDashboardBtn.addEventListener('click', testDashboardConnection);
    }
}); 