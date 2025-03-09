/**
 * AI Dashboard - Main JavaScript File
 * 
 * Comprehensive dashboard for displaying AI-powered financial analytics
 */

// Immediately log that the script has started loading
console.log('AI Dashboard: Script file starting to load');

// Global state for dashboard data
const dashboardState = {
    selectedModel: 'ensemble',
    selectedOptimizationStrategy: 'max_sharpe',
    refreshInterval: null,
    lastUpdated: null,
    data: null,
    selectedPredictionModel: 'ensemble'  // Options: 'ensemble', 'transformer'
};

// Log that script has finished initialization
console.log('AI Dashboard: Global state initialized');

// Initialize when the DOM is fully loaded
document.addEventListener('DOMContentLoaded', function() {
    console.log('AI Dashboard: DOM loaded');
    initDashboard();
});

// Main initialization function
function initDashboard() {
    console.log('Initializing AI Dashboard');
    
    // Set up tab navigation if it exists
    setupTabNavigation();
    
    // Set up theme toggle if it exists
    setupThemeToggle();
    
    // Set up event listeners for dashboard controls
    setupEventListeners();
    
    // Load all dashboard components
    loadDashboardData();
    
    // Set up auto-refresh (every 5 minutes)
    setupAutoRefresh(5 * 60 * 1000);
}

// Set up event listeners for dashboard controls
function setupEventListeners() {
    // Model selector change listener
    const modelSelector = document.getElementById('model-selector');
    if (modelSelector) {
        modelSelector.addEventListener('change', function() {
            dashboardState.selectedModel = this.value;
            updatePredictionVisuals();
        });
    }
    
    // Prediction model selector change listener
    const predictionModelSelector = document.getElementById('prediction-model-selector');
    if (predictionModelSelector) {
        predictionModelSelector.addEventListener('change', function() {
            dashboardState.selectedPredictionModel = this.value;
            loadMarketPredictions();
        });
    }
    
    // Optimization strategy selector change listener
    const strategySelector = document.getElementById('optimization-strategy-selector');
    if (strategySelector) {
        strategySelector.addEventListener('change', function() {
            dashboardState.selectedOptimizationStrategy = this.value;
            updatePortfolioOptimization();
        });
    }
    
    // Create alert button click listener
    const createAlertBtn = document.getElementById('create-alert-btn');
    if (createAlertBtn) {
        createAlertBtn.addEventListener('click', function() {
            showAlertCreationModal();
        });
    }
    
    // Save alert button click listener
    const saveAlertBtn = document.getElementById('save-alert-btn');
    if (saveAlertBtn) {
        saveAlertBtn.addEventListener('click', function() {
            saveNewAlert();
        });
    }
}

// Load all dashboard data components
async function loadDashboardData() {
    try {
        console.log('Loading AI dashboard data');
        
        // Update UI to show loading state
        updateLoadingState(true);
        
        // Fetch AI insights data
        console.log('Fetching data from /api/ai-insights...');
        const response = await fetch('/api/ai-insights');
        
        if (!response.ok) {
            console.error(`API Error: ${response.status} - ${response.statusText}`);
            throw new Error(`API Error: ${response.status}`);
        }
        
        // Parse the data
        const data = await response.json();
        console.log('AI insights data received, size:', JSON.stringify(data).length, 'bytes');
        console.log('Data keys present:', Object.keys(data));
        
        // Validate the data structure
        if (!data || typeof data !== 'object') {
            console.error('Invalid data format received:', data);
            throw new Error('Invalid data format received from API');
        }
        
        // Normalize the data to ensure consistent naming between camelCase API and snake_case JS
        const normalizedData = normalizeDataStructure(data);
        console.log('Data normalized, keys now:', Object.keys(normalizedData));
        
        // Store data in the dashboard state
        dashboardState.data = normalizedData;
        dashboardState.lastUpdated = new Date();
        console.log('Data stored in dashboard state');
        
        // Update all dashboard components with the new data
        updateAllComponents();
        
        // Update UI to show success state
        updateLoadingState(false, true);
        console.log('Dashboard data loading completed successfully');
        
    } catch (error) {
        console.error('Error loading AI insights data:', error);
        // Update UI to show error state
        updateLoadingState(false, false);
        showErrorMessage('AI data failed to load. Using fallback display. Please try refreshing or contact support.');
        
        // Load fallback display data
        loadFallbackData();
    }
}

// Function to normalize data structure keys from camelCase to snake_case
function normalizeDataStructure(data) {
    // Create a deep copy of the data
    const normalizedData = JSON.parse(JSON.stringify(data));
    
    // Handle specific known field mappings
    const fieldMappings = {
        'newsSentiment': 'news_sentiment',
        'featureImportance': 'feature_importance',
        'economicIndicators': 'economic_indicators',
        'portfolioOptimization': 'portfolio_optimization',
        'predictionHistory': 'prediction_history',
        'predictionConfidence': 'prediction_confidence',
        'returnPrediction': 'return_prediction',
        'modelMetrics': 'model_metrics'
    };
    
    // Apply mappings
    for (const [camelCase, snakeCase] of Object.entries(fieldMappings)) {
        if (normalizedData[camelCase] !== undefined) {
            normalizedData[snakeCase] = normalizedData[camelCase];
            // Keep the original for backward compatibility
            // delete normalizedData[camelCase];
        }
    }
    
    console.log('Data normalized with field mappings');
    return normalizedData;
}

// New function to load fallback data when API fails
function loadFallbackData() {
    console.log('Loading fallback data for AI dashboard');
    
    // Create a comprehensive fallback data structure
    const fallbackData = {
        indices: {
            "^DJI": { price: "34,500.00", change: "+0.00", percentChange: "+0.00%" },
            "^GSPC": { price: "4,500.00", change: "+0.00", percentChange: "+0.00%" },
            "^IXIC": { price: "14,000.00", change: "+0.00", percentChange: "+0.00%" }
        },
        predictions: {
            trend: "neutral",
            confidence: 50,
            details: "Market data unavailable. Using fallback display.",
            models: {
                ensemble: { prediction: "neutral", confidence: 50 },
                random_forest: { prediction: "neutral", confidence: 40 },
                gradient_boosting: { prediction: "neutral", confidence: 60 },
                neural_network: { prediction: "neutral", confidence: 55 }
            },
            history: [
                { date: "2025-03-01", prediction: "neutral", actual: "neutral" },
                { date: "2025-03-02", prediction: "neutral", actual: "neutral" },
                { date: "2025-03-03", prediction: "neutral", actual: "neutral" }
            ]
        },
        news_sentiment: {
            positive: [
                { entity: "Technology Sector", sentiment: 0.8 },
                { entity: "Consumer Staples", sentiment: 0.7 }
            ],
            negative: [
                { entity: "Energy Sector", sentiment: -0.6 },
                { entity: "Real Estate", sentiment: -0.5 }
            ],
            distribution: {
                very_positive: 10,
                positive: 30,
                neutral: 40,
                negative: 15,
                very_negative: 5
            },
            headlines: [
                { title: "Market Analysis Unavailable", url: "#", sentiment: 0 }
            ]
        },
        feature_importance: [
            { feature: "Previous Close", importance: 0.2 },
            { feature: "Volume", importance: 0.18 },
            { feature: "PE Ratio", importance: 0.15 },
            { feature: "50-Day MA", importance: 0.12 },
            { feature: "RSI", importance: 0.1 }
        ],
        portfolio_optimization: {
            max_sharpe: {
                weights: { AAPL: 0.2, MSFT: 0.2, AMZN: 0.2, GOOGL: 0.2, META: 0.2 },
                metrics: { expected_return: 0.1, volatility: 0.15, sharpe_ratio: 0.67 }
            },
            min_vol: {
                weights: { AAPL: 0.2, MSFT: 0.2, AMZN: 0.2, GOOGL: 0.2, META: 0.2 },
                metrics: { expected_return: 0.08, volatility: 0.12, sharpe_ratio: 0.67 }
            },
            risk_parity: {
                weights: { AAPL: 0.2, MSFT: 0.2, AMZN: 0.2, GOOGL: 0.2, META: 0.2 },
                metrics: { expected_return: 0.09, volatility: 0.14, sharpe_ratio: 0.64 }
            }
        },
        economic_indicators: {
            gdp: { value: "Data unavailable", change: "0%", trend: "neutral" },
            inflation: { value: "Data unavailable", change: "0%", trend: "neutral" },
            unemployment: { value: "Data unavailable", change: "0%", trend: "neutral" },
            interest_rate: { value: "Data unavailable", change: "0%", trend: "neutral" }
        },
        alerts: []
    };
    
    // Set fallback data
    dashboardState.data = fallbackData;
    dashboardState.lastUpdated = new Date();
    
    // Update components with fallback data
    updateAllComponents();
    
    console.log('Fallback data loaded successfully');
}

// New function to update loading state of UI components
function updateLoadingState(isLoading, isSuccess = null) {
    // Update debug badges
    const jsBadge = document.getElementById('debug-js-badge');
    const apiBadge = document.getElementById('debug-api-badge');
    const errorMessage = document.getElementById('debug-error-message');
    
    if (jsBadge) {
        jsBadge.className = 'badge rounded-pill me-2 loaded';
    }
    
    if (apiBadge) {
        if (isLoading) {
            apiBadge.className = 'badge rounded-pill me-3';
            apiBadge.textContent = 'API...';
        } else if (isSuccess === true) {
            apiBadge.className = 'badge rounded-pill me-3 loaded';
            apiBadge.textContent = 'API';
        } else if (isSuccess === false) {
            apiBadge.className = 'badge rounded-pill me-3 error';
            apiBadge.textContent = 'API';
        }
    }
    
    if (errorMessage) {
        if (isSuccess === false) {
            errorMessage.textContent = 'AI data failed to load. Using fallback display.';
            errorMessage.classList.remove('d-none');
        } else {
            errorMessage.classList.add('d-none');
        }
    }
    
    // Update timestamp
    const timestamp = document.getElementById('debug-timestamp');
    if (timestamp) {
        timestamp.textContent = new Date().toLocaleTimeString();
    }
}

// Show error message in the UI - Improved to show more user-friendly messages
function showErrorMessage(message) {
    const containers = [
        'market-indices-container',
        'market-prediction-container',
        'news-sentiment-container',
        'feature-importance-container',
        'portfolio-optimization-container',
        'economic-indicators-container',
        'alerts-container'
    ];
    
    // Display error in all containers that are in error state
    containers.forEach(id => {
        const container = document.getElementById(id);
        if (container && container.querySelector('.loader')) {
            container.innerHTML = `<div class="alert alert-warning">
                <i class="fas fa-exclamation-triangle me-2"></i>
                ${message}
                <button type="button" class="btn btn-sm btn-outline-secondary mt-2" onclick="retryLoadComponent('${id}')">
                    <i class="fas fa-sync-alt me-1"></i> Retry
                </button>
            </div>`;
        }
    });
}

// New function to retry loading a specific component
function retryLoadComponent(containerId) {
    console.log(`Retrying load for component: ${containerId}`);
    
    // Show loading state
    const container = document.getElementById(containerId);
    if (container) {
        container.innerHTML = '<div class="loader"></div> Retrying...';
    }
    
    // Attempt to reload dashboard data
    loadDashboardData();
}

// Update all dashboard components with current data
function updateAllComponents() {
    // Only update if we have data
    if (!dashboardState.data) {
        console.error('No data available for updating components');
        return;
    }
    
    console.log('Updating all dashboard components');
    
    try {
        // Update each component with try/catch blocks to prevent cascading failures
        try {
            console.log('Loading market indices...');
    loadMarketIndices();
        } catch (error) {
            console.error('Failed to load market indices:', error);
            displayErrorInContainer('market-indices-container', 'Failed to load market indices');
        }
        
        try {
            console.log('Updating market prediction...');
            updateMarketPrediction();
        } catch (error) {
            console.error('Failed to update market prediction:', error);
            displayErrorInContainer('market-prediction-container', 'Failed to load market prediction');
        }
        
        try {
            console.log('Updating news sentiment...');
            updateNewsSentiment();
        } catch (error) {
            console.error('Failed to update news sentiment:', error);
            displayErrorInContainer('news-sentiment-container', 'Failed to load news sentiment');
        }
        
        try {
            console.log('Updating feature importance...');
            updateFeatureImportance();
        } catch (error) {
            console.error('Failed to update feature importance:', error);
            displayErrorInContainer('feature-importance-container', 'Failed to load feature importance');
        }
        
        try {
            console.log('Updating portfolio optimization...');
            updatePortfolioOptimization();
        } catch (error) {
            console.error('Failed to update portfolio optimization:', error);
            displayErrorInContainer('portfolio-optimization-container', 'Failed to load portfolio optimization');
        }
        
        try {
            console.log('Updating economic indicators...');
            updateEconomicIndicators();
        } catch (error) {
            console.error('Failed to update economic indicators:', error);
            displayErrorInContainer('economic-indicators-container', 'Failed to load economic indicators');
        }
        
        try {
            console.log('Updating alert system...');
            updateAlertSystem();
        } catch (error) {
            console.error('Failed to update alert system:', error);
            displayErrorInContainer('alerts-container', 'Failed to load alert system');
        }
        
        try {
            console.log('Loading alternative data...');
            loadAlternativeData();
        } catch (error) {
            console.error('Failed to load alternative data:', error);
            // Handle alternative data containers individually
        }
    } catch (error) {
        console.error('Error updating dashboard components:', error);
    }
    
    // Update last updated timestamp
    updateLastUpdatedTimestamp();
    
    console.log('All dashboard components updated');
}

// Set up auto-refresh for dashboard data
function setupAutoRefresh(interval) {
    // Clear any existing interval
    if (dashboardState.refreshInterval) {
        clearInterval(dashboardState.refreshInterval);
    }
    
    // Set new interval
    dashboardState.refreshInterval = setInterval(() => {
        loadDashboardData();
    }, interval);
    
    console.log(`Auto-refresh set for every ${interval/1000} seconds`);
}

// Update the last updated timestamp display
function updateLastUpdatedTimestamp() {
    const timestamp = dashboardState.lastUpdated ? 
        `Last updated: ${dashboardState.lastUpdated.toLocaleString()}` : 
        'Not yet updated';
    
    // Add timestamp to each container
    const containers = document.querySelectorAll('.card-body');
    containers.forEach(container => {
        let timestampEl = container.querySelector('.last-updated');
        
        if (!timestampEl) {
            timestampEl = document.createElement('div');
            timestampEl.className = 'last-updated text-muted small mt-2 text-end';
            container.appendChild(timestampEl);
        }
        
        timestampEl.textContent = timestamp;
    });
}

// Market Indices - Load and render market indices
async function loadMarketIndices() {
    console.log('Loading market indices');
    
    // Find the container
    const container = document.getElementById('market-indices-container');
    if (!container) {
        console.error('CRITICAL: Market indices container not found');
        return;
    }
    
    try {
        // Show loading state if we don't have data yet
        if (!dashboardState.data) {
        container.innerHTML = '<div class="loader"></div> Loading market data...';
            return;
        }
        
        // Check if we have indices data
        if (!dashboardState.data || !dashboardState.data.indices) {
            console.error('No indices data found in API response');
            container.innerHTML = '<div class="alert alert-warning">No market data available</div>';
            return;
        }
        
        // Clear the container
        container.innerHTML = '';
        
        // Create and display index cards
        const indices = dashboardState.data.indices;
        console.log('Displaying indices:', Object.keys(indices));
        
        // Create row container with proper spacing for the cards
        const grid = document.createElement('div');
        grid.className = 'row row-cols-1 row-cols-sm-2 row-cols-lg-3 g-3';
        container.appendChild(grid);
        
        // Create card for each index
        for (const symbol in indices) {
            if (indices.hasOwnProperty(symbol)) {
                const indexData = indices[symbol];
                createIndexCard(grid, symbol, indexData);
            }
        }
        
        // Add last updated indicator
        const lastUpdatedEl = document.createElement('div');
        lastUpdatedEl.className = 'text-end mt-2 small text-muted';
        lastUpdatedEl.innerHTML = `Last updated: ${new Date().toLocaleTimeString()}`;
        container.appendChild(lastUpdatedEl);
        
    } catch (error) {
        console.error('Error displaying market indices:', error);
        container.innerHTML = `<div class="alert alert-warning">
            Failed to display market data: ${error.message}
        </div>`;
    }
}

// Create a card for a market index
function createIndexCard(container, symbol, data) {
    const col = document.createElement('div');
    col.className = 'col'; // Let the grid system handle sizing
    
    // Create the card with all needed information
    const changeValue = parseFloat(data.change);
    const changePercent = parseFloat(data.changePercent);
    const isPositive = changeValue >= 0;
    const changeColor = isPositive ? 'text-success' : 'text-danger';
    const changeIcon = isPositive ? 'fa-arrow-up' : 'fa-arrow-down';
    
    // Format the index name to be more readable
    const indexNames = {
        'SPX': 'S&P 500',
        'DJI': 'Dow Jones',
        'IXIC': 'NASDAQ',
        'VIX': 'VIX Volatility',
        'TNX': '10-Year Treasury'
    };
    
    const indexName = indexNames[symbol] || symbol;
    
    // Create card HTML with horizontal layout
    col.innerHTML = `
        <div class="card index-card">
            <div class="card-body p-2">
                <div class="d-flex justify-content-between align-items-center">
                    <h5 class="card-title mb-0">${indexName}</h5>
                    <h4 class="mb-0">${parseFloat(data.price).toLocaleString()}</h4>
                </div>
                <div class="d-flex justify-content-between align-items-center mt-2">
                    <div class="d-flex align-items-center">
                        <span class="text-muted small me-2">L: ${parseFloat(data.low).toLocaleString()}</span>
                        <span class="text-muted small">H: ${parseFloat(data.high).toLocaleString()}</span>
                    </div>
                    <div class="${changeColor}">
                        <i class="fas ${changeIcon}"></i>
                        ${Math.abs(changeValue).toLocaleString()} (${Math.abs(changePercent).toFixed(2)}%)
                    </div>
                </div>
                <div class="progress mt-2" style="height: 4px;">
                    <div class="progress-bar ${isPositive ? 'bg-success' : 'bg-danger'}" 
                         role="progressbar" 
                         style="width: ${calculateProgressPercentage(data.low, data.high, data.price)}%"
                         aria-valuenow="${data.price}" 
                         aria-valuemin="${data.low}" 
                         aria-valuemax="${data.high}">
                    </div>
                </div>
            </div>
        </div>
    `;
    
    container.appendChild(col);
}

// Calculate percentage position for progress bar
function calculateProgressPercentage(low, high, current) {
    low = parseFloat(low);
    high = parseFloat(high);
    current = parseFloat(current);
    
    // Handle edge cases
    if (low === high) return 50;
    if (current <= low) return 0;
    if (current >= high) return 100;
    
    // Calculate position
    return ((current - low) / (high - low)) * 100;
}

// Market Prediction - Update the market prediction panel
function updateMarketPrediction() {
    const container = document.getElementById('market-prediction-container');
    if (!container) return;
    
    try {
        // Handle missing data gracefully
        if (!dashboardState.data || !dashboardState.data.predictions) {
            console.log('No prediction data available, showing fallback');
            container.innerHTML = `
                <div class="alert alert-info">
                    <i class="fas fa-info-circle me-2"></i>
                    Market prediction data is currently unavailable. 
                    We're working to restore this feature.
                </div>
                <div class="fallback-prediction">
                    <p>Based on historical patterns, markets typically experience:</p>
                    <ul>
                        <li>Short-term volatility during earnings seasons</li>
                        <li>Sensitivity to economic reports and Federal Reserve announcements</li>
                        <li>Sector rotation based on economic cycles</li>
                    </ul>
                    <p>For personalized predictions, please check back soon.</p>
                </div>
            `;
            return;
        }
        
        console.log('Updating market prediction');
        
        // Check if we have data
        if (!dashboardState.data) {
            container.innerHTML = '<div class="loader"></div> Initializing AI models...';
            return;
        }
        
        // Get data based on selected model
        const model = dashboardState.selectedModel;
        
        // Get prediction confidence and other metrics
        const confidence = dashboardState.data.predictionConfidence || 50;
        const modelMetrics = dashboardState.data.modelMetrics?.[model] || {
            accuracy: 0,
            precision: 0,
            recall: 0,
            f1: 0
        };
        
        // Determine sentiment based on confidence
        let sentiment = 'neutral';
        let sentimentText = 'Neutral';
        
        if (confidence >= 70) {
            sentiment = 'bullish';
            sentimentText = 'Bullish';
        } else if (confidence >= 60) {
            sentiment = 'slightly-bullish';
            sentimentText = 'Slightly Bullish';
        } else if (confidence <= 30) {
            sentiment = 'bearish';
            sentimentText = 'Bearish';
        } else if (confidence <= 40) {
            sentiment = 'slightly-bearish';
            sentimentText = 'Slightly Bearish';
        }
        
        // Create dashboard HTML
        container.innerHTML = `
            <div class="row">
                <div class="col-md-6 text-center">
                    <h6 class="mb-3">S&P 500 Market Direction Prediction</h6>
                    <div class="gauge-container">
                        <div class="gauge-background"></div>
                        <div class="gauge-cover"></div>
                        <div class="gauge-needle" style="transform: rotate(${confidence-50}deg)"></div>
                        <div class="gauge-value">${confidence}%</div>
        </div>
                    <p class="sentiment-label ${sentiment}">
                        <strong>${sentimentText}</strong> 
                        <i class="fas ${sentiment.includes('bull') ? 'fa-arrow-up' : sentiment.includes('bear') ? 'fa-arrow-down' : 'fa-minus'}"></i>
                    </p>
            </div>
                <div class="col-md-6">
                    <h6 class="mb-3">Model Metrics</h6>
                    <div class="model-metrics">
                        <div class="row">
                            <div class="col-6">
                                <div class="metric-box">
                                    <div class="metric-title">Accuracy</div>
                                    <div class="metric-value">${(modelMetrics.accuracy * 100).toFixed(1)}%</div>
                                </div>
                            </div>
                            <div class="col-6">
                                <div class="metric-box">
                                    <div class="metric-title">Precision</div>
                                    <div class="metric-value">${(modelMetrics.precision * 100).toFixed(1)}%</div>
                                </div>
                            </div>
                        </div>
                        <div class="row mt-2">
                            <div class="col-6">
                                <div class="metric-box">
                                    <div class="metric-title">Recall</div>
                                    <div class="metric-value">${(modelMetrics.recall * 100).toFixed(1)}%</div>
                                </div>
                            </div>
                            <div class="col-6">
                                <div class="metric-box">
                                    <div class="metric-title">F1 Score</div>
                                    <div class="metric-value">${(modelMetrics.f1 * 100).toFixed(1)}%</div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            <div class="row mt-4">
                <div class="col-12">
                    <div class="d-flex justify-content-between align-items-center mb-2">
                        <h6 class="mb-0">Prediction History</h6>
                        <button class="btn btn-sm btn-outline-secondary prediction-history-toggle" id="toggle-prediction-history">
                            <i class="fas fa-chevron-up"></i>
                        </button>
                    </div>
                    <div class="prediction-history-container" id="prediction-history-container">
                        <canvas id="prediction-history-chart" height="80"></canvas>
                    </div>
            </div>
        </div>
    `;
    
        // Draw the prediction history chart
        drawPredictionHistoryChart();
        
        // Set up toggle for prediction history chart
        setupPredictionHistoryToggle();
    } catch (error) {
        console.error('Error updating market prediction:', error);
        container.innerHTML = `
            <div class="alert alert-warning">
                <i class="fas fa-exclamation-triangle me-2"></i>
                Unable to display market predictions. 
                <button type="button" class="btn btn-sm btn-outline-secondary mt-2" onclick="retryLoadComponent('market-prediction-container')">
                    <i class="fas fa-sync-alt me-1"></i> Retry
                </button>
            </div>
        `;
    }
}

// Setup toggle functionality for prediction history chart
function setupPredictionHistoryToggle() {
    const toggleBtn = document.getElementById('toggle-prediction-history');
    const container = document.getElementById('prediction-history-container');
    
    if (!toggleBtn || !container) return;
    
    toggleBtn.addEventListener('click', function() {
        container.classList.toggle('collapsed');
        
        // Update button icon
        const icon = toggleBtn.querySelector('i');
        if (container.classList.contains('collapsed')) {
            icon.classList.remove('fa-chevron-up');
            icon.classList.add('fa-chevron-down');
            toggleBtn.setAttribute('title', 'Expand chart');
        } else {
            icon.classList.remove('fa-chevron-down');
            icon.classList.add('fa-chevron-up');
            toggleBtn.setAttribute('title', 'Collapse chart');
        }
    });
    
    // Start collapsed by default to improve initial page performance
    container.classList.add('collapsed');
    const icon = toggleBtn.querySelector('i');
    icon.classList.remove('fa-chevron-up');
    icon.classList.add('fa-chevron-down');
    toggleBtn.setAttribute('title', 'Expand chart');
}

// Draw prediction history chart using Chart.js
function drawPredictionHistoryChart() {
    if (!dashboardState.data || !dashboardState.data.predictionHistory) return;
    
    const chartCanvas = document.getElementById('prediction-history-chart');
    if (!chartCanvas) return;
    
    const ctx = chartCanvas.getContext('2d');
    const history = dashboardState.data.predictionHistory;
    
    // Performance optimization: limit the number of data points
    const maxDataPoints = 20; // Limit to 20 data points
    let dates = history.dates;
    let actual = history.actual;
    let predicted = history.predicted;
    
    // If we have more than maxDataPoints, only use the most recent ones
    if (dates.length > maxDataPoints) {
        dates = dates.slice(-maxDataPoints);
        actual = actual.slice(-maxDataPoints);
        predicted = predicted.slice(-maxDataPoints);
    }
    
    // Create new chart
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: dates,
            datasets: [
                {
                    label: 'Actual',
                    data: actual.map(val => val ? 1 : 0),
                    borderColor: 'rgba(54, 162, 235, 1)',
                    backgroundColor: 'rgba(54, 162, 235, 0.2)',
                    borderWidth: 2,
                    pointRadius: 3, // Smaller points
                    pointBackgroundColor: 'rgba(54, 162, 235, 1)',
                    tension: 0.1
                },
                {
                    label: 'Predicted',
                    data: predicted.map(val => val ? 1 : 0),
                    borderColor: 'rgba(255, 99, 132, 1)',
                    backgroundColor: 'rgba(255, 99, 132, 0.2)',
                    borderWidth: 2,
                    borderDash: [5, 5],
                    pointRadius: 3, // Smaller points
                    pointBackgroundColor: 'rgba(255, 99, 132, 1)',
                    tension: 0.1
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: {
                duration: 500 // Faster animations
            },
            scales: {
                y: {
                    ticks: {
                        callback: function(value) {
                            return value === 1 ? 'Up' : 'Down';
                        }
                    },
                    min: -0.1,
                    max: 1.1
                },
                x: {
                    ticks: {
                        maxRotation: 0, // Don't rotate x-axis labels
                        autoSkip: true, // Skip labels that don't fit
                        maxTicksLimit: 10 // Limit the number of ticks shown
                    }
                }
            },
            plugins: {
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const value = context.raw;
                            return `${context.dataset.label}: ${value === 1 ? 'Up' : 'Down'}`;
                        }
                    }
                },
                legend: {
                    labels: {
                        boxWidth: 10 // Smaller legend icons
                    }
                }
            }
        }
    });
}

// Optional: Set up tab navigation if it exists
function setupTabNavigation() {
    const tabLinks = document.querySelectorAll('.nav-tabs .nav-link');
    if (tabLinks.length === 0) {
        console.log('No tab navigation present');
        return;
    }
    
    tabLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            
            // Remove active class from all tabs
            tabLinks.forEach(l => l.classList.remove('active'));
            
            // Add active class to clicked tab
            this.classList.add('active');
            
            // Show the corresponding tab content
            const target = this.getAttribute('href');
            const tabContents = document.querySelectorAll('.tab-content .tab-pane');
            
            tabContents.forEach(content => {
                content.classList.remove('active', 'show');
                if (`#${content.id}` === target) {
                    content.classList.add('active', 'show');
                }
            });
        });
    });
}

// Optional: Set up theme toggle if it exists
function setupThemeToggle() {
    const themeToggle = document.getElementById('theme-toggle');
    if (!themeToggle) {
        console.log('No theme toggle present');
        return;
    }
    
    themeToggle.addEventListener('click', function() {
        document.body.classList.toggle('dark-theme');
        
        // Save preference to localStorage
        const isDarkTheme = document.body.classList.contains('dark-theme');
        localStorage.setItem('darkTheme', isDarkTheme ? 'true' : 'false');
        
        // Update toggle text
        this.textContent = isDarkTheme ? 'Light Mode' : 'Dark Mode';
    });
}

// News Sentiment Analysis - Update the news sentiment panel
function updateNewsSentiment() {
    const container = document.getElementById('news-sentiment-container');
    if (!container) return;
    
    try {
        if (!ensureDataProperty(dashboardState.data, 'news_sentiment', 'News Sentiment')) {
            throw new Error('News sentiment data missing or invalid');
        }
        
        const sentimentData = dashboardState.data.news_sentiment;
        
        // Check if we have the required properties
        if (!sentimentData.positive || !sentimentData.negative || !sentimentData.distribution) {
            throw new Error('News sentiment data incomplete');
        }
        
        // Build the sentiment display
        let html = '<div class="row">';
        
        // Positive sentiment column
        html += '<div class="col-md-6 mb-3">';
        html += '<h6>Most Positive</h6>';
        html += '<ul class="list-group">';
        
        sentimentData.positive.forEach(item => {
            const sentimentPercent = Math.round(item.sentiment * 100);
            html += `
                <li class="list-group-item d-flex justify-content-between align-items-center">
                    ${item.entity}
                    <span class="badge bg-success rounded-pill">${sentimentPercent}%</span>
                </li>
            `;
        });
        
        html += '</ul></div>';
        
        // Negative sentiment column
        html += '<div class="col-md-6 mb-3">';
        html += '<h6>Most Negative</h6>';
        html += '<ul class="list-group">';
        
        sentimentData.negative.forEach(item => {
            const sentimentPercent = Math.round(Math.abs(item.sentiment) * 100);
            html += `
                <li class="list-group-item d-flex justify-content-between align-items-center">
                    ${item.entity}
                    <span class="badge bg-danger rounded-pill">${sentimentPercent}%</span>
                </li>
            `;
        });
        
        html += '</ul></div>';
        
        // Create sentiment distribution visualization
        html += '<div class="col-12 mt-3">';
        html += '<h6>Sentiment Distribution</h6>';
        html += '<div class="d-flex">';
        
        const distribution = sentimentData.distribution;
        const total = distribution.very_positive + distribution.positive + 
                      distribution.neutral + distribution.negative + distribution.very_negative;
        
        if (total > 0) {
            const vpPct = Math.round((distribution.very_positive / total) * 100);
            const pPct = Math.round((distribution.positive / total) * 100);
            const neuPct = Math.round((distribution.neutral / total) * 100);
            const nPct = Math.round((distribution.negative / total) * 100);
            const vnPct = Math.round((distribution.very_negative / total) * 100);
            
            html += `
                <div class="progress flex-grow-1" style="height: 24px;">
                    <div class="progress-bar bg-success" style="width: ${vpPct}%" title="Very Positive: ${vpPct}%"></div>
                    <div class="progress-bar bg-info" style="width: ${pPct}%" title="Positive: ${pPct}%"></div>
                    <div class="progress-bar bg-secondary" style="width: ${neuPct}%" title="Neutral: ${neuPct}%"></div>
                    <div class="progress-bar bg-warning" style="width: ${nPct}%" title="Negative: ${nPct}%"></div>
                    <div class="progress-bar bg-danger" style="width: ${vnPct}%" title="Very Negative: ${vnPct}%"></div>
                </div>
            `;
        } else {
            html += '<div class="alert alert-info w-100">No sentiment distribution data available</div>';
        }
        
        html += '</div>'; // Close d-flex
        html += '</div>'; // Close col-12
        
        html += '</div>'; // Close row
        
        // Update the container
        container.innerHTML = html;
        
    } catch (error) {
        console.error('Error updating news sentiment:', error);
        displayErrorInContainer('news-sentiment-container', 'Unable to display news sentiment data');
    }
}

// Feature Importance - Update the feature importance chart
function updateFeatureImportance() {
    const container = document.getElementById('feature-importance-container');
    if (!container) return;
    
    try {
        if (!ensureDataProperty(dashboardState.data, 'feature_importance', 'Feature Importance')) {
            throw new Error('Feature importance data missing or invalid');
        }
        
        const features = dashboardState.data.feature_importance;
        if (!Array.isArray(features) || features.length === 0) {
            throw new Error('Feature importance data is empty or invalid format');
        }
        
        // Build the HTML for the feature list
        let html = '<ul class="list-group">';
        
        // Sort features by importance (highest first)
        const sortedFeatures = [...features].sort((a, b) => b.importance - a.importance);
        
        // Add each feature to the list
        sortedFeatures.forEach((feature, index) => {
            const importancePercent = Math.round(feature.importance * 100);
            const colorClass = index < 3 ? 'bg-primary' : 'bg-secondary';
            
            html += `
                <li class="list-group-item d-flex justify-content-between align-items-center">
                    <span>${index + 1}. ${feature.feature}</span>
                    <div class="d-flex align-items-center">
                        <div class="me-2">${importancePercent}%</div>
                        <div class="progress" style="width: 60px; height: 8px;">
                            <div class="progress-bar ${colorClass}" style="width: ${importancePercent}%"></div>
                        </div>
                    </div>
                </li>
            `;
        });
        
        html += '</ul>';
        
        // Update the container
        container.innerHTML = html;
        
    } catch (error) {
        console.error('Error updating feature importance:', error);
        displayErrorInContainer('feature-importance-container', 'Unable to display feature importance data');
    }
}

// Portfolio Optimization - Update the portfolio optimization panel
function updatePortfolioOptimization() {
    console.log('Updating portfolio optimization');
    
    const container = document.getElementById('portfolio-optimization-container');
    if (!container) {
        console.error('Portfolio optimization container not found');
        return;
    }
    
    // Check if we have data
    if (!dashboardState.data || !dashboardState.data.portfolioOptimization) {
        container.innerHTML = '<div class="loader"></div> Running portfolio optimization...';
        return;
    }
    
    // Get portfolio optimization data for selected strategy
    const strategy = dashboardState.selectedOptimizationStrategy;
    const portfolioData = dashboardState.data.portfolioOptimization?.[strategy] || {};
    const weights = portfolioData.weights || {};
    const stats = portfolioData.stats || {};
    
    // Create HTML for portfolio optimization
    let html = `
        <div class="row">
            <div class="col-md-7">
                <h6 class="mb-3">Optimized Asset Allocation</h6>
                <div class="chart-container">
                    <canvas id="portfolio-weights-chart" height="200"></canvas>
                </div>
            </div>
            <div class="col-md-5">
                <h6 class="mb-3">Portfolio Statistics</h6>
                <div class="stats-list">
                    <div class="stat-item">
                        <div class="stat-label">Expected Return</div>
                        <div class="stat-value">${(stats.expectedReturn * 100).toFixed(2)}%</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Annual Volatility</div>
                        <div class="stat-value">${(stats.volatility * 100).toFixed(2)}%</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Sharpe Ratio</div>
                        <div class="stat-value">${stats.sharpeRatio?.toFixed(2) || 'N/A'}</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Max Drawdown</div>
                        <div class="stat-value">${(stats.maxDrawdown * 100).toFixed(2)}%</div>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    // Set the HTML
    container.innerHTML = html;
    
    // Draw portfolio weights chart
    drawPortfolioChart(weights);
}

// Draw portfolio weights chart using Chart.js
function drawPortfolioChart(weights) {
    if (!weights || Object.keys(weights).length === 0) return;
    
    const chartCanvas = document.getElementById('portfolio-weights-chart');
    if (!chartCanvas) return;
    
    const ctx = chartCanvas.getContext('2d');
    
    // Extract labels and data
    const labels = Object.keys(weights);
    const data = Object.values(weights);
    
    // Generate colors
    const backgroundColors = generateChartColors(labels.length);
    
    // Create chart
    new Chart(ctx, {
        type: 'pie',
        data: {
            labels: labels,
            datasets: [{
                data: data.map(w => (w * 100).toFixed(1)),
                backgroundColor: backgroundColors,
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'right',
                    labels: {
                        boxWidth: 15,
                        padding: 10
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const value = context.raw;
                            return `${context.label}: ${value}%`;
                        }
                    }
                }
            }
        }
    });
}

// Generate colors for charts
function generateChartColors(count) {
    const baseColors = [
        'rgba(54, 162, 235, 0.8)',   // Blue
        'rgba(255, 99, 132, 0.8)',    // Red
        'rgba(75, 192, 192, 0.8)',    // Green
        'rgba(255, 159, 64, 0.8)',    // Orange
        'rgba(153, 102, 255, 0.8)',   // Purple
        'rgba(255, 205, 86, 0.8)',    // Yellow
        'rgba(201, 203, 207, 0.8)',   // Grey
        'rgba(99, 255, 132, 0.8)',    // Light Green
        'rgba(255, 99, 255, 0.8)',    // Pink
        'rgba(54, 235, 162, 0.8)'     // Teal
    ];
    
    // If we need more colors than our base set, generate them
    if (count <= baseColors.length) {
        return baseColors.slice(0, count);
    } else {
        const colors = [...baseColors];
        
        // Generate additional colors
        for (let i = baseColors.length; i < count; i++) {
            const hue = (i * 137) % 360; // Use golden ratio to spread colors
            colors.push(`hsla(${hue}, 70%, 60%, 0.8)`);
        }
        
        return colors;
    }
}

// Economic Indicators - Update the economic indicators panel
function updateEconomicIndicators() {
    console.log('Updating economic indicators');
    
    const container = document.getElementById('economic-indicators-container');
    if (!container) {
        console.error('Economic indicators container not found');
        return;
    }
    
    // Check if we have data
    if (!dashboardState.data || !dashboardState.data.economicIndicators) {
        container.innerHTML = '<div class="loader"></div> Loading economic indicators...';
        return;
    }
    
    // Get economic indicators data
    const indicators = dashboardState.data.economicIndicators || [];
    
    // Clear the container
    container.innerHTML = '';
    
    // If no indicators, show a message
    if (indicators.length === 0) {
        container.innerHTML = `
            <div class="alert alert-info">
                <i class="fas fa-info-circle"></i> Economic indicator data is not available.
            </div>
        `;
        return;
    }
    
    // Create a row container with proper spacing for the cards
    const grid = document.createElement('div');
    grid.className = 'row row-cols-1 row-cols-sm-2 row-cols-lg-3 g-3';
    container.appendChild(grid);
    
    // Add each indicator as a card
    indicators.forEach(indicator => {
        createEconomicIndicatorCard(grid, indicator);
    });
    
    // Add last updated indicator
    const lastUpdatedEl = document.createElement('div');
    lastUpdatedEl.className = 'text-end mt-2 small text-muted';
    lastUpdatedEl.innerHTML = `Last updated: ${new Date().toLocaleTimeString()}`;
    container.appendChild(lastUpdatedEl);
}

// Create a card for an economic indicator
function createEconomicIndicatorCard(container, indicator) {
    const col = document.createElement('div');
    col.className = 'col'; // Let the grid system handle sizing
    
    // Get previous value if available
    const previousValue = indicator.previous || indicator.previousValue;
    
    // Determine if the value is positive or negative compared to previous
    const hasChange = previousValue !== undefined && previousValue !== null;
    const isPositive = hasChange ? (indicator.value > previousValue) : false;
    const changeColor = isPositive ? 'text-success' : 'text-danger';
    const changeIcon = isPositive ? 'fa-arrow-up' : 'fa-arrow-down';
    
    // Calculate change percentage if possible
    let changePercent = '';
    if (hasChange && previousValue !== 0) {
        const percentChange = ((indicator.value - previousValue) / Math.abs(previousValue)) * 100;
        changePercent = ` (${percentChange.toFixed(1)}%)`;
    }
    
    // Format display values
    const displayValue = typeof indicator.value === 'number' ? indicator.value.toFixed(2) : indicator.value;
    const displayPrevious = hasChange ? (typeof previousValue === 'number' ? previousValue.toFixed(2) : previousValue) : 'N/A';
    
    // Create indicator importance stars
    const importanceLevel = indicator.importance || 1;
    const importanceStars = Array(importanceLevel).fill('<i class="fas fa-star text-warning"></i>').join(' ');
    
    // Create card HTML with horizontal layout
    col.innerHTML = `
        <div class="card indicator-card">
            <div class="card-body p-2">
                <div class="d-flex justify-content-between align-items-center">
                    <h5 class="card-title mb-0" title="${indicator.name || indicator.label}">
                        ${indicator.name || indicator.label}
                    </h5>
                    <h4 class="mb-0">${displayValue}</h4>
                </div>
                <div class="d-flex justify-content-between align-items-center mt-2">
                    <div>
                        <small class="text-muted">Previous: ${displayPrevious}</small>
                        <div class="small">${importanceStars}</div>
                    </div>
                    ${hasChange ? `
                    <div class="${changeColor}">
                        <i class="fas ${changeIcon}"></i>
                        ${Math.abs(indicator.value - previousValue).toFixed(2)}${changePercent}
                    </div>
                    ` : ''}
                </div>
                <small class="text-muted d-block mt-1">${indicator.category || ''}</small>
            </div>
        </div>
    `;
    
    container.appendChild(col);
}

// Alert System - Update the alerts system panel
function updateAlertSystem() {
    console.log('Updating alert system');
    
    const container = document.getElementById('alerts-container');
    if (!container) {
        console.error('Alerts container not found');
        return;
    }
    
    // Check if we have data
    if (!dashboardState.data || !dashboardState.data.alerts) {
        container.innerHTML = '<div class="loader"></div> Loading alert configuration...';
        return;
    }
    
    // Get alerts data
    const alerts = dashboardState.data.alerts || [];
    
    // Create HTML structure
    if (alerts.length === 0) {
        container.innerHTML = `
            <div class="alert alert-info">
                <i class="fas fa-info-circle"></i> No alerts configured. Create an alert to get notified about market conditions.
            </div>
        `;
        return;
    }
    
    // Create HTML for alerts
    let html = `
        <div class="table-responsive">
            <table class="table table-hover">
                <thead>
                    <tr>
                        <th>Alert Name</th>
                        <th>Condition</th>
                        <th>Status</th>
                        <th>Last Triggered</th>
                        <th>Actions</th>
                    </tr>
                </thead>
                <tbody>
    `;
    
    // Add each alert
    alerts.forEach(alert => {
        const statusClass = alert.status === 'active' ? 'success' : 
                          alert.status === 'triggered' ? 'warning' : 'secondary';
        
        html += `
            <tr data-alert-id="${alert.id}">
                <td>
                    <div class="d-flex align-items-center">
                        <i class="fas fa-${alert.icon || 'bell'} me-2"></i>
                        ${alert.name}
                    </div>
                </td>
                <td>${alert.condition}</td>
                <td>
                    <span class="badge bg-${statusClass}">
                        ${alert.status.charAt(0).toUpperCase() + alert.status.slice(1)}
                    </span>
                </td>
                <td>${alert.lastTriggered || 'Never'}</td>
                <td>
                    <div class="btn-group btn-group-sm">
                        <button class="btn btn-outline-primary edit-alert-btn" data-alert-id="${alert.id}">
                            <i class="fas fa-edit"></i>
                        </button>
                        <button class="btn btn-outline-danger delete-alert-btn" data-alert-id="${alert.id}">
                            <i class="fas fa-trash"></i>
                        </button>
                    </div>
                </td>
            </tr>
        `;
    });
    
    html += `
                </tbody>
            </table>
        </div>
    `;
    
    // Set the HTML
    container.innerHTML = html;
    
    // Add event listeners to the edit and delete buttons
    const editButtons = container.querySelectorAll('.edit-alert-btn');
    editButtons.forEach(button => {
        button.addEventListener('click', function() {
            const alertId = this.getAttribute('data-alert-id');
            editAlert(alertId);
        });
    });
    
    const deleteButtons = container.querySelectorAll('.delete-alert-btn');
    deleteButtons.forEach(button => {
        button.addEventListener('click', function() {
            const alertId = this.getAttribute('data-alert-id');
            deleteAlert(alertId);
        });
    });
}

// Show alert creation modal
function showAlertCreationModal() {
    // Get the modal element
    const modal = document.getElementById('alert-modal');
    if (!modal) {
        console.error('Alert modal not found');
        return;
    }
    
    // Get the form element
    const form = document.getElementById('alert-form');
    if (!form) {
        console.error('Alert form not found');
        return;
    }
    
    // Reset the form
    form.reset();
    
    // Update the form with alert creation fields
    form.innerHTML = `
        <div class="mb-3">
            <label for="alert-name" class="form-label">Alert Name</label>
            <input type="text" class="form-control" id="alert-name" required>
        </div>
        <div class="mb-3">
            <label for="alert-type" class="form-label">Alert Type</label>
            <select class="form-select" id="alert-type" required>
                <option value="">Select Alert Type</option>
                <option value="price">Price Alert</option>
                <option value="technical">Technical Indicator</option>
                <option value="sentiment">Sentiment Alert</option>
                <option value="prediction">AI Prediction Alert</option>
            </select>
        </div>
        <div id="alert-conditions-container">
            <!-- Dynamic conditions will be inserted here based on alert type -->
        </div>
        <div class="mb-3">
            <label for="alert-notification" class="form-label">Notification Method</label>
            <select class="form-select" id="alert-notification" required>
                <option value="ui">Dashboard Only</option>
                <option value="email">Email</option>
                <option value="both">Both</option>
            </select>
        </div>
    `;
    
    // Add event listener to alert type selector
    const alertTypeSelector = document.getElementById('alert-type');
    if (alertTypeSelector) {
        alertTypeSelector.addEventListener('change', function() {
            updateAlertConditionsForm(this.value);
        });
    }
    
    // Update the modal title and save button text
    const modalTitle = modal.querySelector('.modal-title');
    if (modalTitle) {
        modalTitle.textContent = 'Create New AI Alert';
    }
    
    const saveButton = document.getElementById('save-alert-btn');
    if (saveButton) {
        saveButton.textContent = 'Create Alert';
    }
    
    // Show the modal
    const modalInstance = new bootstrap.Modal(modal);
    modalInstance.show();
}

// Update alert conditions form based on selected alert type
function updateAlertConditionsForm(alertType) {
    const container = document.getElementById('alert-conditions-container');
    if (!container) return;
    
    let html = '';
    
    switch (alertType) {
        case 'price':
            html = `
                <div class="mb-3">
                    <label for="alert-symbol" class="form-label">Symbol</label>
                    <input type="text" class="form-control" id="alert-symbol" placeholder="e.g. AAPL, SPX" required>
                </div>
                <div class="mb-3">
                    <label for="alert-condition" class="form-label">Condition</label>
                    <select class="form-select" id="alert-condition" required>
                        <option value="above">Above Price</option>
                        <option value="below">Below Price</option>
                        <option value="percent_change">Percent Change</option>
                    </select>
                </div>
                <div class="mb-3">
                    <label for="alert-value" class="form-label">Value</label>
                    <input type="number" step="0.01" class="form-control" id="alert-value" required>
                </div>
            `;
            break;
            
        case 'technical':
            html = `
                <div class="mb-3">
                    <label for="alert-symbol" class="form-label">Symbol</label>
                    <input type="text" class="form-control" id="alert-symbol" placeholder="e.g. AAPL, SPX" required>
                </div>
                <div class="mb-3">
                    <label for="alert-indicator" class="form-label">Indicator</label>
                    <select class="form-select" id="alert-indicator" required>
                        <option value="rsi">RSI</option>
                        <option value="macd">MACD</option>
                        <option value="sma">Moving Average Crossover</option>
                        <option value="bollinger">Bollinger Bands</option>
                    </select>
                </div>
                <div class="mb-3">
                    <label for="alert-condition" class="form-label">Condition</label>
                    <select class="form-select" id="alert-condition" required>
                        <option value="above">Above Value</option>
                        <option value="below">Below Value</option>
                        <option value="cross_above">Crosses Above</option>
                        <option value="cross_below">Crosses Below</option>
                    </select>
                </div>
                <div class="mb-3">
                    <label for="alert-value" class="form-label">Value</label>
                    <input type="number" step="0.01" class="form-control" id="alert-value" required>
                </div>
            `;
            break;
            
        case 'sentiment':
            html = `
                <div class="mb-3">
                    <label for="alert-source" class="form-label">Sentiment Source</label>
                    <select class="form-select" id="alert-source" required>
                        <option value="news">News Sentiment</option>
                        <option value="social">Social Media</option>
                        <option value="overall">Overall Market Sentiment</option>
                    </select>
                </div>
                <div class="mb-3">
                    <label for="alert-condition" class="form-label">Condition</label>
                    <select class="form-select" id="alert-condition" required>
                        <option value="above">Above Threshold</option>
                        <option value="below">Below Threshold</option>
                        <option value="change">Significant Change</option>
                    </select>
                </div>
                <div class="mb-3">
                    <label for="alert-value" class="form-label">Value</label>
                    <input type="number" step="0.01" min="-1" max="1" class="form-control" id="alert-value" required>
                </div>
            `;
            break;
            
        case 'prediction':
            html = `
                <div class="mb-3">
                    <label for="alert-model" class="form-label">AI Model</label>
                    <select class="form-select" id="alert-model" required>
                        <option value="ensemble">Ensemble Model</option>
                        <option value="random_forest">Random Forest</option>
                        <option value="gradient_boosting">Gradient Boosting</option>
                        <option value="neural_network">Neural Network</option>
                    </select>
                </div>
                <div class="mb-3">
                    <label for="alert-prediction" class="form-label">Prediction Type</label>
                    <select class="form-select" id="alert-prediction" required>
                        <option value="direction">Direction Prediction</option>
                        <option value="confidence">Confidence Level</option>
                        <option value="return">Return Prediction</option>
                    </select>
                </div>
                <div class="mb-3">
                    <label for="alert-condition" class="form-label">Condition</label>
                    <select class="form-select" id="alert-condition" required>
                        <option value="above">Above Threshold</option>
                        <option value="below">Below Threshold</option>
                        <option value="change">Changes Direction</option>
                    </select>
                </div>
                <div class="mb-3">
                    <label for="alert-value" class="form-label">Value</label>
                    <input type="number" step="0.01" class="form-control" id="alert-value" required>
                </div>
            `;
            break;
            
        default:
            html = `
                <div class="alert alert-info">
                    Please select an alert type to continue.
                </div>
            `;
    }
    
    container.innerHTML = html;
}

// Save a new alert
function saveNewAlert() {
    // Get the form and validate
    const form = document.getElementById('alert-form');
    if (!form) return;
    
    // Simple form validation
    const requiredInputs = form.querySelectorAll('[required]');
    let isValid = true;
    
    requiredInputs.forEach(input => {
        if (!input.value) {
            input.classList.add('is-invalid');
            isValid = false;
        } else {
            input.classList.remove('is-invalid');
        }
    });
    
    if (!isValid) {
        alert('Please fill in all required fields');
        return;
    }
    
    // Get form values
    const alertName = document.getElementById('alert-name')?.value;
    const alertType = document.getElementById('alert-type')?.value;
    const alertNotification = document.getElementById('alert-notification')?.value;
    
    // Additional fields based on alert type
    let alertData = {};
    
    switch (alertType) {
        case 'price':
            alertData = {
                symbol: document.getElementById('alert-symbol')?.value,
                condition: document.getElementById('alert-condition')?.value,
                value: document.getElementById('alert-value')?.value
            };
            break;
            
        case 'technical':
            alertData = {
                symbol: document.getElementById('alert-symbol')?.value,
                indicator: document.getElementById('alert-indicator')?.value,
                condition: document.getElementById('alert-condition')?.value,
                value: document.getElementById('alert-value')?.value
            };
            break;
            
        case 'sentiment':
            alertData = {
                source: document.getElementById('alert-source')?.value,
                condition: document.getElementById('alert-condition')?.value,
                value: document.getElementById('alert-value')?.value
            };
            break;
            
        case 'prediction':
            alertData = {
                model: document.getElementById('alert-model')?.value,
                prediction: document.getElementById('alert-prediction')?.value,
                condition: document.getElementById('alert-condition')?.value,
                value: document.getElementById('alert-value')?.value
            };
            break;
    }
    
    // Create the alert object
    const newAlert = {
        id: Date.now().toString(), // Generate a unique ID
        name: alertName,
        type: alertType,
        notification: alertNotification,
        status: 'active',
        lastTriggered: null,
        createdAt: new Date().toISOString(),
        ...alertData
    };
    
    console.log('New alert:', newAlert);
    
    // In a real application, we would submit this to the server
    // For now, we'll just update the UI
    if (!dashboardState.data.alerts) {
        dashboardState.data.alerts = [];
    }
    
    dashboardState.data.alerts.push(newAlert);
    
    // Close the modal
    const modal = document.getElementById('alert-modal');
    if (modal) {
        const modalInstance = bootstrap.Modal.getInstance(modal);
        if (modalInstance) {
            modalInstance.hide();
        }
    }
    
    // Update the alerts UI
    updateAlertSystem();
}

// Edit an existing alert
function editAlert(alertId) {
    console.log(`Editing alert ${alertId}`);
    // In a real implementation, we would populate the form with the alert data
    // and update it on the server when saved
}

// Delete an alert
function deleteAlert(alertId) {
    console.log(`Deleting alert ${alertId}`);
    
    // Confirm deletion
    if (!confirm('Are you sure you want to delete this alert?')) {
        return;
    }
    
    // Remove the alert from the data
    if (dashboardState.data?.alerts) {
        dashboardState.data.alerts = dashboardState.data.alerts.filter(alert => alert.id !== alertId);
    }
    
    // Update the UI
    updateAlertSystem();
}

// Update the loadMarketPredictions function to handle different model types
function loadMarketPredictions() {
    const container = document.getElementById('market-predictions');
    if (!container) return;
    
    container.innerHTML = '<div class="text-center my-5"><div class="spinner-border" role="status"></div><p class="mt-2">Loading predictions...</p></div>';
    
    // Determine which endpoint to use based on selected model
    const endpoint = dashboardState.selectedPredictionModel === 'transformer' 
        ? '/api/market/transformer-predictions'
        : '/api/market/predictions';
    
    fetch(endpoint)
        .then(response => response.json())
        .then(data => {
            if (!data || !data.predictions) {
                container.innerHTML = '<div class="alert alert-warning">No prediction data available.</div>';
                return;
            }
            
            // Update last generated timestamp
            const lastUpdated = new Date(data.generated_at);
            container.innerHTML = `
                <div class="mb-3">
                    <div class="d-flex justify-content-between align-items-center mb-3">
                        <h3 class="mb-0">Market Predictions</h3>
                        <div class="form-group mb-0">
                            <select id="prediction-model-selector" class="form-select form-select-sm">
                                <option value="ensemble" ${dashboardState.selectedPredictionModel === 'ensemble' ? 'selected' : ''}>Ensemble Model</option>
                                <option value="transformer" ${dashboardState.selectedPredictionModel === 'transformer' ? 'selected' : ''}>Transformer Model</option>
                            </select>
                        </div>
                    </div>
                    <p class="text-muted small">Generated: ${lastUpdated.toLocaleString()}</p>
                </div>
                <div class="row prediction-cards">
                    ${renderPredictionCards(data.predictions)}
                </div>
                <div class="mt-3 mb-2">
                    <p class="text-muted small">
                        <i class="fas fa-info-circle me-1"></i>
                        ${getModelDescription(dashboardState.selectedPredictionModel)}
                    </p>
                </div>
            `;
            
            // Re-attach event listener for the prediction model selector
            const modelSelector = document.getElementById('prediction-model-selector');
            if (modelSelector) {
                modelSelector.addEventListener('change', function() {
                    dashboardState.selectedPredictionModel = this.value;
                    loadMarketPredictions();
                });
            }
        })
        .catch(error => {
            console.error('Error loading market predictions:', error);
            container.innerHTML = `<div class="alert alert-danger">Error loading predictions: ${error.message}</div>`;
        });
}

// Helper function to get model description
function getModelDescription(modelType) {
    if (modelType === 'transformer') {
        return 'Transformer model uses multi-headed attention mechanisms to capture complex temporal patterns in market data.';
    } else {
        return 'Ensemble model combines multiple algorithms to produce more stable and accurate predictions.';
    }
}

// Update the renderPredictionCards function to handle different model types
function renderPredictionCards(predictions) {
    if (!predictions || Object.keys(predictions).length === 0) {
        return '<div class="col-12"><div class="alert alert-warning">No predictions available.</div></div>';
    }
    
    let html = '';
    for (const [key, prediction] of Object.entries(predictions)) {
        // Determine confidence display based on model type
        let confidenceHtml = '';
        if (dashboardState.selectedPredictionModel === 'transformer') {
            confidenceHtml = `
                <div class="confidence-meter">
                    <div class="confidence-label">Confidence:</div>
                    <div class="progress" style="height: 6px;">
                        <div class="progress-bar ${getConfidenceClass(prediction.confidence)}" 
                            role="progressbar" 
                            style="width: ${prediction.confidence * 100}%" 
                            aria-valuenow="${prediction.confidence * 100}" 
                            aria-valuemin="0" 
                            aria-valuemax="100"></div>
                    </div>
                    <span class="confidence-value">${Math.round(prediction.confidence * 100)}%</span>
                </div>
                <div class="model-badge">
                    <span class="badge bg-primary">Transformer</span>
                </div>
            `;
        } else {
            confidenceHtml = `
                <div class="confidence-meter">
                    <div class="confidence-label">Confidence:</div>
                    <div class="progress" style="height: 6px;">
                        <div class="progress-bar ${getConfidenceClass(prediction.confidence)}" 
                            role="progressbar" 
                            style="width: ${prediction.confidence * 100}%" 
                            aria-valuenow="${prediction.confidence * 100}" 
                            aria-valuemin="0" 
                            aria-valuemax="100"></div>
                    </div>
                    <span class="confidence-value">${Math.round(prediction.confidence * 100)}%</span>
                </div>
            `;
        }
        
        // Create prediction card
        html += `
            <div class="col-md-6 col-lg-4 mb-3">
                <div class="card prediction-card h-100">
                    <div class="card-body">
                        <h5 class="card-title">
                            ${prediction.symbol}
                            <span class="prediction-direction-badge ${prediction.direction === 'up' ? 'up' : 'down'}">
                                <i class="fas fa-arrow-${prediction.direction}"></i>
                                ${prediction.magnitude.toFixed(2)}%
                            </span>
                        </h5>
                        <div class="prediction-details">
                            <div class="current-price">
                                Current: $${prediction.latest_close.toFixed(2)}
                                <span class="prediction-date">${prediction.latest_date}</span>
                            </div>
                            <div class="predicted-price">
                                Predicted: $${prediction.predicted_prices[0].toFixed(2)}
                                <span class="prediction-date">${prediction.prediction_dates[0]}</span>
                            </div>
                            ${confidenceHtml}
                        </div>
                    </div>
                </div>
            </div>
        `;
    }
    
    return html;
}

// Helper function for confidence class
function getConfidenceClass(confidence) {
    if (confidence >= 0.8) return 'bg-success';
    if (confidence >= 0.6) return 'bg-info';
    if (confidence >= 0.4) return 'bg-warning';
    return 'bg-danger';
}

// Alternative Data Sources Functions
function loadAlternativeData() {
    try {
        console.log('Loading alternative data components');
        
        // Load each alternative data component
        loadNewsSentiment();
        loadRedditSentiment();
        loadRetailSatelliteData();
        loadAgriculturalSatelliteData();
    } catch (error) {
        console.error('Error loading alternative data:', error);
    }
}

// Load and render news sentiment data
function loadNewsSentiment() {
    console.log('Loading news sentiment data');
    
    try {
        fetch('/api/alternative-data/news-sentiment')
            .then(response => {
                if (!response.ok) {
                    throw new Error(`API Error: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                console.log('News sentiment data received:', data);
                
                // Process and normalize data if needed
                const processedData = processNewsSentimentData(data);
                
                updateNewsSentimentTab(processedData);
            })
            .catch(error => {
                console.error('Error loading news sentiment data:', error);
                displayErrorInContainer('positive-sentiment-entities', 'Failed to load sentiment data');
                displayErrorInContainer('negative-sentiment-entities', 'Failed to load sentiment data');
                displayErrorInContainer('sentiment-distribution-chart', 'Failed to load sentiment chart');
                displayErrorInContainer('recent-headlines', 'Failed to load recent headlines');
            });
    } catch (error) {
        console.error('Error in loadNewsSentiment:', error);
    }
}

// Process news sentiment data to ensure it's in the expected format
function processNewsSentimentData(data) {
    if (!data || !data.entities) {
        console.error('Invalid news sentiment data format:', data);
        return {
            positive: [],
            negative: [],
            distribution: { very_positive: 0, positive: 0, neutral: 0, negative: 0, very_negative: 0 },
            headlines: []
        };
    }
    
    // Extract positive and negative entities
    const positive = [];
    const negative = [];
    const headlines = [];
    
    // Process entity data from the API endpoint
    Object.entries(data.entities).forEach(([symbol, entity]) => {
        if (entity.avg_sentiment > 0) {
            positive.push({
                entity: symbol, 
                sentiment: entity.avg_sentiment
            });
        } else if (entity.avg_sentiment < 0) {
            negative.push({
                entity: symbol, 
                sentiment: entity.avg_sentiment
            });
        }
        
        // Process headlines
        if (entity.recent_headlines && entity.recent_headlines.length > 0) {
            entity.recent_headlines.forEach(headline => {
                headlines.push({
                    ...headline,
                    symbol: symbol
                });
            });
        }
    });
    
    // Sort by sentiment (absolute value for negative)
    positive.sort((a, b) => b.sentiment - a.sentiment);
    negative.sort((a, b) => a.sentiment - b.sentiment);
    
    // Create distribution data
    const distribution = {
        very_positive: 0,
        positive: 0,
        neutral: 0,
        negative: 0,
        very_negative: 0
    };
    
    // Aggregate sentiment distribution
    Object.values(data.entities).forEach(entity => {
        distribution.very_positive += entity.very_positive || 0;
        distribution.positive += entity.positive || 0;
        distribution.neutral += entity.neutral || 0;
        distribution.negative += entity.negative || 0;
        distribution.very_negative += entity.very_negative || 0;
    });
    
    return {
        positive: positive.slice(0, 5),
        negative: negative.slice(0, 5),
        distribution,
        headlines: headlines.sort((a, b) => new Date(b.date) - new Date(a.date)).slice(0, 5)
    };
}

// Update the news sentiment tab with data
function updateNewsSentimentTab(data) {
    try {
        // Update timestamp
        const timestamp = document.getElementById('news-sentiment-updated');
        if (timestamp) {
            timestamp.textContent = `Last updated: ${formatDate(new Date(data.timestamp))}`;
        }
        
        // Update positive sentiment entities
        const positiveContainer = document.getElementById('positive-sentiment-entities');
        if (positiveContainer) {
            let html = '<ul class="list-group">';
            let entities = Object.entries(data.entities || {})
                .filter(([_, entity]) => entity.avg_sentiment > 0)
                .sort((a, b) => b[1].avg_sentiment - a[1].avg_sentiment)
                .slice(0, 5);
            
            if (entities.length > 0) {
                entities.forEach(([symbol, entity]) => {
                    const sentiment = Math.round(entity.avg_sentiment * 100);
                    html += `
                        <li class="list-group-item d-flex justify-content-between align-items-center">
                            ${symbol}
                            <span class="badge bg-success rounded-pill">${sentiment}%</span>
                        </li>
                    `;
                });
            } else {
                html += `
                    <li class="list-group-item text-center text-muted">
                        No positive sentiment data available
                    </li>
                `;
            }
            
            html += '</ul>';
            positiveContainer.innerHTML = html;
        }
        
        // Update negative sentiment entities
        const negativeContainer = document.getElementById('negative-sentiment-entities');
        if (negativeContainer) {
            let html = '<ul class="list-group">';
            let entities = Object.entries(data.entities || {})
                .filter(([_, entity]) => entity.avg_sentiment < 0)
                .sort((a, b) => a[1].avg_sentiment - b[1].avg_sentiment)
                .slice(0, 5);
            
            if (entities.length > 0) {
                entities.forEach(([symbol, entity]) => {
                    const sentiment = Math.round(Math.abs(entity.avg_sentiment) * 100);
                    html += `
                        <li class="list-group-item d-flex justify-content-between align-items-center">
                            ${symbol}
                            <span class="badge bg-danger rounded-pill">${sentiment}%</span>
                        </li>
                    `;
                });
            } else {
                html += `
                    <li class="list-group-item text-center text-muted">
                        No negative sentiment data available
                    </li>
                `;
            }
            
            html += '</ul>';
            negativeContainer.innerHTML = html;
        }
        
        // Update sentiment distribution chart
        const chartContainer = document.getElementById('sentiment-distribution-chart');
        if (chartContainer) {
            // Create distribution data
            const labels = ['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive'];
            
            // Get total counts for each sentiment category across all entities
            let totals = {
                very_negative: 0,
                negative: 0,
                neutral: 0,
                positive: 0,
                very_positive: 0
            };
            
            Object.values(data.entities || {}).forEach(entity => {
                totals.very_negative += entity.very_negative || 0;
                totals.negative += entity.negative || 0;
                totals.neutral += entity.neutral || 0;
                totals.positive += entity.positive || 0;
                totals.very_positive += entity.very_positive || 0;
            });
            
            const values = [
                totals.very_negative,
                totals.negative,
                totals.neutral,
                totals.positive,
                totals.very_positive
            ];
            
            // If we have no data, show a message
            if (values.reduce((a, b) => a + b, 0) === 0) {
                chartContainer.innerHTML = `
                    <div class="alert alert-info">
                        No sentiment distribution data available
                    </div>
                `;
                return;
            }
            
            // Create chart config
            const config = {
                type: 'doughnut',
                data: {
                    labels: labels,
                    datasets: [{
                        data: values,
                        backgroundColor: [
                            '#dc3545', // very negative - red
                            '#fd7e14', // negative - orange
                            '#6c757d', // neutral - gray
                            '#20c997', // positive - teal
                            '#198754'  // very positive - green
                        ]
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: 'bottom',
                            labels: {
                                boxWidth: 12
                            }
                        }
                    }
                }
            };
            
            // Create chart safely
            safelyCreateChart(chartContainer, config, "Sentiment distribution chart could not be loaded");
        }
        
        // Update recent headlines
        const headlinesContainer = document.getElementById('recent-headlines');
        if (headlinesContainer) {
            let html = '<ul class="list-group">';
            let headlines = [];
            
            // Collect headlines from all entities
            Object.entries(data.entities || {}).forEach(([symbol, entity]) => {
                (entity.recent_headlines || []).forEach(headline => {
                    headlines.push({
                        ...headline,
                        symbol
                    });
                });
            });
            
            // Sort by recency
            headlines.sort((a, b) => new Date(b.date || 0) - new Date(a.date || 0));
            headlines = headlines.slice(0, 5);
            
            if (headlines.length > 0) {
                headlines.forEach(headline => {
                    const sentimentValue = headline.sentiment || 0;
                    let sentimentClass = 'secondary';
                    let sentimentLabel = 'Neutral';
                    
                    if (sentimentValue >= 0.3) {
                        sentimentClass = 'success';
                        sentimentLabel = 'Positive';
                    } else if (sentimentValue <= -0.3) {
                        sentimentClass = 'danger';
                        sentimentLabel = 'Negative';
                    }
                    
                    html += `
                        <li class="list-group-item">
                            <div class="d-flex justify-content-between align-items-start">
                                <div class="ms-2 me-auto">
                                    <div>
                                        <a href="${headline.url || '#'}" target="_blank" class="text-decoration-none">
                                            ${headline.title || 'Untitled Article'}
                                        </a>
                                    </div>
                                    <small class="text-muted">
                                        ${formatDate(new Date(headline.date || Date.now()))} - ${headline.symbol}
                                    </small>
                                </div>
                                <span class="badge bg-${sentimentClass} rounded-pill">${sentimentLabel}</span>
                            </div>
                        </li>
                    `;
                });
            } else {
                html += `
                    <li class="list-group-item text-center text-muted">
                        No recent headlines available
                    </li>
                `;
            }
            
            html += '</ul>';
            headlinesContainer.innerHTML = html;
        }
    } catch (error) {
        console.error('Error updating news sentiment tab:', error);
    }
}

// Reddit Sentiment Functions
function loadRedditSentiment() {
    fetch('/api/alternative-data/reddit-sentiment')
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            updateRedditSentiment(data);
        })
        .catch(error => {
            console.error('Error fetching Reddit sentiment data:', error);
            displayErrorInContainer('top-reddit-entities', 'Error loading Reddit data');
            displayErrorInContainer('subreddit-sentiment', 'Error loading Reddit data');
            displayErrorInContainer('reddit-sentiment-chart', 'Error loading sentiment chart');
            displayErrorInContainer('top-reddit-posts', 'Error loading Reddit posts');
        });
}

function updateRedditSentiment(data) {
    // Update timestamp
    document.getElementById('reddit-sentiment-updated').textContent = `Last updated: ${formatDate(new Date())}`;
    
    // Clear placeholders
    document.getElementById('top-reddit-entities').innerHTML = '';
    document.getElementById('subreddit-sentiment').innerHTML = '';
    document.getElementById('top-reddit-posts').innerHTML = '';
    
    // Display most mentioned entities
    const entitiesContainer = document.getElementById('top-reddit-entities');
    if (!data.top_entities || data.top_entities.length === 0) {
        entitiesContainer.innerHTML = '<div class="text-muted">No entities found</div>';
    } else {
        data.top_entities.slice(0, 5).forEach(entity => {
            let sentimentClass = 'neutral';
            if (entity.sentiment > 0.05) sentimentClass = 'positive';
            if (entity.sentiment < -0.05) sentimentClass = 'negative';
            
            const sentimentPercentage = Math.min(Math.abs(entity.sentiment) * 100, 100).toFixed(0);
            
            const entityElement = document.createElement('div');
            entityElement.className = `sentiment-entity-item ${sentimentClass}`;
            entityElement.innerHTML = `
                <div class="d-flex justify-content-between align-items-center">
                    <strong>${entity.name}</strong>
                    <span class="badge bg-${sentimentClass === 'positive' ? 'success' : sentimentClass === 'negative' ? 'danger' : 'secondary'}">${sentimentPercentage}%</span>
                </div>
                <div class="small text-muted">${entity.mentions} mentions</div>
                <div class="sentiment-bar">
                    <div class="sentiment-bar-fill sentiment-${sentimentClass}" style="width: ${sentimentPercentage}%"></div>
                </div>
            `;
            entitiesContainer.appendChild(entityElement);
        });
    }
    
    // Display subreddit sentiment
    const subredditContainer = document.getElementById('subreddit-sentiment');
    if (!data.subreddit_sentiment || Object.keys(data.subreddit_sentiment).length === 0) {
        subredditContainer.innerHTML = '<div class="text-muted">No subreddit data found</div>';
    } else {
        Object.entries(data.subreddit_sentiment).slice(0, 5).forEach(([subreddit, sentiment]) => {
            let sentimentClass = 'neutral';
            if (sentiment > 0.05) sentimentClass = 'positive';
            if (sentiment < -0.05) sentimentClass = 'negative';
            
            const sentimentPercentage = Math.min(Math.abs(sentiment) * 100, 100).toFixed(0);
            
            const subredditElement = document.createElement('div');
            subredditElement.className = `sentiment-entity-item ${sentimentClass}`;
            subredditElement.innerHTML = `
                <div class="d-flex justify-content-between align-items-center">
                    <strong>r/${subreddit}</strong>
                    <span class="badge bg-${sentimentClass === 'positive' ? 'success' : sentimentClass === 'negative' ? 'danger' : 'secondary'}">${sentimentPercentage}%</span>
                </div>
                <div class="sentiment-bar">
                    <div class="sentiment-bar-fill sentiment-${sentimentClass}" style="width: ${sentimentPercentage}%"></div>
                </div>
            `;
            subredditContainer.appendChild(subredditElement);
        });
    }
    
    // Display top posts
    const postsContainer = document.getElementById('top-reddit-posts');
    if (!data.top_posts || data.top_posts.length === 0) {
        postsContainer.innerHTML = '<div class="text-muted">No posts found</div>';
    } else {
        data.top_posts.slice(0, 5).forEach(post => {
            let sentimentClass = 'neutral';
            if (post.sentiment > 0.05) sentimentClass = 'positive';
            if (post.sentiment < -0.05) sentimentClass = 'negative';
            
            const postElement = document.createElement('div');
            postElement.className = 'reddit-post';
            postElement.innerHTML = `
                <div class="d-flex justify-content-between align-items-start">
                    <div>
                        <div>${post.title}</div>
                        <div class="post-stats">r/${post.subreddit} · ${post.upvotes} upvotes · ${formatRelativeTime(new Date(post.date))}</div>
                    </div>
                    <span class="badge bg-${sentimentClass === 'positive' ? 'success' : sentimentClass === 'negative' ? 'danger' : 'secondary'} ms-2">
                        ${sentimentClass}
                    </span>
                </div>
            `;
            postsContainer.appendChild(postElement);
        });
    }
    
    // Create Reddit sentiment chart
    createRedditSentimentChart(data);
}

function createRedditSentimentChart(data) {
    // Extract sentiment data
    let positiveCount = 0;
    let neutralCount = 0;
    let negativeCount = 0;
    
    if (data.top_posts) {
        data.top_posts.forEach(post => {
            if (post.sentiment > 0.05) positiveCount++;
            else if (post.sentiment < -0.05) negativeCount++;
            else neutralCount++;
        });
    }
    
    // Create chart using Chart.js
    const ctx = document.getElementById('reddit-sentiment-chart').getContext('2d');
    
    // Destroy existing chart if it exists
    if (window.redditSentimentChart) {
        window.redditSentimentChart.destroy();
    }
    
    window.redditSentimentChart = new Chart(ctx, {
        type: 'pie',
        data: {
            labels: ['Positive', 'Neutral', 'Negative'],
            datasets: [{
                data: [positiveCount, neutralCount, negativeCount],
                backgroundColor: ['#28a745', '#6c757d', '#dc3545'],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom'
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const value = context.raw;
                            const total = context.dataset.data.reduce((a, b) => a + b, 0);
                            const percentage = total > 0 ? Math.round((value / total) * 100) : 0;
                            return `${context.label}: ${value} (${percentage}%)`;
                        }
                    }
                }
            }
        }
    });
}

// Retail Satellite Data Functions
function loadRetailSatelliteData() {
    fetch('/api/alternative-data/retail-satellite')
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            updateRetailSatelliteData(data);
        })
        .catch(error => {
            console.error('Error fetching retail satellite data:', error);
            displayErrorInContainer('high-traffic-locations', 'Error loading retail traffic data');
            displayErrorInContainer('retail-stock-impact', 'Error loading stock impact data');
            displayErrorInContainer('retail-traffic-chart', 'Error loading traffic chart');
        });
}

function updateRetailSatelliteData(data) {
    // Update timestamp
    document.getElementById('retail-satellite-updated').textContent = `Last updated: ${formatDate(new Date())}`;
    
    // Clear placeholders
    document.getElementById('high-traffic-locations').innerHTML = '';
    document.getElementById('retail-stock-impact').innerHTML = '';
    
    // Display high traffic locations
    const locationsContainer = document.getElementById('high-traffic-locations');
    if (!data.locations || data.locations.length === 0) {
        locationsContainer.innerHTML = '<div class="text-muted">No traffic data found</div>';
    } else {
        // Sort by traffic
        const sortedLocations = [...data.locations].sort((a, b) => b.traffic_change - a.traffic_change);
        
        sortedLocations.slice(0, 5).forEach(location => {
            let trafficClass = 'traffic-medium';
            if (location.traffic_change > 15) trafficClass = 'traffic-high';
            if (location.traffic_change < 0) trafficClass = 'traffic-low';
            
            const trafficPercentage = location.traffic_change.toFixed(1);
            const sign = location.traffic_change > 0 ? '+' : '';
            
            const locationElement = document.createElement('div');
            locationElement.className = `traffic-location ${trafficClass}`;
            locationElement.innerHTML = `
                <div class="d-flex justify-content-between align-items-center">
                    <strong>${location.retailer} - ${location.location}</strong>
                    <span class="badge ${location.traffic_change > 0 ? 'bg-success' : location.traffic_change < 0 ? 'bg-danger' : 'bg-warning'}">
                        ${sign}${trafficPercentage}%
                    </span>
                </div>
                <div class="small text-muted">Traffic Level: ${location.cars_detected} vehicles</div>
            `;
            locationsContainer.appendChild(locationElement);
        });
    }
    
    // Display stock impact
    const stockContainer = document.getElementById('retail-stock-impact');
    if (!data.stock_impact || data.stock_impact.length === 0) {
        stockContainer.innerHTML = '<div class="text-muted">No stock impact data found</div>';
    } else {
        // Sort by predicted impact
        const sortedStocks = [...data.stock_impact].sort((a, b) => Math.abs(b.predicted_impact) - Math.abs(a.predicted_impact));
        
        sortedStocks.slice(0, 5).forEach(stock => {
            const impactClass = stock.predicted_impact >= 0 ? 'price-increase' : 'price-decrease';
            const impactPercentage = stock.predicted_impact.toFixed(2);
            const sign = stock.predicted_impact > 0 ? '+' : '';
            
            const stockElement = document.createElement('div');
            stockElement.className = `price-impact ${impactClass}`;
            stockElement.innerHTML = `
                <div class="d-flex justify-content-between align-items-center">
                    <strong>${stock.symbol} (${stock.company})</strong>
                    <span class="badge ${stock.predicted_impact > 0 ? 'bg-success' : 'bg-danger'}">
                        ${sign}${impactPercentage}%
                    </span>
                </div>
                <div class="small text-muted">Based on ${stock.locations_analyzed} locations</div>
            `;
            stockContainer.appendChild(stockElement);
        });
    }
    
    // Create retail traffic chart
    createRetailTrafficChart(data);
}

function createRetailTrafficChart(data) {
    if (!data.historical_traffic || data.historical_traffic.length === 0) {
        displayErrorInContainer('retail-traffic-chart', 'No historical traffic data available');
        return;
    }
    
    // Prepare data for the chart
    const labels = data.historical_traffic.map(item => item.date);
    const trafficData = data.historical_traffic.map(item => item.traffic_index);
    
    // Create chart using Chart.js
    const ctx = document.getElementById('retail-traffic-chart').getContext('2d');
    
    // Destroy existing chart if it exists
    if (window.retailTrafficChart) {
        window.retailTrafficChart.destroy();
    }
    
    window.retailTrafficChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Retail Traffic Index',
                data: trafficData,
                backgroundColor: 'rgba(13, 110, 253, 0.2)',
                borderColor: 'rgba(13, 110, 253, 1)',
                borderWidth: 2,
                tension: 0.3,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    grid: {
                        display: false
                    },
                    ticks: {
                        maxTicksLimit: 8
                    }
                },
                y: {
                    beginAtZero: false,
                    grid: {
                        borderDash: [2]
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `Traffic Index: ${context.raw}`;
                        }
                    }
                }
            }
        }
    });
}

// Agricultural Satellite Data Functions
function loadAgriculturalSatelliteData() {
    fetch('/api/alternative-data/agricultural-satellite')
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            updateAgriculturalSatelliteData(data);
        })
        .catch(error => {
            console.error('Error fetching agricultural satellite data:', error);
            displayErrorInContainer('agricultural-yield-changes', 'Error loading yield data');
            displayErrorInContainer('agricultural-price-impact', 'Error loading price impact data');
            displayErrorInContainer('agricultural-yield-chart', 'Error loading yield chart');
        });
}

function updateAgriculturalSatelliteData(data) {
    // Update timestamp
    document.getElementById('agricultural-satellite-updated').textContent = `Last updated: ${formatDate(new Date())}`;
    
    // Clear placeholders
    document.getElementById('agricultural-yield-changes').innerHTML = '';
    document.getElementById('agricultural-price-impact').innerHTML = '';
    
    // Display yield changes
    const yieldContainer = document.getElementById('agricultural-yield-changes');
    if (!data.regions || data.regions.length === 0) {
        yieldContainer.innerHTML = '<div class="text-muted">No yield data found</div>';
    } else {
        // Sort by yield change
        const sortedRegions = [...data.regions].sort((a, b) => Math.abs(b.yield_change) - Math.abs(a.yield_change));
        
        sortedRegions.slice(0, 5).forEach(region => {
            const yieldClass = region.yield_change >= 0 ? 'yield-increase' : 'yield-decrease';
            const yieldPercentage = region.yield_change.toFixed(1);
            const sign = region.yield_change > 0 ? '+' : '';
            
            const regionElement = document.createElement('div');
            regionElement.className = `yield-change ${yieldClass}`;
            regionElement.innerHTML = `
                <div class="d-flex justify-content-between align-items-center">
                    <strong>${region.crop} - ${region.region}</strong>
                    <span class="badge ${region.yield_change > 0 ? 'bg-success' : 'bg-danger'}">
                        ${sign}${yieldPercentage}%
                    </span>
                </div>
                <div class="small text-muted">NDVI: ${region.vegetation_index.toFixed(2)}</div>
            `;
            yieldContainer.appendChild(regionElement);
        });
    }
    
    // Display price impact
    const priceContainer = document.getElementById('agricultural-price-impact');
    if (!data.price_impact || data.price_impact.length === 0) {
        priceContainer.innerHTML = '<div class="text-muted">No price impact data found</div>';
    } else {
        // Sort by price impact
        const sortedPrices = [...data.price_impact].sort((a, b) => Math.abs(b.predicted_impact) - Math.abs(a.predicted_impact));
        
        sortedPrices.slice(0, 5).forEach(price => {
            const priceClass = price.predicted_impact >= 0 ? 'price-increase' : 'price-decrease';
            const pricePercentage = price.predicted_impact.toFixed(2);
            const sign = price.predicted_impact > 0 ? '+' : '';
            
            const priceElement = document.createElement('div');
            priceElement.className = `price-impact ${priceClass}`;
            priceElement.innerHTML = `
                <div class="d-flex justify-content-between align-items-center">
                    <strong>${price.commodity}</strong>
                    <span class="badge ${price.predicted_impact > 0 ? 'bg-success' : 'bg-danger'}">
                        ${sign}${pricePercentage}%
                    </span>
                </div>
                <div class="small text-muted">Based on satellite data from ${price.regions_analyzed} regions</div>
            `;
            priceContainer.appendChild(priceElement);
        });
    }
    
    // Create agricultural yield chart
    createAgriculturalYieldChart(data);
}

function createAgriculturalYieldChart(data) {
    if (!data.historical_yields || data.historical_yields.length === 0) {
        displayErrorInContainer('agricultural-yield-chart', 'No historical yield data available');
        return;
    }
    
    // Prepare data for the chart
    const labels = data.historical_yields.map(item => item.date);
    
    // Unique crops
    const crops = [...new Set(data.historical_yields.map(item => item.crop))];
    
    // Create datasets
    const datasets = crops.map((crop, index) => {
        // Select color based on index
        const colors = [
            { bg: 'rgba(40, 167, 69, 0.2)', border: 'rgb(40, 167, 69)' },
            { bg: 'rgba(0, 123, 255, 0.2)', border: 'rgb(0, 123, 255)' },
            { bg: 'rgba(255, 193, 7, 0.2)', border: 'rgb(255, 193, 7)' },
            { bg: 'rgba(220, 53, 69, 0.2)', border: 'rgb(220, 53, 69)' },
            { bg: 'rgba(111, 66, 193, 0.2)', border: 'rgb(111, 66, 193)' }
        ];
        const colorSet = colors[index % colors.length];
        
        // Get yield data for this crop
        const cropData = labels.map(date => {
            const entry = data.historical_yields.find(item => item.date === date && item.crop === crop);
            return entry ? entry.yield_index : null;
        });
        
        return {
            label: crop,
            data: cropData,
            backgroundColor: colorSet.bg,
            borderColor: colorSet.border,
            borderWidth: 2,
            tension: 0.3
        };
    });
    
    // Create chart using Chart.js
    const ctx = document.getElementById('agricultural-yield-chart').getContext('2d');
    
    // Destroy existing chart if it exists
    if (window.agriculturalYieldChart) {
        window.agriculturalYieldChart.destroy();
    }
    
    window.agriculturalYieldChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    grid: {
                        display: false
                    },
                    ticks: {
                        maxTicksLimit: 8
                    }
                },
                y: {
                    beginAtZero: false,
                    grid: {
                        borderDash: [2]
                    }
                }
            },
            plugins: {
                legend: {
                    position: 'bottom'
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `${context.dataset.label}: ${context.raw}`;
                        }
                    }
                }
            }
        }
    });
}

// Helper function to format date
function formatDate(date) {
    return new Intl.DateTimeFormat('en-US', {
        month: 'short',
        day: 'numeric',
        year: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    }).format(date);
}

// Helper function to format relative time
function formatRelativeTime(date) {
    const now = new Date();
    const diffInSeconds = Math.floor((now - date) / 1000);
    
    if (diffInSeconds < 60) {
        return `${diffInSeconds}s ago`;
    } else if (diffInSeconds < 3600) {
        return `${Math.floor(diffInSeconds / 60)}m ago`;
    } else if (diffInSeconds < 86400) {
        return `${Math.floor(diffInSeconds / 3600)}h ago`;
    } else {
        return `${Math.floor(diffInSeconds / 86400)}d ago`;
    }
}

// Helper function to display error messages
function displayErrorInContainer(containerId, message) {
    const container = document.getElementById(containerId);
    if (container) {
        container.innerHTML = `
            <div class="alert alert-warning">
                <i class="fas fa-exclamation-triangle me-2"></i>
                ${message}
                <button type="button" class="btn btn-sm btn-outline-secondary mt-2" onclick="retryLoadComponent('${containerId}')">
                    <i class="fas fa-sync-alt me-1"></i> Retry
                </button>
            </div>
            <div class="fallback-message text-muted small mt-3">
                <p>We're experiencing some technical difficulties loading this data.</p>
                <p>Our team has been notified and is working to resolve the issue.</p>
            </div>
        `;
    }
}

// Attach event listener to refresh button
document.addEventListener('DOMContentLoaded', function() {
    // Existing code...
    
    // Add Alternative Data refresh handler
    const refreshAltDataBtn = document.getElementById('refresh-alt-data');
    if (refreshAltDataBtn) {
        refreshAltDataBtn.addEventListener('click', function() {
            loadAlternativeData();
        });
    }
    
    // Load alternative data on page load
    loadAlternativeData();
});

// A utility function to check if a data property exists and log if it doesn't
function ensureDataProperty(data, property, componentName) {
    if (!data || typeof data !== 'object' || !data.hasOwnProperty(property)) {
        console.error(`Required property '${property}' missing for ${componentName}. Data:`, data);
        return false;
    }
    return true;
}

// Safely initialize a chart or return fallback content if Chart.js isn't available
function safelyCreateChart(chartContainer, chartConfig, fallbackMessage = "Chart could not be loaded") {
    if (!chartContainer) return false;
    
    try {
        // Check if Chart is available
        if (typeof Chart === 'undefined') {
            console.error('Chart.js is not available');
            chartContainer.innerHTML = `
                <div class="alert alert-warning">
                    <i class="fas fa-exclamation-triangle me-2"></i>
                    ${fallbackMessage}
                </div>
                <div class="fallback-content p-3 text-center text-muted">
                    <i class="fas fa-chart-line fa-3x mb-3"></i>
                    <p>Chart visualization library not loaded.</p>
                </div>
            `;
            return false;
        }
        
        // Create canvas if needed
        let canvas = chartContainer.querySelector('canvas');
        if (!canvas) {
            canvas = document.createElement('canvas');
            chartContainer.innerHTML = '';
            chartContainer.appendChild(canvas);
        }
        
        // Check if there's an existing chart instance and destroy it
        if (canvas._chart) {
            canvas._chart.destroy();
        }
        
        // Create new chart
        const ctx = canvas.getContext('2d');
        const chart = new Chart(ctx, chartConfig);
        canvas._chart = chart;
        return chart;
    } catch (error) {
        console.error('Error creating chart:', error);
        chartContainer.innerHTML = `
            <div class="alert alert-warning">
                <i class="fas fa-exclamation-triangle me-2"></i>
                ${fallbackMessage}
            </div>
            <div class="fallback-content p-3 text-center text-muted">
                <i class="fas fa-chart-line fa-3x mb-3"></i>
                <p>Chart rendering failed: ${error.message}</p>
            </div>
        `;
        return false;
    }
} 