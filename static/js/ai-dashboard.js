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
        
        // Fetch AI insights data
        const response = await fetch('/api/ai-insights');
        
        if (!response.ok) {
            throw new Error(`API Error: ${response.status}`);
        }
        
        // Parse the data
        const data = await response.json();
        console.log('AI insights data received:', data);
        
        // Store data in the dashboard state
        dashboardState.data = data;
        dashboardState.lastUpdated = new Date();
        
        // Update all dashboard components with the new data
        updateAllComponents();
        
    } catch (error) {
        console.error('Error loading AI insights data:', error);
        showErrorMessage('Failed to load AI data. Please try again later.');
    }
}

// Update all dashboard components with current data
function updateAllComponents() {
    // Only update if we have data
    if (!dashboardState.data) {
        return;
    }
    
    // Update each component
    loadMarketIndices();
    updateMarketPrediction();
    updateNewsSentiment();
    updateFeatureImportance();
    updatePortfolioOptimization();
    updateEconomicIndicators();
    updateAlertSystem();
    
    // Update last updated timestamp
    updateLastUpdatedTimestamp();
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

// Show error message in the UI
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
                ${message}
            </div>`;
        }
    });
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
    console.log('Updating market prediction');
    
    const container = document.getElementById('market-prediction-container');
    if (!container) {
        console.error('Market prediction container not found');
        return;
    }
    
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
    console.log('Updating news sentiment analysis');
    
    const container = document.getElementById('news-sentiment-container');
    if (!container) {
        console.error('News sentiment container not found');
        return;
    }
    
    // Check if we have data
    if (!dashboardState.data || !dashboardState.data.newsSentiment) {
        container.innerHTML = '<div class="loader"></div> Analyzing news sentiment...';
        return;
    }
    
    // Extract news sentiment data
    const sentiment = dashboardState.data.newsSentiment || {};
    const overallSentiment = sentiment.overall || 0;
    const topSources = sentiment.topSources || [];
    const recentArticles = sentiment.recentArticles || [];
    
    // Calculate sentiment class
    let sentimentClass = 'neutral';
    if (overallSentiment >= 0.3) sentimentClass = 'positive';
    else if (overallSentiment <= -0.3) sentimentClass = 'negative';
    
    // Create HTML for sentiment analysis
    let html = `
        <div class="overall-sentiment">
            <h6 class="mb-3">Overall Market Sentiment</h6>
            <div class="sentiment-meter ${sentimentClass}">
                <div class="sentiment-value">${(overallSentiment * 100).toFixed(1)}%</div>
                <div class="sentiment-label">
                    ${overallSentiment >= 0.3 ? 'Positive' : overallSentiment <= -0.3 ? 'Negative' : 'Neutral'}
                </div>
            </div>
        </div>
    `;
    
    // Add top news sources with their sentiment
    if (topSources && topSources.length > 0) {
        html += `
            <div class="source-sentiment mt-4">
                <h6 class="mb-3">Top News Sources</h6>
                <ul class="list-group source-list">
        `;
        
        topSources.forEach(source => {
            const sourceClass = source.sentiment >= 0.2 ? 'positive' : 
                              source.sentiment <= -0.2 ? 'negative' : 'neutral';
            
            html += `
                <li class="list-group-item d-flex justify-content-between align-items-center">
                    <span>${source.name}</span>
                    <span class="badge rounded-pill bg-${sourceClass === 'positive' ? 'success' : 
                                                         sourceClass === 'negative' ? 'danger' : 'secondary'}">
                        ${(source.sentiment * 100).toFixed(0)}%
                    </span>
                </li>
            `;
        });
        
        html += `
                </ul>
            </div>
        `;
    }
    
    // Add recent articles section
    if (recentArticles && recentArticles.length > 0) {
        html += `
            <div class="recent-articles mt-4">
                <h6 class="mb-3">Recent Market News</h6>
                <div class="article-list">
        `;
        
        recentArticles.forEach(article => {
            const articleClass = article.sentiment >= 0.2 ? 'positive' : 
                               article.sentiment <= -0.2 ? 'negative' : 'neutral';
            
            html += `
                <div class="article-item mb-2 ${articleClass}">
                    <div class="article-title">
                        <a href="${article.url}" target="_blank" rel="noopener noreferrer">
                            ${article.title}
                        </a>
                    </div>
                    <div class="article-meta">
                        <small>${article.source} | ${article.date}</small>
                        <span class="article-sentiment ${articleClass}">
                            ${article.sentiment >= 0.2 ? 'Positive' : 
                              article.sentiment <= -0.2 ? 'Negative' : 'Neutral'}
                        </span>
                    </div>
                </div>
            `;
        });
        
        html += `
                </div>
            </div>
        `;
    }
    
    // If we don't have specific news data, show a message
    if ((!topSources || topSources.length === 0) && (!recentArticles || recentArticles.length === 0)) {
        html += `
            <div class="alert alert-info mt-3">
                <i class="fas fa-info-circle"></i> Detailed news sentiment data is not available.
            </div>
        `;
    }
    
    // Set the HTML
    container.innerHTML = html;
}

// Feature Importance - Update the feature importance chart
function updateFeatureImportance() {
    console.log('Updating feature importance');
    
    const container = document.getElementById('feature-importance-container');
    if (!container) {
        console.error('Feature importance container not found');
        return;
    }
    
    // Check if we have data
    if (!dashboardState.data || !dashboardState.data.featureImportance) {
        container.innerHTML = '<div class="loader"></div> Loading feature importance...';
        return;
    }
    
    // Get feature importance data
    const featureImportance = dashboardState.data.featureImportance || [];
    
    // Sort features by importance value (descending)
    const sortedFeatures = [...featureImportance].sort((a, b) => b.value - a.value);
    
    // Create HTML structure
    container.innerHTML = `
        <div class="feature-list">
            ${sortedFeatures.map((feature, index) => `
                <div class="feature-item" data-bs-toggle="tooltip" title="${feature.name}: ${(feature.value * 100).toFixed(1)}%">
                    <div class="feature-name">${feature.name}</div>
                    <div class="feature-bar-container">
                        <div class="feature-bar" style="width: ${feature.value * 100}%"></div>
                    </div>
                    <div class="feature-value">${(feature.value * 100).toFixed(1)}%</div>
                </div>
            `).join('')}
        </div>
    `;
    
    // Initialize tooltips
    const tooltips = [].slice.call(container.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltips.forEach(tooltip => {
        new bootstrap.Tooltip(tooltip);
    });
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