/**
 * AI Dashboard - Main JavaScript File
 * 
 * Handles tab navigation, theme toggling, and initializes the dashboard components.
 * This is the central file that coordinates all dashboard functionality.
 */

// Cache frequently accessed DOM elements
const dashboardContainer = document.querySelector('.container');
const tabNavItems = document.querySelectorAll('.tab');
const tabContents = document.querySelectorAll('.tab-content');
const themeToggleBtn = document.getElementById('dark-mode-toggle');
const aiStatusIndicator = document.querySelector('.status .status-indicator');
const aiStatusText = document.querySelector('.status');

// Global dashboard state
const dashboardState = {
  // Data cache to minimize API calls
  cache: {
    marketIndices: null,
    aiInsights: null,
    portfolioOptimizations: null,
    alerts: null,
  },
  
  // UI state
  ui: {
    currentTab: 'market-insights',
    isDarkMode: false, // Will be initialized based on system preference
    isAiActive: true,
  },
  
  // Last updated timestamps
  lastUpdated: {
    marketIndices: null,
    aiInsights: null,
    portfolioOptimizations: null,
    alerts: null,
  }
};

/**
 * Initialize the dashboard
 */
function initDashboard() {
  // Setup event listeners
  setupTabNavigation();
  setupThemeToggle();
  
  // Set initial theme based on user preference
  initializeTheme();
  
  // Load initial data
  loadInitialData();
  
  // Setup refresh intervals
  setupDataRefreshIntervals();
}

/**
 * Set up tab navigation
 */
function setupTabNavigation() {
  tabNavItems.forEach(item => {
    item.addEventListener('click', () => {
      const tabId = item.getAttribute('data-tab');
      setActiveTab(tabId);
    });
  });
}

/**
 * Set active tab
 * @param {string} tabId - ID of the tab to activate
 */
function setActiveTab(tabId) {
  // Update UI state
  dashboardState.ui.currentTab = tabId;
  
  // Update tab navigation
  tabNavItems.forEach(item => {
    if (item.getAttribute('data-tab') === tabId) {
      item.classList.add('active');
    } else {
      item.classList.remove('active');
    }
  });
  
  // Update tab content
  tabContents.forEach(content => {
    if (content.id === `${tabId}-tab-content`) {
      content.classList.add('active');
    } else {
      content.classList.remove('active');
    }
  });
  
  // Load tab-specific data if needed
  loadTabData(tabId);
}

/**
 * Load data specific to the active tab
 * @param {string} tabId - ID of the tab to load data for
 */
function loadTabData(tabId) {
  switch (tabId) {
    case 'market-insights':
      if (!dashboardState.cache.marketIndices) {
        fetchMarketIndices();
      }
      if (!dashboardState.cache.aiInsights) {
        fetchAiInsights();
      }
      break;
      
    case 'portfolio':
      // Portfolio data is loaded on demand when users submit the form
      break;
      
    case 'alerts':
      // Alerts are handled by the alerts-system.js module
      break;
      
    case 'economic':
      // Economic data is handled by the news-economic.js module
      break;
      
    case 'news-sentiment':
      // News sentiment is handled by the news-economic.js module
      break;
  }
}

/**
 * Set up theme toggle functionality
 */
function setupThemeToggle() {
  if (!themeToggleBtn) return;
  
  themeToggleBtn.addEventListener('click', () => {
    toggleDarkMode();
  });
}

/**
 * Initialize theme based on user preference
 */
function initializeTheme() {
  // Check for saved theme preference
  const savedTheme = localStorage.getItem('theme');
  
  if (savedTheme) {
    // Use saved preference
    dashboardState.ui.isDarkMode = savedTheme === 'dark';
  } else {
    // Check for system preference
    const prefersDarkMode = window.matchMedia('(prefers-color-scheme: dark)').matches;
    dashboardState.ui.isDarkMode = prefersDarkMode;
  }
  
  // Apply theme
  applyTheme();
  
  // Listen for system theme changes
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', e => {
    if (localStorage.getItem('theme') === null) {
      dashboardState.ui.isDarkMode = e.matches;
      applyTheme();
    }
  });
}

/**
 * Toggle dark mode
 */
function toggleDarkMode() {
  dashboardState.ui.isDarkMode = !dashboardState.ui.isDarkMode;
  applyTheme();
  
  // Save preference
  localStorage.setItem('theme', dashboardState.ui.isDarkMode ? 'dark' : 'light');
}

/**
 * Apply current theme to the dashboard
 */
function applyTheme() {
  if (dashboardState.ui.isDarkMode) {
    document.body.classList.add('dark-mode');
    document.documentElement.setAttribute('data-theme', 'dark');
    themeToggleBtn.innerHTML = '<i class="fas fa-sun"></i>';
  } else {
    document.body.classList.remove('dark-mode');
    document.documentElement.setAttribute('data-theme', 'light');
    themeToggleBtn.innerHTML = '<i class="fas fa-moon"></i>';
  }
  
  // Update charts theme if ai-charts is loaded
  if (window.aiCharts && typeof window.aiCharts.updateChartsTheme === 'function') {
    window.aiCharts.updateChartsTheme(dashboardState.ui.isDarkMode);
  }
}

/**
 * Load initial data for the dashboard
 */
function loadInitialData() {
  // Check AI status first
  checkAiStatus();
  
  // Load data for the active tab
  loadTabData(dashboardState.ui.currentTab);
}

/**
 * Set up intervals for refreshing data
 */
function setupDataRefreshIntervals() {
  // Refresh market data every 5 minutes
  setInterval(() => {
    if (dashboardState.ui.currentTab === 'market-insights') {
      fetchMarketIndices();
      fetchAiInsights();
    }
  }, 5 * 60 * 1000);
  
  // Check AI status every minute
  setInterval(checkAiStatus, 60 * 1000);
}

/**
 * Check AI engine status
 */
async function checkAiStatus() {
  try {
    const response = await fetch('/api/ai-status');
    
    if (!response.ok) {
      throw new Error('AI status check failed');
    }
    
    const data = await response.json();
    updateAiStatus(data.status === 'active');
    
  } catch (error) {
    console.error('Failed to check AI status:', error);
    updateAiStatus(false);
  }
}

/**
 * Update AI status indicator
 * @param {boolean} isActive - Whether the AI engine is active
 */
function updateAiStatus(isActive) {
  dashboardState.ui.isAiActive = isActive;
  
  // Update indicator
  if (isActive) {
    aiStatusIndicator.classList.remove('inactive');
    aiStatusIndicator.classList.add('active');
    aiStatusText.textContent = 'AI Engine: Active';
  } else {
    aiStatusIndicator.classList.remove('active');
    aiStatusIndicator.classList.add('inactive');
    aiStatusText.textContent = 'AI Engine: Inactive';
  }
}

/**
 * Show a loading state for a section
 * @param {string} sectionId - ID of the section to show loader for
 */
function showLoading(sectionId) {
  const loader = document.getElementById(`${sectionId}-loader`);
  const error = document.getElementById(`${sectionId}-error`);
  
  if (loader) {
    loader.style.display = 'flex';
  }
  
  if (error) {
    error.style.display = 'none';
  }
}

/**
 * Hide loading state for a section
 * @param {string} sectionId - ID of the section to hide loader for
 */
function hideLoading(sectionId) {
  const loader = document.getElementById(`${sectionId}-loader`);
  
  if (loader) {
    loader.style.display = 'none';
  }
}

/**
 * Show error message for a section
 * @param {string} sectionId - ID of the section to show error for
 * @param {string} message - Error message to display
 */
function showError(sectionId, message) {
  const loader = document.getElementById(`${sectionId}-loader`);
  const error = document.getElementById(`${sectionId}-error`);
  
  if (loader) {
    loader.style.display = 'none';
  }
  
  if (error) {
    error.style.display = 'block';
    error.textContent = message || `Failed to load ${sectionId.replace('-', ' ')}. Please try again.`;
  }
}

/**
 * Format date for display
 * @param {Date|string} date - Date to format
 * @returns {string} Formatted date string
 */
function formatDate(date) {
  if (!date) return '--';
  
  const dateObj = typeof date === 'string' ? new Date(date) : date;
  
  return dateObj.toLocaleString(undefined, {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit'
  });
}

/**
 * Format percentage for display
 * @param {number} value - Percentage value (0-1)
 * @returns {string} Formatted percentage string
 */
function formatPercent(value) {
  if (value === null || value === undefined) return '--';
  return (value * 100).toFixed(2) + '%';
}

/**
 * Format price for display
 * @param {number} value - Price value
 * @returns {string} Formatted price string
 */
function formatPrice(value) {
  if (value === null || value === undefined) return '--';
  return '$' + value.toFixed(2);
}

/**
 * Get appropriate CSS class for a percentage change
 * @param {number} change - Percentage change value
 * @returns {string} CSS class name ('positive', 'negative', or '')
 */
function getChangeClass(change) {
  if (change > 0) return 'positive';
  if (change < 0) return 'negative';
  return '';
}

/**
 * Fetch market indices data from API
 */
async function fetchMarketIndices() {
  showLoading('market');
  
  try {
    const response = await fetch('/api/market-indices');
    
    if (!response.ok) {
      throw new Error('Failed to fetch market indices');
    }
    
    const data = await response.json();
    dashboardState.cache.marketIndices = data;
    dashboardState.lastUpdated.marketIndices = new Date();
    
    displayMarketIndices(data);
    
  } catch (error) {
    console.error('Error fetching market indices:', error);
    showError('market', error.message);
  } finally {
    hideLoading('market');
  }
}

/**
 * Display market indices
 * @param {Object} data - Market indices data
 */
function displayMarketIndices(data) {
  const indicesGrid = document.getElementById('indices-grid');
  const indicesUpdated = document.getElementById('indices-updated');
  
  if (indicesGrid) {
    indicesGrid.innerHTML = '';
    
    Object.entries(data.indices).forEach(([symbol, indexData]) => {
      const indexItem = document.createElement('div');
      indexItem.className = `index-item ${getChangeClass(indexData.percent_change)}`;
      
      const arrowIcon = indexData.percent_change > 0 ? 'fa-caret-up' : 
                        (indexData.percent_change < 0 ? 'fa-caret-down' : 'fa-minus');
      
      indexItem.innerHTML = `
        <div class="index-name">${symbol}</div>
        <div class="index-price">${formatPrice(indexData.price)}</div>
        <div class="index-change">
          <i class="fas ${arrowIcon}"></i>
          ${formatPercent(indexData.percent_change)}
        </div>
      `;
      
      indicesGrid.appendChild(indexItem);
    });
  }
  
  if (indicesUpdated) {
    indicesUpdated.textContent = `Last updated: ${formatDate(dashboardState.lastUpdated.marketIndices)}`;
  }
}

/**
 * Fetch AI insights data from API
 */
async function fetchAiInsights() {
  try {
    const response = await fetch('/api/ai-insights');
    
    if (!response.ok) {
      throw new Error('Failed to fetch AI insights');
    }
    
    const data = await response.json();
    dashboardState.cache.aiInsights = data;
    dashboardState.lastUpdated.aiInsights = new Date();
    
    displayAiPredictions(data);
    displayFeatureImportance(data);
    displayMarketMetrics(data);
    displayPredictionHistory(data);
    displayReturnPredictions(data);
    
  } catch (error) {
    console.error('Error fetching AI insights:', error);
  }
}

/**
 * Display AI predictions
 * @param {Object} data - AI insights data
 */
function displayAiPredictions(data) {
  const predictionsGrid = document.getElementById('predictions-grid');
  const accuracyElement = document.getElementById('prediction-accuracy');
  
  if (predictionsGrid && data.predictions) {
    predictionsGrid.innerHTML = '';
    
    Object.entries(data.predictions).forEach(([symbol, prediction]) => {
      const predictionItem = document.createElement('div');
      predictionItem.className = `prediction-item ${prediction.direction.toLowerCase()}`;
      
      const directionIcon = prediction.direction === 'UP' ? 'fa-arrow-up' : 'fa-arrow-down';
      
      predictionItem.innerHTML = `
        <div class="prediction-symbol">${symbol}</div>
        <div class="prediction-direction">
          <i class="fas ${directionIcon}"></i>
          ${prediction.direction}
        </div>
        <div class="prediction-confidence">${formatPercent(prediction.confidence)}</div>
      `;
      
      predictionsGrid.appendChild(predictionItem);
    });
  }
  
  if (accuracyElement && data.model_metrics) {
    accuracyElement.textContent = `Accuracy: ${formatPercent(data.model_metrics.accuracy)}`;
  }
}

/**
 * Display feature importance chart
 * @param {Object} data - AI insights data
 */
function displayFeatureImportance(data) {
  const featureImportanceChart = document.getElementById('feature-importance-chart');
  
  if (featureImportanceChart && data.feature_importance && window.aiCharts) {
    const isDarkMode = document.body.classList.contains('dark-mode');
    window.aiCharts.createFeatureImportanceChart(featureImportanceChart, data.feature_importance, isDarkMode);
  }
}

/**
 * Display market metrics
 * @param {Object} data - AI insights data
 */
function displayMarketMetrics(data) {
  const metricsGrid = document.getElementById('metrics-grid');
  
  if (metricsGrid && data.market_metrics) {
    metricsGrid.innerHTML = '';
    
    // Define metrics to display with icons
    const metrics = [
      { name: 'Market Volatility', value: data.market_metrics.volatility, icon: 'chart-line' },
      { name: 'Sentiment Score', value: data.market_metrics.sentiment_score, icon: 'smile' },
      { name: 'Trading Volume', value: data.market_metrics.volume, format: 'volume', icon: 'chart-bar' },
      { name: 'Market Breadth', value: data.market_metrics.market_breadth, icon: 'balance-scale' }
    ];
    
    metrics.forEach(metric => {
      const metricItem = document.createElement('div');
      metricItem.className = 'metric-item';
      
      let formattedValue;
      if (metric.format === 'volume') {
        formattedValue = (metric.value / 1000000).toFixed(1) + 'M';
      } else if (metric.name === 'Sentiment Score') {
        formattedValue = metric.value.toFixed(2);
        // Add class based on sentiment
        if (metric.value > 0.2) metricItem.classList.add('positive');
        else if (metric.value < -0.2) metricItem.classList.add('negative');
      } else {
        formattedValue = formatPercent(metric.value);
      }
      
      metricItem.innerHTML = `
        <div class="metric-icon">
          <i class="fas fa-${metric.icon}"></i>
        </div>
        <div class="metric-details">
          <div class="metric-name">${metric.name}</div>
          <div class="metric-value">${formattedValue}</div>
        </div>
      `;
      
      metricsGrid.appendChild(metricItem);
    });
  }
}

/**
 * Display prediction history chart
 * @param {Object} data - AI insights data
 */
function displayPredictionHistory(data) {
  const predictionHistoryChart = document.getElementById('prediction-history-chart');
  
  if (predictionHistoryChart && data.prediction_history && window.aiCharts) {
    const isDarkMode = document.body.classList.contains('dark-mode');
    window.aiCharts.createPredictionHistoryChart(predictionHistoryChart, data.prediction_history, isDarkMode);
  }
}

/**
 * Display return predictions chart
 * @param {Object} data - AI insights data
 */
function displayReturnPredictions(data) {
  const returnsPredictionChart = document.getElementById('returns-prediction-chart');
  const timeframeButtons = document.querySelectorAll('#return-timeframe button');
  
  if (returnsPredictionChart && data.return_predictions && window.aiCharts) {
    const isDarkMode = document.body.classList.contains('dark-mode');
    
    // Default to 1d timeframe
    let currentTimeframe = '1d';
    const timeframeData = data.return_predictions[currentTimeframe];
    
    if (timeframeData) {
      window.aiCharts.createReturnsPredictionChart(returnsPredictionChart, timeframeData, isDarkMode);
    }
    
    // Set up timeframe buttons
    if (timeframeButtons) {
      timeframeButtons.forEach(button => {
        button.addEventListener('click', () => {
          // Update active button
          timeframeButtons.forEach(btn => btn.classList.remove('active'));
          button.classList.add('active');
          
          // Update chart
          const timeframe = button.getAttribute('data-timeframe');
          if (data.return_predictions[timeframe]) {
            window.aiCharts.createReturnsPredictionChart(returnsPredictionChart, data.return_predictions[timeframe], isDarkMode);
          }
        });
      });
    }
  }
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', initDashboard); 