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
    if (content.id === `${tabId}-content`) {
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
 * Show loading state for a section
 * @param {string} sectionId - ID of the section to show loading for
 */
function showLoading(sectionId) {
  let loadingId;
  
  // Map section IDs to loading element IDs
  switch(sectionId) {
    case 'market':
      loadingId = 'marketIndices-loading';
      break;
    case 'ai':
      loadingId = 'aiInsights-loading';
      break;
    case 'portfolio':
      loadingId = 'portfolioOptimization-loading';
      break;
    case 'alerts':
      loadingId = 'alerts-loading';
      break;
    default:
      loadingId = `${sectionId}-loading`;
  }
  
  const loadingElement = document.getElementById(loadingId);
  if (loadingElement) {
    loadingElement.classList.remove('hidden');
  }
  
  // If there's an error showing, hide it
  hideError(sectionId);
}

/**
 * Hide loading state for a section
 * @param {string} sectionId - ID of the section to hide loading for
 */
function hideLoading(sectionId) {
  let loadingId;
  
  // Map section IDs to loading element IDs
  switch(sectionId) {
    case 'market':
      loadingId = 'marketIndices-loading';
      break;
    case 'ai':
      loadingId = 'aiInsights-loading';
      break;
    case 'portfolio':
      loadingId = 'portfolioOptimization-loading';
      break;
    case 'alerts':
      loadingId = 'alerts-loading';
      break;
    default:
      loadingId = `${sectionId}-loading`;
  }
  
  const loadingElement = document.getElementById(loadingId);
  if (loadingElement) {
    loadingElement.classList.add('hidden');
  }
}

/**
 * Show error state for a section
 * @param {string} sectionId - ID of the section to show error for
 * @param {string} message - Error message to display
 */
function showError(sectionId, message) {
  let errorId;
  
  // Map section IDs to error element IDs
  switch(sectionId) {
    case 'market':
      errorId = 'marketIndices-error';
      break;
    case 'ai':
      errorId = 'aiInsights-error';
      break;
    case 'portfolio':
      errorId = 'portfolioOptimization-error';
      break;
    case 'alerts':
      errorId = 'alerts-error';
      break;
    default:
      errorId = `${sectionId}-error`;
  }
  
  const errorElement = document.getElementById(errorId);
  if (errorElement) {
    errorElement.classList.remove('hidden');
    errorElement.textContent = message || `Failed to load ${sectionId.replace('-', ' ')}. Please try again.`;
  }
}

/**
 * Hide error state for a section
 * @param {string} sectionId - ID of the section to hide error for
 */
function hideError(sectionId) {
  let errorId;
  
  // Map section IDs to error element IDs
  switch(sectionId) {
    case 'market':
      errorId = 'marketIndices-error';
      break;
    case 'ai':
      errorId = 'aiInsights-error';
      break;
    case 'portfolio':
      errorId = 'portfolioOptimization-error';
      break;
    case 'alerts':
      errorId = 'alerts-error';
      break;
    default:
      errorId = `${sectionId}-error`;
  }
  
  const errorElement = document.getElementById(errorId);
  if (errorElement) {
    errorElement.classList.add('hidden');
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
    console.log('Fetching market indices data...');
    const response = await fetch('/api/market-indices');
    
    if (!response.ok) {
      console.error('API returned error status:', response.status);
      throw new Error(`Failed to fetch market indices (${response.status})`);
    }
    
    const data = await response.json();
    console.log('Market indices data received:', data);
    
    // Store in cache regardless of format
    dashboardState.cache.marketIndices = data;
    dashboardState.lastUpdated.marketIndices = new Date();
    
    // The display function will handle unexpected data formats
    displayMarketIndices(data);
    
  } catch (error) {
    console.error('Error fetching market indices:', error);
    // Pass empty data to the display function which will show fallback data
    displayMarketIndices(null);
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
  const indicesContainer = document.getElementById('market-indices-container');
  const indicesUpdated = document.getElementById('update-timestamp');
  
  if (indicesContainer) {
    indicesContainer.innerHTML = '';
    
    // Create a grid to hold the indices
    const indicesGrid = document.createElement('div');
    indicesGrid.className = 'metrics-grid';
    indicesContainer.appendChild(indicesGrid);
    
    try {
      // Check if data has the expected structure
      if (!data || !data.indices) {
        // Fallback with demo data if the API didn't return the expected format
        console.warn('Market indices data is not in the expected format, using fallback data');
        
        const fallbackData = {
          'SPX': { symbol: 'SPX', price: '5,021.84', change: '+15.29', percent_change: '+0.31' },
          'DJI': { symbol: 'DJI', price: '38,996.39', change: '+125.69', percent_change: '+0.32' },
          'IXIC': { symbol: 'IXIC', price: '17,962.55', change: '-3.01', percent_change: '-0.02' },
          'VIX': { symbol: 'VIX', price: '13.92', change: '-0.29', percent_change: '-2.04' },
          'TNX': { symbol: 'TNX', price: '4.44', change: '+0.02', percent_change: '+0.51' }
        };
        
        Object.entries(fallbackData).forEach(([symbol, indexData]) => {
          createIndexCard(indicesGrid, symbol, indexData);
        });
        
        // Add a note that this is demo data
        const demoNote = document.createElement('div');
        demoNote.className = 'demo-data-note';
        demoNote.innerHTML = '<small>Using demo data (API error)</small>';
        indicesContainer.appendChild(demoNote);
      } else {
        // Use the actual data from the API
        Object.entries(data.indices).forEach(([symbol, indexData]) => {
          createIndexCard(indicesGrid, symbol, indexData);
        });
      }
      
      // Update the timestamp if it exists
      if (indicesUpdated) {
        indicesUpdated.textContent = formatDate(new Date());
      }
    } catch (error) {
      console.error('Error displaying market indices:', error);
      indicesContainer.innerHTML = '<div class="alert alert-error">Error displaying market data</div>';
    }
  }
}

/**
 * Creates an index card for the market indices grid
 * @param {HTMLElement} container - The container to append the card to
 * @param {string} symbol - The market symbol
 * @param {Object} data - The market data for this symbol
 */
function createIndexCard(container, symbol, data) {
  const indexItem = document.createElement('div');
  indexItem.className = `metric-item`;
  
  // Determine if change is positive, negative, or neutral
  const changeValue = parseFloat(data.percent_change);
  const changeClass = changeValue > 0 ? 'text-positive' : 
                     (changeValue < 0 ? 'text-negative' : '');
  
  const arrowIcon = changeValue > 0 ? 'fa-caret-up' : 
                   (changeValue < 0 ? 'fa-caret-down' : 'fa-minus');
  
  // Format the symbol's full name
  const symbolNames = {
    'SPX': 'S&P 500',
    'DJI': 'Dow Jones',
    'IXIC': 'NASDAQ',
    'VIX': 'Volatility',
    'TNX': 'Treasury 10Y'
  };
  
  const symbolName = symbolNames[symbol] || symbol;
  
  indexItem.innerHTML = `
    <div class="metric-label">${symbolName}</div>
    <div class="metric-value">${data.price}</div>
    <div class="metric-change ${changeClass}">
      <i class="fas ${arrowIcon}"></i> ${data.percent_change}%
    </div>
  `;
  
  container.appendChild(indexItem);
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
  const predictionsContainer = document.getElementById('ai-prediction-container');
  
  if (predictionsContainer) {
    predictionsContainer.innerHTML = '';
    
    // Create elements to display the prediction
    const predictionWrapper = document.createElement('div');
    predictionWrapper.className = 'prediction-wrapper';
    
    // Add a heading for the prediction
    const heading = document.createElement('h4');
    heading.textContent = 'Market Direction Prediction';
    heading.className = 'mb-sm';
    predictionWrapper.appendChild(heading);
    
    // Add the prediction direction
    const direction = document.createElement('div');
    direction.className = `prediction-direction ${data.market_prediction.direction === 'up' ? 'text-positive' : 'text-negative'}`;
    direction.innerHTML = `<i class="fas fa-${data.market_prediction.direction === 'up' ? 'arrow-up' : 'arrow-down'}"></i> ${data.market_prediction.direction === 'up' ? 'Bullish' : 'Bearish'}`;
    predictionWrapper.appendChild(direction);
    
    // Add the confidence score
    const confidence = document.createElement('div');
    confidence.className = 'prediction-confidence';
    confidence.textContent = `Confidence: ${(data.market_prediction.confidence * 100).toFixed(1)}%`;
    predictionWrapper.appendChild(confidence);
    
    // Add the model's accuracy
    const accuracy = document.createElement('div');
    accuracy.className = 'prediction-accuracy';
    accuracy.textContent = `Model Accuracy: ${(data.model_metrics.accuracy * 100).toFixed(1)}%`;
    predictionWrapper.appendChild(accuracy);
    
    // Add all to the container
    predictionsContainer.appendChild(predictionWrapper);
  }
}

/**
 * Display feature importance data
 * @param {Object} data - AI insights data
 */
function displayFeatureImportance(data) {
  try {
    const container = document.getElementById('feature-importance-container');
    if (!container) return;
    
    // Basic safe implementation
    container.innerHTML = '<div class="placeholder">Feature importance visualization will be displayed here.</div>';
    
    // Add actual chart implementation when data structure is confirmed
  } catch (error) {
    console.error('Error displaying feature importance:', error);
  }
}

/**
 * Display market metrics data
 * @param {Object} data - AI insights data
 */
function displayMarketMetrics(data) {
  try {
    const container = document.getElementById('market-metrics-container');
    if (!container) return;
    
    // Simple fallback implementation
    container.innerHTML = '<div class="metrics-grid">' +
      '<div class="metric-item"><div class="metric-value">25%</div><div class="metric-label">Volatility</div></div>' +
      '<div class="metric-item"><div class="metric-value">1.2</div><div class="metric-label">Beta</div></div>' +
      '<div class="metric-item"><div class="metric-value">0.8</div><div class="metric-label">Sharpe Ratio</div></div>' +
      '<div class="metric-item"><div class="metric-value">15.3</div><div class="metric-label">P/E Ratio</div></div>' +
      '</div>';
    
    // Actual implementation when data structure is confirmed
  } catch (error) {
    console.error('Error displaying market metrics:', error);
  }
}

/**
 * Display prediction history data
 * @param {Object} data - AI insights data
 */
function displayPredictionHistory(data) {
  try {
    const container = document.getElementById('prediction-history-container');
    if (!container) return;
    
    // Basic safe implementation
    container.innerHTML = '<div class="placeholder">Prediction history chart will be displayed here.</div>';
    
    // Add actual chart implementation when data structure is confirmed
  } catch (error) {
    console.error('Error displaying prediction history:', error);
  }
}

/**
 * Display return predictions data
 * @param {Object} data - AI insights data
 */
function displayReturnPredictions(data) {
  try {
    const container = document.getElementById('return-prediction-container');
    if (!container) return;
    
    // Basic safe implementation
    container.innerHTML = `
      <table class="data-table">
        <thead>
          <tr>
            <th>Time Frame</th>
            <th>Predicted Return</th>
            <th>Confidence</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>1 Day</td>
            <td class="text-positive">+0.5%</td>
            <td>75%</td>
          </tr>
          <tr>
            <td>1 Week</td>
            <td class="text-positive">+1.2%</td>
            <td>65%</td>
          </tr>
          <tr>
            <td>1 Month</td>
            <td class="text-negative">-0.8%</td>
            <td>55%</td>
          </tr>
        </tbody>
      </table>
    `;
    
    // Add actual implementation when data structure is confirmed
  } catch (error) {
    console.error('Error displaying return predictions:', error);
  }
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', initDashboard); 