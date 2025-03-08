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
 * Debug logging function to help troubleshoot issues
 * @param {string} component - The component or function name
 * @param {string} message - The debug message
 * @param {any} data - Optional data to log
 */
function debugLog(component, message, data = null) {
    const timestamp = new Date().toISOString();
    const prefix = `[${timestamp}] [${component}]`;
    
    if (data) {
        console.log(`${prefix} ${message}`, data);
    } else {
        console.log(`${prefix} ${message}`);
    }
}

/**
 * Initialize the dashboard
 */
function initDashboard() {
  debugLog('INIT', 'Dashboard initialization started');
  
  // Setup event listeners
  setupTabNavigation();
  setupThemeToggle();
  
  // Set initial theme based on user preference
  initializeTheme();
  
  // Load initial data
  loadInitialData();
  
  // Setup refresh intervals
  setupDataRefreshIntervals();
  
  debugLog('INIT', 'Dashboard initialization completed');
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
  debugLog('DATA', 'Loading initial data');
  
  // Check AI status first
  checkAiStatus();
  
  // Load data for the active tab
  const currentTab = dashboardState.ui.currentTab;
  debugLog('DATA', `Loading data for active tab: ${currentTab}`);
  loadTabData(currentTab);
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
 * Global error handler to catch unhandled errors
 */
window.addEventListener('error', function(event) {
    console.error('CRITICAL ERROR:', event.error);
    // Display a user-friendly message
    const errorContainer = document.createElement('div');
    errorContainer.className = 'alert alert-error';
    errorContainer.innerHTML = '<strong>Something went wrong</strong><p>The dashboard encountered an error. Please try refreshing the page.</p>';
    document.querySelector('.container')?.prepend(errorContainer);
});

/**
 * Fetches market indices data from the API
 * @returns {Promise<Object>} The market indices data
 */
async function fetchMarketIndices() {
    debugLog('fetchMarketIndices', 'Starting to fetch market indices data');
    
    try {
        // Find the loading indicator and show it
        const loadingEl = document.getElementById('marketIndices-loading');
        if (loadingEl) {
            loadingEl.classList.remove('hidden');
        }
        
        // Hide any error messages
        const errorEl = document.getElementById('marketIndices-error');
        if (errorEl) {
            errorEl.classList.add('hidden');
        }
        
        // If we have cached data and it's less than 5 minutes old, use it
        const now = new Date();
        if (dashboardState.cache.marketIndices && 
            (now - dashboardState.lastUpdated.marketIndices) < 300000) {
            debugLog('fetchMarketIndices', 'Using cached market indices data', dashboardState.cache.marketIndices);
            
            // Hide loading
            if (loadingEl) {
                loadingEl.classList.add('hidden');
            }
            
            return dashboardState.cache.marketIndices;
        }
        
        debugLog('fetchMarketIndices', 'Making API request to /api/market-indices');
        const response = await fetch('/api/market-indices', {
            method: 'GET',
            headers: {
                'Accept': 'application/json',
            },
            credentials: 'same-origin' // Include cookies for authentication
        });
        
        debugLog('fetchMarketIndices', `Response status: ${response.status}`);
        
        // Check if response is ok (status in the range 200-299)
        if (!response.ok) {
            if (response.status === 401) {
                debugLog('fetchMarketIndices', 'Authentication required - showing demo data');
                
                // Hide loading
                if (loadingEl) {
                    loadingEl.classList.add('hidden');
                }
                
                return showDemoData();
            }
            
            const errorText = await response.text();
            debugLog('fetchMarketIndices', 'Error response', errorText);
            throw new Error(`API request failed with status ${response.status}: ${errorText}`);
        }
        
        // Parse JSON response
        const data = await response.json();
        debugLog('fetchMarketIndices', 'Successfully parsed API response', data);
        
        // Cache the data and update the timestamp
        dashboardState.cache.marketIndices = data;
        dashboardState.lastUpdated.marketIndices = now;
        
        // Hide loading
        if (loadingEl) {
            loadingEl.classList.add('hidden');
        }
        
        // Show a banner if this is demo data
        if (data.demo === true) {
            debugLog('fetchMarketIndices', 'Displaying demo data banner');
            // Display the indices data first
            displayMarketIndices(data);
        } else {
            // Display the indices data
            displayMarketIndices(data);
        }
        
        return data;
    } catch (error) {
        debugLog('fetchMarketIndices', 'Error fetching market indices', error);
        
        // Hide loading indicator
        const loadingEl = document.getElementById('marketIndices-loading');
        if (loadingEl) {
            loadingEl.classList.add('hidden');
        }
        
        // Show error message
        const errorEl = document.getElementById('marketIndices-error');
        if (errorEl) {
            errorEl.classList.remove('hidden');
            errorEl.innerHTML = `Failed to load market data: ${error.message}`;
        }
        
        // Fall back to demo data if we have an error
        return showDemoData();
    }
}

/**
 * Display fallback demo data for market indices
 */
function showDemoData() {
  debugLog('UI', 'Showing demo market data');
  
  // Create fallback data
  const fallbackData = {
    demo: true,
    indices: {
      'SPX': { symbol: 'SPX', price: '5,021.84', change: '+15.29', percent_change: '+0.31', source: 'demo' },
      'DJI': { symbol: 'DJI', price: '38,996.39', change: '+125.69', percent_change: '+0.32', source: 'demo' },
      'IXIC': { symbol: 'IXIC', price: '17,962.55', change: '-3.01', percent_change: '-0.02', source: 'demo' },
      'VIX': { symbol: 'VIX', price: '13.92', change: '-0.29', percent_change: '-2.04', source: 'demo' },
      'TNX': { symbol: 'TNX', price: '4.44', change: '+0.02', percent_change: '+0.51', source: 'demo' }
    }
  };
  
  // Store in cache
  dashboardState.cache.marketIndices = fallbackData;
  dashboardState.lastUpdated.marketIndices = new Date();
  
  // Display the demo data
  displayMarketIndices(fallbackData);
  
  return fallbackData;
}

/**
 * Displays market indices data in the UI
 * @param {Object} data - The market indices data
 */
function displayMarketIndices(data) {
    debugLog('displayMarketIndices', 'Displaying market indices data', data);
    
    if (!data) {
        debugLog('displayMarketIndices', 'No data to display');
        return;
    }
    
    try {
        // Find the container by ID
        const container = document.getElementById('market-indices-container');
        
        // If container not found, try searching by class or creating it
        if (!container) {
            debugLog('displayMarketIndices', 'Container #market-indices-container not found, looking for alternatives');
            
            // Try to find by class
            let altContainer = document.querySelector('.market-indices');
            
            // If that doesn't work, try the card body
            if (!altContainer) {
                altContainer = document.querySelector('.card-body');
            }
            
            // If we still can't find it, create a placeholder
            if (!altContainer) {
                debugLog('displayMarketIndices', 'No suitable container found, adding placeholder to body');
                altContainer = document.createElement('div');
                altContainer.id = 'market-indices-container';
                document.body.appendChild(altContainer);
            }
            
            // Add an ID so we can find it next time
            if (!altContainer.id) {
                altContainer.id = 'market-indices-container';
            }
            
            displayMarketIndicesInContainer(altContainer, data);
        } else {
            // Use the found container
            displayMarketIndicesInContainer(container, data);
        }
    } catch (error) {
        debugLog('displayMarketIndices', 'Error displaying market indices', error);
        
        // Try one more fallback approach - add directly to the body
        try {
            const fallbackContainer = document.createElement('div');
            fallbackContainer.id = 'emergency-market-indices';
            fallbackContainer.className = 'alert alert-warning';
            fallbackContainer.innerHTML = '<strong>Market Indices (Emergency Fallback)</strong>';
            document.body.prepend(fallbackContainer);
            
            displayMarketIndicesInContainer(fallbackContainer, data);
        } catch (fallbackError) {
            console.error('Critical failure in displaying market indices', fallbackError);
        }
    }
}

/**
 * Helper function to actually display the indices in a container
 */
function displayMarketIndicesInContainer(container, data) {
    // Clear the container
    container.innerHTML = '';
    
    // Check if data has the indices property (new structure)
    const indices = data.indices || data;
    
    debugLog('displayMarketIndicesInContainer', 'Processing indices data', indices);
    
    // Create cards for each index
    if (indices.SPX) createIndexCard(container, 'SPX', indices.SPX);
    if (indices.DJI) createIndexCard(container, 'DJI', indices.DJI);
    if (indices.IXIC) createIndexCard(container, 'IXIC', indices.IXIC);
    if (indices.VIX) createIndexCard(container, 'VIX', indices.VIX);
    if (indices.TNX) createIndexCard(container, 'TNX', indices.TNX);
    
    // Update last updated timestamp
    const lastUpdated = document.getElementById('market-last-updated');
    if (lastUpdated) {
        const timestamp = new Date().toLocaleString();
        lastUpdated.textContent = `Last updated: ${timestamp}`;
    }
}

/**
 * Creates a card for a market index
 * @param {HTMLElement} container - The container to add the card to
 * @param {string} symbol - The index symbol
 * @param {Object} data - The index data
 */
function createIndexCard(container, symbol, data) {
    debugLog('createIndexCard', `Creating card for ${symbol}`, data);
    
    try {
        // Make sure we have valid data
        if (!data || typeof data !== 'object') {
            debugLog('createIndexCard', `Invalid data for ${symbol}`, data);
            return;
        }
        
        // Create card elements
        const card = document.createElement('div');
        card.className = 'card index-card';
        
        // Format the card content
        let symbolName = symbol;
        switch(symbol) {
            case 'SPX': symbolName = 'S&P 500'; break;
            case 'DJI': symbolName = 'Dow Jones'; break;
            case 'IXIC': symbolName = 'NASDAQ'; break;
            case 'VIX': symbolName = 'Volatility Index'; break;
            case 'TNX': symbolName = '10-Year Treasury'; break;
        }
        
        // Safely parse the values with fallbacks
        let price = data.price || '0.00';
        let change = 0;
        let percentChange = 0;
        
        try {
            // Handle different formats of the change data
            if (typeof data.change === 'string') {
                change = parseFloat(data.change.replace(/[+$,]/g, '')) || 0;
            } else if (typeof data.change === 'number') {
                change = data.change;
            }
            
            if (typeof data.percent_change === 'string') {
                percentChange = parseFloat(data.percent_change.replace(/[+%,]/g, '')) || 0;
            } else if (typeof data.percent_change === 'number') {
                percentChange = data.percent_change;
            }
        } catch (parseError) {
            debugLog('createIndexCard', `Error parsing values for ${symbol}`, parseError);
            // Use default values
            change = 0;
            percentChange = 0;
        }
        
        const changeClass = getChangeClass(change);
        const source = data.source || 'live';
        
        // Set the card content with proper formatting
        card.innerHTML = `
            <div class="card-header">
                <h3 class="card-title">${symbolName}</h3>
                <span class="source-tag">${source === 'demo' ? 'DEMO' : 'LIVE'}</span>
            </div>
            <div class="card-body">
                <div class="price">${price}</div>
                <div class="change ${changeClass}">
                    ${change >= 0 ? '+' : ''}${change.toFixed(2)} (${percentChange >= 0 ? '+' : ''}${percentChange.toFixed(2)}%)
                </div>
            </div>
        `;
        
        // Add the card to the container
        container.appendChild(card);
        
        // Log success
        debugLog('createIndexCard', `Successfully created card for ${symbol}`);
        
    } catch (error) {
        debugLog('createIndexCard', `Error creating card for ${symbol}`, error);
        
        // Create a simple fallback card
        try {
            const fallbackCard = document.createElement('div');
            fallbackCard.className = 'card index-card';
            fallbackCard.innerHTML = `
                <div class="card-header">
                    <h3 class="card-title">${symbol}</h3>
                </div>
                <div class="card-body">
                    <div class="price">Data Unavailable</div>
                </div>
            `;
            container.appendChild(fallbackCard);
        } catch (fallbackError) {
            console.error(`Complete failure creating card for ${symbol}`, fallbackError);
        }
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

// Make sure to call initDashboard when the page loads
document.addEventListener('DOMContentLoaded', function() {
  debugLog('GLOBAL', 'DOM content loaded, initializing dashboard');
  initDashboard();
}); 