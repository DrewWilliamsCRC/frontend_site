/**
 * Portfolio Optimization Module
 * Handles portfolio optimization forms, API calls, and displaying results
 */

// Cache DOM elements
let portfolioForm;
let symbolsInput;
let riskToleranceInput;
let methodSelect;
let constraintsSection;
let minWeightInputs = {};
let maxWeightInputs = {};
let portfolioResult;
let portfolioError;
let portfolioLoader;
let allocationChart;
let metricsTable;
let optimizationHistory;

// Store optimization history
const optimizationHistoryData = [];

/**
 * Initialize the portfolio optimization module
 */
function initPortfolioOptimization() {
  // Cache DOM elements
  portfolioForm = document.getElementById('portfolio-form');
  symbolsInput = document.getElementById('portfolio-symbols');
  riskToleranceInput = document.getElementById('risk-tolerance');
  methodSelect = document.getElementById('optimization-method');
  constraintsSection = document.getElementById('constraints-section');
  portfolioResult = document.getElementById('portfolio-result');
  portfolioError = document.getElementById('portfolio-error');
  portfolioLoader = document.getElementById('portfolio-loader');
  allocationChart = document.getElementById('allocation-chart');
  metricsTable = document.getElementById('metrics-table');
  optimizationHistory = document.getElementById('optimization-history');

  // Set up event listeners
  if (portfolioForm) {
    portfolioForm.addEventListener('submit', handlePortfolioSubmit);
  }
  
  if (symbolsInput) {
    symbolsInput.addEventListener('blur', handleSymbolsChange);
  }
  
  if (methodSelect) {
    methodSelect.addEventListener('change', handleMethodChange);
  }
  
  // Initialize tooltips
  initTooltips();
}

/**
 * Handle changes to the symbols input
 * This will dynamically create constraint inputs for each symbol
 */
function handleSymbolsChange() {
  if (!symbolsInput || !constraintsSection) return;
  
  // Clear current constraints
  constraintsSection.innerHTML = '';
  minWeightInputs = {};
  maxWeightInputs = {};
  
  // Get symbols from input (comma-separated)
  const symbols = symbolsInput.value
    .split(',')
    .map(s => s.trim())
    .filter(s => s.length > 0);
  
  if (symbols.length === 0) return;
  
  // Create header
  const header = document.createElement('div');
  header.className = 'constraints-header';
  header.innerHTML = `
    <h4>Weight Constraints</h4>
    <p class="constraint-description">
      Optionally specify minimum and maximum allocation percentages for each symbol.
      Leave blank for default constraints.
    </p>
  `;
  constraintsSection.appendChild(header);
  
  // Create constraints table
  const table = document.createElement('table');
  table.className = 'constraints-table';
  
  // Table header
  const thead = document.createElement('thead');
  thead.innerHTML = `
    <tr>
      <th>Symbol</th>
      <th>Min Weight (%)</th>
      <th>Max Weight (%)</th>
    </tr>
  `;
  table.appendChild(thead);
  
  // Table body
  const tbody = document.createElement('tbody');
  
  // Create rows for each symbol
  symbols.forEach(symbol => {
    const row = document.createElement('tr');
    
    // Symbol cell
    const symbolCell = document.createElement('td');
    symbolCell.textContent = symbol;
    row.appendChild(symbolCell);
    
    // Min weight cell
    const minCell = document.createElement('td');
    const minInput = document.createElement('input');
    minInput.type = 'number';
    minInput.min = '0';
    minInput.max = '100';
    minInput.step = '1';
    minInput.placeholder = '0%';
    minInput.className = 'min-weight-input';
    minInput.dataset.symbol = symbol;
    minWeightInputs[symbol] = minInput;
    minCell.appendChild(minInput);
    row.appendChild(minCell);
    
    // Max weight cell
    const maxCell = document.createElement('td');
    const maxInput = document.createElement('input');
    maxInput.type = 'number';
    maxInput.min = '0';
    maxInput.max = '100';
    maxInput.step = '1';
    maxInput.placeholder = '100%';
    maxInput.className = 'max-weight-input';
    maxInput.dataset.symbol = symbol;
    maxWeightInputs[symbol] = maxInput;
    maxCell.appendChild(maxInput);
    row.appendChild(maxCell);
    
    tbody.appendChild(row);
  });
  
  table.appendChild(tbody);
  constraintsSection.appendChild(table);
  
  // Show the constraints section
  constraintsSection.style.display = 'block';
}

/**
 * Handle changes to the optimization method
 */
function handleMethodChange() {
  const method = methodSelect.value;
  
  // Show/hide risk tolerance based on selected method
  if (riskToleranceInput) {
    const riskToleranceContainer = riskToleranceInput.closest('.form-group');
    
    if (method === 'maximum_sharpe' || method === 'minimum_volatility') {
      riskToleranceContainer.style.display = 'none';
    } else {
      riskToleranceContainer.style.display = 'block';
    }
  }
}

/**
 * Handle portfolio optimization form submission
 * @param {Event} e - Form submission event
 */
async function handlePortfolioSubmit(e) {
  e.preventDefault();
  
  // Show loader, hide results and errors
  portfolioLoader.style.display = 'block';
  portfolioResult.style.display = 'none';
  portfolioError.style.display = 'none';
  
  // Get form data
  const symbols = symbolsInput.value
    .split(',')
    .map(s => s.trim())
    .filter(s => s.length > 0);
  
  const method = methodSelect.value;
  let riskTolerance = parseFloat(riskToleranceInput.value);
  
  // Validate inputs
  if (symbols.length < 2) {
    showPortfolioError('Please enter at least 2 stock symbols');
    return;
  }
  
  // Prepare constraints
  const constraints = { min_weights: {}, max_weights: {} };
  
  symbols.forEach(symbol => {
    if (minWeightInputs[symbol] && minWeightInputs[symbol].value) {
      const minValue = parseFloat(minWeightInputs[symbol].value) / 100;
      constraints.min_weights[symbol] = minValue;
    }
    
    if (maxWeightInputs[symbol] && maxWeightInputs[symbol].value) {
      const maxValue = parseFloat(maxWeightInputs[symbol].value) / 100;
      constraints.max_weights[symbol] = maxValue;
    }
  });
  
  // Create request data
  const requestData = {
    symbols: symbols,
    method: method,
    constraints: constraints
  };
  
  // Only include risk_tolerance for methods that use it
  if (method !== 'maximum_sharpe' && method !== 'minimum_volatility') {
    requestData.risk_tolerance = riskTolerance;
  }
  
  try {
    // Make API call
    const response = await fetch('/api/portfolio-optimization', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(requestData)
    });
    
    // Check if response is OK
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || 'Failed to optimize portfolio');
    }
    
    // Parse response data
    const data = await response.json();
    
    // Add to optimization history
    addToOptimizationHistory({
      id: optimizationHistoryData.length + 1,
      date: new Date().toLocaleString(),
      symbols: symbols,
      method: getMethodDisplayName(method),
      data: data
    });
    
    // Display the results
    displayPortfolioResults(data);
    
  } catch (error) {
    showPortfolioError(error.message);
  }
}

/**
 * Display portfolio optimization results
 * @param {Object} data - Portfolio optimization results from API
 */
function displayPortfolioResults(data) {
  // Hide loader, show results
  portfolioLoader.style.display = 'none';
  portfolioResult.style.display = 'block';
  
  // Draw allocation chart
  if (allocationChart && window.aiCharts && data.weights) {
    const isDarkMode = document.body.classList.contains('dark-mode');
    window.aiCharts.createPortfolioAllocationChart(allocationChart, data, isDarkMode);
  }
  
  // Display metrics
  if (metricsTable && data.metrics) {
    displayPortfolioMetrics(data.metrics);
  }
}

/**
 * Display portfolio metrics in a table
 * @param {Object} metrics - Portfolio metrics
 */
function displayPortfolioMetrics(metrics) {
  // Clear existing table rows
  metricsTable.innerHTML = '';
  
  // Create metrics rows
  const rows = [
    { label: 'Expected Annual Return', value: formatPercent(metrics.expected_return), icon: 'chart-line' },
    { label: 'Annual Volatility', value: formatPercent(metrics.volatility), icon: 'chart-area' },
    { label: 'Sharpe Ratio', value: metrics.sharpe_ratio.toFixed(2), icon: 'balance-scale' },
    { label: 'Maximum Drawdown', value: formatPercent(metrics.max_drawdown), icon: 'arrow-down' },
    { label: 'Value at Risk (95%)', value: formatPercent(metrics.var_95), icon: 'exclamation-triangle' },
    { label: 'Sortino Ratio', value: metrics.sortino_ratio.toFixed(2), icon: 'sort-amount-up' }
  ];
  
  // Add rows to table
  rows.forEach(row => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td><i class="fas fa-${row.icon}"></i> ${row.label}</td>
      <td>${row.value}</td>
    `;
    metricsTable.appendChild(tr);
  });
}

/**
 * Add an optimization result to history
 * @param {Object} optimization - Optimization details
 */
function addToOptimizationHistory(optimization) {
  // Add to history array (limit to 10 most recent)
  optimizationHistoryData.unshift(optimization);
  if (optimizationHistoryData.length > 10) {
    optimizationHistoryData.pop();
  }
  
  // Update history display
  updateOptimizationHistory();
}

/**
 * Update the optimization history display
 */
function updateOptimizationHistory() {
  if (!optimizationHistory) return;
  
  // Clear current history
  optimizationHistory.innerHTML = '';
  
  if (optimizationHistoryData.length === 0) {
    // Show empty state
    const emptyState = document.createElement('div');
    emptyState.className = 'empty-state';
    emptyState.innerHTML = `
      <i class="fas fa-history"></i>
      <p>No optimization history yet</p>
    `;
    optimizationHistory.appendChild(emptyState);
    return;
  }
  
  // Create history items
  optimizationHistoryData.forEach(item => {
    const historyItem = document.createElement('div');
    historyItem.className = 'history-item';
    historyItem.dataset.id = item.id;
    
    // Truncate long symbol lists
    let symbolsText = item.symbols.join(', ');
    if (symbolsText.length > 30) {
      symbolsText = symbolsText.substring(0, 27) + '...';
    }
    
    // Create history item content
    historyItem.innerHTML = `
      <div class="history-item-header">
        <span class="history-date">${item.date}</span>
        <button class="history-reload" title="Reload this optimization">
          <i class="fas fa-sync-alt"></i>
        </button>
      </div>
      <div class="history-item-body">
        <div class="history-method">${item.method}</div>
        <div class="history-symbols">${symbolsText}</div>
        <div class="history-metrics">
          <span title="Expected Return">
            <i class="fas fa-chart-line"></i> ${formatPercent(item.data.metrics.expected_return)}
          </span>
          <span title="Volatility">
            <i class="fas fa-chart-area"></i> ${formatPercent(item.data.metrics.volatility)}
          </span>
          <span title="Sharpe Ratio">
            <i class="fas fa-balance-scale"></i> ${item.data.metrics.sharpe_ratio.toFixed(2)}
          </span>
        </div>
      </div>
    `;
    
    // Add event listener to reload button
    const reloadButton = historyItem.querySelector('.history-reload');
    reloadButton.addEventListener('click', () => reloadOptimization(item));
    
    optimizationHistory.appendChild(historyItem);
  });
}

/**
 * Reload a previous optimization
 * @param {Object} optimization - Previous optimization to reload
 */
function reloadOptimization(optimization) {
  // Display the results
  displayPortfolioResults(optimization.data);
  
  // Scroll to the results section
  portfolioResult.scrollIntoView({ behavior: 'smooth' });
}

/**
 * Initialize tooltips for additional information
 */
function initTooltips() {
  const tooltips = document.querySelectorAll('[data-tooltip]');
  
  tooltips.forEach(tooltip => {
    const tooltipElement = document.createElement('div');
    tooltipElement.className = 'tooltip';
    tooltipElement.textContent = tooltip.dataset.tooltip;
    tooltip.appendChild(tooltipElement);
    
    tooltip.addEventListener('mouseenter', () => {
      tooltipElement.style.opacity = '1';
      tooltipElement.style.visibility = 'visible';
    });
    
    tooltip.addEventListener('mouseleave', () => {
      tooltipElement.style.opacity = '0';
      tooltipElement.style.visibility = 'hidden';
    });
  });
}

/**
 * Show error message
 * @param {string} message - Error message to display
 */
function showPortfolioError(message) {
  portfolioLoader.style.display = 'none';
  portfolioError.style.display = 'block';
  portfolioError.textContent = message;
}

/**
 * Format a number as a percentage
 * @param {number} value - Value to format
 * @returns {string} Formatted percentage
 */
function formatPercent(value) {
  return (value * 100).toFixed(2) + '%';
}

/**
 * Get a display friendly name for optimization methods
 * @param {string} method - Method name from API
 * @returns {string} Display name for the method
 */
function getMethodDisplayName(method) {
  const methodMap = {
    'maximum_sharpe': 'Maximum Sharpe Ratio',
    'minimum_volatility': 'Minimum Volatility',
    'efficient_risk': 'Efficient Risk',
    'efficient_return': 'Efficient Return',
    'maximum_diversification': 'Maximum Diversification',
    'equal_weighted': 'Equal Weighted'
  };
  
  return methodMap[method] || method;
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
  // Initialize if we're on the portfolio tab
  const portfolioTab = document.getElementById('portfolio-tab-content');
  if (portfolioTab) {
    initPortfolioOptimization();
  }
}); 