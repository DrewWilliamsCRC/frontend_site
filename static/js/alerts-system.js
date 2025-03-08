/**
 * Alerts System Module
 * Handles creating, viewing, and managing market alerts
 */

// Cache DOM elements
let alertsList;
let createAlertForm;
let alertsLoader;
let alertsError;
let alertTypeSelect;
let alertParamsContainer;
let activeAlertCount;

// Store alerts data
let userAlerts = [];

/**
 * Initialize the alerts system module
 */
function initAlertsSystem() {
  // Cache DOM elements
  alertsList = document.getElementById('alerts-list');
  createAlertForm = document.getElementById('create-alert-form');
  alertsLoader = document.getElementById('alerts-loader');
  alertsError = document.getElementById('alerts-error');
  alertTypeSelect = document.getElementById('alert-type');
  alertParamsContainer = document.getElementById('alert-params-container');
  activeAlertCount = document.getElementById('active-alert-count');
  
  // Set up event listeners
  if (createAlertForm) {
    createAlertForm.addEventListener('submit', handleCreateAlert);
  }
  
  if (alertTypeSelect) {
    alertTypeSelect.addEventListener('change', updateAlertParams);
  }
  
  // Load existing alerts
  loadAlerts();
  
  // Set up auto refresh (every 60 seconds)
  setInterval(loadAlerts, 60000);
}

/**
 * Load alerts from API
 */
async function loadAlerts() {
  if (!alertsList || !alertsLoader) return;
  
  // Show loader
  alertsLoader.style.display = 'block';
  
  try {
    // Make API call
    const response = await fetch('/api/alerts', {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json'
      }
    });
    
    // Check if response is OK
    if (!response.ok) {
      throw new Error('Failed to load alerts');
    }
    
    // Parse response data
    const data = await response.json();
    userAlerts = data.alerts || [];
    
    // Display alerts
    displayAlerts(userAlerts);
    
    // Update active alert count
    updateActiveAlertCount();
    
  } catch (error) {
    showAlertsError(error.message);
  } finally {
    // Hide loader
    alertsLoader.style.display = 'none';
  }
}

/**
 * Display alerts in the UI
 * @param {Array} alerts - List of user alerts
 */
function displayAlerts(alerts) {
  if (!alertsList) return;
  
  // Clear current alerts
  alertsList.innerHTML = '';
  
  if (alerts.length === 0) {
    // Show empty state
    const emptyState = document.createElement('div');
    emptyState.className = 'empty-state';
    emptyState.innerHTML = `
      <i class="fas fa-bell-slash"></i>
      <p>No alerts configured</p>
      <p class="empty-state-help">Create your first alert using the form below</p>
    `;
    alertsList.appendChild(emptyState);
    return;
  }
  
  // Create alert cards
  alerts.forEach(alert => {
    const alertCard = createAlertCard(alert);
    alertsList.appendChild(alertCard);
  });
}

/**
 * Create an alert card element
 * @param {Object} alert - Alert data
 * @returns {HTMLElement} Alert card element
 */
function createAlertCard(alert) {
  const card = document.createElement('div');
  card.className = `alert-card ${alert.status.toLowerCase()}`;
  card.dataset.id = alert.id;
  
  // Create alert status badge
  const statusBadge = document.createElement('div');
  statusBadge.className = `status-badge ${alert.status.toLowerCase()}`;
  statusBadge.innerHTML = `<i class="fas fa-${getAlertStatusIcon(alert.status)}"></i> ${alert.status}`;
  
  // Create delete button
  const deleteButton = document.createElement('button');
  deleteButton.className = 'delete-alert';
  deleteButton.innerHTML = `<i class="fas fa-trash"></i>`;
  deleteButton.title = 'Delete alert';
  deleteButton.addEventListener('click', (e) => {
    e.stopPropagation();
    deleteAlert(alert.id);
  });
  
  // Create toggle button
  const toggleButton = document.createElement('button');
  toggleButton.className = 'toggle-alert';
  const isActive = alert.status !== 'DISABLED';
  toggleButton.innerHTML = `<i class="fas fa-${isActive ? 'pause' : 'play'}"></i>`;
  toggleButton.title = isActive ? 'Disable alert' : 'Enable alert';
  toggleButton.addEventListener('click', (e) => {
    e.stopPropagation();
    toggleAlert(alert.id, isActive);
  });
  
  // Create alert header
  const header = document.createElement('div');
  header.className = 'alert-header';
  header.innerHTML = `
    <h4 class="alert-name">${alert.name}</h4>
    <div class="alert-actions">
      ${statusBadge.outerHTML}
      ${toggleButton.outerHTML}
      ${deleteButton.outerHTML}
    </div>
  `;
  
  // Create alert details
  const details = document.createElement('div');
  details.className = 'alert-details';
  
  // Format alert details based on type
  const detailsHTML = formatAlertDetails(alert);
  details.innerHTML = detailsHTML;
  
  // Create alert footer with dates
  const footer = document.createElement('div');
  footer.className = 'alert-footer';
  footer.innerHTML = `
    <div class="alert-dates">
      <span title="Created">
        <i class="fas fa-calendar-plus"></i> ${new Date(alert.created_at).toLocaleString()}
      </span>
      ${alert.last_triggered_at ? `
        <span title="Last triggered">
          <i class="fas fa-bell"></i> ${new Date(alert.last_triggered_at).toLocaleString()}
        </span>
      ` : ''}
    </div>
  `;
  
  // Add collapse/expand functionality
  const toggleDetails = () => {
    if (details.style.display === 'none') {
      details.style.display = 'block';
      card.classList.add('expanded');
    } else {
      details.style.display = 'none';
      card.classList.remove('expanded');
    }
  };
  
  header.addEventListener('click', toggleDetails);
  
  // Assemble card
  card.appendChild(header);
  card.appendChild(details);
  card.appendChild(footer);
  
  // Start collapsed
  details.style.display = 'none';
  
  return card;
}

/**
 * Format alert details based on alert type
 * @param {Object} alert - Alert data
 * @returns {string} HTML for alert details
 */
function formatAlertDetails(alert) {
  let detailsHTML = `<div class="alert-type-label">${getAlertTypeLabel(alert.type)}</div>`;
  
  switch (alert.type) {
    case 'price_threshold':
      detailsHTML += `
        <div class="alert-param">
          <span class="param-label">Symbol:</span>
          <span class="param-value">${alert.params.symbol}</span>
        </div>
        <div class="alert-param">
          <span class="param-label">Condition:</span>
          <span class="param-value">${alert.params.direction === 'above' ? 'Above' : 'Below'} ${alert.params.threshold}</span>
        </div>
      `;
      break;
      
    case 'price_change':
      detailsHTML += `
        <div class="alert-param">
          <span class="param-label">Symbol:</span>
          <span class="param-value">${alert.params.symbol}</span>
        </div>
        <div class="alert-param">
          <span class="param-label">Percent Change:</span>
          <span class="param-value">${alert.params.direction === 'increase' ? '+' : '-'}${alert.params.percent_change}%</span>
        </div>
        <div class="alert-param">
          <span class="param-label">Time Frame:</span>
          <span class="param-value">${formatTimeFrame(alert.params.time_frame)}</span>
        </div>
      `;
      break;
      
    case 'volatility':
      detailsHTML += `
        <div class="alert-param">
          <span class="param-label">Symbol:</span>
          <span class="param-value">${alert.params.symbol}</span>
        </div>
        <div class="alert-param">
          <span class="param-label">Threshold:</span>
          <span class="param-value">${alert.params.threshold}%</span>
        </div>
        <div class="alert-param">
          <span class="param-label">Time Frame:</span>
          <span class="param-value">${formatTimeFrame(alert.params.time_frame)}</span>
        </div>
      `;
      break;
      
    case 'volume_spike':
      detailsHTML += `
        <div class="alert-param">
          <span class="param-label">Symbol:</span>
          <span class="param-value">${alert.params.symbol}</span>
        </div>
        <div class="alert-param">
          <span class="param-label">Multiple:</span>
          <span class="param-value">${alert.params.multiple}x average</span>
        </div>
      `;
      break;
      
    case 'moving_average_cross':
      detailsHTML += `
        <div class="alert-param">
          <span class="param-label">Symbol:</span>
          <span class="param-value">${alert.params.symbol}</span>
        </div>
        <div class="alert-param">
          <span class="param-label">Fast Period:</span>
          <span class="param-value">${alert.params.fast_period} days</span>
        </div>
        <div class="alert-param">
          <span class="param-label">Slow Period:</span>
          <span class="param-value">${alert.params.slow_period} days</span>
        </div>
        <div class="alert-param">
          <span class="param-label">Direction:</span>
          <span class="param-value">${alert.params.direction === 'above' ? 'Fast crosses above slow' : 'Fast crosses below slow'}</span>
        </div>
      `;
      break;
      
    case 'rsi_threshold':
      detailsHTML += `
        <div class="alert-param">
          <span class="param-label">Symbol:</span>
          <span class="param-value">${alert.params.symbol}</span>
        </div>
        <div class="alert-param">
          <span class="param-label">Period:</span>
          <span class="param-value">${alert.params.period} days</span>
        </div>
        <div class="alert-param">
          <span class="param-label">Condition:</span>
          <span class="param-value">RSI ${alert.params.direction === 'above' ? 'above' : 'below'} ${alert.params.threshold}</span>
        </div>
      `;
      break;
      
    default:
      detailsHTML += `<div class="alert-param">Custom alert configuration</div>`;
  }
  
  // Add notification settings
  detailsHTML += `
    <div class="alert-notifications">
      <div class="notification-label">Notifications:</div>
      <div class="notification-methods">
        ${alert.notification_methods.includes('email') ? 
          `<span class="notification-method active"><i class="fas fa-envelope"></i> Email</span>` : 
          `<span class="notification-method inactive"><i class="fas fa-envelope"></i> Email</span>`}
        ${alert.notification_methods.includes('app') ? 
          `<span class="notification-method active"><i class="fas fa-bell"></i> App</span>` : 
          `<span class="notification-method inactive"><i class="fas fa-bell"></i> App</span>`}
      </div>
    </div>
  `;
  
  return detailsHTML;
}

/**
 * Update the alert parameters form based on selected alert type
 */
function updateAlertParams() {
  if (!alertTypeSelect || !alertParamsContainer) return;
  
  const alertType = alertTypeSelect.value;
  alertParamsContainer.innerHTML = '';
  
  let paramsHTML = '';
  
  switch (alertType) {
    case 'price_threshold':
      paramsHTML = `
        <div class="form-group">
          <label for="alert-symbol">Symbol</label>
          <input type="text" id="alert-symbol" name="symbol" required placeholder="e.g. AAPL">
        </div>
        <div class="form-group">
          <label for="alert-direction">Direction</label>
          <select id="alert-direction" name="direction" required>
            <option value="above">Above</option>
            <option value="below">Below</option>
          </select>
        </div>
        <div class="form-group">
          <label for="alert-threshold">Price Threshold</label>
          <input type="number" id="alert-threshold" name="threshold" step="0.01" required placeholder="e.g. 150.00">
        </div>
      `;
      break;
      
    case 'price_change':
      paramsHTML = `
        <div class="form-group">
          <label for="alert-symbol">Symbol</label>
          <input type="text" id="alert-symbol" name="symbol" required placeholder="e.g. AAPL">
        </div>
        <div class="form-group">
          <label for="alert-direction">Direction</label>
          <select id="alert-direction" name="direction" required>
            <option value="increase">Increase</option>
            <option value="decrease">Decrease</option>
          </select>
        </div>
        <div class="form-group">
          <label for="alert-percent-change">Percent Change</label>
          <input type="number" id="alert-percent-change" name="percent_change" step="0.1" min="0.1" required placeholder="e.g. 5.0">
        </div>
        <div class="form-group">
          <label for="alert-time-frame">Time Frame</label>
          <select id="alert-time-frame" name="time_frame" required>
            <option value="1h">1 Hour</option>
            <option value="24h">24 Hours</option>
            <option value="7d">7 Days</option>
            <option value="30d">30 Days</option>
          </select>
        </div>
      `;
      break;
      
    case 'volatility':
      paramsHTML = `
        <div class="form-group">
          <label for="alert-symbol">Symbol</label>
          <input type="text" id="alert-symbol" name="symbol" required placeholder="e.g. AAPL">
        </div>
        <div class="form-group">
          <label for="alert-threshold">Volatility Threshold (%)</label>
          <input type="number" id="alert-threshold" name="threshold" step="0.1" min="1" required placeholder="e.g. 20">
        </div>
        <div class="form-group">
          <label for="alert-time-frame">Time Frame</label>
          <select id="alert-time-frame" name="time_frame" required>
            <option value="24h">24 Hours</option>
            <option value="7d">7 Days</option>
            <option value="30d">30 Days</option>
          </select>
        </div>
      `;
      break;
      
    case 'volume_spike':
      paramsHTML = `
        <div class="form-group">
          <label for="alert-symbol">Symbol</label>
          <input type="text" id="alert-symbol" name="symbol" required placeholder="e.g. AAPL">
        </div>
        <div class="form-group">
          <label for="alert-multiple">Volume Multiple</label>
          <input type="number" id="alert-multiple" name="multiple" step="0.5" min="1.5" required placeholder="e.g. 3">
          <small>Alert when volume is X times the average</small>
        </div>
      `;
      break;
      
    case 'moving_average_cross':
      paramsHTML = `
        <div class="form-group">
          <label for="alert-symbol">Symbol</label>
          <input type="text" id="alert-symbol" name="symbol" required placeholder="e.g. AAPL">
        </div>
        <div class="form-group">
          <label for="alert-fast-period">Fast MA Period (days)</label>
          <input type="number" id="alert-fast-period" name="fast_period" min="1" max="50" required placeholder="e.g. 10">
        </div>
        <div class="form-group">
          <label for="alert-slow-period">Slow MA Period (days)</label>
          <input type="number" id="alert-slow-period" name="slow_period" min="5" max="200" required placeholder="e.g. 50">
        </div>
        <div class="form-group">
          <label for="alert-direction">Cross Direction</label>
          <select id="alert-direction" name="direction" required>
            <option value="above">Fast crosses above slow</option>
            <option value="below">Fast crosses below slow</option>
          </select>
        </div>
      `;
      break;
      
    case 'rsi_threshold':
      paramsHTML = `
        <div class="form-group">
          <label for="alert-symbol">Symbol</label>
          <input type="text" id="alert-symbol" name="symbol" required placeholder="e.g. AAPL">
        </div>
        <div class="form-group">
          <label for="alert-period">RSI Period (days)</label>
          <input type="number" id="alert-period" name="period" min="2" max="30" value="14" required>
        </div>
        <div class="form-group">
          <label for="alert-direction">Direction</label>
          <select id="alert-direction" name="direction" required>
            <option value="above">Above</option>
            <option value="below">Below</option>
          </select>
        </div>
        <div class="form-group">
          <label for="alert-threshold">RSI Threshold</label>
          <input type="number" id="alert-threshold" name="threshold" min="1" max="99" required placeholder="e.g. 70">
        </div>
      `;
      break;
  }
  
  // Add notification methods - always the same for all alert types
  paramsHTML += `
    <div class="form-group">
      <label>Notification Methods</label>
      <div class="checkbox-group">
        <label class="checkbox-label">
          <input type="checkbox" name="notification_email" checked>
          <span>Email</span>
        </label>
        <label class="checkbox-label">
          <input type="checkbox" name="notification_app" checked>
          <span>App</span>
        </label>
      </div>
    </div>
  `;
  
  alertParamsContainer.innerHTML = paramsHTML;
}

/**
 * Handle create alert form submission
 * @param {Event} e - Form submission event
 */
async function handleCreateAlert(e) {
  e.preventDefault();
  
  // Show loader, hide errors
  alertsLoader.style.display = 'block';
  alertsError.style.display = 'none';
  
  // Get alert name and type
  const name = document.getElementById('alert-name').value;
  const type = alertTypeSelect.value;
  
  // Get notification methods
  const notificationMethods = [];
  if (document.querySelector('input[name="notification_email"]').checked) {
    notificationMethods.push('email');
  }
  if (document.querySelector('input[name="notification_app"]').checked) {
    notificationMethods.push('app');
  }
  
  // Get parameters based on alert type
  const params = {};
  
  switch (type) {
    case 'price_threshold':
      params.symbol = document.getElementById('alert-symbol').value;
      params.direction = document.getElementById('alert-direction').value;
      params.threshold = parseFloat(document.getElementById('alert-threshold').value);
      break;
      
    case 'price_change':
      params.symbol = document.getElementById('alert-symbol').value;
      params.direction = document.getElementById('alert-direction').value;
      params.percent_change = parseFloat(document.getElementById('alert-percent-change').value);
      params.time_frame = document.getElementById('alert-time-frame').value;
      break;
      
    case 'volatility':
      params.symbol = document.getElementById('alert-symbol').value;
      params.threshold = parseFloat(document.getElementById('alert-threshold').value);
      params.time_frame = document.getElementById('alert-time-frame').value;
      break;
      
    case 'volume_spike':
      params.symbol = document.getElementById('alert-symbol').value;
      params.multiple = parseFloat(document.getElementById('alert-multiple').value);
      break;
      
    case 'moving_average_cross':
      params.symbol = document.getElementById('alert-symbol').value;
      params.fast_period = parseInt(document.getElementById('alert-fast-period').value);
      params.slow_period = parseInt(document.getElementById('alert-slow-period').value);
      params.direction = document.getElementById('alert-direction').value;
      break;
      
    case 'rsi_threshold':
      params.symbol = document.getElementById('alert-symbol').value;
      params.period = parseInt(document.getElementById('alert-period').value);
      params.direction = document.getElementById('alert-direction').value;
      params.threshold = parseFloat(document.getElementById('alert-threshold').value);
      break;
  }
  
  // Create request data
  const requestData = {
    name: name,
    type: type,
    params: params,
    notification_methods: notificationMethods
  };
  
  try {
    // Make API call
    const response = await fetch('/api/alerts', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(requestData)
    });
    
    // Check if response is OK
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || 'Failed to create alert');
    }
    
    // Reset form
    createAlertForm.reset();
    
    // Reload alerts
    loadAlerts();
    
  } catch (error) {
    showAlertsError(error.message);
  } finally {
    // Hide loader
    alertsLoader.style.display = 'none';
  }
}

/**
 * Delete an alert
 * @param {string} alertId - ID of the alert to delete
 */
async function deleteAlert(alertId) {
  if (!confirm('Are you sure you want to delete this alert?')) {
    return;
  }
  
  // Show loader
  alertsLoader.style.display = 'block';
  
  try {
    // Make API call
    const response = await fetch(`/api/alerts/${alertId}`, {
      method: 'DELETE',
      headers: {
        'Content-Type': 'application/json'
      }
    });
    
    // Check if response is OK
    if (!response.ok) {
      throw new Error('Failed to delete alert');
    }
    
    // Reload alerts
    loadAlerts();
    
  } catch (error) {
    showAlertsError(error.message);
  } finally {
    // Hide loader
    alertsLoader.style.display = 'none';
  }
}

/**
 * Toggle alert enabled/disabled state
 * @param {string} alertId - ID of the alert to toggle
 * @param {boolean} isCurrentlyActive - Whether the alert is currently active
 */
async function toggleAlert(alertId, isCurrentlyActive) {
  // Show loader
  alertsLoader.style.display = 'block';
  
  const newStatus = isCurrentlyActive ? 'DISABLED' : 'ACTIVE';
  
  try {
    // Make API call
    const response = await fetch(`/api/alerts/${alertId}/status`, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ status: newStatus })
    });
    
    // Check if response is OK
    if (!response.ok) {
      throw new Error('Failed to update alert status');
    }
    
    // Reload alerts
    loadAlerts();
    
  } catch (error) {
    showAlertsError(error.message);
  } finally {
    // Hide loader
    alertsLoader.style.display = 'none';
  }
}

/**
 * Update the active alert count badge
 */
function updateActiveAlertCount() {
  if (!activeAlertCount) return;
  
  const count = userAlerts.filter(alert => alert.status === 'ACTIVE').length;
  activeAlertCount.textContent = count.toString();
  activeAlertCount.style.display = count > 0 ? 'inline-flex' : 'none';
}

/**
 * Show error message
 * @param {string} message - Error message to display
 */
function showAlertsError(message) {
  alertsLoader.style.display = 'none';
  alertsError.style.display = 'block';
  alertsError.textContent = message;
}

/**
 * Get the icon for an alert status
 * @param {string} status - Alert status
 * @returns {string} Icon name
 */
function getAlertStatusIcon(status) {
  switch (status) {
    case 'ACTIVE':
      return 'check-circle';
    case 'TRIGGERED':
      return 'bell';
    case 'DISABLED':
      return 'times-circle';
    case 'ERROR':
      return 'exclamation-triangle';
    default:
      return 'question-circle';
  }
}

/**
 * Get a display friendly name for alert types
 * @param {string} type - Alert type
 * @returns {string} Display name for the alert type
 */
function getAlertTypeLabel(type) {
  const typeMap = {
    'price_threshold': 'Price Threshold',
    'price_change': 'Price Change',
    'volatility': 'Volatility',
    'volume_spike': 'Volume Spike',
    'moving_average_cross': 'Moving Average Cross',
    'rsi_threshold': 'RSI Threshold'
  };
  
  return typeMap[type] || type;
}

/**
 * Format a time frame for display
 * @param {string} timeFrame - Time frame code
 * @returns {string} Formatted time frame
 */
function formatTimeFrame(timeFrame) {
  const timeMap = {
    '1h': '1 Hour',
    '24h': '24 Hours',
    '7d': '7 Days',
    '30d': '30 Days'
  };
  
  return timeMap[timeFrame] || timeFrame;
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
  // Initialize if we're on the alerts tab
  const alertsTab = document.getElementById('alerts-tab-content');
  if (alertsTab) {
    initAlertsSystem();
  }
}); 