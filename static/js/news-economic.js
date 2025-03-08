/**
 * News and Economic Indicators Module
 * Handles news sentiment analysis and economic indicators dashboard sections
 */

// Cache DOM elements
let sentimentList;
let sentimentLoader;
let sentimentError;
let sentimentChart;
let economicData;
let economicLoader;
let economicError;
let economicFilters;
let economicTable;

// Store data
let newsData = [];
let economicIndicators = [];

/**
 * Initialize the news and economic indicators module
 */
function initNewsEconomic() {
  // Cache sentiment DOM elements
  sentimentList = document.getElementById('sentiment-list');
  sentimentLoader = document.getElementById('sentiment-loader');
  sentimentError = document.getElementById('sentiment-error');
  sentimentChart = document.getElementById('sentiment-chart');
  
  // Cache economic DOM elements
  economicData = document.getElementById('economic-data');
  economicLoader = document.getElementById('economic-loader');
  economicError = document.getElementById('economic-error');
  economicFilters = document.getElementById('economic-filters');
  economicTable = document.getElementById('economic-table');
  
  // Add event listeners for filters
  if (economicFilters) {
    const categoryFilter = economicFilters.querySelector('#category-filter');
    const importanceFilter = economicFilters.querySelector('#importance-filter');
    
    if (categoryFilter) {
      categoryFilter.addEventListener('change', filterEconomicData);
    }
    
    if (importanceFilter) {
      importanceFilter.addEventListener('change', filterEconomicData);
    }
  }
  
  // Load data when tabs are activated
  document.addEventListener('click', function(e) {
    if (e.target.getAttribute('data-tab') === 'news-sentiment') {
      loadNewsSentiment();
    } else if (e.target.getAttribute('data-tab') === 'economic') {
      loadEconomicIndicators();
    }
  });
  
  // Auto-refresh data every 5 minutes
  setInterval(() => {
    if (isTabActive('news-sentiment')) {
      loadNewsSentiment();
    } else if (isTabActive('economic')) {
      loadEconomicIndicators();
    }
  }, 300000); // 5 minutes
}

/**
 * Check if a specific tab is currently active
 * @param {string} tabId - ID of the tab to check
 * @returns {boolean} Whether the tab is active
 */
function isTabActive(tabId) {
  const activeTab = document.querySelector('.tab-item.active');
  return activeTab && activeTab.getAttribute('data-tab') === tabId;
}

/**
 * Load news sentiment data from API
 */
async function loadNewsSentiment() {
  if (!sentimentList || !sentimentLoader) return;
  
  // Show loader
  sentimentLoader.style.display = 'block';
  sentimentError.style.display = 'none';
  
  try {
    // Make API call
    const response = await fetch('/api/news-sentiment', {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json'
      }
    });
    
    // Check if response is OK
    if (!response.ok) {
      throw new Error('Failed to load news sentiment data');
    }
    
    // Parse response data
    const data = await response.json();
    newsData = data.articles || [];
    
    // Display news sentiment
    displayNewsSentiment(newsData);
    
    // Create sentiment chart
    createSentimentChart(data.sentiment_summary);
    
  } catch (error) {
    showSentimentError(error.message);
  } finally {
    // Hide loader
    sentimentLoader.style.display = 'none';
  }
}

/**
 * Display news sentiment items
 * @param {Array} articles - News articles with sentiment
 */
function displayNewsSentiment(articles) {
  if (!sentimentList) return;
  
  // Clear current list
  sentimentList.innerHTML = '';
  
  if (articles.length === 0) {
    // Show empty state
    const emptyState = document.createElement('div');
    emptyState.className = 'empty-state';
    emptyState.innerHTML = `
      <i class="fas fa-newspaper"></i>
      <p>No news articles found</p>
    `;
    sentimentList.appendChild(emptyState);
    return;
  }
  
  // Create news items
  articles.forEach(article => {
    const newsItem = createNewsItem(article);
    sentimentList.appendChild(newsItem);
  });
}

/**
 * Create a news item element
 * @param {Object} article - News article data
 * @returns {HTMLElement} News item element
 */
function createNewsItem(article) {
  const item = document.createElement('div');
  item.className = `news-item ${getSentimentClass(article.sentiment)}`;
  
  // Format date
  const publishedDate = new Date(article.published_at);
  const formattedDate = publishedDate.toLocaleString();
  
  // Sentiment score badge
  const scoreBadge = document.createElement('div');
  scoreBadge.className = `sentiment-badge ${getSentimentClass(article.sentiment)}`;
  scoreBadge.innerHTML = `
    <i class="fas fa-${getSentimentIcon(article.sentiment)}"></i>
    <span>${article.sentiment_score.toFixed(2)}</span>
  `;
  
  // Create news item content
  item.innerHTML = `
    <div class="news-header">
      <h4 class="news-title">
        <a href="${article.url}" target="_blank" rel="noopener noreferrer">
          ${article.title}
        </a>
      </h4>
      ${scoreBadge.outerHTML}
    </div>
    <div class="news-meta">
      <span class="news-source">
        <i class="fas fa-globe"></i> ${article.source}
      </span>
      <span class="news-date">
        <i class="fas fa-clock"></i> ${formattedDate}
      </span>
    </div>
    <p class="news-summary">${article.summary}</p>
    <div class="news-entities">
      ${article.entities.slice(0, 5).map(entity => `
        <span class="entity-tag" title="${entity.type}">
          ${entity.name}
        </span>
      `).join('')}
    </div>
  `;
  
  return item;
}

/**
 * Create sentiment summary chart
 * @param {Object} sentimentSummary - Summary of sentiment data
 */
function createSentimentChart(sentimentSummary) {
  if (!sentimentChart || !window.aiCharts) return;
  
  const isDarkMode = document.body.classList.contains('dark-mode');
  
  // Create custom chart since this isn't one of the standard ones in ai-charts.js
  if (window.sentimentChartInstance) {
    window.sentimentChartInstance.destroy();
  }
  
  const ctx = sentimentChart.getContext('2d');
  
  // Prepare data
  const labels = ['Positive', 'Neutral', 'Negative'];
  const data = [
    sentimentSummary.positive_count,
    sentimentSummary.neutral_count,
    sentimentSummary.negative_count
  ];
  
  // Define colors
  const colors = isDarkMode ? [
    'rgba(105, 222, 222, 0.8)',
    'rgba(200, 200, 200, 0.8)',
    'rgba(255, 129, 152, 0.8)'
  ] : [
    'rgba(75, 192, 192, 0.8)',
    'rgba(231, 233, 237, 0.8)',
    'rgba(255, 99, 132, 0.8)'
  ];
  
  // Create chart
  window.sentimentChartInstance = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: labels,
      datasets: [{
        data: data,
        backgroundColor: colors,
        borderColor: isDarkMode ? '#1e1e1e' : '#ffffff',
        borderWidth: 1
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'right'
        },
        tooltip: {
          callbacks: {
            label: (context) => {
              const value = context.raw;
              const total = data.reduce((a, b) => a + b, 0);
              const percentage = total > 0 ? Math.round((value / total) * 100) : 0;
              return `${context.label}: ${value} (${percentage}%)`;
            }
          }
        }
      },
      cutout: '60%'
    }
  });
  
  // Add sentiment score in center
  if (sentimentSummary.average_score !== undefined) {
    const avgSentiment = sentimentSummary.average_score;
    
    // Plugin to display text in center
    const textCenter = {
      id: 'textCenter',
      beforeDraw: function(chart) {
        const width = chart.width;
        const height = chart.height;
        const ctx = chart.ctx;
        
        ctx.restore();
        ctx.font = '16px Roboto, sans-serif';
        ctx.textBaseline = 'middle';
        
        const text = `${avgSentiment.toFixed(2)}`;
        const textLabel = 'Avg. Sentiment';
        const textX = width / 2;
        const textY = height / 2 - 10;
        
        ctx.fillStyle = isDarkMode ? '#e0e0e0' : '#333333';
        ctx.textAlign = 'center';
        ctx.fillText(text, textX, textY);
        
        ctx.font = '12px Roboto, sans-serif';
        ctx.fillText(textLabel, textX, textY + 20);
        
        ctx.save();
      }
    };
    
    // Register plugin
    Chart.register(textCenter);
  }
}

/**
 * Load economic indicators data from API
 */
async function loadEconomicIndicators() {
  if (!economicTable || !economicLoader) return;
  
  // Show loader
  economicLoader.style.display = 'block';
  economicError.style.display = 'none';
  
  try {
    // Make API call
    const response = await fetch('/api/economic-indicators', {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json'
      }
    });
    
    // Check if response is OK
    if (!response.ok) {
      throw new Error('Failed to load economic indicators');
    }
    
    // Parse response data
    const data = await response.json();
    economicIndicators = data.indicators || [];
    
    // Populate category filter
    populateCategoryFilter(economicIndicators);
    
    // Display economic indicators
    displayEconomicIndicators(economicIndicators);
    
  } catch (error) {
    showEconomicError(error.message);
  } finally {
    // Hide loader
    economicLoader.style.display = 'none';
  }
}

/**
 * Populate category filter select with unique categories
 * @param {Array} indicators - Economic indicators data
 */
function populateCategoryFilter(indicators) {
  const categoryFilter = economicFilters.querySelector('#category-filter');
  if (!categoryFilter) return;
  
  // Get unique categories
  const categories = [...new Set(indicators.map(item => item.category))];
  
  // Keep the first option (All)
  categoryFilter.innerHTML = '<option value="all">All Categories</option>';
  
  // Add category options
  categories.forEach(category => {
    const option = document.createElement('option');
    option.value = category;
    option.textContent = category;
    categoryFilter.appendChild(option);
  });
}

/**
 * Filter economic data based on selected filters
 */
function filterEconomicData() {
  if (!economicTable || !economicFilters) return;
  
  const categoryFilter = economicFilters.querySelector('#category-filter');
  const importanceFilter = economicFilters.querySelector('#importance-filter');
  
  const category = categoryFilter ? categoryFilter.value : 'all';
  const importance = importanceFilter ? importanceFilter.value : 'all';
  
  // Apply filters
  let filteredData = [...economicIndicators];
  
  if (category !== 'all') {
    filteredData = filteredData.filter(item => item.category === category);
  }
  
  if (importance !== 'all') {
    const importanceValue = parseInt(importance);
    filteredData = filteredData.filter(item => item.importance === importanceValue);
  }
  
  // Display filtered data
  displayEconomicIndicators(filteredData);
}

/**
 * Display economic indicators in table
 * @param {Array} indicators - Economic indicators data
 */
function displayEconomicIndicators(indicators) {
  if (!economicTable) return;
  
  // Clear table
  economicTable.innerHTML = '';
  
  if (indicators.length === 0) {
    // Show empty state
    const emptyRow = document.createElement('tr');
    emptyRow.innerHTML = `
      <td colspan="6" class="empty-table">
        <div class="empty-state">
          <i class="fas fa-chart-line"></i>
          <p>No economic indicators found</p>
        </div>
      </td>
    `;
    economicTable.appendChild(emptyRow);
    return;
  }
  
  // Create table rows
  indicators.forEach(indicator => {
    const row = createEconomicRow(indicator);
    economicTable.appendChild(row);
  });
}

/**
 * Create table row for economic indicator
 * @param {Object} indicator - Economic indicator data
 * @returns {HTMLElement} Table row element
 */
function createEconomicRow(indicator) {
  const row = document.createElement('tr');
  
  // Format date
  const releaseDate = new Date(indicator.release_date);
  const formattedDate = releaseDate.toLocaleDateString();
  
  // Create importance stars
  let importanceStars = '';
  for (let i = 0; i < 3; i++) {
    if (i < indicator.importance) {
      importanceStars += '<i class="fas fa-star"></i>';
    } else {
      importanceStars += '<i class="far fa-star"></i>';
    }
  }
  
  // Determine value status
  let valueStatus = '';
  if (indicator.value !== null && indicator.previous !== null) {
    if (indicator.value > indicator.previous) {
      valueStatus = 'increase';
    } else if (indicator.value < indicator.previous) {
      valueStatus = 'decrease';
    }
  }
  
  // Create row content
  row.innerHTML = `
    <td class="indicator-name">
      <div>${indicator.name}</div>
      <span class="indicator-category">${indicator.category}</span>
    </td>
    <td class="indicator-importance">
      <div class="importance-stars">${importanceStars}</div>
    </td>
    <td class="indicator-value ${valueStatus}">
      ${indicator.value !== null ? indicator.value.toFixed(2) : 'N/A'}
      ${valueStatus ? `<i class="fas fa-caret-${valueStatus === 'increase' ? 'up' : 'down'}"></i>` : ''}
    </td>
    <td class="indicator-previous">
      ${indicator.previous !== null ? indicator.previous.toFixed(2) : 'N/A'}
    </td>
    <td class="indicator-forecast">
      ${indicator.forecast !== null ? indicator.forecast.toFixed(2) : 'N/A'}
    </td>
    <td class="indicator-date">${formattedDate}</td>
  `;
  
  return row;
}

/**
 * Show sentiment error message
 * @param {string} message - Error message to display
 */
function showSentimentError(message) {
  sentimentLoader.style.display = 'none';
  sentimentError.style.display = 'block';
  sentimentError.textContent = message;
}

/**
 * Show economic error message
 * @param {string} message - Error message to display
 */
function showEconomicError(message) {
  economicLoader.style.display = 'none';
  economicError.style.display = 'block';
  economicError.textContent = message;
}

/**
 * Get CSS class for sentiment
 * @param {string} sentiment - Sentiment label
 * @returns {string} CSS class name
 */
function getSentimentClass(sentiment) {
  switch (sentiment.toLowerCase()) {
    case 'positive':
      return 'positive';
    case 'negative':
      return 'negative';
    default:
      return 'neutral';
  }
}

/**
 * Get icon for sentiment
 * @param {string} sentiment - Sentiment label
 * @returns {string} Icon name
 */
function getSentimentIcon(sentiment) {
  switch (sentiment.toLowerCase()) {
    case 'positive':
      return 'smile';
    case 'negative':
      return 'frown';
    default:
      return 'meh';
  }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
  // Check if we're on a page with news/economic tabs
  const newsTab = document.querySelector('[data-tab="news-sentiment"]');
  const economicTab = document.querySelector('[data-tab="economic"]');
  
  if (newsTab || economicTab) {
    initNewsEconomic();
    
    // Load data for the active tab
    if (newsTab && newsTab.classList.contains('active')) {
      loadNewsSentiment();
    } else if (economicTab && economicTab.classList.contains('active')) {
      loadEconomicIndicators();
    }
  }
}); 