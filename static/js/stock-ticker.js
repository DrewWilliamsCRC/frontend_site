class StockTicker {
  constructor() {
    this.tickerContent = document.querySelector('.ticker-content');
    // All 30 stocks in the Dow Jones Industrial Average with their exchanges
    this.stocks = [
      { symbol: 'AAPL', exchange: 'NASDAQ' },  // Apple
      { symbol: 'AMGN', exchange: 'NASDAQ' },  // Amgen
      { symbol: 'AXP', exchange: 'NYSE' },     // American Express
      { symbol: 'BA', exchange: 'NYSE' },      // Boeing
      { symbol: 'CAT', exchange: 'NYSE' },     // Caterpillar
      { symbol: 'CRM', exchange: 'NYSE' },     // Salesforce
      { symbol: 'CSCO', exchange: 'NASDAQ' },  // Cisco
      { symbol: 'CVX', exchange: 'NYSE' },     // Chevron
      { symbol: 'DIS', exchange: 'NYSE' },     // Disney
      { symbol: 'DOW', exchange: 'NYSE' },     // Dow Inc
      { symbol: 'GS', exchange: 'NYSE' },      // Goldman Sachs
      { symbol: 'HD', exchange: 'NYSE' },      // Home Depot
      { symbol: 'HON', exchange: 'NASDAQ' },   // Honeywell
      { symbol: 'IBM', exchange: 'NYSE' },     // IBM
      { symbol: 'INTC', exchange: 'NASDAQ' },  // Intel
      { symbol: 'JNJ', exchange: 'NYSE' },     // Johnson & Johnson
      { symbol: 'JPM', exchange: 'NYSE' },     // JPMorgan Chase
      { symbol: 'KO', exchange: 'NYSE' },      // Coca-Cola
      { symbol: 'MCD', exchange: 'NYSE' },     // McDonald's
      { symbol: 'MMM', exchange: 'NYSE' },     // 3M
      { symbol: 'MRK', exchange: 'NYSE' },     // Merck
      { symbol: 'MSFT', exchange: 'NASDAQ' },  // Microsoft
      { symbol: 'NKE', exchange: 'NYSE' },     // Nike
      { symbol: 'PG', exchange: 'NYSE' },      // Procter & Gamble
      { symbol: 'TRV', exchange: 'NYSE' },     // Travelers
      { symbol: 'UNH', exchange: 'NYSE' },     // UnitedHealth
      { symbol: 'V', exchange: 'NYSE' },       // Visa
      { symbol: 'VZ', exchange: 'NYSE' },      // Verizon
      { symbol: 'WBA', exchange: 'NASDAQ' },   // Walgreens
      { symbol: 'WMT', exchange: 'NYSE' }      // Walmart
    ];
    this.updateInterval = 60000; // Update every minute
    this.lastUpdateTime = null;
    this.init();
  }

  async init() {
    await this.updateStockData();
    setInterval(() => this.updateStockData(), this.updateInterval);
  }

  async fetchStockData(stockInfo) {
    try {
      console.log(`Fetching data for ${stockInfo.symbol}`);  // Debug log
      const response = await fetch(`/api/stock/${stockInfo.symbol}`);
      if (!response.ok) throw new Error('Network response was not ok');
      const data = await response.json();
      console.log(`Received data for ${stockInfo.symbol}:`, data);  // Debug log
      return { ...data, exchange: stockInfo.exchange };
    } catch (error) {
      console.error('Error fetching stock data:', error);
      return null;
    }
  }

  async updateStockData() {
    const now = new Date();
    console.log('Starting stock data update at:', now.toLocaleTimeString());  // Debug log
    
    // Clear existing content
    this.tickerContent.innerHTML = '';

    try {
      // Fetch all stock data in parallel - we're within rate limits (75/min for 30 stocks)
      const stockDataPromises = this.stocks.map(stockInfo => this.fetchStockData(stockInfo));
      const stocksData = await Promise.all(stockDataPromises);
      console.log('All stock data received:', stocksData);  // Debug log

      // Check for API errors
      if (stocksData.length > 0) {
        const firstError = stocksData[0]?.error;
        if (firstError === 'API key not configured') {
          this.tickerContent.innerHTML = `
            <div class="stock-error">
              Stock data unavailable - API key not configured
            </div>`;
          return;
        } else if (firstError && firstError.includes('API rate limit')) {
          this.tickerContent.innerHTML = `
            <div class="stock-error">
              Stock data temporarily unavailable - API rate limit reached. Please try again in a few minutes.
            </div>`;
          return;
        }
      }

      // Create stock items HTML
      const stockItemsHTML = stocksData
        .filter(data => data !== null && !data.error)
        .map(data => {
          const changeClass = parseFloat(data.change) >= 0 ? 'positive' : 'negative';
          const arrowIcon = parseFloat(data.change) >= 0 ? 'fa-arrow-up' : 'fa-arrow-down';
          const googleFinanceUrl = `https://www.google.com/finance/quote/${data.symbol}:${data.exchange}`;
          
          console.log(`Creating HTML for ${data.symbol}:`, {  // Debug log
            price: data.price,
            change: data.change,
            url: googleFinanceUrl
          });

          return `
            <a href="${googleFinanceUrl}" 
               class="stock-item" 
               target="_blank" 
               rel="noopener noreferrer"
               title="View ${data.symbol} details on Google Finance">
              <span class="stock-symbol">${data.symbol}</span>
              <span class="stock-price">$${parseFloat(data.price).toFixed(2)}</span>
              <span class="stock-change ${changeClass}">
                <i class="fas ${arrowIcon}"></i>
                ${Math.abs(parseFloat(data.change)).toFixed(2)}%
              </span>
            </a>
          `;
        })
        .join('');
      
      // Add the stock items twice for smooth infinite scrolling
      if (stockItemsHTML) {
        this.tickerContent.innerHTML = stockItemsHTML + stockItemsHTML;
        this.lastUpdateTime = now;
      } else {
        this.tickerContent.innerHTML = `
          <div class="stock-error">
            Unable to load stock data
          </div>`;
      }
      console.log('Stock ticker update complete at:', now.toLocaleTimeString());  // Debug log
    } catch (error) {
      console.error('Error updating stock data:', error);
      this.tickerContent.innerHTML = `
        <div class="stock-error">
          Error updating stock data: ${error.message}
        </div>`;
    }
  }
}

// Initialize the stock ticker when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
  new StockTicker();
});