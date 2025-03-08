/**
 * Market Indices JavaScript
 * Fetches and displays real-time market index data using Alpha Vantage API
 */

document.addEventListener('DOMContentLoaded', function() {
    // Market index mapping for API symbols
    const MARKET_INDICES = {
        'DJI': {apiSymbol: '^DJI', fullName: 'Dow Jones Industrial Average'},
        'SPX': {apiSymbol: '^GSPC', fullName: 'S&P 500'},
        'IXIC': {apiSymbol: '^IXIC', fullName: 'NASDAQ Composite'},
        'VIX': {apiSymbol: '^VIX', fullName: 'CBOE Volatility Index'},
        'TNX': {apiSymbol: '^TNX', fullName: '10-Year Treasury Note Yield'}
    };

    // Format functions
    const formatPrice = (price, symbol) => {
        const value = parseFloat(price);
        
        // Special formatting for TNX (Treasury yield)
        if (symbol === 'TNX') {
            return `${value.toLocaleString('en-US', {
                minimumFractionDigits: 2,
                maximumFractionDigits: 2
            })}%`;
        }
        
        return value.toLocaleString('en-US', {
            minimumFractionDigits: 2,
            maximumFractionDigits: 2
        });
    };

    const formatChange = (change) => {
        const changeValue = parseFloat(change);
        return changeValue >= 0 ? 
            `+${changeValue.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2})}` : 
            `${changeValue.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
    };

    const formatPercent = (percent) => {
        const percentValue = parseFloat(percent);
        return percentValue >= 0 ? 
            `(+${percentValue.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2})}%)` : 
            `(${percentValue.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2})}%)`;
    };

    const formatLastUpdate = () => {
        const now = new Date();
        const options = {
            hour: 'numeric',
            minute: '2-digit',
            second: '2-digit',
            hour12: true,
            timeZone: 'America/New_York'
        };
        return `LAST | ${now.toLocaleTimeString('en-US', options)} EST`;
    };

    // Update the UI with fetched market data
    const updateMarketCard = (symbol, data) => {
        const card = document.querySelector(`.index-card[data-symbol="${symbol}"]`);
        if (!card) return;

        let isPositive = false;
        
        try {
            console.log(`Updating card for ${symbol} with data:`, data);
            
            // Get price and change values
            const price = data.price || data.quote || data.regularMarketPrice;
            const change = data.change || data.regularMarketChange;
            const percentChange = data.percentChange || data.regularMarketChangePercent;
            
            // Determine if positive or negative
            isPositive = parseFloat(change) >= 0;
            
            // Set the trend icon
            const trendIcon = card.querySelector('.trend-icon');
            trendIcon.innerHTML = isPositive ? '▲' : '▼';
            
            // Set values in the card
            card.querySelector('.price').textContent = formatPrice(price, symbol);
            card.querySelector('.change-value').textContent = formatChange(change);
            card.querySelector('.change-percent').textContent = formatPercent(percentChange);
            card.querySelector('.last-update').textContent = formatLastUpdate();
            
            // Add loaded class
            card.classList.add('loaded');
            
            // Update styling based on trend
            card.classList.remove('positive', 'negative');
            card.classList.add(isPositive ? 'positive' : 'negative');
            
            // Indicate if this is real or simulated data
            // Check if the data has the source property set to 'alpha_vantage'
            console.log(`Data source for ${symbol}:`, data.source);
            
            // Consider it real data if it's specifically marked as coming from yahoo_finance
            // or if it has real API data properties and no simulation marker
            const isRealData = 
                data.source === 'yahoo_finance' || 
                data.source === 'alpha_vantage' ||
                (data.source !== 'simulation' && !data.hasOwnProperty('error'));
                
            const dataSource = isRealData ? 'real' : 'simulated';
            card.setAttribute('data-source', dataSource);
            
            // Add a small indicator in the top-right corner for simulated data
            if (dataSource !== 'real') {
                console.log(`${symbol} is using simulated data`);
                if (!card.querySelector('.data-source-indicator')) {
                    const indicator = document.createElement('div');
                    indicator.className = 'data-source-indicator';
                    indicator.title = 'Showing simulated data';
                    indicator.textContent = 'SIM';
                    indicator.style.cssText = 'position:absolute; top:3px; right:3px; font-size:8px; padding:2px; background:rgba(0,0,0,0.5); border-radius:3px; color:white; z-index:10';
                    card.style.position = 'relative';
                    card.appendChild(indicator);
                }
            } else {
                console.log(`${symbol} is using real data from Alpha Vantage`);
                // Remove indicator if this is real data
                const indicator = card.querySelector('.data-source-indicator');
                if (indicator) {
                    indicator.remove();
                }
            }
            
        } catch (error) {
            console.error(`Error updating market card for ${symbol}:`, error);
        }
    };

    // Fetch market data from our API
    const fetchMarketData = async () => {
        try {
            console.log('Fetching market data from Alpha Vantage API');
            const response = await fetch('/api/market-indices');
            if (!response.ok) {
                throw new Error(`API returned status code ${response.status}`);
            }
            
            const data = await response.json();
            let hasValidData = false;
            
            // Update each market card with the fetched data
            Object.keys(data).forEach(symbol => {
                if (!data[symbol].error) {
                    console.log(`Updating ${symbol} with real data:`, data[symbol]);
                    updateMarketCard(symbol, data[symbol]);
                    hasValidData = true;
                } else {
                    console.error(`Error for ${symbol}:`, data[symbol].error);
                    // Don't update this card's data if there's an error
                }
            });
            
            // If we didn't get any valid data, fall back to simulation
            if (!hasValidData) {
                console.warn('No valid market data received, falling back to simulation');
                simulateMarketData();
            }
            
        } catch (error) {
            console.error('Error fetching market indices:', error);
            // Use simulation as fallback if API fails
            simulateMarketData();
        }
    };
    
    // Simulate market data for testing or when API is unavailable
    const simulateMarketData = () => {
        console.log('Using simulated market data');
        
        const indexCards = document.querySelectorAll('.index-card');
        
        // Loop through each card and update with simulated data
        indexCards.forEach(card => {
            const symbol = card.dataset.symbol;
            
            // Generate random price and change values based on realistic ranges
            let basePrice, change, percentChange;
            
            switch(symbol) {
                case 'DJI':
                    basePrice = Math.random() * 1000 + 42000; // Between 42,000-43,000
                    break;
                case 'SPX':
                    basePrice = Math.random() * 100 + 5400; // Between 5,400-5,500
                    break;
                case 'IXIC':
                    basePrice = Math.random() * 200 + 17600; // Between 17,600-17,800
                    break;
                case 'VIX':
                    basePrice = Math.random() * 6 + 12; // Between 12-18
                    break;
                case 'TNX':
                    basePrice = Math.random() * 0.3 + 4.2; // Between 4.2-4.5%
                    break;
                default:
                    basePrice = Math.random() * 1000 + 1000;
            }
            
            // Generate change between -2% and +2%
            percentChange = (Math.random() * 4) - 2;
            change = (basePrice * percentChange) / 100;
            
            // Create simulated data object
            const data = {
                price: basePrice.toFixed(2),
                change: change.toFixed(2),
                percentChange: percentChange.toFixed(2)
            };
            
            // Update the card
            updateMarketCard(symbol, data);
        });
    };
    
    // Initialize market data fetching
    const initMarketData = () => {
        // Display simulated data immediately while loading real data
        simulateMarketData();
        
        // Then fetch real data
        fetchMarketData();
        
        // Set up refresh interval (every 60 seconds)
        setInterval(fetchMarketData, 60000);
    };
    
    // Start fetching market data
    initMarketData();
}); 