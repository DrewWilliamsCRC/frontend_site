class MarketIndices {
    constructor() {
        this.indices = [
            { 
                symbol: 'DJI', 
                name: 'Dow Jones',
                etf: 'DIA',
                showDollar: true,
                description: 'Based on DIA ETF'
            },
            { 
                symbol: 'SPX', 
                name: 'S&P 500',
                etf: 'SPY',
                showDollar: true,
                description: 'Based on SPY ETF'
            },
            { 
                symbol: 'IXIC', 
                name: 'NASDAQ',
                etf: 'QQQ',
                showDollar: true,
                description: 'Based on QQQ ETF'
            },
            { 
                symbol: 'VIXY', 
                name: 'VIX',
                showDollar: true,
                description: 'ProShares VIX Short-Term Futures ETF'
            },
            {
                symbol: 'UST10Y',
                name: '10Y Treasury',
                showDollar: false,
                isTreasury: true,
                description: '10-Year Treasury Yield'
            }
        ];
        this.updateInterval = 300000; // 5 minutes
        this.init();
    }

    async init() {
        await this.updateIndices();
        setInterval(() => this.updateIndices(), this.updateInterval);
    }

    async fetchIndexData(symbol) {
        try {
            const response = await fetch(`/api/stock/${symbol}`);
            if (!response.ok) throw new Error('Network response was not ok');
            return await response.json();
        } catch (error) {
            console.error(`Error fetching ${symbol} data:`, error);
            return { error: 'Failed to fetch data' };
        }
    }

    async updateIndices() {
        for (const index of this.indices) {
            const card = document.querySelector(`[data-index="${index.symbol}"]`);
            if (!card) continue;

            const indexData = card.querySelector('.index-data');
            const lastUpdate = card.querySelector('.last-update');
            
            // Show loading state
            indexData.innerHTML = `
                <div class="loading-spinner">
                    <i class="fas fa-spinner fa-spin"></i>
                    <div>Loading...</div>
                </div>`;

            const data = await this.fetchIndexData(index.symbol);

            if (data.error) {
                indexData.innerHTML = `
                    <div class="error-message">
                        <i class="fas fa-exclamation-circle"></i>
                        <span>${data.error}</span>
                    </div>`;
                continue;
            }

            const changeValue = parseFloat(data.change);
            const changeClass = changeValue >= 0 ? 'positive' : 'negative';
            const triangleSymbol = changeValue >= 0 ? '▲' : '▼';
            
            // Format price with commas and exactly 2 decimal places
            const priceValue = parseFloat(data.price).toLocaleString('en-US', {
                minimumFractionDigits: 2,
                maximumFractionDigits: 2
            });

            // Format change value with exactly 2 decimal places
            const changeAmount = Math.abs(parseFloat(data.change)).toFixed(2);
            // Calculate percentage change
            const changePercent = ((changeValue / parseFloat(data.price)) * 100).toFixed(2);
            const changePrefix = changeValue >= 0 ? '+' : '-';

            // Update card class for background color
            card.classList.remove('positive', 'negative');
            card.classList.add(changeClass);

            // Update trend icon
            const trendIcon = card.querySelector('.trend-icon');
            trendIcon.textContent = triangleSymbol;

            // Clear any existing loading spinner
            indexData.innerHTML = '';

            // Create and append price element
            const priceDiv = document.createElement('div');
            priceDiv.className = 'price';
            if (index.isTreasury) {
                priceDiv.textContent = `${priceValue}%`;
            } else {
                priceDiv.textContent = `${index.showDollar ? '$' : ''}${priceValue}`;
            }
            indexData.appendChild(priceDiv);

            // Create and append change element
            const changeDiv = document.createElement('div');
            changeDiv.className = 'change';
            if (index.isTreasury) {
                changeDiv.textContent = `${changePrefix}${changeAmount}%`;
            } else {
                changeDiv.textContent = `${changePrefix}$${changeAmount} (${changePrefix}${Math.abs(changePercent)}%)`;
            }
            indexData.appendChild(changeDiv);

            // Update timestamp with exact format from photo
            const now = new Date();
            lastUpdate.textContent = `LAST | ${now.toLocaleTimeString('en-US', {
                hour: 'numeric',
                minute: '2-digit',
                second: '2-digit',
                hour12: true,
                timeZone: 'America/New_York'
            })} EST`;
        }
    }
}

// Initialize market indices when the DOM is loaded
let marketIndices;
document.addEventListener('DOMContentLoaded', () => {
    marketIndices = new MarketIndices();
}); 