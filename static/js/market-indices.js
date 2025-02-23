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
            
            // Show loading state
            indexData.innerHTML = `
                <div class="loading-spinner">
                    <i class="fas fa-spinner fa-spin"></i>
                    <div>Fetching ${index.name} data${index.etf ? ` (${index.etf})` : ''}...</div>
                    <div class="update-info">Last update: ${new Date().toLocaleTimeString()}</div>
                    <div class="update-info">(Updates every 5 minutes)</div>
                </div>`;

            const data = await this.fetchIndexData(index.symbol);

            if (data.error) {
                indexData.innerHTML = `
                    <div class="error-message">
                        <i class="fas fa-exclamation-circle"></i>
                        <span>${data.error}</span>
                        <div class="retry-button" onclick="marketIndices.updateIndices()">
                            <i class="fas fa-sync"></i> Retry
                        </div>
                    </div>`;
                continue;
            }

            const changeValue = parseFloat(data.change);
            const changeClass = changeValue >= 0 ? 'positive' : 'negative';
            const arrowIcon = changeValue >= 0 ? 'fa-arrow-up' : 'fa-arrow-down';

            indexData.innerHTML = `
                <div class="price">
                    ${index.showDollar ? '$' : ''}${data.price}
                </div>
                <div class="change ${changeClass}">
                    <i class="fas ${arrowIcon}"></i>
                    ${Math.abs(changeValue).toFixed(2)}%
                </div>
                <div class="description">${index.description}</div>
                <div class="update-info">
                    Last update: ${new Date().toLocaleTimeString()}
                </div>`;
        }
    }
}

// Initialize market indices when the DOM is loaded
let marketIndices;
document.addEventListener('DOMContentLoaded', () => {
    marketIndices = new MarketIndices();
}); 