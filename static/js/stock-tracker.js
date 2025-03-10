class StockTracker {
    constructor() {
        this.endpointSelector = document.getElementById('endpointSelector');
        this.optionsSelector = document.getElementById('optionsSelector');
        this.loadingSpinner = document.getElementById('loadingSpinner');
        this.errorMessage = document.getElementById('errorMessage');
        this.tableHeader = document.getElementById('tableHeader');
        this.tableBody = document.getElementById('tableBody');
        this.stockForm = document.getElementById('stockForm');
        this.searchInput = document.getElementById('tableSearch');
        this.currentData = null; // Store current data for searching
        
        // Add DOW companies list
        this.dowCompanies = [
            { symbol: 'AAPL', name: 'Apple Inc.' },
            { symbol: 'AMGN', name: 'Amgen Inc.' },
            { symbol: 'AXP', name: 'American Express Co.' },
            { symbol: 'BA', name: 'Boeing Co.' },
            { symbol: 'CAT', name: 'Caterpillar Inc.' },
            { symbol: 'CRM', name: 'Salesforce Inc.' },
            { symbol: 'CSCO', name: 'Cisco Systems Inc.' },
            { symbol: 'CVX', name: 'Chevron Corp.' },
            { symbol: 'DIS', name: 'Walt Disney Co.' },
            { symbol: 'DOW', name: 'Dow Inc.' },
            { symbol: 'GS', name: 'Goldman Sachs Group Inc.' },
            { symbol: 'HD', name: 'Home Depot Inc.' },
            { symbol: 'HON', name: 'Honeywell International Inc.' },
            { symbol: 'IBM', name: 'International Business Machines Corp.' },
            { symbol: 'INTC', name: 'Intel Corp.' },
            { symbol: 'JNJ', name: 'Johnson & Johnson' },
            { symbol: 'JPM', name: 'JPMorgan Chase & Co.' },
            { symbol: 'KO', name: 'Coca-Cola Co.' },
            { symbol: 'MCD', name: "McDonald's Corp." },
            { symbol: 'MMM', name: '3M Co.' },
            { symbol: 'MRK', name: 'Merck & Co. Inc.' },
            { symbol: 'MSFT', name: 'Microsoft Corp.' },
            { symbol: 'NKE', name: 'Nike Inc.' },
            { symbol: 'PG', name: 'Procter & Gamble Co.' },
            { symbol: 'TRV', name: 'Travelers Companies Inc.' },
            { symbol: 'UNH', name: 'UnitedHealth Group Inc.' },
            { symbol: 'V', name: 'Visa Inc.' },
            { symbol: 'VZ', name: 'Verizon Communications Inc.' },
            { symbol: 'WBA', name: 'Walgreens Boots Alliance Inc.' },
            { symbol: 'WMT', name: 'Walmart Inc.' }
        ];
        
        // Add cache storage
        this.cache = {
            DOW_COMPANIES: {
                data: null,
                timestamp: null,
                expiryTime: 5 * 60 * 1000 // 5 minutes in milliseconds
            }
        };
        
        // Add pagination state
        this.currentPage = 1;
        this.rowsPerPage = 50;
        
        // Configure endpoint options
        this.endpointConfig = {
            'STOCK_SYMBOL': {
                options: ['all'],
                optionLabels: {
                    'all': 'Current Price'
                },
                columns: ['Symbol', 'Price', 'Change %']
            },
            'DOW_COMPANIES': {
                options: ['all'],
                optionLabels: {
                    'all': 'All Companies'
                },
                columns: ['Symbol', 'Name', 'Price', 'Change', 'Change %', 'Actions']
            },
            'TOP_GAINERS_LOSERS': {
                options: ['top_gainers', 'top_losers', 'most_actively_traded'],
                optionLabels: {
                    'top_gainers': 'Top Gainers',
                    'top_losers': 'Top Losers',
                    'most_actively_traded': 'Most Active'
                },
                columns: ['Symbol', 'Name', 'Price', 'Change', 'Change %', 'Volume']
            },
            'SECTOR': {
                options: ['real_time', '1day', '5day', '1month', '3month', 'ytd', '1year', '3year', '5year', '10year'],
                optionLabels: {
                    'real_time': 'Real-Time',
                    '1day': '1 Day',
                    '5day': '5 Days',
                    '1month': '1 Month',
                    '3month': '3 Months',
                    'ytd': 'Year to Date',
                    '1year': '1 Year',
                    '3year': '3 Years',
                    '5year': '5 Years',
                    '10year': '10 Years'
                },
                columns: ['Sector', 'Performance']
            },
            'MARKET_STATUS': {
                options: ['global'],
                optionLabels: {
                    'global': 'Global Markets'
                },
                columns: ['Market', 'Region', 'Status', 'Local Time']
            },
            'IPO_CALENDAR': {
                options: ['upcoming'],
                optionLabels: {
                    'upcoming': 'Upcoming IPOs'
                },
                columns: ['Symbol', 'Name', 'IPO Date', 'Price Range', 'Exchange']
            },
            'EARNINGS_CALENDAR': {
                options: ['upcoming'],
                optionLabels: {
                    'upcoming': 'Upcoming Earnings'
                },
                columns: ['Symbol', 'Name', 'Report Date', 'Estimate', 'Currency']
            },
            'CRYPTO_INTRADAY': {
                options: ['BTC', 'ETH', 'DOGE', 'ADA'],
                optionLabels: {
                    'BTC': 'Bitcoin (BTC)',
                    'ETH': 'Ethereum (ETH)',
                    'DOGE': 'Dogecoin (DOGE)',
                    'ADA': 'Cardano (ADA)'
                },
                columns: ['Timestamp', 'Price', 'Volume']
            },
            'INSIDER_TRANSACTIONS': {
                options: ['all'],
                optionLabels: {
                    'all': 'All Transactions'
                },
                columns: ['Filing Date', 'Transaction Date', 'Insider Name', 'Title', 'Type', 'Price', 'Shares', 'Security Type', 'Ticker']
            }
        };
        
        this.init();
    }
    
    init() {
        // Add event listeners
        this.endpointSelector.addEventListener('change', () => this.handleEndpointChange());
        
        // Add form submit handler
        this.stockForm.addEventListener('submit', (e) => {
            e.preventDefault();
            this.handleFormSubmit();
        });
        
        // Add search handler
        this.searchInput.addEventListener('input', () => this.handleSearch());
        
        // Initialize with default selection if present in URL
        const urlParams = new URLSearchParams(window.location.search);
        const endpoint = urlParams.get('endpoint');
        const option = urlParams.get('option');
        
        if (endpoint && this.endpointConfig[endpoint]) {
            this.endpointSelector.value = endpoint;
            this.handleEndpointChange();
            
            if (option) {
                this.optionsSelector.value = option;
            }
        }
    }
    
    handleEndpointChange() {
        const endpoint = this.endpointSelector.value;
        
        // Reset search input
        this.searchInput.value = '';
        this.currentData = null;
        
        // Add labels to the containers
        this.endpointSelector.parentElement.setAttribute('data-label', 'Select Endpoint');
        this.optionsSelector.parentElement.setAttribute('data-label', 'Select Option');
        
        this.optionsSelector.innerHTML = '<option value="">Select Option</option>';
        this.optionsSelector.disabled = true;
        
        if (endpoint && this.endpointConfig[endpoint]) {
            const config = this.endpointConfig[endpoint];
            
            // Populate options
            config.options.forEach(option => {
                const label = config.optionLabels[option] || option;
                const optionElement = document.createElement('option');
                optionElement.value = option;
                optionElement.textContent = label;
                this.optionsSelector.appendChild(optionElement);
            });
            
            this.optionsSelector.disabled = false;
            
            // Show/hide symbol input based on endpoint type
            const symbolInputContainer = document.getElementById('symbolInputContainer');
            if (symbolInputContainer) {
                symbolInputContainer.style.display = (endpoint === 'INSIDER_TRANSACTIONS' || endpoint === 'STOCK_SYMBOL') ? 'block' : 'none';
            }
        }
        
        // Clear table
        this.tableHeader.innerHTML = '';
        this.tableBody.innerHTML = '';
    }
    
    showLoading(show = true) {
        this.loadingSpinner.classList.toggle('d-none', !show);
    }
    
    showError(message, details = '') {
        this.errorMessage.innerHTML = `
            <div class="alert-content">
                <i class="fas fa-exclamation-circle"></i>
                <div class="alert-text">
                    <strong>${message}</strong>
                    ${details ? `<div class="alert-details">${details}</div>` : ''}
                </div>
            </div>
        `;
        this.errorMessage.classList.remove('d-none');
        this.errorMessage.classList.add('custom-alert');
    }
    
    hideError() {
        this.errorMessage.classList.add('d-none');
    }
    
    async fetchDataFromApi(endpoint, option, symbol = null) {
        try {
            let url;
            if (endpoint === 'DOW_COMPANIES') {
                url = `/api/stock/${symbol}`;
            } else if (endpoint === 'INSIDER_TRANSACTIONS') {
                url = `/api/insider-transactions/${symbol}`;
            } else {
                url = `/api/market-data/${endpoint}/${option}`;
            }

            const response = await fetch(url);
            if (!response.ok) throw new Error('Network response was not ok');
            return await response.json();
        } catch (error) {
            console.error('Error fetching data:', error);
            return { error: 'Failed to fetch data' };
        }
    }
    
    async fetchData() {
        const endpoint = this.endpointSelector.value;
        const option = this.optionsSelector.value;
        
        if (!endpoint) return;
        
        // For stock symbol or insider transactions, we need a symbol
        if (endpoint === 'STOCK_SYMBOL' || endpoint === 'INSIDER_TRANSACTIONS') {
            const symbol = document.getElementById('symbolInput').value.trim().toUpperCase();
            if (!symbol) {
                this.showError('Please enter a stock symbol');
                return;
            }
            console.log(`Fetching data for symbol: ${symbol}`);
        } else if (!option && endpoint !== 'DOW_COMPANIES') {
            this.showError('Please select an option');
            return;
        }
        
        this.showLoading(true);
        this.hideError();
        
        try {
            let data;
            
            if (endpoint === 'DOW_COMPANIES') {
                data = await this.fetchDowCompaniesData();
            } else if (endpoint === 'STOCK_SYMBOL') {
                const symbol = document.getElementById('symbolInput').value.trim().toUpperCase();
                console.log(`Making API request for stock symbol: ${symbol}`);
                
                const response = await fetch(`/api/stock/${symbol}`);
                if (!response.ok) throw new Error('Failed to fetch stock data');
                const stockData = await response.json();
                
                if (stockData.error) {
                    throw new Error(stockData.error);
                }
                
                // Format the stock data for display
                data = [{
                    symbol: symbol,
                    price: stockData.price,
                    change: stockData.change,
                    error: stockData.error
                }];
                
            } else if (endpoint === 'INSIDER_TRANSACTIONS') {
                const symbol = document.getElementById('symbolInput').value.trim().toUpperCase();
                console.log(`Making API request for insider transactions: ${symbol}`);
                
                const response = await this.fetchDataFromApi('INSIDER_TRANSACTIONS', 'all', symbol);
                console.log('Received response:', response);
                
                if (response.error) {
                    throw new Error(response.error);
                }
                data = response.data;
                
                if (!data || data.length === 0) {
                    throw new Error('No data available for this symbol');
                }
                console.log(`Found ${data.length} transactions`);
            } else {
                const response = await this.fetchDataFromApi(endpoint, option);
                if (response.error) {
                    throw new Error(response.error);
                }
                data = response.data;
            }
            
            this.updateTable(endpoint, data);
            
        } catch (error) {
            console.error('Error fetching data:', error);
            this.showError(error.message);
            this.tableHeader.innerHTML = '';
            this.tableBody.innerHTML = '';
        } finally {
            this.showLoading(false);
        }
    }
    
    async fetchDowCompaniesData(forceRefresh = false) {
        // Check cache if not forcing refresh
        if (!forceRefresh && this.cache.DOW_COMPANIES.data && this.cache.DOW_COMPANIES.timestamp) {
            const now = Date.now();
            const elapsed = now - this.cache.DOW_COMPANIES.timestamp;
            
            if (elapsed < this.cache.DOW_COMPANIES.expiryTime) {
                console.log('Using cached DOW companies data');
                return this.cache.DOW_COMPANIES.data;
            }
        }
        
        console.log('Fetching fresh DOW companies data');
        const results = [];
        const batchSize = 5;
        
        for (let i = 0; i < this.dowCompanies.length; i += batchSize) {
            const batch = this.dowCompanies.slice(i, i + batchSize);
            const batchPromises = batch.map(async (company) => {
                try {
                    const response = await fetch(`/api/stock/${company.symbol}`);
                    if (!response.ok) throw new Error(`Failed to fetch ${company.symbol}`);
                    const data = await response.json();
                    
                    if (data.error) {
                        console.warn(`Error fetching ${company.symbol}:`, data.error);
                        return null;
                    }
                    
                    return {
                        symbol: company.symbol,
                        name: company.name,
                        price: data.price,
                        change: data.change
                    };
                } catch (error) {
                    console.error(`Error fetching ${company.symbol}:`, error);
                    return null;
                }
            });
            
            const batchResults = await Promise.all(batchPromises);
            results.push(...batchResults.filter(result => result !== null));
            
            if (i + batchSize < this.dowCompanies.length) {
                await new Promise(resolve => setTimeout(resolve, 1000));
            }
        }
        
        // Update cache
        this.cache.DOW_COMPANIES.data = results;
        this.cache.DOW_COMPANIES.timestamp = Date.now();
        
        return results;
    }
    
    updateTable(endpoint, data) {
        if (!data || (Array.isArray(data) && data.length === 0)) {
            this.showError('No data available');
            return;
        }
        
        // Store the current data for searching and pagination
        this.currentData = Array.isArray(data) ? data : [data];
        this.currentPage = 1;
        
        const config = this.endpointConfig[endpoint];
        
        // Update header
        this.tableHeader.innerHTML = `
            <tr>
                ${config.columns.map(col => `<th class="table-header">${col}</th>`).join('')}
            </tr>
        `;
        
        // Add pagination controls
        this.updatePaginationControls();
        
        // Update body with first page
        this.updateTableBody(this.getPageData());
    }
    
    getPageData() {
        if (!this.currentData) return [];
        
        const startIndex = (this.currentPage - 1) * this.rowsPerPage;
        const endIndex = startIndex + this.rowsPerPage;
        return this.currentData.slice(startIndex, endIndex);
    }
    
    updatePaginationControls() {
        if (!this.currentData) return;
        
        const totalPages = Math.ceil(this.currentData.length / this.rowsPerPage);
        
        // Remove existing pagination if any
        const existingPagination = document.querySelector('.pagination-controls');
        if (existingPagination) {
            existingPagination.remove();
        }
        
        if (totalPages <= 1) return;
        
        const paginationHtml = `
            <div class="pagination-controls">
                <button class="btn btn-outline-secondary" ${this.currentPage === 1 ? 'disabled' : ''} onclick="stockTracker.changePage(${this.currentPage - 1})">
                    <i class="fas fa-chevron-left"></i>
                </button>
                <span class="pagination-info">Page ${this.currentPage} of ${totalPages}</span>
                <button class="btn btn-outline-secondary" ${this.currentPage === totalPages ? 'disabled' : ''} onclick="stockTracker.changePage(${this.currentPage + 1})">
                    <i class="fas fa-chevron-right"></i>
                </button>
            </div>
        `;
        
        this.tableHeader.parentElement.insertAdjacentHTML('afterend', paginationHtml);
    }
    
    changePage(newPage) {
        if (!this.currentData) return;
        
        const totalPages = Math.ceil(this.currentData.length / this.rowsPerPage);
        if (newPage < 1 || newPage > totalPages) return;
        
        this.currentPage = newPage;
        this.updatePaginationControls();
        this.updateTableBody(this.getPageData());
    }
    
    updateTableBody(data) {
        const endpoint = this.endpointSelector.value;
        let bodyHTML = '';
        
        switch (endpoint) {
            case 'DOW_COMPANIES':
                bodyHTML = data.map(item => `
                    <tr class="table-row">
                        <td class="symbol-cell">${item.symbol}</td>
                        <td class="name-cell">${item.name}</td>
                        <td class="price-cell">$${parseFloat(item.price).toFixed(2)}</td>
                        <td class="change-cell ${parseFloat(item.change) >= 0 ? 'positive-change' : 'negative-change'}">
                            ${parseFloat(item.change).toFixed(2)}%
                        </td>
                    </tr>
                `).join('');
                break;
                
            case 'TOP_GAINERS_LOSERS':
                bodyHTML = data.map(item => `
                    <tr class="table-row">
                        <td class="symbol-cell">${item.ticker}</td>
                        <td class="name-cell">${item.name}</td>
                        <td class="price-cell">$${parseFloat(item.price).toFixed(2)}</td>
                        <td class="change-cell">$${parseFloat(item.change_amount).toFixed(2)}</td>
                        <td class="change-percent-cell ${parseFloat(item.change_percentage) >= 0 ? 'positive-change' : 'negative-change'}">
                            ${parseFloat(item.change_percentage).toFixed(2)}%
                        </td>
                        <td class="volume-cell">${parseInt(item.volume).toLocaleString()}</td>
                    </tr>
                `).join('');
                break;
                
            case 'SECTOR':
                bodyHTML = data.map(item => `
                    <tr class="table-row">
                        <td class="sector-cell">${item.sector}</td>
                        <td class="performance-cell ${parseFloat(item.performance) >= 0 ? 'positive-change' : 'negative-change'}">
                            ${parseFloat(item.performance).toFixed(2)}%
                        </td>
                    </tr>
                `).join('');
                break;
                
            case 'MARKET_STATUS':
                bodyHTML = data.map(item => `
                    <tr class="table-row">
                        <td class="market-cell">${item.market_type}</td>
                        <td class="region-cell">${item.region}</td>
                        <td class="status-cell">
                            <span class="status-indicator ${item.current_status.toLowerCase()}">${item.current_status}</span>
                        </td>
                        <td class="time-cell">${item.local_time}</td>
                    </tr>
                `).join('');
                break;
                
            case 'IPO_CALENDAR':
                bodyHTML = data.map(item => `
                    <tr class="table-row">
                        <td class="symbol-cell">${item.symbol}</td>
                        <td class="name-cell">${item.name}</td>
                        <td class="date-cell">${item.ipo_date}</td>
                        <td class="price-range-cell">${item.price_range}</td>
                        <td class="exchange-cell">${item.exchange}</td>
                    </tr>
                `).join('');
                break;
                
            case 'EARNINGS_CALENDAR':
                bodyHTML = data.map(item => `
                    <tr class="table-row">
                        <td class="symbol-cell">${item.symbol}</td>
                        <td class="name-cell">${item.name}</td>
                        <td class="date-cell">${item.report_date}</td>
                        <td class="estimate-cell">${item.estimate}</td>
                        <td class="currency-cell">${item.currency}</td>
                    </tr>
                `).join('');
                break;
                
            case 'CRYPTO_INTRADAY':
                bodyHTML = data.map(item => `
                    <tr class="table-row">
                        <td class="timestamp-cell">${new Date(item.timestamp).toLocaleString()}</td>
                        <td class="price-cell">$${parseFloat(item.price).toFixed(2)}</td>
                        <td class="volume-cell">${parseFloat(item.volume).toFixed(4)}</td>
                    </tr>
                `).join('');
                break;
                
            case 'STOCK_SYMBOL':
                bodyHTML = data.map(item => `
                    <tr class="table-row">
                        <td class="symbol-cell">${item.symbol}</td>
                        <td class="price-cell">$${parseFloat(item.price).toFixed(2)}</td>
                        <td class="change-cell ${parseFloat(item.change) >= 0 ? 'positive-change' : 'negative-change'}">
                            ${parseFloat(item.change).toFixed(2)}%
                        </td>
                    </tr>
                `).join('');
                break;

            case 'INSIDER_TRANSACTIONS':
                bodyHTML = data.map(item => `
                    <tr class="table-row">
                        <td class="date-cell">${item.filing_date}</td>
                        <td class="date-cell">${item.transaction_date}</td>
                        <td class="name-cell">${item.insider_name}</td>
                        <td class="title-cell">${item.insider_title}</td>
                        <td class="type-cell">${item.transaction_type}</td>
                        <td class="price-cell">$${parseFloat(item.price || 0).toFixed(2)}</td>
                        <td class="shares-cell">${parseInt(item.shares || 0).toLocaleString()}</td>
                        <td class="type-cell">${item.security_type}</td>
                        <td class="symbol-cell">${item.ticker}</td>
                    </tr>
                `).join('');
                break;
        }
        
        this.tableBody.innerHTML = bodyHTML;
    }
    
    async handleFormSubmit() {
        const endpoint = this.endpointSelector.value;
        const option = this.optionsSelector.value;
        
        if (!endpoint) {
            this.showError('Please select an endpoint');
            return;
        }
        
        if (!option && endpoint !== 'DOW_COMPANIES') {
            this.showError('Please select an option');
            return;
        }
        
        // For insider transactions and stock symbol, validate symbol
        if (endpoint === 'INSIDER_TRANSACTIONS' || endpoint === 'STOCK_SYMBOL') {
            const symbol = document.getElementById('symbolInput').value.trim().toUpperCase();
            if (!symbol) {
                this.showError('Please enter a stock symbol');
                return;
            }
            if (!/^[A-Z]{1,5}$/.test(symbol)) {
                this.showError('Invalid symbol format. Please enter 1-5 capital letters.');
                return;
            }
        }
        
        // All validation passed, fetch the data
        await this.fetchData();
    }

    handleSearch() {
        const searchTerm = this.searchInput.value.toLowerCase().trim();
        
        if (!this.currentData) {
            return;
        }

        if (!searchTerm) {
            // If search is empty, show first page of all data
            this.currentPage = 1;
            this.updatePaginationControls();
            this.updateTableBody(this.getPageData());
            return;
        }

        const filteredData = this.currentData.filter(item => {
            return Object.values(item).some(value => 
                String(value).toLowerCase().includes(searchTerm)
            );
        });

        // Update pagination for filtered data
        this.currentData = filteredData;
        this.currentPage = 1;
        this.updatePaginationControls();
        this.updateTableBody(this.getPageData());
    }
}

// Initialize the stock tracker when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.stockTracker = new StockTracker();
});

// Add styles to the document
const style = document.createElement('style');
style.textContent = `
    .custom-alert {
        background-color: var(--bg-color);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
        color: var(--text-color);
    }

    .alert-content {
        display: flex;
        align-items: flex-start;
        gap: 0.75rem;
    }

    .alert-content i {
        color: #dc2626;
        font-size: 1.25rem;
        margin-top: 0.125rem;
    }

    .alert-text {
        flex: 1;
    }

    .alert-details {
        margin-top: 0.25rem;
        font-size: 0.875rem;
        color: var(--text-muted);
    }

    .table {
        background-color: var(--bg-color);
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid var(--border-color);
    }

    .table-header {
        background-color: var(--bg-secondary);
        color: var(--text-muted);
        font-weight: 600;
        text-transform: uppercase;
        font-size: 0.75rem;
        padding: 0.75rem 1rem;
        border-bottom: 1px solid var(--border-color);
    }

    .table-row {
        transition: background-color 0.2s ease;
    }

    .table-row:hover {
        background-color: var(--bg-hover);
    }

    .table td {
        padding: 0.75rem 1rem;
        border-bottom: 1px solid var(--border-color);
        color: var(--text-color);
    }

    .symbol-cell {
        font-weight: 600;
        color: var(--text-color);
    }

    .name-cell {
        max-width: 300px;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }

    .positive-change {
        color: #10B981 !important;
    }

    .negative-change {
        color: #EF4444 !important;
    }

    .status-indicator {
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-size: 0.875rem;
        font-weight: 500;
    }

    .status-indicator.open {
        background-color: rgba(16, 185, 129, 0.1);
        color: #10B981;
    }

    .status-indicator.closed {
        background-color: rgba(239, 68, 68, 0.1);
        color: #EF4444;
    }

    .volume-cell {
        font-family: monospace;
        font-size: 0.875rem;
    }

    .price-cell, .change-cell {
        font-family: monospace;
        font-weight: 500;
    }

    .date-cell {
        font-family: monospace;
        font-size: 0.875rem;
    }

    /* Enhanced Dropdown Styles */
    .row.mb-3 {
        display: flex;
        gap: 1rem;
        margin-bottom: 1.5rem !important;
    }

    .col-md-6 {
        flex: 1;
        min-width: 200px;
    }

    .form-select {
        width: 100%;
        background-color: var(--bg-color);
        border: 1px solid var(--border-color);
        color: var(--text-color);
        border-radius: 8px;
        padding: 0.75rem 1rem;
        font-size: 0.9rem;
        line-height: 1.5;
        transition: all 0.2s ease;
        cursor: pointer;
        -webkit-appearance: none;
        -moz-appearance: none;
        appearance: none;
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='6 9 12 15 18 9'%3E%3C/polyline%3E%3C/svg%3E");
        background-repeat: no-repeat;
        background-position: right 1rem center;
        background-size: 1em;
        padding-right: 2.5rem;
    }

    .form-select:hover:not(:disabled) {
        border-color: var(--primary-color);
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }

    .form-select:focus {
        border-color: var(--primary-color);
        outline: none;
        box-shadow: 0 0 0 3px rgba(var(--primary-color-rgb), 0.15);
    }

    .form-select:disabled {
        background-color: var(--bg-secondary);
        opacity: 0.75;
        cursor: not-allowed;
        border-color: var(--border-color);
    }

    .form-select option {
        padding: 0.75rem;
        background-color: var(--bg-color);
        color: var(--text-color);
    }

    /* Dark mode enhancements for dropdowns */
    .dark-mode .form-select {
        background-color: rgba(255, 255, 255, 0.05);
        border-color: rgba(255, 255, 255, 0.1);
        color: rgba(255, 255, 255, 0.9);
        /* Updated chevron SVG with better visibility */
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='rgba(255, 255, 255, 0.6)' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='6 9 12 15 18 9'%3E%3C/polyline%3E%3C/svg%3E");
    }

    .dark-mode .form-select:hover:not(:disabled) {
        background-color: rgba(255, 255, 255, 0.1);
        border-color: var(--primary-color);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }

    .dark-mode .form-select:focus {
        background-color: rgba(255, 255, 255, 0.1);
        border-color: var(--primary-color);
        box-shadow: 0 0 0 3px rgba(var(--primary-color-rgb), 0.3),
                    0 4px 12px rgba(0, 0, 0, 0.3);
    }

    .dark-mode .form-select option {
        background-color: var(--bg-color);
        color: rgba(255, 255, 255, 0.9);
        padding: 1rem;
    }

    .dark-mode .form-select option:hover,
    .dark-mode .form-select option:focus,
    .dark-mode .form-select option:active,
    .dark-mode .form-select option:checked {
        background-color: rgba(var(--primary-color-rgb), 0.2) !important;
    }

    .dark-mode .form-select:disabled {
        background-color: rgba(255, 255, 255, 0.02);
        border-color: rgba(255, 255, 255, 0.05);
        color: rgba(255, 255, 255, 0.4);
        opacity: 1;
    }

    /* Enhanced label visibility in dark mode */
    .dark-mode .select-container::before {
        color: rgba(255, 255, 255, 0.6);
    }

    /* Improved placeholder visibility in dark mode */
    .dark-mode .form-select option[value=""] {
        color: rgba(255, 255, 255, 0.5);
    }

    /* Add a subtle gradient background effect in dark mode */
    .dark-mode .form-select {
        background-image: 
            linear-gradient(to bottom, rgba(255, 255, 255, 0.05), rgba(255, 255, 255, 0.02)),
            url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='rgba(255, 255, 255, 0.6)' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='6 9 12 15 18 9'%3E%3C/polyline%3E%3C/svg%3E");
        background-position: center, right 1rem center;
        background-size: 100% 100%, 1em;
        background-repeat: no-repeat;
    }

    .dark-mode .form-select:hover:not(:disabled) {
        background-image: 
            linear-gradient(to bottom, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0.05)),
            url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='rgba(255, 255, 255, 0.8)' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='6 9 12 15 18 9'%3E%3C/polyline%3E%3C/svg%3E");
    }

    /* Add a subtle glow effect on focus in dark mode */
    .dark-mode .form-select:focus {
        background-image: 
            linear-gradient(to bottom, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0.05)),
            url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='rgba(255, 255, 255, 0.8)' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='6 9 12 15 18 9'%3E%3C/polyline%3E%3C/svg%3E");
    }

    /* Enhance the dropdown options in dark mode */
    .dark-mode .form-select optgroup {
        background-color: var(--bg-color);
        color: rgba(255, 255, 255, 0.7);
        font-weight: 600;
    }

    /* Add a subtle separator between options in dark mode */
    .dark-mode .form-select option:not(:last-child) {
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }

    /* Dark mode specific styles */
    .dark-mode .custom-alert {
        background-color: var(--bg-secondary);
    }

    .dark-mode .table {
        background-color: var(--bg-secondary);
    }

    .dark-mode .table-header {
        background-color: var(--bg-color);
    }

    .dark-mode .positive-change {
        color: #34D399 !important;
    }

    .dark-mode .negative-change {
        color: #F87171 !important;
    }

    .dark-mode .status-indicator.open {
        background-color: rgba(52, 211, 153, 0.1);
        color: #34D399;
    }

    .dark-mode .status-indicator.closed {
        background-color: rgba(248, 113, 113, 0.1);
        color: #F87171;
    }

    .refresh-all-container {
        display: flex;
        justify-content: flex-end;
        margin-bottom: 1rem;
    }

    .btn-refresh-all {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        border: 1px solid var(--border-color);
        border-radius: 6px;
        background-color: var(--bg-color);
        color: var(--text-color);
        cursor: pointer;
        transition: all 0.2s ease;
    }

    .btn-refresh-all:hover {
        background-color: var(--bg-hover);
        border-color: var(--primary-color);
    }

    .btn-refresh-all i {
        font-size: 0.875rem;
    }

    .last-updated {
        font-size: 0.75rem;
        color: var(--text-muted);
        margin-left: 0.5rem;
    }

    .btn-refresh {
        padding: 0.25rem;
        border: none;
        background: none;
        color: var(--text-muted);
        cursor: pointer;
        transition: all 0.2s ease;
    }

    .btn-refresh:hover {
        color: var(--primary-color);
        transform: rotate(180deg);
    }

    .btn-refresh i {
        font-size: 0.875rem;
    }

    /* Dark mode enhancements */
    .dark-mode .btn-refresh-all {
        background-color: rgba(255, 255, 255, 0.05);
        border-color: rgba(255, 255, 255, 0.1);
    }

    .dark-mode .btn-refresh-all:hover {
        background-color: rgba(255, 255, 255, 0.1);
        border-color: var(--primary-color);
    }

    .dark-mode .last-updated {
        color: rgba(255, 255, 255, 0.5);
    }

    .dark-mode .btn-refresh {
        color: rgba(255, 255, 255, 0.5);
    }

    .dark-mode .btn-refresh:hover {
        color: var(--primary-color);
    }
`;

document.head.appendChild(style); 