/**
 * AI Dashboard - Main JavaScript File
 * 
 * Simplified version that focuses on reliably showing market indices
 */

// Initialize when the DOM is fully loaded
document.addEventListener('DOMContentLoaded', function() {
    console.log('AI Dashboard: DOM loaded');
    initDashboard();
});

// Main initialization function
function initDashboard() {
    console.log('Initializing AI Dashboard');
    
    // Set up tab navigation if it exists
    setupTabNavigation();
    
    // Set up theme toggle if it exists
    setupThemeToggle();
    
    // Load market indices (primary goal)
    loadMarketIndices();
}

// Market Indices - Primary Goal
async function loadMarketIndices() {
    console.log('Loading market indices');
    
    // Find the container
    const container = document.getElementById('market-indices-container');
    if (!container) {
        console.error('CRITICAL: Market indices container not found');
        return;
    }
    
    try {
        // Show loading state
        container.innerHTML = '<div class="loader"></div> Loading market data...';
        
        // Fetch market indices data
        console.log('Fetching market indices data');
        const response = await fetch('/api/market-indices');
        
        if (!response.ok) {
            throw new Error(`API Error: ${response.status}`);
        }
        
        // Parse the data
        const data = await response.json();
        console.log('Market indices data received:', data);
        
        // Clear the container
        container.innerHTML = '';
        
        // Check if we have indices data
        if (!data || !data.indices) {
            console.error('No indices data found in API response');
            container.innerHTML = '<div class="alert alert-warning">No market data available</div>';
            return;
        }
        
        // Create and display index cards
        const indices = data.indices;
        console.log('Displaying indices:', Object.keys(indices));
        
        // Simple grid container
        const grid = document.createElement('div');
        grid.style.display = 'flex';
        grid.style.flexWrap = 'wrap';
        grid.style.gap = '10px';
        container.appendChild(grid);
        
        // Create card for each index
        for (const symbol in indices) {
            if (indices.hasOwnProperty(symbol)) {
                const indexData = indices[symbol];
                createIndexCard(grid, symbol, indexData);
            }
        }
        
        // Add last updated info
        const timestamp = document.createElement('div');
        timestamp.style.width = '100%';
        timestamp.style.marginTop = '10px';
        timestamp.style.fontSize = '0.8em';
        timestamp.style.textAlign = 'right';
        timestamp.style.color = '#666';
        timestamp.textContent = `Last updated: ${new Date().toLocaleString()}`;
        container.appendChild(timestamp);
        
    } catch (error) {
        console.error('Error loading market indices:', error);
        container.innerHTML = `<div class="alert alert-warning">
            Failed to load market data: ${error.message}
        </div>`;
    }
}

// Create a card for a market index
function createIndexCard(container, symbol, data) {
    console.log(`Creating card for ${symbol}:`, data);
    
    // Create card element
    const card = document.createElement('div');
    card.className = 'card index-card';
    card.style.minWidth = '150px';
    card.style.textAlign = 'center';
    card.style.margin = '5px';
    
    // Get display name for the symbol
    let displayName = symbol;
    switch(symbol) {
        case 'SPX': displayName = 'S&P 500'; break;
        case 'DJI': displayName = 'Dow Jones'; break;
        case 'IXIC': displayName = 'NASDAQ'; break;
        case 'VIX': displayName = 'Volatility'; break;
        case 'TNX': displayName = '10Y Treasury'; break;
    }
    
    // Parse change values
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
    } catch (error) {
        console.warn(`Error parsing values for ${symbol}:`, error);
    }
    
    // Determine color based on change
    const changeClass = change >= 0 ? 'positive' : 'negative';
    const textColor = change >= 0 ? '#4caf50' : '#f44336';
    
    // Set card content
    card.innerHTML = `
        <div class="card-header" style="font-weight: bold; padding: 10px; background-color: #f5f5f5; border-bottom: 1px solid #ddd;">
            ${displayName}
        </div>
        <div class="card-body" style="padding: 15px;">
            <div style="font-size: 1.2em; font-weight: bold; margin-bottom: 5px;">
                ${price}
            </div>
            <div style="color: ${textColor};">
                ${change >= 0 ? '+' : ''}${change.toFixed(2)} (${percentChange >= 0 ? '+' : ''}${percentChange.toFixed(2)}%)
            </div>
        </div>
    `;
    
    // Add card to container
    container.appendChild(card);
}

// Optional: Set up tab navigation if it exists
function setupTabNavigation() {
    const tabs = document.querySelectorAll('.tab');
    const tabContents = document.querySelectorAll('.tab-content');
    
    if (tabs.length === 0) return;
    
    console.log('Setting up tab navigation');
    
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            // Get the target tab content
            const target = tab.getAttribute('data-tab');
            
            // Remove active class from all tabs and contents
            tabs.forEach(t => t.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));
            
            // Add active class to selected tab and content
            tab.classList.add('active');
            document.getElementById(target).classList.add('active');
        });
    });
}

// Optional: Set up theme toggle if it exists
function setupThemeToggle() {
    const themeToggleBtn = document.getElementById('dark-mode-toggle');
    
    if (!themeToggleBtn) return;
    
    console.log('Setting up theme toggle');
    
    themeToggleBtn.addEventListener('click', () => {
        document.body.classList.toggle('dark-mode');
        
        // Save preference
        const isDarkMode = document.body.classList.contains('dark-mode');
        localStorage.setItem('darkMode', isDarkMode ? 'true' : 'false');
    });
    
    // Check for saved preference
    const savedDarkMode = localStorage.getItem('darkMode') === 'true';
    if (savedDarkMode) {
        document.body.classList.add('dark-mode');
    }
} 