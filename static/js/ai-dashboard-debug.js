/**
 * AI Dashboard Debug Script
 * This script helps diagnose issues with loading the AI dashboard
 */

console.log('AI Dashboard Debug Script Loaded!');

// Add a global debug object
window.AIDashboardDebug = {
    version: '1.0.0',
    loadTime: new Date().toISOString(),
    
    // Test function to verify the script is loaded
    test: function() {
        console.log('Debug test function called at: ' + new Date().toISOString());
        return true;
    },
    
    // Function to check the main dashboard script
    checkMainScript: function() {
        if (typeof dashboardState !== 'undefined') {
            console.log('Main dashboard script loaded: dashboardState exists');
            return true;
        } else {
            console.log('Main dashboard script NOT loaded: dashboardState is undefined');
            return false;
        }
    },
    
    // Function to trigger data loading
    loadData: function() {
        console.log('Debug: Triggering data load from /api/ai-insights');
        
        return fetch('/api/ai-insights')
            .then(response => {
                console.log('Debug: Response status:', response.status);
                if (!response.ok) {
                    throw new Error(`API Error: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                console.log('Debug: Data loaded successfully:', data);
                document.dispatchEvent(new CustomEvent('ai-debug-data-loaded', { detail: data }));
                return data;
            })
            .catch(error => {
                console.error('Debug: Error loading data:', error);
                throw error;
            });
    }
};

// Add an event listener to check for the main script loading
document.addEventListener('DOMContentLoaded', function() {
    console.log('Debug: DOM loaded, checking for main script in 1 second...');
    setTimeout(function() {
        AIDashboardDebug.checkMainScript();
        
        // Try to load data directly
        AIDashboardDebug.loadData()
            .then(data => {
                console.log('Debug: Successfully loaded data directly');
                
                // Create a visible notification
                const debugNotification = document.createElement('div');
                debugNotification.style.position = 'fixed';
                debugNotification.style.bottom = '20px';
                debugNotification.style.right = '20px';
                debugNotification.style.backgroundColor = '#4caf50';
                debugNotification.style.color = 'white';
                debugNotification.style.padding = '10px 15px';
                debugNotification.style.borderRadius = '4px';
                debugNotification.style.boxShadow = '0 2px 5px rgba(0,0,0,0.2)';
                debugNotification.style.zIndex = '9999';
                debugNotification.textContent = 'Debug: Data loaded successfully!';
                
                document.body.appendChild(debugNotification);
                
                // Remove after 5 seconds
                setTimeout(() => {
                    document.body.removeChild(debugNotification);
                }, 5000);
            })
            .catch(error => {
                console.error('Debug: Failed to load data directly:', error);
                
                // Create a visible error notification
                const errorNotification = document.createElement('div');
                errorNotification.style.position = 'fixed';
                errorNotification.style.bottom = '20px';
                errorNotification.style.right = '20px';
                errorNotification.style.backgroundColor = '#f44336';
                errorNotification.style.color = 'white';
                errorNotification.style.padding = '10px 15px';
                errorNotification.style.borderRadius = '4px';
                errorNotification.style.boxShadow = '0 2px 5px rgba(0,0,0,0.2)';
                errorNotification.style.zIndex = '9999';
                errorNotification.textContent = `Debug Error: ${error.message}`;
                
                document.body.appendChild(errorNotification);
                
                // Remove after 10 seconds
                setTimeout(() => {
                    document.body.removeChild(errorNotification);
                }, 10000);
            });
    }, 1000);
});

// AI Dashboard Debug Script

// Function to check dashboard data loading
function debugDashboardData() {
    console.log('DEBUG: Starting dashboard data debugging');
    
    // Check if we can access the AI API
    return fetch('/api/ai-insights')
        .then(response => {
            console.log('DEBUG: API Response status:', response.status);
            if (!response.ok) {
                throw new Error(`API Error: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log('DEBUG: Full API data received:', data);
            
            // Check for specific data elements
            checkDataElement(data, 'indices', 'Market indices');
            checkDataElement(data, 'economicIndicators', 'Economic indicators');
            checkDataElement(data, 'newsSentiment', 'News sentiment');
            checkDataElement(data, 'featureImportance', 'Feature importance');
            checkDataElement(data, 'portfolioOptimization', 'Portfolio optimization');
            checkDataElement(data, 'predictionConfidence', 'Prediction confidence');
            
            return data;
        })
        .catch(error => {
            console.error('DEBUG ERROR: Failed to fetch data:', error);
            document.getElementById('debug-output').innerHTML = 
                `<div class="alert alert-danger">Error fetching data: ${error.message}</div>`;
        });
}

// Function to check individual data elements
function checkDataElement(data, key, description) {
    if (!data[key]) {
        console.error(`DEBUG: Missing data - ${description} (${key})`);
    } else {
        console.log(`DEBUG: Found data - ${description}:`, data[key]);
    }
}

// Function to check dashboard containers
function debugDashboardContainers() {
    console.log('DEBUG: Checking dashboard containers');
    
    const containers = [
        'market-indices-container',
        'market-prediction-container',
        'news-sentiment-container',
        'feature-importance-container',
        'portfolio-optimization-container',
        'economic-indicators-container',
        'alerts-container'
    ];
    
    containers.forEach(id => {
        const container = document.getElementById(id);
        if (!container) {
            console.error(`DEBUG: Container not found - ${id}`);
        } else {
            console.log(`DEBUG: Container found - ${id}`);
            if (container.querySelector('.loader')) {
                console.log(`DEBUG: Container has loader - ${id}`);
            }
        }
    });
}

// Add a button to run debugging
document.addEventListener('DOMContentLoaded', function() {
    console.log('DEBUG: DOM loaded for debugging');
    
    // Create debug container if it doesn't exist
    let debugContainer = document.getElementById('debug-container');
    if (!debugContainer) {
        debugContainer = document.createElement('div');
        debugContainer.id = 'debug-container';
        debugContainer.className = 'container mt-4';
        document.body.appendChild(debugContainer);
        
        // Add debug header
        debugContainer.innerHTML = `
            <div class="card">
                <div class="card-header bg-primary text-white">
                    <h4>Dashboard Debugging</h4>
                </div>
                <div class="card-body">
                    <button id="run-debug" class="btn btn-primary mb-3">Run Debug</button>
                    <div id="debug-output" class="border p-3 bg-light"></div>
                </div>
            </div>
        `;
        
        // Add event listener
        document.getElementById('run-debug').addEventListener('click', function() {
            const output = document.getElementById('debug-output');
            output.innerHTML = '<div class="spinner-border text-primary" role="status"></div> Running debug...';
            
            debugDashboardContainers();
            debugDashboardData()
                .then(data => {
                    output.innerHTML = '<div class="alert alert-success">Debug completed. Check console for details.</div>';
                })
                .catch(error => {
                    output.innerHTML = `<div class="alert alert-danger">Debug error: ${error.message}</div>`;
                });
        });
    }
}); 