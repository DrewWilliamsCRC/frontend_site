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