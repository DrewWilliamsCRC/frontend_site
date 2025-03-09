/**
 * AI Dashboard Diagnostic Script
 * This script provides detailed logging on component updates
 */

console.log('AI Dashboard Diagnostic Script Loaded');

// Add global diagnostic object
window.DiagnosticTools = {
    // Log data with component name
    logComponentAttempt: function(name, data) {
        console.group(`🔍 Diagnostic: ${name}`);
        console.log('Attempting to update component');
        console.log('Data available:', !!data);
        if (data) {
            console.log('Data keys:', Object.keys(data));
        }
        console.groupEnd();
    },
    
    // Log success for component
    logComponentSuccess: function(name) {
        console.log(`✅ Diagnostic: ${name} updated successfully`);
    },
    
    // Log error for component
    logComponentError: function(name, error) {
        console.group(`❌ Diagnostic: ${name} update failed`);
        console.error(error);
        console.groupEnd();
    },
    
    // Check if an element has a loader
    checkLoader: function(containerId) {
        const container = document.getElementById(containerId);
        if (!container) {
            console.warn(`Container not found: ${containerId}`);
            return false;
        }
        
        const loader = container.querySelector('.loader');
        return !!loader;
    },
    
    // Get all components with loaders
    getComponentsWithLoaders: function() {
        const components = [
            'market-indices-container',
            'market-prediction-container',
            'news-sentiment-container',
            'feature-importance-container',
            'portfolio-optimization-container',
            'economic-indicators-container',
            'alerts-container'
        ];
        
        const result = components.filter(id => this.checkLoader(id));
        console.log('Components still loading:', result);
        return result;
    },
    
    // Test data field existence
    checkDataField: function(data, fieldPath) {
        if (!data) return false;
        
        const path = fieldPath.split('.');
        let current = data;
        
        for (const segment of path) {
            if (current[segment] === undefined) {
                return false;
            }
            current = current[segment];
        }
        
        return true;
    },
    
    // Analyze data for missing fields
    analyzeData: function(data) {
        if (!data) {
            console.error('No data provided for analysis');
            return { valid: false, reason: 'No data' };
        }
        
        const requiredFields = [
            'indices',
            'lastUpdated',
            'modelMetrics',
            'newsSentiment',
            'featureImportance',
            'portfolioOptimization',
            'economicIndicators',
            'alerts'
        ];
        
        const missingFields = requiredFields.filter(field => !this.checkDataField(data, field));
        
        return {
            valid: missingFields.length === 0,
            missingFields: missingFields,
            dataType: data.status || 'Unknown'
        };
    },
    
    // Examine dashboard state
    examineDashboardState: function() {
        if (typeof dashboardState === 'undefined') {
            console.error('dashboardState is not defined');
            return null;
        }
        
        console.group('Dashboard State Analysis');
        console.log('dashboardState defined:', !!dashboardState);
        console.log('dashboardState.data defined:', !!dashboardState.data);
        
        if (dashboardState.data) {
            const analysis = this.analyzeData(dashboardState.data);
            console.log('Data analysis:', analysis);
        }
        
        console.groupEnd();
        
        return dashboardState;
    },
    
    // Run full diagnostics
    runDiagnostics: function() {
        console.group('🔍 AI Dashboard Diagnostics');
        
        // Check basic state
        console.log('Script loaded at:', new Date().toISOString());
        console.log('Window location:', window.location.href);
        
        // Check dashboard state
        this.examineDashboardState();
        
        // Check components with loaders
        this.getComponentsWithLoaders();
        
        // Try to load data
        console.log('Attempting to load fresh data...');
        try {
            // Use the original loadDashboardData function if available
            if (typeof loadDashboardData === 'function') {
                loadDashboardData();
            } else {
                console.error('loadDashboardData function not found');
            }
        } catch (e) {
            console.error('Error loading dashboard data:', e);
        }
        
        console.groupEnd();
    }
};

// Monitor dashboardState for changes
let previousState = null;
setInterval(() => {
    if (typeof dashboardState !== 'undefined' && dashboardState.data !== previousState) {
        previousState = dashboardState.data;
        console.log('Dashboard state updated:', new Date().toISOString());
        DiagnosticTools.examineDashboardState();
    }
}, 1000);

// Run diagnostics after a delay to ensure page is loaded
setTimeout(() => {
    DiagnosticTools.runDiagnostics();
}, 2000);

// Add diagnostic button to page
document.addEventListener('DOMContentLoaded', function() {
    // Create diagnostic controls
    const controls = document.createElement('div');
    controls.style.position = 'fixed';
    controls.style.bottom = '20px';
    controls.style.right = '20px';
    controls.style.zIndex = '9999';
    controls.style.display = 'flex';
    controls.style.flexDirection = 'column';
    controls.style.gap = '10px';
    
    // Create diagnostic button
    const diagButton = document.createElement('button');
    diagButton.innerText = '🔍 Run Diagnostics';
    diagButton.className = 'btn btn-primary';
    diagButton.onclick = () => DiagnosticTools.runDiagnostics();
    
    // Create reload data button
    const reloadButton = document.createElement('button');
    reloadButton.innerText = '🔄 Reload Data';
    reloadButton.className = 'btn btn-success';
    reloadButton.onclick = () => {
        if (typeof loadDashboardData === 'function') {
            loadDashboardData();
        } else {
            alert('loadDashboardData function not found');
        }
    };
    
    // Add buttons to controls
    controls.appendChild(diagButton);
    controls.appendChild(reloadButton);
    
    // Add controls to page
    document.body.appendChild(controls);
}); 