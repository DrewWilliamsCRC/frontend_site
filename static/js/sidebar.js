console.log('Sidebar script loaded');

window.addEventListener('load', function() {
    console.log('Window loaded, initializing sidebar...');
    
    // Get DOM elements
    const toggleButton = document.querySelector('.mobile-menu-toggle');
    const sidebar = document.querySelector('.sidebar');
    
    // Debug check for elements
    console.log('Toggle button found:', toggleButton !== null, toggleButton);
    console.log('Sidebar found:', sidebar !== null, sidebar);
    
    if (!toggleButton || !sidebar) {
        console.error('Required elements not found for sidebar toggle');
        return;
    }
    
    // Simple toggle function
    function toggleSidebar(e) {
        if (e) {
            e.preventDefault();
            e.stopPropagation();
        }
        
        const isActive = sidebar.classList.contains('active');
        console.log('Toggle clicked, current state:', isActive);
        
        if (isActive) {
            sidebar.classList.remove('active');
        } else {
            sidebar.classList.add('active');
        }
    }
    
    // Add click handler
    toggleButton.addEventListener('click', toggleSidebar);
    
    // Add touch handler for mobile
    toggleButton.addEventListener('touchend', function(e) {
        e.preventDefault();
        toggleSidebar(e);
    });
    
    console.log('Sidebar functionality initialized');
});
