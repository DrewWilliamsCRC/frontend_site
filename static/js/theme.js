// Theme management functionality
document.addEventListener('DOMContentLoaded', function() {
    const darkModeToggle = document.getElementById('darkModeToggle');
    const themeText = document.getElementById('themeText');
    const body = document.body;
    const DARK_MODE_KEY = 'darkMode';

    // Function to update UI elements based on theme
    function updateThemeUI(isDark) {
        const icon = darkModeToggle.querySelector('i');
        if (isDark) {
            icon.classList.remove('fa-moon');
            icon.classList.add('fa-sun');
            themeText.textContent = 'Light Mode';
        } else {
            icon.classList.remove('fa-sun');
            icon.classList.add('fa-moon');
            themeText.textContent = 'Dark Mode';
        }
    }

    // Function to toggle theme
    function toggleTheme() {
        const isDark = body.classList.toggle('dark-mode');
        localStorage.setItem(DARK_MODE_KEY, isDark);
        updateThemeUI(isDark);
    }

    // Initialize theme from localStorage
    const storedPref = localStorage.getItem(DARK_MODE_KEY) === 'true';
    if (storedPref) {
        body.classList.add('dark-mode');
        updateThemeUI(true);
    }

    // Add click event listener
    darkModeToggle.addEventListener('click', toggleTheme);
});
