// Common Font Awesome icons grouped by category
const iconGroups = {
    'Common': ['fa-home', 'fa-user', 'fa-cog', 'fa-envelope', 'fa-bell', 'fa-star', 'fa-heart', 'fa-check', 'fa-times'],
    'Media': ['fa-play', 'fa-pause', 'fa-stop', 'fa-volume-up', 'fa-camera', 'fa-video', 'fa-music', 'fa-image', 'fa-film'],
    'System': ['fa-server', 'fa-database', 'fa-network-wired', 'fa-hdd', 'fa-memory', 'fa-microchip', 'fa-desktop', 'fa-laptop', 'fa-mobile'],
    'Files': ['fa-file', 'fa-folder', 'fa-file-pdf', 'fa-file-word', 'fa-file-excel', 'fa-file-image', 'fa-file-video', 'fa-file-audio'],
    'Actions': ['fa-plus', 'fa-minus', 'fa-edit', 'fa-trash', 'fa-download', 'fa-upload', 'fa-sync', 'fa-search', 'fa-save']
};

function updateIconPreview(inputElement, icon) {
    const preview = document.getElementById('iconPreview');
    if (preview) {
        // Remove all existing fa- classes but keep fas
        preview.className = 'fas fa-fw ' + icon;
    }
}

function createIconSelector(inputElement) {
    // Create dropdown container
    const dropdownContainer = document.createElement('div');
    dropdownContainer.className = 'icon-selector-dropdown';
    inputElement.parentNode.insertBefore(dropdownContainer, inputElement.nextSibling);

    // Make input field read-only to prevent direct typing
    inputElement.setAttribute('readonly', 'readonly');
    inputElement.style.cursor = 'pointer';
    
    // Create search input
    const searchInput = document.createElement('input');
    searchInput.type = 'text';
    searchInput.className = 'icon-search';
    searchInput.placeholder = 'Search icons...';
    dropdownContainer.appendChild(searchInput);

    // Create icons container
    const iconsContainer = document.createElement('div');
    iconsContainer.className = 'icons-container';
    dropdownContainer.appendChild(iconsContainer);

    // Add icons by category
    Object.entries(iconGroups).forEach(([category, icons]) => {
        const categoryDiv = document.createElement('div');
        categoryDiv.className = 'icon-category';
        
        const categoryTitle = document.createElement('h4');
        categoryTitle.textContent = category;
        categoryDiv.appendChild(categoryTitle);

        const iconsGrid = document.createElement('div');
        iconsGrid.className = 'icons-grid';

        icons.forEach(icon => {
            const iconWrapper = document.createElement('div');
            iconWrapper.className = 'icon-option';
            iconWrapper.innerHTML = `
                <div class="icon-preview-box">
                    <i class="fas fa-fw ${icon}"></i>
                </div>
                <span class="icon-name">${icon}</span>
            `;
            iconWrapper.setAttribute('data-icon', icon);
            
            iconWrapper.addEventListener('click', () => {
                inputElement.value = icon;
                updateIconPreview(inputElement, icon);
                dropdownContainer.style.display = 'none';
                
                // Trigger a change event on the input
                const event = new Event('change', { bubbles: true });
                inputElement.dispatchEvent(event);
            });

            iconsGrid.appendChild(iconWrapper);
        });

        categoryDiv.appendChild(iconsGrid);
        iconsContainer.appendChild(categoryDiv);
    });

    // Handle search functionality
    searchInput.addEventListener('input', (e) => {
        const searchTerm = e.target.value.toLowerCase();
        document.querySelectorAll('.icon-option').forEach(option => {
            const iconName = option.getAttribute('data-icon').toLowerCase();
            option.style.display = iconName.includes(searchTerm) ? 'flex' : 'none';
        });

        // Show/hide categories based on whether they have visible icons
        document.querySelectorAll('.icon-category').forEach(category => {
            const hasVisibleIcons = category.querySelector('.icon-option[style="display: flex;"]');
            category.style.display = hasVisibleIcons ? 'block' : 'none';
        });
    });

    // Show/hide dropdown
    inputElement.addEventListener('click', () => {
        dropdownContainer.style.display = dropdownContainer.style.display === 'block' ? 'none' : 'block';
        if (dropdownContainer.style.display === 'block') {
            searchInput.focus();
        }
    });

    // Close dropdown when clicking outside
    document.addEventListener('click', (e) => {
        if (!dropdownContainer.contains(e.target) && e.target !== inputElement) {
            dropdownContainer.style.display = 'none';
        }
    });

    // Initialize with current value if it exists
    if (inputElement.value) {
        updateIconPreview(inputElement, inputElement.value);
    }
}

// Initialize icon selectors when the document is ready
document.addEventListener('DOMContentLoaded', () => {
    const iconInput = document.getElementById('icon');
    if (iconInput) {
        createIconSelector(iconInput);
    }
});
