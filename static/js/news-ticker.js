class NewsTicker {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.tickerContent = document.createElement('div');
        this.tickerContent.className = 'news-ticker-content';
        this.container.appendChild(this.tickerContent);
        
        // Check if in development mode
        this.isDev = document.body.classList.contains('dev-mode');
        
        // Update interval (30 minutes in dev, 5 minutes in prod)
        this.updateInterval = this.isDev ? 1800000 : 300000;
        
        // Initialize the ticker
        this.init();
        
        // Add styles
        this.addStyles();

        // Adjust position if dev banner exists
        this.adjustForDevBanner();

        // Listen for sidebar toggle
        document.addEventListener('sidebarToggle', this.handleSidebarToggle.bind(this));
    }

    adjustForDevBanner() {
        const devBanner = document.querySelector('.dev-banner');
        if (devBanner) {
            // Get the banner height and add any margin/padding
            const bannerHeight = devBanner.offsetHeight;
            this.container.style.top = `${bannerHeight}px`;
        }
    }

    addStyles() {
        const style = document.createElement('style');
        style.textContent = `
            .news-ticker {
                width: calc(100% - var(--sidebar-width));
                height: 40px;
                background-color: var(--bg-primary);
                border-bottom: 1px solid var(--border-color);
                overflow: hidden;
                position: fixed;
                top: 0;
                left: var(--sidebar-width);
                z-index: 1060;
                transition: all 0.3s ease;
                box-shadow: var(--shadow-sm);
            }

            .news-ticker-content {
                height: 100%;
                display: flex;
                align-items: center;
                animation: ticker 30s linear infinite;
                white-space: nowrap;
                color: var(--text-color);
                padding: 0 20px;
            }

            .news-item {
                display: inline-block;
                padding: 0 30px;
                color: var(--text-color);
                font-size: 14px;
            }

            .news-item:first-child {
                padding-left: 30px;
            }

            .news-item a {
                color: var(--text-color);
                text-decoration: none;
                transition: color 0.2s ease;
            }

            .news-item a:hover {
                text-decoration: underline;
                color: var(--primary);
            }

            .news-error {
                width: 100%;
                text-align: center;
                color: var(--text-color);
                padding: 10px;
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 8px;
                font-size: 14px;
                background-color: var(--bg-primary);
            }

            .news-error i {
                color: #dc2626;
            }

            @keyframes ticker {
                0% {
                    transform: translateX(100%);
                }
                100% {
                    transform: translateX(-100%);
                }
            }

            /* Pause animation on hover */
            .news-ticker:hover .news-ticker-content {
                animation-play-state: paused;
            }

            /* Dark mode styles */
            .dark-mode .news-ticker {
                background-color: var(--bg-dark);
                border-color: var(--border-medium);
            }

            .dark-mode .news-item a {
                color: var(--text-dark);
            }

            .dark-mode .news-item a:hover {
                color: var(--primary);
            }

            /* Mobile styles */
            @media (max-width: 768px) {
                .news-ticker {
                    width: 100%;
                    left: 0;
                    top: 60px; /* Account for mobile menu button */
                }

                /* Hide ticker when sidebar is open on mobile */
                body.sidebar-active .news-ticker {
                    opacity: 0;
                    visibility: hidden;
                }
            }
        `;
        document.head.appendChild(style);
    }

    async fetchNews() {
        try {
            const response = await fetch('/api/news');
            const data = await response.json();
            
            console.log('News API Response:', {
                status: response.status,
                ok: response.ok,
                data: data
            });
            
            if (!response.ok || data.error) {
                if (this.isDev) {
                    console.warn('Development mode: News API call skipped or failed');
                }
                return { error: data.error || 'Failed to fetch news' };
            }
            
            return data;
        } catch (error) {
            console.error('Error fetching news:', error);
            return { error: 'Failed to fetch news' };
        }
    }

    async updateNews() {
        try {
            const response = await this.fetchNews();
            
            console.log('News Ticker Response:', response);
            
            // Clear existing content
            this.tickerContent.innerHTML = '';
            
            if (response.error) {
                console.log('Error type:', response.error);
                let errorMessage;
                if (response.error === 'Daily API limit reached') {
                    errorMessage = this.isDev ? 
                        'Development mode: News updates paused to preserve API quota' :
                        'News updates paused - Daily API limit reached. Updates will resume tomorrow.';
                } else if (response.error === 'API key invalid or expired' || response.error === 'API key not configured') {
                    errorMessage = 'News updates unavailable - Please check API configuration.';
                } else {
                    errorMessage = 'Unable to fetch news at this time. Please try again later.';
                }
                
                console.log('Displaying error message:', errorMessage);
                
                this.container.innerHTML = `
                    <div class="news-error">
                        <i class="fas fa-exclamation-circle"></i>
                        <span>${errorMessage}</span>
                    </div>`;
                return;
            }

            if (!response.articles || response.articles.length === 0) {
                this.container.innerHTML = `
                    <div class="news-error">
                        <span>No news available at this time</span>
                    </div>`;
                return;
            }

            // Create news items
            const newsItems = response.articles.map(article => `
                <div class="news-item">
                    <a href="${article.url}" target="_blank" rel="noopener noreferrer">
                        ${article.title}
                    </a>
                </div>
            `).join('');

            this.tickerContent.innerHTML = newsItems;

        } catch (error) {
            console.error('Error updating news:', error);
            this.container.innerHTML = `
                <div class="news-error">
                    <i class="fas fa-exclamation-circle"></i>
                    <span>Error loading news</span>
                </div>`;
        }
    }

    async init() {
        await this.updateNews();
        setInterval(() => this.updateNews(), this.updateInterval);
    }

    handleSidebarToggle(event) {
        if (window.innerWidth <= 768) {
            document.body.classList.toggle('sidebar-active', event.detail.isOpen);
        }
    }
} 