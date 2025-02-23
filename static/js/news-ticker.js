class NewsTicker {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.tickerContent = document.createElement('div');
        this.tickerContent.className = 'news-ticker-content';
        this.container.appendChild(this.tickerContent);
        
        // Update interval (5 minutes = 300000 ms)
        this.updateInterval = 300000;
        
        // Initialize the ticker
        this.init();
        
        // Add styles
        this.addStyles();

        // Adjust position if dev banner exists
        this.adjustForDevBanner();
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
                width: 100%;
                height: 40px;
                background-color: var(--bg-color);
                border-bottom: 1px solid var(--border-color);
                overflow: hidden;
                position: fixed;
                top: 0;
                left: 0;
                z-index: 999;
                transition: top 0.3s ease;
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
            }

            .news-item:first-child {
                padding-left: 30px;
            }

            .news-item a {
                color: var(--text-color);
                text-decoration: none;
            }

            .news-item a:hover {
                text-decoration: underline;
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
                    errorMessage = 'News updates paused - Daily API limit reached. Updates will resume tomorrow.';
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
} 