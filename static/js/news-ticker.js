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
            if (!response.ok) throw new Error('Network response was not ok');
            return await response.json();
        } catch (error) {
            console.error('Error fetching news:', error);
            return { error: 'Failed to fetch news' };
        }
    }

    async updateNews() {
        const now = new Date();
        console.log('Starting news update at:', now.toLocaleTimeString());

        try {
            const data = await this.fetchNews();

            // Clear existing content
            this.tickerContent.innerHTML = '';

            if (data.error) {
                this.tickerContent.innerHTML = `
                    <div class="news-error">
                        ${data.error}
                    </div>`;
                return;
            }

            if (!data.articles || data.articles.length === 0) {
                this.tickerContent.innerHTML = `
                    <div class="news-error">
                        No news available at this time
                    </div>`;
                return;
            }

            // Create news items
            const newsItems = data.articles.map(article => `
                <div class="news-item">
                    <a href="${article.url}" target="_blank" rel="noopener noreferrer">
                        ${article.title}
                    </a>
                </div>
            `).join('');

            this.tickerContent.innerHTML = newsItems;

        } catch (error) {
            console.error('Error updating news:', error);
            this.tickerContent.innerHTML = `
                <div class="news-error">
                    Error loading news
                </div>`;
        }
    }

    async init() {
        await this.updateNews();
        setInterval(() => this.updateNews(), this.updateInterval);
    }
} 