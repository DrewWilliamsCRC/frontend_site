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
                white-space: nowrap;
                color: var(--text-color);
                padding: 0 20px;
                will-change: transform;
            }

            .news-content-wrapper {
                display: flex;
                align-items: center;
            }

            .news-content-base {
                display: flex;
                align-items: center;
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
                    transform: translate3d(100%, 0, 0);
                }
                100% {
                    transform: translate3d(-100%, 0, 0);
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

    // Fisher-Yates shuffle algorithm
    shuffleArray(array) {
        for (let i = array.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [array[i], array[j]] = [array[j], array[i]];
        }
        return array;
    }

    async updateNews() {
        try {
            console.log('Fetching news...');
            const result = await this.fetchNews();
            
            if (result.error) {
                console.error('Error from news API:', result.error);
                this.showError(result.error);
                return;
            }
            
            if (!result.articles || !Array.isArray(result.articles) || result.articles.length === 0) {
                console.warn('No articles returned from API');
                this.showError('No news articles available');
                return;
            }

            console.log(`Received ${result.articles.length} articles`);
            
            // Shuffle the articles array
            const shuffledArticles = this.shuffleArray([...result.articles]);
            
            // Clear existing content
            this.tickerContent.innerHTML = '';
            
            // Create a container for repeated content
            const contentWrapper = document.createElement('div');
            contentWrapper.className = 'news-content-wrapper';
            
            // Create the base content first
            const baseContent = document.createElement('div');
            baseContent.className = 'news-content-base';
            
            // Add each article to the base content
            shuffledArticles.forEach(article => {
                const newsItem = document.createElement('div');
                newsItem.className = 'news-item';
                
                const link = document.createElement('a');
                link.href = article.url;
                link.target = '_blank';
                link.rel = 'noopener noreferrer';
                link.textContent = article.title;
                
                newsItem.appendChild(link);
                baseContent.appendChild(newsItem);
            });

            // Add the base content to wrapper 10 times
            for (let i = 0; i < 10; i++) {
                contentWrapper.appendChild(baseContent.cloneNode(true));
            }
            
            // Add the wrapper to the ticker
            this.tickerContent.appendChild(contentWrapper);

            // Calculate animation duration based on total content width
            const contentWidth = baseContent.scrollWidth * 10; // Multiply by 10 for repeated content
            const viewportWidth = this.container.offsetWidth;
            const pixelsPerSecond = 800;
            const totalDistance = contentWidth + viewportWidth;
            const duration = totalDistance / pixelsPerSecond;

            // Apply the animation
            this.tickerContent.style.animation = `ticker ${duration}s linear infinite`;
            
            console.log('News ticker updated successfully');
        } catch (error) {
            console.error('Error updating news:', error);
            this.showError('Failed to update news');
        }
    }

    showError(message) {
        this.tickerContent.innerHTML = `
            <div class="news-error">
                <i class="fas fa-exclamation-circle"></i>
                ${message}
            </div>
        `;
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