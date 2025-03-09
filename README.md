# Frontend Site with User Auth (Flask) - last updated 3-8-2024

![Docker Build Status](https://github.com/dawttu00/frontend_site/actions/workflows/docker-publish.yml/badge.svg)
![CodeQL Status](https://github.com/dawttu00/frontend_site/actions/workflows/codeql.yml/badge.svg)

A modern web application built with Flask that provides secure user authentication, dynamic content, and interactive dashboards. Features AI-powered financial analytics, market insights, weather forecasts, news aggregation, random pet images, and quick access to various services.

## Key Features

- 🔐 **Secure Authentication** - User registration and login with Werkzeug password hashing and rate limiting
- 📈 **AI Financial Dashboard** - Advanced analytics with market indices, economic indicators, and AI-driven insights
- 📊 **Market Data** - Real-time market indices with visual indicators and data visualization
- 🌤️ **Weather Dashboard** - 5-day forecast using OpenWeatherMap API with city customization
- 📰 **News Aggregation** - News headlines from various sources with caching and rate limit handling
- 🐱 **Random Pet Images** - Integration with Dog and Cat APIs with instant refresh
- 🎯 **Service Quick Links** - Customizable dashboard for media and system services
- 👤 **User Management** - Admin dashboard for user administration with CSRF protection
- 🌙 **Dark Mode** - System-aware dark mode with smooth transitions
- 📱 **Responsive Design** - Mobile-first design with modern UI components
- 🐳 **Docker Ready** - Production-grade containerized deployment
- 🔒 **Security First** - HTTPS enforcement, secure sessions, and input validation
- 🚀 **Performance Optimized** - Caching mechanisms, API rate limiting, and fallback to mock data

## Quick Start

### Prerequisites
- Python 3.10+
- PostgreSQL
- Docker & Docker Compose (optional)

### Local Setup

1. Clone and setup environment:
```bash
git clone <repository_url>
cd frontend_site
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Configure `.env`:
```
SECRET_KEY="your-secret-key"
OWM_API_KEY="your-openweather-api-key"
GUARDIAN_API_KEY="your-guardian-api-key"
ALPHA_VANTAGE_API_KEY="your-alpha-vantage-api-key"
FLASK_DEBUG=1
FLASK_ENV=development
PORT=5001
DATABASE_URL=postgresql://user:password@localhost:5432/dbname
```

3. Initialize database:
```bash
createdb frontend_db
python3 -c "from app import init_db; init_db()"
```

### Docker Setup

1. Build and run:
```bash
docker compose up -d
```

### Docker Development Environment

This project includes a local Docker development setup that's optimized for fast iteration and easy setup.

#### Using the Development Script

We've included a convenient script to manage the Docker development environment:

1. Make the script executable:
```bash
chmod +x dev.sh
```

2. Start the development environment:
```bash
./dev.sh up
```

3. Access the application:
   - Frontend: http://localhost:5001
   - PostgreSQL: localhost:5432 (using credentials from .env)

4. Other useful commands:
```bash
./dev.sh down       # Stop the environment
./dev.sh rebuild    # Rebuild and restart containers
./dev.sh logs       # View logs
./dev.sh exec       # Open a shell in the frontend container
./dev.sh db         # Open PostgreSQL CLI
./dev.sh help       # Show all commands
```

## New AI Dashboard Features

The application now includes a comprehensive AI Financial Dashboard with the following features:

- **Market Indices**: Real-time market data displayed in a responsive card layout showing price, change, and high/low values with visual indicators
- **Economic Indicators**: Key economic data presented in an easy-to-scan format with trend visualization
- **AI Market Prediction**: Predictive analytics for market movements with confidence metrics
- **News Sentiment Analysis**: AI-powered analysis of news sentiment impact on markets
- **Portfolio Optimization**: AI-driven portfolio strategy recommendations
- **Predictive Feature Importance**: Visualization of the most important factors in market predictions
- **Alert System**: Configurable alerts for market conditions

### AI Dashboard Components

- **Debug Panel**: Compact status indicator showing loading state of components
- **Market Indices Cards**: Short, wide cards showing key market information at a glance
- **Prediction History Chart**: Collapsible chart showing prediction accuracy over time
- **Economic Indicators**: Visual representation of economic data with trend indicators

### Performance Optimizations

- **Ticker Disabling**: Market and news tickers are automatically hidden when using intensive dashboards
- **API Caching**: Intelligent caching of API responses with appropriate timeouts
- **Mock Data Fallbacks**: Automatic fallback to mock data when API rate limits are reached
- **Exponential Backoff**: Retry logic with exponential backoff for API calls
- **Responsive Layout**: Optimized layouts for various screen sizes

## Project Structure

```
├── app.py                 # Main Flask application with routes and API integrations
├── static/                # Static assets including CSS and JavaScript
│   ├── css/               # Stylesheets including ai-dashboard.css
│   ├── js/                # JavaScript files for interactive features
│   └── img/               # Image assets
├── templates/             # Jinja2 templates for rendering pages
│   ├── base.html          # Base template with common elements
│   ├── index.html         # Homepage template
│   ├── ai_dashboard.html  # AI dashboard template
│   └── ...                # Other page templates
├── ai_experiments/        # AI functionality and analysis modules
├── src/                   # Core application modules
├── Dockerfile             # Production-ready Docker configuration
├── docker-compose.yml     # Production Docker Compose configuration
├── docker-compose.dev.yml # Development Docker Compose configuration
└── dev.sh                 # Development environment management script
```

## API Integrations

- **Alpha Vantage API** - Market data and stock information
- **Guardian API** - News articles and headlines
- **OpenWeather API** - Weather forecasts
- **Dog API** - Random dog images
- **Cat API** - Random cat images

## Error Handling and Resilience

The application includes comprehensive error handling:

- **API Rate Limiting**: Automatic detection and handling of API rate limits
- **Mock Data Generation**: Fallback to realistic mock data when APIs are unavailable
- **Logging**: Detailed logging of errors and warnings
- **Retry Logic**: Intelligent retry with backoff for transient errors
- **User Feedback**: Clear error messages and status indicators

## Security Features

- **Authentication & Authorization**
  - Password hashing using Werkzeug's secure implementation
  - Rate limiting on login attempts (5 per minute)
  - Session-based authentication with secure cookie handling
  - Admin-only routes protection

- **Web Security**
  - CSRF protection on all forms
  - XSS protection through proper escaping
  - SQL injection protection using parameterized queries
  - Secure headers configuration
  - HTTPS enforcement in production

- **Session Security**
  - HttpOnly cookie flags
  - Secure cookie flags in production
  - SameSite=Lax cookie attribute
  - Domain-specific cookie scope
  - Configurable session timeouts

- **Infrastructure Security**
  - Docker container isolation
  - Environment-based configuration
  - Secure secret management
  - Database connection pooling
  - Regular security updates

## Cache Configuration

- Intelligent caching strategy based on data type
- News headlines cached for up to 30 minutes
- Market data cached based on market hours
- Weather data cached for 30 minutes
- API usage tracking for rate limits
- Automatic cache invalidation on relevant updates

## Contributing

Please refer to [CONTRIBUTING.md](CONTRIBUTING.md) for detailed information on how to contribute to this project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Security

For security-related information, please see [SECURITY.md](SECURITY.md).

## Acknowledgements

- Alpha Vantage API for market data
- Guardian API for news content
- OpenWeather API for weather data
- Dog and Cat APIs for pet images
- Flask and its extensions for the web framework
- Werkzeug for security features
- Docker for containerization
- PostgreSQL for database management
- Bootstrap and Font Awesome for UI components
- Chart.js for data visualization

