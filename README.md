# Frontend Site with User Auth (Flask) - last updated 2-23-2024

![Docker Build Status](https://github.com/dawttu00/frontend_site/actions/workflows/docker-publish.yml/badge.svg)
![CodeQL Status](https://github.com/dawttu00/frontend_site/actions/workflows/codeql.yml/badge.svg)

A modern web application built with Flask that provides secure user authentication, dynamic content, and an interactive dashboard. Features weather forecasts, random pet images, and quick access to various services.

## Key Features

- 🔐 **Secure Authentication** - User registration and login with Werkzeug password hashing and rate limiting
- 🌤️ **Weather Dashboard** - 5-day forecast using OpenWeatherMap API with city customization
- 🐱 **Random Pet Images** - Integration with Dog and Cat APIs with instant refresh
- 🎯 **Service Quick Links** - Customizable dashboard for media and system services
- 👤 **User Management** - Admin dashboard for user administration with CSRF protection
- 🌙 **Dark Mode** - System-aware dark mode with smooth transitions
- 📱 **Responsive Design** - Mobile-first design with modern UI components
- 🐳 **Docker Ready** - Production-grade containerized deployment
- 🔒 **Security First** - HTTPS enforcement, secure sessions, and input validation

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

#### Features of the Docker Development Environment

- 🔄 **Live Reloading** - Changes to your code apply immediately
- 📂 **Volume Mounting** - Your local files are mounted into the container
- 🛠️ **Development Mode** - Flask debug mode is enabled
- 📊 **Database Persistence** - Your database data persists between restarts
- 🐳 **Isolated Environment** - Consistent development experience across machines

#### Manual Docker Compose (Alternative)

If you prefer to use Docker Compose directly:

```bash
# Start services
docker-compose -f docker-compose.dev.yml up -d

# Rebuild and start services
docker-compose -f docker-compose.dev.yml up -d --build

# Stop services
docker-compose -f docker-compose.dev.yml down
```

### Local Setup Without Docker

// ... existing code ...

## Project Structure

```
├── app.py                 # Main Flask application with security middleware
├── src/
│   ├── user_manager.py    # User management blueprint with auth checks
│   └── manage_db.py       # Database management with security practices
├── templates/             # Jinja2 templates with CSRF protection
├── static/               # Static assets and secure CSP headers
├── Dockerfile            # Production-ready Docker configuration
└── docker-compose.yml    # Orchestration with security best practices
```

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

## API Integration

- **OpenWeather API** - Weather forecasts
- **Dog API** - Random dog images
- **Cat API** - Random cat images
- **Gnews API** - News headlines ticker (100 requests/day limit)

## Cache Configuration

- 5-minute default timeout
- In-memory storage
- Automatic invalidation
- News headlines cached for 5 minutes
- API usage tracking for rate limits

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Push to the branch
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

For detailed documentation and API references, visit our [Wiki](link-to-wiki).

---

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
  - [Local Setup](#local-setup)
  - [Docker Setup](#docker-setup)
- [Running the Application](#running-the-application)
  - [Development Mode](#development-mode)
  - [Production Deployment](#production-deployment)
- [Database Management](#database-management)
- [Configuration](#configuration)
- [Frontend Details](#frontend-details)
- [Admin User Management](#admin-user-management)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Features

- **User Authentication:** Secure registration and login with hashed passwords (using Werkzeug) and session-based mechanisms.
- **Dynamic Home Page:** Displays a randomized dog image and a random cat image, both of which can be refreshed with a simple click.
- **News Ticker:** Real-time news headlines from Gnews API, displayed in a smooth scrolling ticker at the top of the page.
- **Weather Forecast:** Provides a dynamic 5-day weather forecast (sourced from the OpenWeatherMap API) based on a user-defined city. Each forecast card shows temperature in Celsius, weather description, and weather icon.
- **Service Dashboard:** Quick access buttons for external services such as Sonarr, Radarr, NZBGet, Unifi Controller, Code Server, and monitoring tools (e.g., Portainer and Glances).
- **User Settings:** Allows users to update their preferred city to tailor the weather information.
- **Admin User Management:** A dedicated administration section (accessible under `/admin`) enables listing, adding, editing, and deleting users.
- **Database Support:** Uses PostgreSQL for robust data storage in both development and production environments.
- **Caching:** Uses Flask-Caching to cache responses from external APIs (news headlines, dog and cat images, weather forecasts) for improved performance.
- **Dark Mode:** A dark mode toggle is provided on every page, allowing users to switch between light and dark themes.
- **Improved UI & Responsiveness:** Modern templates and CSS styles with enhanced dark mode support and improved responsiveness across devices.
- **Dockerized Environment:** Comes with Docker and Docker Compose configurations to ease containerized deployment in production.
- **Rate Limiting:** Implements Flask-Limiter for API rate limiting (currently disabled but configurable).
- **CSRF Protection:** Implements Flask-WTF CSRF protection for enhanced security.
- **API Usage Monitoring:** Tracks and displays usage statistics for external APIs to prevent rate limit issues.

---

## Project Structure

- **app.py:** Main Flask application that handles routes, sessions, API calls, and error handling.
- **src/user_manager_blueprint.py:** Blueprint for user management functionality.
- **wsgi.py:** WSGI entry point (for servers such as Gunicorn).
- **templates/**: Contains the HTML templates:
  - `base.html` – Common layout with navigation, messages, and dark mode toggle.
  - `index.html` – Landing page with weather forecast and random dog/cat images.
  - `login.html` – Login form.
  - `settings.html` – User settings form.
  - `user_manager_index.html`, `add_user.html`, `edit_user.html` – Templates for administering users.
- **static/**: Contains CSS files and assets
- **.env:** Contains environment variables including:
  - SECRET_KEY
  - OWM_API_KEY
  - FLASK_DEBUG
  - FLASK_ENV
  - PORT
  - DATABASE_URL
- **Dockerfile:** Instructions to containerize the application.
- **docker-compose.yml:** Defines the Docker services and configuration.
- **requirements.txt:** Lists the required Python dependencies.

---

## Prerequisites

- **Python 3.10+**
- **pip**
- **Virtual Environment** (recommended for local development)
- **Docker & Docker Compose** (for containerized deployments)
- **PostgreSQL**

---

## Installation

### Local Setup

1. **Clone the Repository:**
   ```bash
   git clone <repository_url>
   cd frontend_site
   ```

2. **Create and Activate a Virtual Environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables:**
   Create a `.env` file with the following required variables:
   ```
   SECRET_KEY="your-secret-key"
   OWM_API_KEY="your-openweather-api-key"
   FLASK_DEBUG=1
   FLASK_ENV=development
   PORT=5001
   DATABASE_URL=postgresql://user:password@localhost:5432/dbname
   ```

### Docker Setup
1. **Build the Docker Image:**
   ```bash
   docker build -t frontend-site .
   ```

2. **Configure Environment Variables:**
   Create a `.env` file with the same variables as local setup, but update the DATABASE_URL to use the Docker service name:
   ```
   DATABASE_URL=postgresql://db:password@postgres:5432/frontend_db
   ```

3. **Configure Docker Network:**
   The application and database containers will automatically be connected via the network defined in docker-compose.yml.

1. **Build and Run with Docker Compose:**
   ```bash
   docker compose up -d
   ```

---
## Database Setup

1. **Create PostgreSQL Database:**
   ```bash
   createdb frontend_db
   ```

2. **Initialize Database Schema:**
   The application will automatically create required tables on startup. Alternatively, you can manually initialize:
   ```bash
   python3 -c "from app import init_db; init_db()"
   ```

## Security Configuration

1. **Generate a Strong Secret Key:**
   ```bash
   python3 -c "import secrets; print(secrets.token_hex(32))"
   ```
   Add the generated key to your `.env` file.

2. **Configure Session Security:**
   - Production settings enforce HTTPS-only cookies
   - Local development allows HTTP cookies
   - All cookies are HttpOnly and use SameSite=Lax

3. **Rate Limiting:**
   Default limits are configured to:
   - 200 requests per day per IP
   - 50 requests per hour per IP
   
4. **CSRF Protection:**
   All forms are protected against Cross-Site Request Forgery attacks.

## API Keys

1. **OpenWeather API:**
   - Sign up at [OpenWeather](https://openweathermap.org/api)
   - Get your API key from the account dashboard
   - Add to `.env` as `OWM_API_KEY`

## Caching

The application uses SimpleCache with:
- 5-minute default timeout
- In-memory storage
- Automatic cache invalidation

## Running the Application

### Development Mode

Run the Flask development server:

```
python3 app.py
```

### Production Deployment

Use Docker Compose to run the application in production:

```bash
docker compose up -d
```

---

## Database Management

### PostgreSQL

The application uses PostgreSQL as its database. Ensure you have PostgreSQL installed and running on your system.

### Database Schema

The application will automatically create the required database schema on startup. You can also manually initialize the database schema using the following command:

```bash
python3 -c "from app import init_db; init_db()"
```

---

## Configuration

### Environment Variables

The application uses environment variables to configure its behavior. You can set these variables in a `.env` file or using your operating system's environment variable settings.

### Required Environment Variables

- `SECRET_KEY`: A secure random key for session management (32+ bytes recommended)
- `OWM_API_KEY`: OpenWeatherMap API key for weather data
- `FLASK_DEBUG`: Debug mode flag (disabled in production)
- `FLASK_ENV`: Runtime environment (development/production)
- `PORT`: Application port number
- `DATABASE_URL`: PostgreSQL connection string (with SSL in production)
- `SESSION_COOKIE_DOMAIN`: Domain for session cookies
- `SESSION_COOKIE_SECURE`: HTTPS-only cookie flag
- `RATELIMIT_ENABLED`: Toggle rate limiting functionality

### Security Configuration

1. **Generate a Strong Secret Key:**
   ```bash
   python3 -c "import secrets; print(secrets.token_hex(32))"
   ```
   Add the generated key to your `.env` file.

2. **Configure Session Security:**
   - Production settings enforce HTTPS-only cookies
   - Local development allows HTTP cookies
   - All cookies are HttpOnly and use SameSite=Lax
   - Session cookies are domain-scoped

3. **Rate Limiting:**
   Default limits are configured to:
   - 200 requests per day per IP
   - 50 requests per hour per IP
   - 5 login attempts per minute

4. **CSRF Protection:**
   - All forms include CSRF tokens
   - Tokens are rotated regularly
   - Invalid tokens return 403 errors

## Production Deployment

### Security Checklist

1. **Environment Setup**
   - [ ] Generate new SECRET_KEY
   - [ ] Configure secure DATABASE_URL
   - [ ] Set FLASK_DEBUG=0
   - [ ] Enable rate limiting
   - [ ] Configure HTTPS

2. **Docker Security**
   - [ ] Use non-root user
   - [ ] Set resource limits
   - [ ] Enable security options
   - [ ] Regular base image updates

3. **Database Security**
   - [ ] Strong passwords
   - [ ] SSL connections
   - [ ] Regular backups
   - [ ] Access control

4. **Monitoring**
   - [ ] Error logging
   - [ ] Access logging
   - [ ] Rate limit alerts
   - [ ] Security scanning

## Production Deployment Checklist

Before deploying to production, ensure the following:

### Security
- [ ] Generate new strong SECRET_KEY
- [ ] Update all API keys with production credentials
- [ ] Set FLASK_DEBUG=0 and FLASK_ENV=production
- [ ] Enable HTTPS/SSL
- [ ] Configure secure cookie settings
- [ ] Set up proper firewall rules
- [ ] Enable rate limiting

### Database
- [ ] Use strong PostgreSQL password
- [ ] Enable SSL for database connections
- [ ] Configure automated backups
- [ ] Set up database monitoring
- [ ] Configure connection pooling
- [ ] Set appropriate resource limits

### Docker
- [ ] Update image tags to use specific versions
- [ ] Configure container resource limits
- [ ] Enable security options
- [ ] Set up container monitoring
- [ ] Configure logging
- [ ] Set up container restart policies

### Monitoring
- [ ] Set up error tracking
- [ ] Configure access logging
- [ ] Set up performance monitoring
- [ ] Enable security scanning
- [ ] Configure alerts
- [ ] Set up uptime monitoring

### Backup
- [ ] Configure automated database backups
- [ ] Set up backup verification
- [ ] Configure backup retention
- [ ] Test backup restoration
- [ ] Document recovery procedures

### Documentation
- [ ] Update deployment documentation
- [ ] Document rollback procedures
- [ ] Update API documentation
- [ ] Document monitoring setup
- [ ] Create incident response plan
- [ ] Document security procedures

## Acknowledgements

* OpenWeatherMap API for weather data
* Dog and Cat APIs for pet images
* Flask and its extensions for the web framework
* Werkzeug for security features
* Docker for containerization
* PostgreSQL for database management
* Bootstrap and Font Awesome for UI components
* The security community for best practices
* Alpha Vantage API for stock market data

