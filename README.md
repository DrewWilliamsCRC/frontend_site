# Frontend Site with User Auth (Flask)

A modern web application built with Flask that provides secure user authentication, dynamic content, and an interactive dashboard. Features weather forecasts, random pet images, and quick access to various services.

## Key Features

- 🔐 **Secure Authentication** - User registration and login with password hashing
- 🌤️ **Weather Dashboard** - 5-day forecast using OpenWeatherMap API
- 🐱 **Random Pet Images** - Integration with Dog and Cat APIs
- 🎯 **Service Quick Links** - Easy access to media and system services
- 👤 **User Management** - Admin dashboard for user administration
- 🌙 **Dark Mode** - Toggle between light and dark themes
- 📱 **Responsive Design** - Optimized for all device sizes
- 🐳 **Docker Ready** - Containerized deployment support

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

## Project Structure

```
├── app.py                 # Main Flask application
├── src/
│   └── user_manager.py    # User management blueprint
├── templates/             # HTML templates
├── static/               # CSS and assets
├── Dockerfile
└── docker-compose.yml
```

## Security Features

- Password hashing with Werkzeug
- CSRF protection
- Rate limiting
- Secure session configuration
- HTTPS-only cookies in production

## API Integration

- **OpenWeather API** - Weather forecasts
- **Dog API** - Random dog images
- **Cat API** - Random cat images

## Cache Configuration

- 5-minute default timeout
- In-memory storage
- Automatic invalidation

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
- **Weather Forecast:** Provides a dynamic 5-day weather forecast (sourced from the OpenWeatherMap API) based on a user-defined city. Each forecast card shows temperature in Celsius, weather description, and weather icon.
- **Service Dashboard:** Quick access buttons for external services such as Sonarr, Radarr, NZBGet, Unifi Controller, Code Server, and monitoring tools (e.g., Portainer and Glances).
- **User Settings:** Allows users to update their preferred city to tailor the weather information.
- **Admin User Management:** A dedicated administration section (accessible under `/admin`) enables listing, adding, editing, and deleting users.
- **Database Support:** Uses PostgreSQL for robust data storage in both development and production environments.
- **Caching:** Uses Flask-Caching to cache responses from external APIs (dog and cat images, weather forecasts) for improved performance.
- **Dark Mode:** A dark mode toggle is provided on every page, allowing users to switch between light and dark themes.
- **Improved UI & Responsiveness:** Modern templates and CSS styles with enhanced dark mode support and improved responsiveness across devices.
- **Dockerized Environment:** Comes with Docker and Docker Compose configurations to ease containerized deployment in production.
- **Rate Limiting:** Implements Flask-Limiter for API rate limiting (currently disabled but configurable).
- **CSRF Protection:** Implements Flask-WTF CSRF protection for enhanced security.

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
   - 200 requests per day
   - 50 requests per hour
   
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
