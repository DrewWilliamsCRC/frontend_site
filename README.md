# Frontend Site with User Auth (Flask)

This project is a lightweight web application built with Flask that provides user authentication and a customizable landing page. It integrates external APIs to display a dynamic 5-day weather forecast and a random dog image while offering quick links to various media management, system, and monitoring services.

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
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Features

- **User Authentication:** Secure registration and login using hashed passwords (via Werkzeug) with a simple session-based mechanism.
- **Dynamic Home Page:** Displays a random dog image (from Dog CEO API) and a 5-day weather forecast based on a user-specified city using the OpenWeatherMap API.
- **Service Dashboard:** Quick access buttons linking to external services like Sonarr, Radarr, NZBGet, Unifi Controller, Code Server, and various monitoring tools.
- **User Settings:** Update your preferred city to customize your weather forecast.
- **SQLite Database:** A lightweight database that automatically initializes (or upgrades) and stores user information and settings.
- **Caching:** Implements Flask-Caching to cache responses from external APIs (weather and dog image) for improved performance.
- **Dockerized Environment:** Comes with Docker and Docker Compose configurations to simplify deployment in production environments.

---

## Project Structure

- **app.py:** Main Flask application handling routes, sessions, API calls, and error handling.
- **manage_db.py:** Command-line interface for managing the SQLite database (adding, updating, listing, and deleting users).
- **wsgi.py:** Entry point for WSGI servers (e.g., Gunicorn) used in production.
- **templates/**: HTML templates, including:
  - `base.html` – The base layout with navigation and flash message support.
  - `index.html` – The landing page displaying weather forecast and random dog image.
  - `login.html` – Login form.
  - `settings.html` – User settings form.
- **static/**: Directory containing CSS files for styling:
  - `css/style.css` – Custom styles for the application.
  - `style.css` – Additional styling for modern color schemes and responsive components.
- **Dockerfile:** Instructions to containerize the application. It installs dependencies, copies the code, and sets up Gunicorn as the production server.
- **docker-compose.yml:** Defines the Docker service, volume mounts (for persisting the SQLite database), and port mappings.
- **deploy.sh:** A script to sync code and redeploy the containerized services.
- **requirements.txt:** Lists all Python dependencies required by the project.
- **.gitignore:** Specifies files and directories to be ignored by Git (virtual environments, database files, logs, etc.).

---

## Prerequisites

- **Python 3.10** or higher
- **pip** for managing Python packages
- **Virtual Environment** (recommended for local development)
- **Docker & Docker Compose** (for containerized deployment)

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

4. **Initialize the Database:**
   - The SQLite database (`users.db`) is automatically created and updated when the application starts.
   - Alternatively, manage users via the CLI using:
     ```bash
     python manage_db.py
     ```

### Docker Setup

This project includes Docker support for seamless production deployment.

1. **Build the Docker Image:**
   ```bash
   docker build -t frontend_site .
   ```

2. **Use Docker Compose to Run the Application:**
   ```bash
   docker compose up -d
   ```
   - This command creates and runs a container named `frontend` by mapping the local `./data` directory to persist the SQLite database (`/app/data`) and exposing port 5001.

3. **Deploy Script:**
   - A sample deploy script (`deploy.sh`) is provided:
     ```bash
     ./deploy.sh
     ```

---

## Running the Application

### Development Mode

- **Run the Flask Development Server:**
  ```bash
  python app.py
  ```
  - The server will start on port 5001 by default.
  - Access the application at [http://0.0.0.0:5001](http://0.0.0.0:5001).

### Production Deployment

- **Run with Gunicorn:**
  ```bash
  gunicorn --bind 0.0.0.0:5001 app:app
  ```
  - This is configured in the Dockerfile for containerized production deployments.

- **Nginx Reverse Proxy Example:**
  ```nginx
  server {
      listen 80;
      server_name yourdomain.com;

      location / {
          proxy_pass http://127.0.0.1:5001;
          proxy_set_header Host $host;
          proxy_set_header X-Real-IP $remote_addr;
          proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
          proxy_set_header X-Forwarded-Proto $scheme;
      }
  }
  ```
- **Security Recommendations:**
  - Update `app.secret_key` in `app.py` with a strong, random string.
  - Always use HTTPS in production.
  - Consider stronger password policies and two-factor authentication for enhanced security.

---

## Database Management

- **SQLite Database:**
  - The database file is stored as `users.db` (or within the Docker container at `/app/data/users.db`).
  - Use `manage_db.py` to perform CRUD operations on user accounts:
    ```bash
    python manage_db.py
    ```

---

## Configuration

- **API Keys:**
  - Update the OpenWeatherMap API key in `app.py` (variable `OWM_API_KEY`) with your own key from [OpenWeatherMap](https://openweathermap.org/api).
  
- **Database Location:**
  - When running via Docker, the SQLite database is persisted by mounting `./data` to `/app/data` in the container.

---

## Frontend Details

- **Templates:**
  - The HTML templates make use of Jinja2 templating and include:
    - `base.html` for the common layout, including navigation and flash messages.
    - `index.html` for the dynamic home page with weather forecasts and dog images.
    - `login.html` and `settings.html` for user authentication and configuration.
  
- **Static Files:**
  - Custom styles are defined in `static/css/style.css` and `static/style.css`.
  - The design leverages Bootstrap for responsive layout and Font Awesome for icons.

---

## Contributing

Contributions are welcome! Please fork the repository, make your changes, and submit a pull request. For major changes, please open an issue first to discuss what you would like to change.

---

## License

[Include your license information here. For example: MIT License]

---

## Acknowledgements

- [Flask](https://flask.palletsprojects.com/)
- [Werkzeug](https://werkzeug.palletsprojects.com/)
- [OpenWeatherMap API](https://openweathermap.org/api)
- [Dog CEO's Dog API](https://dog.ceo/dog-api/)
- [Bootstrap](https://getbootstrap.com/)
- [Font Awesome](https://fontawesome.com/)

---

This README provides comprehensive instructions on setting up, running, and deploying the project in various environments. For any issues, improvements, or further discussions, please open an issue in the repository.

