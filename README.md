# Frontend Site with User Auth (Flask)

This project is a lightweight web application built with Flask that provides secure user authentication, a customizable landing page, and an interactive dashboard. It integrates multiple external APIs to display a dynamic 5-day weather forecast, a random dog image, and a random cat image—all of which can be refreshed on demand. In addition, the application provides quick access links to various media management, system, and monitoring services. An administration section (via a user management blueprint) is available to manage users from a dedicated dashboard.

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
- **Weather Forecast:** Provides a dynamic 5-day weather forecast (sourced from the OpenWeatherMap API) based on a user-defined city. Each forecast card links to detailed weather data on the National Weather Service website.
- **Service Dashboard:** Quick access buttons for external services such as Sonarr, Radarr, NZBGet, Unifi Controller, Code Server, and monitoring tools (e.g., Portainer and Glances).
- **User Settings:** Allows users to update their preferred city to tailor the weather information.
- **Admin User Management:** A dedicated administration section (accessible under `/admin`) enables listing, adding, editing, and deleting users.
- **SQLite Database:** A lightweight SQLite database is automatically created and updated on startup to store user information and settings.
- **Caching:** Uses Flask-Caching to cache responses from external APIs (dog and cat images, weather forecasts) for improved performance.
- **Dark Mode:** A dark mode toggle is provided on every page, allowing users to switch themes on the fly.
- **Dockerized Environment:** Comes with Docker and Docker Compose configurations to ease containerized deployment in production.

---

## Project Structure

- **app.py:** Main Flask application that handles routes, sessions, API calls, and error handling.
- **manage_db.py:** A CLI tool for database management (adding, updating, listing, and deleting users).
- **wsgi.py:** WSGI entry point (for servers such as Gunicorn).
- **templates/**: Contains the HTML templates:
  - `base.html` – Common layout with navigation, messages, and dark mode toggle.
  - `index.html` – Landing page with weather forecast and random dog/cat images.
  - `login.html` – Login form.
  - `settings.html` – User settings form.
  - `user_manager_index.html`, `add_user.html`, `edit_user.html` – Templates for administering users.
- **static/**: Contains CSS files:
  - `css/style.css` – Custom styles for the application.
  - `style.css` – Additional modern styling.
- **Dockerfile:** Instructions to containerize the application using Gunicorn as the production server.
- **docker-compose.yml:** Defines the Docker service, volume mounts (for persisting the SQLite database), and port mappings.
- **deploy.sh:** A deploy script to sync code and redeploy containerized services.
- **requirements.txt:** Lists the required Python dependencies.
- **.gitignore:** Specifies files and directories (e.g., virtual environments, database files, logs) to ignore.

---

## Prerequisites

- **Python 3.10+**
- **pip**
- **Virtual Environment** (recommended for local development)
- **Docker & Docker Compose** (for containerized deployments)

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
   - Alternatively, you can manage users via the CLI using:
     ```bash
     python manage_db.py
     ```

### Docker Setup

1. **Build the Docker Image:**
   ```bash
   docker build -t frontend_site .
   ```

2. **Run with Docker Compose:**
   ```bash
   docker compose up -d
   ```
   - This command runs a container named `frontend` while mapping the local `./data` directory to persist the SQLite database at `/app/data`, and exposing port `5001`.

3. **Using the Deploy Script:**
   ```bash
   ./deploy.sh
   ```

---

## Running the Application

### Development Mode

Run the Flask development server:
```bash
python app.py
```
- By default, the app listens on port 5001.
- Open your browser at [http://0.0.0.0:5001](http://0.0.0.0:5001).

### Production Deployment

**Using Gunicorn:**
```bash
gunicorn --bind 0.0.0.0:5001 app:app
```
- This is also configured within the Dockerfile for production deployments.

A sample **Nginx reverse proxy** configuration:
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

**Security Recommendations:**

- Set a strong `SECRET_KEY` via environment variables.
- Always use HTTPS in production.
- Consider implementing two-factor authentication or advanced security measures as needed.

---

## Database Management

- **SQLite Database:** Located as `users.db` (or at `/app/data/users.db` in Docker).
- **CLI Management:** Use `manage_db.py` for adding, updating, listing, and deleting users:
  ```bash
  python manage_db.py
  ```

---

## Configuration

- **API Keys:**
  - Update the OpenWeatherMap API key in `app.py` (the variable `OWM_API_KEY`) with your own key available from [OpenWeatherMap](https://openweathermap.org/api).
- **Environment Variables:**
  - Ensure that `SECRET_KEY` (for session security) and `OWM_API_KEY` are set in your environment or via the `.env` file.
- **Database Persistence:**
  - When using Docker, the SQLite database is persisted by mounting `./data` to `/app/data` in the container.

---

## Frontend Details

- **Templates & Styling:** 
  - The application uses Jinja2 templates for dynamic content rendering.
  - Custom CSS styles (found in `static/css/style.css` and `static/style.css`) provide a modern look and responsive design.
- **Dynamic Content Refresh:**
  - Clicking on the dog or cat images sends a POST request to `/refresh_dog` or `/refresh_cat` respectively to fetch a new image, leveraging Flask-Caching.
- **Dark Mode Support:** 
  - A dark mode toggle button (located in the navigation bar) allows users to switch between light and dark themes. User preference is stored using local storage.

---

## Admin User Management

- **User Dashboard:** 
  - A dedicated admin section is available via the `/admin` URL prefix.
  - Administrators can view a list of registered users, add new users, edit existing users, and delete users—all via a user-friendly interface.
- **Blueprint Integration:**
  - The user management functionality is provided by a Flask blueprint (`user_manager_blueprint`) that organizes all admin routes under the `/admin` route.

---

## Contributing

Contributions are welcome! Please fork the repository, create a feature branch for your changes, and submit a pull request. For major changes, please open an issue first to discuss what you'd like to change.

---

## License

[Include your license information here. For example: MIT License]

---

## Acknowledgements

- [Flask](https://flask.palletsprojects.com/)
- [Werkzeug](https://werkzeug.palletsprojects.com/)
- [OpenWeatherMap API](https://openweathermap.org/api)
- [Dog CEO's Dog API](https://dog.ceo/dog-api/)
- [The Cat API](https://thecatapi.com/)
- [Bootstrap](https://getbootstrap.com/)
- [Font Awesome](https://fontawesome.com/)

---

This README provides comprehensive instructions on setting up, running, and deploying the project in various environments. For any issues, improvements, or further discussions, please open an issue in the repository.

