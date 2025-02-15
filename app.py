import os
import sqlite3
import random
import requests
import datetime
import time

from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
from flask_caching import Cache
from src.user_manager_blueprint import user_manager_bp

app = Flask(__name__)
app.secret_key = 'CHANGE_ME_TO_SOMETHING_SECURE'  # Replace with your own secret key.

# Ensure cookies are valid for any subdomain if needed
app.config.update({
    'SESSION_COOKIE_DOMAIN': '.drewwilliams.biz'
})

# Initialize cache: In-memory by default, 5-minute default timeout
cache = Cache(app, config={
    'CACHE_TYPE': 'SimpleCache',
    'CACHE_DEFAULT_TIMEOUT': 300  # 5 minutes
})

DB_NAME = '/app/data/users.db'

# --- Hard-coded API Keys ---
OWM_API_KEY = "f869c65af9d218710883f3321b2cf709"

def init_db():
    """
    Initialize the DB, ensuring 'users' table has city_name column.
    """
    with sqlite3.connect(DB_NAME) as conn:
        # Create the users table if not exists
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL
            )
        """)
        # Add city_name column if not present
        try:
            conn.execute("ALTER TABLE users ADD COLUMN city_name TEXT")
        except sqlite3.OperationalError:
            pass

    print("Database initialized or updated.")

init_db()

# -----------------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------------

def get_user_settings(username):
    """
    Retrieves the city_name for the given user.
    Returns a default if blank.
    """
    default_city = "New York"
    with sqlite3.connect(DB_NAME) as conn:
        cur = conn.cursor()
        cur.execute("SELECT city_name FROM users WHERE username=?", (username,))
        row = cur.fetchone()
    if row:
        user_city = row[0] if row[0] else default_city
        return user_city
    else:
        return default_city


@cache.memoize(timeout=300)  # Cache for 5 minutes
def fetch_random_dog():
    """
    Fetch a random dog image from https://dog.ceo/api/breeds/image/random.
    Returns the image URL, or a fallback image URL if the API call fails.
    """
    fallback_url = "https://via.placeholder.com/300?text=No+Dog+Image"
    try:
        r = requests.get("https://dog.ceo/api/breeds/image/random", timeout=5)
        r.raise_for_status()
        data = r.json()
        return data.get("message", fallback_url)
    except Exception as e:
        app.logger.error("Error fetching random dog image: %s", e)
        return fallback_url


@cache.memoize(timeout=300)  # Cache for 5 minutes
def get_weekly_forecast(lat, lon):
    """
    Calls OWM One Call API for daily forecast (up to 7 days),
    but we'll slice to the first 5 days for a "5-day" forecast.
    Returns a list of daily forecast dicts.
    """
    url = (
        f"http://api.openweathermap.org/data/3.0/onecall"
        f"?lat={lat}&lon={lon}&exclude=current,minutely,hourly,alerts"
        f"&units=imperial"
        f"&appid={OWM_API_KEY}"
    )
    forecast_list = []
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        if "daily" in data:
            daily_data = data["daily"][:5]  # Take the first 5 days
            for day in daily_data:
                dt = day["dt"]  # Unix timestamp
                date_str = datetime.datetime.utcfromtimestamp(dt).strftime("%b %d")
                icon_code = day["weather"][0]["icon"]
                icon_url = f"https://openweathermap.org/img/wn/{icon_code}@2x.png"
                description = day["weather"][0].get("description", "")
                temp_min = round(day["temp"]["min"], 1)
                temp_max = round(day["temp"]["max"], 1)

                forecast_list.append({
                    "date_str": date_str,
                    "icon_url": icon_url,
                    "description": description,
                    "temp_min": temp_min,
                    "temp_max": temp_max
                })
    except Exception as e:
        print(f"[ERROR] Failed to fetch daily forecast: {e}")
    return forecast_list

# -----------------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------------

@app.route('/auth-check')
def auth_check():
    """Internal endpoint for NGINX's auth_request directive."""
    if 'user' in session:
        return '', 200
    else:
        return '', 401

@app.route('/')
def home():
    """
    Home page, requires user login.
    Displays random dog pic and 5-day weather forecast.
    """
    if 'user' not in session:
        return redirect(url_for('login'))

    try:
        username = session['user']
        city_name = get_user_settings(username)
        
        # Add error handling and loading states
        dog_image_url = fetch_random_dog()
        lat, lon = get_coordinates_for_city(city_name)
        daily_forecasts = []
        
        if lat is not None and lon is not None:
            daily_forecasts = get_weekly_forecast(lat, lon)
        else:
            flash("Could not fetch weather data for your city. Please check your city name in settings.", "warning")
        
        return render_template(
            "index.html",
            user=username,
            city_name=city_name,
            dog_image_url=dog_image_url,
            daily_forecasts=daily_forecasts
        )
    except Exception as e:
        flash("An error occurred while loading the page. Please try again.", "error")
        return redirect(url_for('login'))

def get_coordinates_for_city(city_name):
    """Uses OpenWeatherMap Geocoding API to get (lat, lon) for a city."""
    url = f"http://api.openweathermap.org/geo/1.0/direct?q={city_name}&limit=1&appid={OWM_API_KEY}"
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        if data and len(data) > 0:
            return (data[0]["lat"], data[0]["lon"])
    except Exception as e:
        print(f"[ERROR] Geocoding city '{city_name}' failed: {e}")
    return (None, None)

@app.route('/settings', methods=['GET', 'POST'])
def settings():
    """User can set their city_name."""
    if 'user' not in session:
        return redirect(url_for('login'))

    username = session['user']
    if request.method == 'POST':
        new_city = request.form.get('city_name', '').strip()
        with sqlite3.connect(DB_NAME) as conn:
            cur = conn.cursor()
            cur.execute("""
                UPDATE users
                SET city_name = ?
                WHERE username = ?
            """, (new_city, username))
            conn.commit()

        flash("Settings updated!", "success")
        return redirect(url_for('home'))
    else:
        # Show current settings
        city_name = get_user_settings(username)
        return render_template('settings.html', city_name=city_name)

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login route."""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        with sqlite3.connect(DB_NAME) as conn:
            cur = conn.cursor()
            cur.execute("SELECT password_hash FROM users WHERE username=?", (username,))
            row = cur.fetchone()

        if row:
            stored_hash = row[0]
            if check_password_hash(stored_hash, password):
                # Valid login
                session['user'] = username
                flash("Login successful!", "success")
                return redirect(url_for('home'))
            else:
                # Wrong password
                app.logger.warning(
                    f"Failed login attempt for existing user {username} from {request.remote_addr}"
                )
                return "Invalid credentials", 401
        else:
            # No such user
            app.logger.warning(
                f"Failed login attempt for non-existent user {username} from {request.remote_addr}"
            )
            return "Invalid credentials", 401

    return render_template('login.html')

@app.route('/logout')
def logout():
    """Logout route."""
    session.pop('user', None)
    flash("Logged out successfully.", "info")
    return redirect(url_for('login'))

# Register the blueprint with a URL prefix (e.g., /admin)
app.register_blueprint(user_manager_bp, url_prefix="/admin")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
