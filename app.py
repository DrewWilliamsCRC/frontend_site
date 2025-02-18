from dotenv import load_dotenv
load_dotenv()

import os
import requests
import datetime
import re
import psycopg2
from psycopg2.extras import RealDictCursor

from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from flask_caching import Cache
from src.user_manager_blueprint import user_manager_bp
from flask_wtf.csrf import CSRFProtect
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

app = Flask(__name__)

# Retrieve a strong, environment-specific secret key.
# Do not provide a fallback in production!
app.secret_key = os.environ.get("SECRET_KEY")
if not app.secret_key:
    raise ValueError("No SECRET_KEY set for Flask application. Please set the SECRET_KEY environment variable.")

# Production secure session cookie configuration.
app.config.update({
    'SESSION_COOKIE_DOMAIN': '.drewwilliams.biz',
    'SESSION_COOKIE_SECURE': True,       # Only send cookies over HTTPS in production.
    'SESSION_COOKIE_HTTPONLY': True,       # Prevent JavaScript access to cookies.
    'SESSION_COOKIE_SAMESITE': 'Lax'
})

# For local development (when running on 127.0.0.1), override the cookie settings.
if os.environ.get("FLASK_ENV") == "development":
    app.config.update({
        'SESSION_COOKIE_DOMAIN': None,  # Use the current domain (i.e., localhost)
        'SESSION_COOKIE_SECURE': False    # Allow cookies over HTTP in development.
    })

# Disable rate limiting for now.
app.config["RATELIMIT_ENABLED"] = False

# Initialize CSRF Protection
csrf = CSRFProtect(app)

# Initialize Flask-Limiter with the client's IP address as the key.
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)
limiter.init_app(app)

# Initialize cache: In-memory cache with 5-minute default timeout
cache = Cache(app, config={
    'CACHE_TYPE': 'SimpleCache',
    'CACHE_DEFAULT_TIMEOUT': 300,  # 5 minutes
})

# Use a relative path for the database file:
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Instead of hard-coding the API key,
OWM_API_KEY = os.environ.get("OWM_API_KEY", "default_api_key_for_dev")

# Select the appropriate DATABASE_URL depending on the environment
if os.environ.get("FLASK_ENV") == "development":
    DATABASE_URL = os.environ.get("DEV_DATABASE_URL", os.environ.get("DATABASE_URL"))
else:
    DATABASE_URL = os.environ.get("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("No DATABASE_URL set for the Flask application.")

def get_db_connection():
    """
    Returns a new database connection using psycopg2 and a RealDictCursor.
    """
    conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
    return conn

def init_db():
    """
    Initializes the database schema for the users table.
    Uses PostgreSQL-specific SQL syntax, with SERIAL for auto-increment.
    """
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    city_name TEXT
                );
            """)
            conn.commit()
        conn.close()
        print("Database initialized or updated.")
    except Exception as e:
        print("Error initializing the database:", e)

# -----------------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------------

def get_user_settings(username):
    """
    Retrieves the city setting for the given username using PostgreSQL.
    Falls back to a default city ("New York") if no value is found.
    """
    default_city = "New York"
    conn = get_db_connection()  # Uses psycopg2 and RealDictCursor for dictionary rows
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT city_name FROM users WHERE username=%s;", (username,))
            row = cur.fetchone()
    except Exception as e:
        print("Error retrieving user settings:", e)
        return default_city
    finally:
        conn.close()

    if row:
        # Assuming a RealDictCursor is used, 'row' is a dictionary.
        user_city = row["city_name"] if row["city_name"] is not None else default_city
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
def fetch_random_cat():
    """
    Fetch a random cat image from The Cat API.
    Returns the image URL, or a fallback image if the API call fails.
    """
    fallback_url = "https://via.placeholder.com/300?text=No+Cat+Image"
    try:
        r = requests.get("https://api.thecatapi.com/v1/images/search", timeout=5)
        r.raise_for_status()
        data = r.json()
        if data and isinstance(data, list) and len(data) > 0:
            return data[0].get("url", fallback_url)
    except Exception as e:
        app.logger.error("Error fetching random cat image: %s", e)
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
                    "dt": dt,
                    "date_str": date_str,
                    "icon_url": icon_url,
                    "description": description,
                    "temp_min": temp_min,
                    "temp_max": temp_max
                })
    except Exception as e:
        print(f"[ERROR] Failed to fetch daily forecast: {e}")
    return forecast_list

def sanitize_city_name(city):
    """
    Validate and sanitize the city name.
    Only allow letters, numbers, spaces, commas, periods, hyphens, and apostrophes.
    If invalid characters are found, they are removed.
    """
    city = city.strip()
    # Define a regex pattern that only allows the specified characters
    pattern = r"^[A-Za-z0-9\s,\.\-']+$"
    if re.match(pattern, city):
        # The city name is valid.
        return city
    else:
        # Remove any disallowed characters.
        sanitized = re.sub(r"[^A-Za-z0-9\s,\.\-']", "", city)
        # Fall back to a default value if nothing remains.
        return sanitized if sanitized else "New York"

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
    Displays random dog pic, cat image, and 5-day weather forecast.
    """
    if 'user' not in session:
        return redirect(url_for('login'))

    try:
        username = session['user']
        city_name = get_user_settings(username)
        
        dog_image_url = fetch_random_dog()
        cat_image_url = fetch_random_cat()
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
            cat_image_url=cat_image_url,
            daily_forecasts=daily_forecasts,
            lat=lat,
            lon=lon
        )
    except Exception as e:
        flash("An error occurred while loading the page. Please try again.", "error")
        return redirect(url_for('login'))

def get_coordinates_for_city(city_name):
    """
    Calls the OpenWeatherMap Geocoding API to get (lat, lon) for a given city;
    the city name is first sanitized to prevent issues with unexpected characters.
    """
    # Sanitize the user-supplied city name
    sanitized_city = sanitize_city_name(city_name)
    
    # Build the URL and parameters dict so that requests encodes values properly
    url = "http://api.openweathermap.org/geo/1.0/direct"
    params = {
        "q": sanitized_city,
        "limit": 1,
        "appid": OWM_API_KEY,  # Ensure this is defined (from your app config)
    }
    
    try:
        resp = requests.get(url, params=params, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        if data and len(data) > 0:
            return (data[0]["lat"], data[0]["lon"])
    except Exception as e:
        print(f"[ERROR] Geocoding city '{sanitized_city}' failed: {e}")
    
    return (None, None)

@app.route('/settings', methods=['GET', 'POST'])
def settings():
    """User can set their city_name."""
    if 'user' not in session:
        return redirect(url_for('login'))

    username = session['user']
    if request.method == 'POST':
        new_city = request.form.get('city_name', '').strip()
        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE users
                    SET city_name = %s
                    WHERE username = %s
                """, (new_city, username))
                conn.commit()
        except Exception as e:
            conn.rollback()
            flash("Error updating settings: {}".format(e), "danger")
        finally:
            conn.close()
        flash("Settings updated!", "success")
        return redirect(url_for('home'))
    else:
        # Show current settings using the updated PostgreSQL connection.
        city_name = get_user_settings(username)
        return render_template('settings.html', city_name=city_name)

@app.route('/login', methods=['GET', 'POST'])
@limiter.limit("5 per minute", methods=["POST"], error_message="Too many login attempts, please try again in a minute.")
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT password_hash FROM users WHERE username=%s;", (username,))
                row = cur.fetchone()
        except Exception as e:
            app.logger.error(f"Database error during login: {e}")
            return "Internal server error", 500
        finally:
            conn.close()

        if row:
            # With RealDictCursor, 'row' is a dict containing the column names.
            stored_hash = row["password_hash"]
            if check_password_hash(stored_hash, password):
                session['user'] = username
                flash("Login successful!", "success")
                return redirect(url_for('home'))
            else:
                app.logger.warning(
                    f"Failed login attempt for existing user {username} from {request.remote_addr}"
                )
                flash("Invalid credentials", "danger")
                return "Invalid credentials", 401
        else:
            app.logger.warning(
                f"Failed login attempt for non-existent user {username} from {request.remote_addr}"
            )
            flash("Invalid credentials", "danger")
            return "Invalid credentials", 401

    return render_template('login.html')

@app.route('/logout')
def logout():
    """Logout route."""
    session.pop('user', None)
    flash("Logged out successfully.", "info")
    return redirect(url_for('login'))

# --- New Refresh Endpoints ---
@app.route('/refresh_dog', methods=['POST'])
def refresh_dog():
    if 'user' not in session:
        return "Unauthorized", 401
    # Clear cached value to make another API call
    cache.delete_memoized(fetch_random_dog)
    new_dog_url = fetch_random_dog()
    return jsonify({"url": new_dog_url})

@app.route('/refresh_cat', methods=['POST'])
def refresh_cat():
    if 'user' not in session:
        return "Unauthorized", 401
    cache.delete_memoized(fetch_random_cat)
    new_cat_url = fetch_random_cat()
    return jsonify({"url": new_cat_url})

# Register the blueprint with a URL prefix (e.g., /admin)
app.register_blueprint(user_manager_bp, url_prefix="/admin")

# Updated route: Redirect to National Weather Service detailed forecast page
@app.route('/weather/details/<int:dt>')
def weather_details(dt):
    """Redirect to detailed weather information on the National Weather Service for the selected forecast day."""
    lat = request.args.get('lat')
    lon = request.args.get('lon')
    if not lat or not lon:
        flash("Coordinates are missing for weather details.", "warning")
        return redirect(url_for('home'))
    nws_url = f"https://forecast.weather.gov/MapClick.php?lat={lat}&lon={lon}"
    return redirect(nws_url)

@app.context_processor
def inject_is_dev_mode():
    # This will be True when FLASK_ENV is set to "development"
    return dict(is_dev_mode=(os.environ.get("FLASK_ENV") == "development"))

if __name__ == '__main__':
    # Determine if debug mode should be on. By default, it's off.
    # Set FLASK_DEBUG=1 (or "true"/"on") in your development environment.
    debug_mode = os.environ.get("FLASK_DEBUG", "0").lower() in ("1", "true", "on")
    
    # Optionally, read the port from an environment variable.
    port = int(os.environ.get("PORT", "5001"))
    
    # Run the Flask app with the appropriate settings.
    app.run(host='0.0.0.0', port=port, debug=debug_mode)
