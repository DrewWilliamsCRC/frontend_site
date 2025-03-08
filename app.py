# Load environment variables from .env file for configuration management
from dotenv import load_dotenv
import os
import random

# Load environment variables before any other configuration
print("Loading environment variables from .env file...")
load_dotenv()

# Debug log for API keys
alpha_vantage_key = os.environ.get("ALPHA_VANTAGE_API_KEY")
guardian_api_key = os.environ.get("GUARDIAN_API_KEY")

if alpha_vantage_key:
    print("Alpha Vantage API key loaded successfully")
    print(f"API key starts with: {alpha_vantage_key[:4]}...")
else:
    print("Warning: ALPHA_VANTAGE_API_KEY not found in environment variables")

if guardian_api_key:
    print("Guardian API key loaded successfully")
    print(f"API key starts with: {guardian_api_key[:4]}...")
else:
    print("Warning: GUARDIAN_API_KEY not found in environment variables")

# Standard library imports
import requests
from datetime import datetime
import re
import json
from functools import lru_cache

# Database related imports
import psycopg2
from psycopg2.extras import RealDictCursor

# Logging functionality
import logging

# Flask and related extension imports
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from flask_caching import Cache
from src.user_manager_blueprint import user_manager_bp
from src.alpha_vantage_blueprint import alpha_vantage_bp
from flask_wtf.csrf import CSRFProtect
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# Initialize Flask application
app = Flask(__name__)

# Security Configuration
# ---------------------
# Configure application secret key from environment variable
# This key is used for session management and CSRF protection
app.secret_key = os.environ.get("SECRET_KEY")
if not app.secret_key:
    raise ValueError("No SECRET_KEY set for Flask application. Please set the SECRET_KEY environment variable.")

# Session Cookie Configuration
# --------------------------
# Production settings for secure cookie handling
app.config.update({
    'SESSION_COOKIE_DOMAIN': '.drewwilliams.biz',  # Domain for which the cookie is valid
    'SESSION_COOKIE_SECURE': True,     # Ensures cookies are only sent over HTTPS
    'SESSION_COOKIE_HTTPONLY': True,   # Prevents JavaScript access to session cookies
    'SESSION_COOKIE_SAMESITE': 'Lax'  # Provides CSRF protection while maintaining usability
})

# Development Environment Configuration
# ----------------------------------
# Override cookie settings for local development environment
if os.environ.get("FLASK_ENV") == "development":
    app.config.update({
        'SESSION_COOKIE_DOMAIN': None,    # Use default domain for local development
        'SESSION_COOKIE_SECURE': False    # Allow HTTP in development environment
    })

# Rate Limiting Configuration
# -------------------------
app.config["RATELIMIT_ENABLED"] = False  # Rate limiting currently disabled

# Security Middleware Setup
# -----------------------
# Initialize CSRF Protection to prevent cross-site request forgery attacks
csrf = CSRFProtect(app)

# Rate Limiting Setup
# -----------------
# Configure request rate limiting based on client IP address
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]  # Default rate limits
)
limiter.init_app(app)

# Caching Configuration
# -------------------
# Setup in-memory cache for improved performance
cache = Cache(app, config={
    'CACHE_TYPE': 'SimpleCache',
    'CACHE_DEFAULT_TIMEOUT': 300,  # Cache entries expire after 5 minutes
})

# Application Base Directory
# ------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# API Configuration
# ---------------
# OpenWeatherMap API key configuration
OWM_API_KEY = os.environ.get("OWM_API_KEY", "default_api_key_for_dev")
# Alpha Vantage API key configuration
ALPHA_VANTAGE_API_KEY = os.environ.get("ALPHA_VANTAGE_API_KEY")

if not ALPHA_VANTAGE_API_KEY:
    print("Warning: ALPHA_VANTAGE_API_KEY not set. Stock ticker will display error message.")
else:
    print("Alpha Vantage API key is configured")

# Database Configuration
# --------------------
# Select appropriate database URL based on environment
if os.environ.get("FLASK_ENV") == "development":
    # When running in Docker, we should use the "db" service name, not localhost
    # The docker-compose.dev.yml already sets DATABASE_URL correctly
    DATABASE_URL = os.environ.get("DATABASE_URL", os.environ.get("DEV_DATABASE_URL"))
else:
    DATABASE_URL = os.environ.get("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("No DATABASE_URL set for the Flask application.")

# Database Connection Functions
# ------------------------
def get_db_connection():
    """
    Establishes and returns a connection to the PostgreSQL database.
    
    Returns:
        psycopg2.extensions.connection: A connection object with RealDictCursor factory set
        for returning results as dictionaries instead of tuples.
    """
    conn = psycopg2.connect(DATABASE_URL)
    conn.cursor_factory = psycopg2.extras.RealDictCursor
    return conn

def init_db():
    """
    Initializes the database schema by creating the required tables if they don't exist.
    
    Creates tables:
    - users: User account information and preferences
    - api_usage: Tracks API calls and limits for various services
    """
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            # Create users table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    city_name TEXT,
                    button_width INTEGER DEFAULT 200,
                    button_height INTEGER DEFAULT 200,
                    news_categories TEXT DEFAULT 'general'
                );
            """)
            
            # Create api_usage table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS api_usage (
                    id SERIAL PRIMARY KEY,
                    api_name TEXT NOT NULL,
                    endpoint TEXT NOT NULL,
                    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                    request_params JSONB,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
                
                -- Index for efficient querying of recent usage
                CREATE INDEX IF NOT EXISTS idx_api_usage_timestamp 
                ON api_usage (api_name, timestamp);
                
                -- Index for JSON querying if needed
                CREATE INDEX IF NOT EXISTS idx_api_usage_request_params 
                ON api_usage USING GIN (request_params);
            """)
            
            conn.commit()
        conn.close()
        print("Database initialized or updated.")
    except Exception as e:
        print("Error initializing the database:", e)

# -----------------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------------

def get_user_settings(username):
    """
    Retrieves user-specific settings from the database.
    
    Args:
        username (str): The username whose settings should be retrieved
        
    Returns:
        tuple: A tuple containing (city_name, button_width, button_height)
               If no settings are found, returns default values:
               - Default city: "New York"
               - Default button dimensions: 200x200
    """
    default_city = "New York"
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT city_name, button_width, button_height, news_categories FROM users WHERE username=%s;", (username,))
            row = cur.fetchone()
            if row and row['city_name']:
                return row['city_name'], row['button_width'], row['button_height'], row['news_categories'].split(',') if row['news_categories'] else ['general']
            return default_city, 200, 200, ['general']
    except Exception as e:
        print("Error retrieving user settings:", e)
        return default_city, 200, 200, ['general']
    finally:
        conn.close()


@cache.memoize(timeout=300)  # Cache for 5 minutes
def fetch_random_dog():
    """
    Fetches a random dog image from the Dog API (dog.ceo).
    
    Uses caching to prevent excessive API calls and improve performance.
    Results are cached for 5 minutes before a new API call is made.
    
    Returns:
        str: URL of a random dog image
             If the API call fails, returns a fallback placeholder image URL
    
    Note:
        - Uses dog.ceo/api/breeds/image/random endpoint
        - Implements a 5-second timeout for API calls
        - Includes error handling for failed API requests
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
    Fetches a random cat image from The Cat API.
    
    Uses caching to prevent excessive API calls and improve performance.
    Results are cached for 5 minutes before a new API call is made.
    
    Returns:
        str: URL of a random cat image
             If the API call fails, returns a fallback placeholder image URL
    
    Note:
        - Uses api.thecatapi.com/v1/images/search endpoint
        - Implements a 5-second timeout for API calls
        - Includes error handling for failed API requests
        - Response format is a list containing image objects
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
    """Retrieves a 5-day weather forecast using OpenWeatherMap's One Call API."""
    url = (
        f"http://api.openweathermap.org/data/3.0/onecall"
        f"?lat={lat}&lon={lon}&exclude=current,minutely,hourly,alerts"
        f"&units=imperial"
        f"&appid={OWM_API_KEY}"
    )
    forecast_list = []
    try:
        # Track API call in database
        track_api_call('openweathermap', 'onecall', {
            'lat': lat,
            'lon': lon,
            'endpoint': 'forecast'
        })
        
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        
        if "daily" in data:
            daily_data = data["daily"][:5]  # Take the first 5 days
            for day in daily_data:
                dt = day["dt"]  # Unix timestamp
                date_str = datetime.fromtimestamp(dt).strftime("%b %d")
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
                    "temp_max": temp_max,
                    "lat": lat,
                    "lon": lon
                })
                
    except Exception as e:
        print(f"[ERROR] Failed to fetch daily forecast: {e}")
        print(f"URL attempted: {url}")
    return forecast_list

@cache.memoize(timeout=300)  # Cache for 5 minutes
def get_current_weather(lat, lon):
    """Retrieves current weather conditions using OpenWeatherMap's Current Weather API."""
    url = (
        f"https://api.openweathermap.org/data/2.5/weather"
        f"?lat={lat}&lon={lon}"
        f"&units=imperial"
        f"&appid={OWM_API_KEY}"
    )
    
    try:
        # Track API call in database
        track_api_call('openweathermap', 'current', {
            'lat': lat,
            'lon': lon,
            'endpoint': 'weather'
        })
        
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        
        if "main" in data and "weather" in data and len(data["weather"]) > 0:
            icon_code = data["weather"][0]["icon"]
            return {
                "temp": round(data["main"]["temp"], 1),
                "description": data["weather"][0]["description"],
                "icon_url": f"https://openweathermap.org/img/wn/{icon_code}@2x.png",
                "feels_like": round(data["main"]["feels_like"], 1)
            }
            
    except Exception as e:
        print(f"[ERROR] Failed to fetch current weather: {e}")
        print(f"URL attempted: {url}")
    
    return None

def sanitize_city_name(city):
    """
    Validates and sanitizes user-provided city names for safe usage in API calls.
    
    Args:
        city (str): The city name to sanitize
        
    Returns:
        str: Sanitized city name containing only allowed characters:
             - Letters (A-Z, a-z)
             - Numbers (0-9)
             - Spaces
             - Commas
             - Periods
             - Hyphens
             - Apostrophes
             
    Note:
        - Strips leading/trailing whitespace
        - Removes any disallowed characters
        - Falls back to "New York" if sanitization results in empty string
        - Preserves common city name formats (e.g., "St. Louis", "Winston-Salem")
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

def create_service_dict(service, is_default=False):
    """
    Creates a standardized dictionary representation of a service entry.
    
    This helper function ensures consistent structure for both default and 
    user-defined services throughout the application.
    
    Args:
        service (dict): Raw service data containing at minimum:
                       - name: Service name
                       - url: Service URL
                       - icon: Icon identifier/URL
                       Optional for user services:
                       - id: Database ID
                       - description: Service description
                       - section: Service category/section
        is_default (bool): Whether this is a default service (True) or 
                          user-defined service (False)
    
    Returns:
        dict: Standardized service dictionary containing:
              - id: Database ID (None for default services)
              - name: Service name
              - url: Service URL
              - icon: Icon identifier/URL
              - description: Service description (empty for default services)
              - is_default: Boolean flag indicating if it's a default service
              - section: Service category/section (empty string if not specified)
    
    Note:
        Default services are built-in services that cannot be modified by users.
        User services are stored in the database and can be customized.
    """
    if is_default:
        return {
            'id': None,
            'name': service['name'],
            'url': service['url'],
            'icon': service['icon'],
            'description': '',
            'is_default': True,
            'section': service.get('section', '')
        }
    return {
        'id': service['id'],
        'name': service['name'],
        'url': service['url'],
        'icon': service['icon'],
        'description': service.get('description', ''),
        'is_default': False,
        'section': service.get('section', '')
    }

@lru_cache(maxsize=100)
def get_cached_stock_data(symbol):
    """
    Fetch and cache stock data from Alpha Vantage API.
    Data is cached for 60 seconds to stay within rate limits.
    Tracks API usage for both regular stocks and market indices.
    """
    print(f"Fetching data for symbol: {symbol}")  # Debug log

    # Return error state if no API key is available
    if not ALPHA_VANTAGE_API_KEY:
        print("No API key available")  # Debug log
        return {
            'price': '0.00',
            'change': '0.00',
            'error': 'API key not configured'
        }

    try:
        url = f"https://www.alphavantage.co/query"
        params = {
            "function": "GLOBAL_QUOTE",
            "symbol": symbol,
            "apikey": ALPHA_VANTAGE_API_KEY
        }

        # Track API call before making request
        now = datetime.now()
        
        # Track by hour
        current_hour = now.strftime('%Y-%m-%d %H')
        hour_calls = cache.get('api_calls_hour_' + current_hour) or []
        hour_calls.append({
            'timestamp': now.strftime('%Y-%m-%d %H:%M:%S'),
            'symbol': symbol
        })
        cache.set('api_calls_hour_' + current_hour, hour_calls, timeout=3600)
        
        # Track by day
        current_day = now.strftime('%Y-%m-%d')
        day_calls = cache.get('api_calls_day_' + current_day) or []
        day_calls.append({
            'timestamp': now.strftime('%Y-%m-%d %H:%M:%S'),
            'symbol': symbol
        })
        cache.set('api_calls_day_' + current_day, day_calls, timeout=86400)

        print(f"[API Usage] {now.strftime('%Y-%m-%d %H:%M:%S')} - Requesting data for {symbol}")

        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        print(f"Raw API response for {symbol}:", data)  # Debug log

        # Check for rate limit messages
        if "Note" in data or "Information" in data:
            error_msg = data.get("Note", data.get("Information", ""))
            print(f"[API Usage] {now.strftime('%Y-%m-%d %H:%M:%S')} - Rate limit hit for {symbol}: {error_msg}")
            return {
                'price': '0.00',
                'change': '0.00',
                'error': 'API rate limit reached - Please try again later'
            }

        if "Global Quote" in data and data["Global Quote"]:
            quote = data["Global Quote"]
            print(f"[API Usage] {now.strftime('%Y-%m-%d %H:%M:%S')} - Successfully retrieved data for {symbol}")
            return {
                'price': quote.get('05. price', '0.00'),
                'change': quote.get('10. change percent', '0.00%').rstrip('%'),
                'error': None
            }
        
        print(f"[API Usage] {now.strftime('%Y-%m-%d %H:%M:%S')} - No data available for {symbol}")
        return {
            'price': '0.00',
            'change': '0.00',
            'error': 'No data available'
        }

    except Exception as e:
        print(f"[API Usage] {now.strftime('%Y-%m-%d %H:%M:%S')} - Error fetching data for {symbol}: {str(e)}")
        return {
            'price': '0.00',
            'change': '0.00',
            'error': 'Error fetching data'
        }

def track_api_call(api_name, endpoint, details=None):
    """
    Track an API call in the database.
    
    Args:
        api_name (str): Name of the API service (e.g., 'alpha_vantage', 'gnews', 'openweathermap')
        endpoint (str): The specific endpoint or operation called
        details (dict, optional): Additional details about the API call (e.g., parameters, response status)
    """
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO api_usage (api_name, endpoint, timestamp, request_params)
                VALUES (%s, %s, CURRENT_TIMESTAMP, %s)
            """, (api_name, endpoint, json.dumps(details) if details else None))
            conn.commit()
    except Exception as e:
        print(f"Error tracking API call: {e}")
    finally:
        conn.close()

def get_api_usage(api_name, period='day'):
    """
    Get API usage statistics for a specific period.
    
    Args:
        api_name (str): Name of the API service
        period (str): Time period to check ('hour' or 'day')
    
    Returns:
        dict: Contains 'used' and 'remaining' counts based on API limits
    """
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            if period == 'hour':
                cur.execute("""
                    SELECT COUNT(*) 
                    FROM api_usage 
                    WHERE api_name = %s 
                    AND timestamp >= DATE_TRUNC('hour', CURRENT_TIMESTAMP)
                """, (api_name,))
            else:  # day
                cur.execute("""
                    SELECT COUNT(*) 
                    FROM api_usage 
                    WHERE api_name = %s 
                    AND timestamp >= DATE_TRUNC('day', CURRENT_TIMESTAMP)
                """, (api_name,))
            
            count = cur.fetchone()['count']
            
            # Define limits for each API
            limits = {
                'alpha_vantage': {'hour': 4500, 'day': 108000},
                'guardian': {'hour': 100, 'day': 5000},  # Guardian API has more generous limits
                'openweathermap': {'hour': 42, 'day': 1000}
            }
            
            limit = limits.get(api_name, {}).get(period, 0)
            
            return {
                'used': count,
                'remaining': max(0, limit - count),
                'period': period
            }
    except Exception as e:
        print(f"Error getting API usage: {e}")
        return {'used': 0, 'remaining': 0, 'period': period}
    finally:
        conn.close()

def cleanup_api_usage(days_to_keep=30):
    """
    Clean up old API usage records.
    
    Args:
        days_to_keep (int): Number of days of history to retain
    """
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute("""
                DELETE FROM api_usage 
                WHERE timestamp < CURRENT_TIMESTAMP - INTERVAL '%s days'
            """, (days_to_keep,))
            conn.commit()
    except Exception as e:
        print(f"Error cleaning up API usage: {e}")
    finally:
        conn.close()

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
    if 'user' not in session:
        return redirect(url_for('login'))
    
    # Get user settings
    city_name, button_width, button_height, news_categories = get_user_settings(session['user'])
    
    # Get coordinates for the city
    lat, lon = get_coordinates_for_city(city_name)
    
    # Get forecast data and current weather if coordinates are available
    forecast_data = None
    current_weather = None
    if lat and lon:
        forecast_data = get_weekly_forecast(lat, lon)
        current_weather = get_current_weather(lat, lon)
    
    # Get services from database
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # Get user's services
            cur.execute("""
                SELECT * FROM custom_services 
                WHERE user_id = (
                    SELECT id FROM users WHERE username = %s
                )
                ORDER BY section, COALESCE(display_order, EXTRACT(EPOCH FROM created_at)::bigint), created_at""",
                (session['user'],)
            )
            user_services = cur.fetchall()
    finally:
        conn.close()
    
    # Organize services by section
    media_services = []
    system_services = []
    
    # Add default services first
    default_media_services = [
        {'name': 'Sonarr', 'url': 'https://drewwilliams.biz/sonarr', 'icon': 'fa-tv'},
        {'name': 'Radarr', 'url': 'https://drewwilliams.biz/radarr', 'icon': 'fa-film'},
        {'name': 'NZBGet', 'url': 'https://drewwilliams.biz/nzbget', 'icon': 'fa-download'}
    ]
    
    default_system_services = [
        {'name': 'Portainer', 'url': 'https://portainer.drewwilliams.biz', 'icon': 'fa-server'},
        {'name': 'Glances', 'url': 'https://glances.drewwilliams.biz', 'icon': 'fa-tachometer-alt'},
        {'name': 'Unifi', 'url': 'https://unifi.ui.com', 'icon': 'fa-wifi'}
    ]
    
    # Convert default services to consistent format
    media_services = [create_service_dict(s, True) for s in default_media_services]
    system_services = [create_service_dict(s, True) for s in default_system_services]
    
    # Add custom services
    media_services.extend([create_service_dict(s) for s in user_services if s['section'] == 'media'])
    system_services.extend([create_service_dict(s) for s in user_services if s['section'] == 'system'])
    
    return render_template('index.html',
                         media_services=media_services,
                         system_services=system_services,
                         forecast_data=forecast_data,
                         current_weather=current_weather,
                         city_name=city_name,
                         button_width=button_width,
                         button_height=button_height,
                         news_categories=news_categories)

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
    """User can set their city_name and news preferences."""
    if 'user' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        city_name = request.form.get('city_name', '')
        button_width = request.form.get('button_width', '200')
        button_height = request.form.get('button_height', '200')
        news_categories = request.form.getlist('news_categories')  # Get multiple selected values
        
        # Convert list to comma-separated string for storage
        categories_str = ','.join(news_categories) if news_categories else 'general'
        
        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """UPDATE users 
                       SET city_name = %s, 
                           button_width = %s,
                           button_height = %s,
                           news_categories = %s
                       WHERE username = %s""",
                    (city_name, button_width, button_height, categories_str, session['user'])
                )
                conn.commit()
            flash('Settings updated successfully!', 'success')
        except Exception as e:
            conn.rollback()
            flash(f'Error updating settings: {str(e)}', 'danger')
        finally:
            conn.close()
        return redirect(url_for('settings'))
        
    # Get current settings
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT city_name, button_width, button_height, news_categories FROM users WHERE username = %s",
                (session['user'],)
            )
            user_settings = cur.fetchone()
            city_name = user_settings['city_name'] if user_settings else ''
            button_width = user_settings.get('button_width', 200)
            button_height = user_settings.get('button_height', 200)
            news_categories = user_settings.get('news_categories', 'general').split(',')
    finally:
        conn.close()
    
    return render_template('settings.html', 
                         city_name=city_name,
                         button_width=button_width,
                         button_height=button_height,
                         news_categories=news_categories)

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

# Register the blueprints with URL prefixes
app.register_blueprint(user_manager_bp)
app.register_blueprint(alpha_vantage_bp, url_prefix="/alpha_vantage")

# Updated route: Redirect to National Weather Service detailed forecast page
@app.route('/weather/details/<int:dt>')
def weather_details(dt):
    """Redirect to detailed weather information on the National Weather Service for the selected forecast day."""
    lat = request.args.get('lat')
    lon = request.args.get('lon')
    try:
        lat = float(lat)
        lon = float(lon)
    except (TypeError, ValueError):
        flash("Invalid coordinates for weather details.", "warning")
        return redirect(url_for('home'))
    nws_url = f"https://forecast.weather.gov/MapClick.php?lat={lat}&lon={lon}"
    return redirect(nws_url)

@app.route('/services/add', methods=['GET', 'POST'])
def add_service():
    if 'user' not in session:
        return redirect(url_for('login'))
        
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        url = request.form.get('url', '').strip()
        icon = request.form.get('icon', '').strip()
        description = request.form.get('description', '').strip()
        section = request.form.get('section', '').strip()
        
        # Debug logging
        print(f"Received form data:")
        print(f"Name: {name}")
        print(f"URL: {url}")
        print(f"Icon: {icon}")
        print(f"Description: {description}")
        print(f"Section: {section}")
        print(f"Current user in session: {session['user']}")
        
        if not all([name, url, icon, section]):
            flash('Name, URL, icon, and section are required.', 'warning')
            return redirect(url_for('add_service'))
        
        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id FROM users WHERE username = %s",
                    (session['user'],)
                )
                user_result = cur.fetchone()
                print(f"Query result: {user_result}")
                
                if user_result is None:
                    flash(f'Error: Could not find user {session["user"]} in database', 'danger')
                    return redirect(url_for('add_service'))
                
                user_id = user_result['id']  # Changed from user_result[0] to user_result['id']
                print(f"Found user_id: {user_id}")
                
                cur.execute(
                    """INSERT INTO custom_services 
                       (user_id, name, url, icon, description, section)
                       VALUES (%s, %s, %s, %s, %s, %s)""",
                    (user_id, name, url, icon, description, section)
                )
                conn.commit()
                print("Insert successful!")
            flash('Service added successfully!', 'success')
            return redirect(url_for('home'))
        except Exception as e:
            conn.rollback()
            print(f"Full error details: {str(e)}")
            import traceback
            print(traceback.format_exc())
            flash(f'Error adding service: {str(e)}', 'danger')
        finally:
            conn.close()
            
    return render_template('add_service.html')

@app.route('/services/delete/<int:service_id>', methods=['POST'])
def delete_service(service_id):
    if 'user' not in session:
        return redirect(url_for('login'))
        
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # First verify the service belongs to the user
            cur.execute(
                """DELETE FROM custom_services 
                   WHERE id = %s AND user_id = (
                       SELECT id FROM users WHERE username = %s
                   )""",
                (service_id, session['user'])
            )
            conn.commit()
            if cur.rowcount > 0:
                flash('Service deleted successfully!', 'success')
            else:
                flash('Service not found or not authorized.', 'warning')
    except Exception as e:
        conn.rollback()
        flash(f'Error deleting service: {str(e)}', 'danger')
    finally:
        conn.close()
    return redirect(url_for('home'))

@app.route('/services/edit/<int:service_id>', methods=['GET', 'POST'])
def edit_service(service_id):
    if 'user' not in session:
        return redirect(url_for('login'))
        
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            # First verify the service belongs to the user
            cur.execute(
                """SELECT s.* FROM custom_services s
                   JOIN users u ON s.user_id = u.id
                   WHERE s.id = %s AND u.username = %s""",
                (service_id, session['user'])
            )
            service = cur.fetchone()
            
            if not service:
                flash('Service not found or not authorized.', 'warning')
                return redirect(url_for('home'))
            
            if request.method == 'POST':
                name = request.form.get('name', '').strip()
                url = request.form.get('url', '').strip()
                icon = request.form.get('icon', '').strip()
                description = request.form.get('description', '').strip()
                section = request.form.get('section', '').strip()
                
                if not all([name, url, icon, section]):
                    flash('Name, URL, icon, and section are required.', 'warning')
                    return redirect(url_for('edit_service', service_id=service_id))
                
                cur.execute(
                    """UPDATE custom_services 
                       SET name = %s, url = %s, icon = %s, 
                           description = %s, section = %s
                       WHERE id = %s""",
                    (name, url, icon, description, section, service_id)
                )
                conn.commit()
                flash('Service updated successfully!', 'success')
                return redirect(url_for('home'))
                
            return render_template('edit_service.html', service=service)
    except Exception as e:
        conn.rollback()
        flash(f'Error updating service: {str(e)}', 'danger')
        return redirect(url_for('home'))
    finally:
        conn.close()

@app.route('/services/reorder', methods=['POST'])
def reorder_services():
    if 'user' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
        
    data = request.get_json()
    section = data.get('section')
    service_ids = data.get('serviceIds')
    
    if not section or not service_ids:
        return jsonify({'error': 'Missing required data'}), 400
        
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            for index, service_id in enumerate(service_ids):
                cur.execute(
                    """UPDATE custom_services 
                       SET display_order = %s 
                       WHERE id = %s AND user_id = (
                           SELECT id FROM users WHERE username = %s
                       )""",
                    (index, service_id, session['user'])
                )
            conn.commit()
        return jsonify({'success': True})
    except Exception as e:
        conn.rollback()
        logging.error("Error in reorder_services: %s", str(e))
        return jsonify({'error': 'An internal error has occurred!'}), 500
    finally:
        conn.close()

@app.route('/services/reorder-default', methods=['POST'])
def reorder_default_services():
    if 'user' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
        
    data = request.get_json()
    section = data.get('section')
    service_names = data.get('serviceNames')
    
    if not section or not service_names:
        return jsonify({'error': 'Missing required data'}), 400
        
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # First, delete existing orders for this user and section
            cur.execute(
                """DELETE FROM default_service_order 
                   WHERE user_id = (SELECT id FROM users WHERE username = %s)
                   AND section = %s""",
                (session['user'], section)
            )
            
            # Insert new orders
            for index, name in enumerate(service_names):
                cur.execute(
                    """INSERT INTO default_service_order 
                       (user_id, service_name, display_order, section)
                       VALUES (
                           (SELECT id FROM users WHERE username = %s),
                           %s, %s, %s
                       )""",
                    (session['user'], name, index, section)
                )
            conn.commit()
        return jsonify({'success': True})
    except Exception as e:
        conn.rollback()
        logging.error("Error in reorder_default_services: %s", str(e))
        return jsonify({'error': 'An internal error has occurred!'}), 500
    finally:
        conn.close()

@app.context_processor
def inject_is_dev_mode():
    # This will be True when FLASK_ENV is set to "development"
    return dict(is_dev_mode=(os.environ.get("FLASK_ENV") == "development"))

@app.route('/api/stock/<symbol>')
def get_stock_data(symbol):
    """API endpoint to get stock data for a given symbol."""
    if 'user' not in session:
        return jsonify({'error': 'Unauthorized'}), 401

    # Handle Treasury yield data
    if symbol == 'UST10Y':
        try:
            url = f"https://www.alphavantage.co/query"
            params = {
                "function": "TREASURY_YIELD",
                "interval": "daily",
                "maturity": "10year",
                "apikey": ALPHA_VANTAGE_API_KEY
            }

            # Track API call
            track_api_call('alpha_vantage', 'treasury_yield', {
                'maturity': '10year'
            })

            response = requests.get(url, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()

            if "data" in data:
                # Get the most recent yield value
                latest_data = data["data"][0]
                current_yield = float(latest_data["value"])
                
                # Get previous day's yield for change calculation
                prev_yield = float(data["data"][1]["value"]) if len(data["data"]) > 1 else current_yield
                change = current_yield - prev_yield

                return jsonify({
                    'symbol': symbol,
                    'price': str(current_yield),
                    'change': str(change),
                    'error': None
                })

            return jsonify({
                'symbol': symbol,
                'price': '0.00',
                'change': '0.00',
                'error': 'No data available'
            })

        except Exception as e:
            print(f"Error fetching Treasury data: {str(e)}")
            return jsonify({
                'symbol': symbol,
                'price': '0.00',
                'change': '0.00',
                'error': 'Error fetching data'
            })

    # Map index symbols to their Alpha Vantage symbols and conversion factors
    index_map = {
        'DJI': {'symbol': 'DIA', 'factor': 100.0},  # DIA ETF tracks Dow/100
        'SPX': {'symbol': 'SPY', 'factor': 10.0},   # SPY ETF tracks S&P 500/10
        'IXIC': {'symbol': 'QQQ', 'factor': 100.0}, # QQQ ETF tracks NASDAQ-100/100
        'VIXY': {'symbol': 'VIXY', 'factor': 1.0}   # ProShares VIX Short-Term Futures ETF
    }

    # Use ETF symbols for indices
    index_info = index_map.get(symbol)
    av_symbol = index_info['symbol'] if index_info else symbol
    print(f"Using symbol: {av_symbol} for {symbol}")  # Debug log

    # Return error state if no API key is available
    if not ALPHA_VANTAGE_API_KEY:
        print("No API key available")  # Debug log
        return jsonify({
            'symbol': symbol,
            'price': '0.00',
            'change': '0.00',
            'error': 'API key not configured'
        })

    try:
        url = f"https://www.alphavantage.co/query"
        params = {
            "function": "GLOBAL_QUOTE",
            "symbol": av_symbol,
            "apikey": ALPHA_VANTAGE_API_KEY
        }

        # Track API call before making request
        track_api_call('alpha_vantage', 'global_quote', {
            'symbol': symbol,
            'mapped_symbol': av_symbol
        })

        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        print(f"Raw API response for {symbol}:", data)  # Debug log

        # Check for rate limit messages
        if "Note" in data or "Information" in data:
            error_msg = data.get("Note", data.get("Information", ""))
            print(f"API rate limit hit for {symbol}: {error_msg}")
            return jsonify({
                'symbol': symbol,
                'price': '0.00',
                'change': '0.00',
                'error': 'API rate limit reached - Please try again later'
            })

        if "Global Quote" in data and data["Global Quote"]:
            quote = data["Global Quote"]
            price = quote.get('05. price', '0.00')
            change = quote.get('10. change percent', '0.00%').rstrip('%')
            
            # Scale values for indices
            if index_info and price != '0.00':
                try:
                    price = float(price) * index_info['factor']
                    price = f"{price:,.2f}"
                except (ValueError, TypeError):
                    pass

            return jsonify({
                'symbol': symbol,
                'price': price,
                'change': change,
                'error': None
            })
        
        return jsonify({
            'symbol': symbol,
            'price': '0.00',
            'change': '0.00',
            'error': 'No data available'
        })

    except Exception as e:
        print(f"Error fetching data for {symbol}: {str(e)}")
        return jsonify({
            'symbol': symbol,
            'price': '0.00',
            'change': '0.00',
            'error': 'Error fetching data'
        })

@app.route('/api/usage')
def api_usage():
    """View API usage statistics for all APIs."""
    if 'user' not in session:
        return redirect(url_for('login'))

    # Get usage statistics for each API
    alpha_vantage_stats = {
        'hour': get_api_usage('alpha_vantage', 'hour'),
        'day': get_api_usage('alpha_vantage', 'day')
    }

    guardian_stats = {
        'hour': get_api_usage('guardian', 'hour'),
        'day': get_api_usage('guardian', 'day')
    }

    owm_stats = {
        'hour': get_api_usage('openweathermap', 'hour'),
        'day': get_api_usage('openweathermap', 'day')
    }

    # Clean up old records (keep 30 days of history)
    cleanup_api_usage(30)
    
    return render_template('api_usage.html', 
                         stats=alpha_vantage_stats,
                         guardian_stats=guardian_stats,
                         owm_stats=owm_stats)

@app.route('/api/news')
def get_news():
    print("\n=== Starting /api/news endpoint ===")
    print(f"Session data: {session}")
    
    if 'user' not in session:
        print("No user in session, returning unauthorized")
        return jsonify({'error': 'Unauthorized'}), 401

    print(f"User authenticated: {session['user']}")
    
    # Check if Guardian API key is configured
    guardian_api_key = os.environ.get('GUARDIAN_API_KEY')
    print(f"Guardian API key loaded: {'Yes' if guardian_api_key else 'No'}")
    
    if not guardian_api_key:
        print("Guardian API key not found in environment")
        return jsonify({'error': 'Guardian API not configured'}), 500

    try:
        # Get user's preferred news sections
        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT news_categories FROM users WHERE username = %s",
                    (session['user'],)
                )
                result = cur.fetchone()
                user_sections = result['news_categories'].split(',') if result and result['news_categories'] else ['news']
                print(f"User sections retrieved: {user_sections}")
        finally:
            conn.close()

        # Initialize articles list
        all_articles = []
        
        # Cache key for all sections
        cache_key = f"news_articles_{','.join(sorted(user_sections))}"
        print(f"Checking cache with key: {cache_key}")
        
        # Try to get cached articles
        cached_articles = cache.get(cache_key)
        if cached_articles and not app.debug:  # Only use cache in production
            print(f"Found {len(cached_articles)} cached articles")
            return jsonify({'articles': cached_articles})
        
        print("No cached articles found or in development mode, fetching from Guardian API")
        
        # Excluded sections and title patterns
        excluded_sections = ['corrections-and-clarifications', 'for-the-record']
        excluded_patterns = ['corrections and clarifications', 'for the record']
        
        # Set page size based on environment
        page_size = 50 if app.debug else 5  # Get more articles in development mode
        
        # Fetch articles for each section
        for section in user_sections:
            if section.lower() in excluded_sections:
                print(f"Skipping excluded section: {section}")
                continue
                
            print(f"\nFetching articles for section: {section}")
            
            params = {
                'api-key': guardian_api_key,
                'section': section,
                'show-fields': 'headline,shortUrl',
                'page-size': page_size,
                'order-by': 'newest'
            }
            
            url = 'https://content.guardianapis.com/search'
            print(f"Making API request to: {url}")
            print(f"With parameters: {params}")
            
            response = requests.get(url, params=params)
            print(f"API response status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"Response data: {data}")
                
                if 'response' in data and 'results' in data['response']:
                    section_articles = data['response']['results']
                    print(f"Found {len(section_articles)} articles in section {section}")
                    
                    for article in section_articles:
                        # Skip articles with excluded patterns in the title
                        title = article['fields']['headline'].lower()
                        if any(pattern in title for pattern in excluded_patterns):
                            print(f"Skipping article with excluded pattern: {title}")
                            continue
                            
                        all_articles.append({
                            'title': article['fields']['headline'],
                            'url': article['fields']['shortUrl']
                        })
                else:
                    print(f"Unexpected response format for section {section}")
            else:
                print(f"Error response for section {section}: {response.text}")
        
        print(f"\nTotal articles collected: {len(all_articles)}")
        
        # Only cache in production
        if not app.debug:
            cache_timeout = 300  # 5 minutes in prod
            cache.set(cache_key, all_articles, timeout=cache_timeout)
            print(f"Articles cached with timeout: {cache_timeout} seconds")
        else:
            print("Skipping cache in development mode")
        
        # Track API usage only in production
        if not app.debug:
            track_api_call('guardian', 'news')
        
        return jsonify({'articles': all_articles})
        
    except Exception as e:
        print(f"Error in get_news: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': 'Failed to fetch news'}), 500

@app.route('/api/news/usage')
def news_api_usage():
    """API usage statistics for news endpoints."""
    if 'user' not in session:
        return jsonify({"error": "Authentication required"}), 401

    try:
        period = request.args.get('period', 'day')
        usage_data = get_api_usage('guardian', period)
        return jsonify(usage_data)
    except Exception as e:
        app.logger.error(f"Error getting news API usage: {str(e)}")
        return jsonify({"error": "Failed to retrieve API usage data"}), 500

# TrendInsight API Routes

@app.route('/api/trend-insight/sentiment')
def get_market_sentiment():
    """Get market sentiment data for TrendInsight dashboard."""
    if 'username' not in session:
        return jsonify({"error": "Authentication required"}), 401

    try:
        # Track API call
        track_api_call('alpha_vantage', 'news_sentiment')
        
        # Get the time range parameter
        time_range = request.args.get('time_range', '1w')
        
        # Call Alpha Vantage API for news sentiment
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "NEWS_SENTIMENT",
            "sort": "RELEVANCE",
            "limit": "50", # Adjust as needed
            "apikey": alpha_vantage_key
        }
        
        if 'tickers' in request.args:
            params['tickers'] = request.args.get('tickers')
            
        if 'topics' in request.args:
            params['topics'] = request.args.get('topics')
            
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Process the sentiment data
        sentiment_data = {
            'overall_sentiment': {
                'bullish': 0,
                'neutral': 0,
                'bearish': 0
            },
            'sentiment_timeline': [],
            'news_items': []
        }
        
        if 'feed' in data:
            # Group by date for timeline
            date_sentiment = {}
            
            for article in data['feed']:
                # Extract sentiment
                sentiment_score = article.get('overall_sentiment_score', 0)
                
                # Categorize sentiment
                sentiment_label = 'neutral'
                if sentiment_score >= 0.25:
                    sentiment_label = 'bullish'
                    sentiment_data['overall_sentiment']['bullish'] += 1
                elif sentiment_score <= -0.25:
                    sentiment_label = 'bearish'
                    sentiment_data['overall_sentiment']['bearish'] += 1
                else:
                    sentiment_data['overall_sentiment']['neutral'] += 1
                
                # Add to news items
                time_published = article.get('time_published', '')
                date_published = time_published.split('T')[0] if 'T' in time_published else time_published[:8]
                
                # Format date from YYYYMMDD to YYYY-MM-DD if needed
                if len(date_published) == 8 and '-' not in date_published:
                    date_published = f"{date_published[:4]}-{date_published[4:6]}-{date_published[6:8]}"
                
                # Add to date sentiment for timeline
                if date_published not in date_sentiment:
                    date_sentiment[date_published] = {
                        'bullish': 0,
                        'neutral': 0,
                        'bearish': 0,
                        'count': 0
                    }
                
                date_sentiment[date_published][sentiment_label] += 1
                date_sentiment[date_published]['count'] += 1
                
                # Add to news items
                sentiment_data['news_items'].append({
                    'title': article.get('title', ''),
                    'summary': article.get('summary', ''),
                    'source': article.get('source', ''),
                    'url': article.get('url', ''),
                    'time_published': time_published,
                    'sentiment': sentiment_label,
                    'sentiment_score': sentiment_score,
                    'tickers': [ticker.get('ticker') for ticker in article.get('ticker_sentiment', [])]
                })
            
            # Create timeline data
            for date, values in date_sentiment.items():
                total = values['count'] if values['count'] > 0 else 1
                sentiment_data['sentiment_timeline'].append({
                    'date': date,
                    'bullish_percentage': (values['bullish'] / total) * 100,
                    'neutral_percentage': (values['neutral'] / total) * 100,
                    'bearish_percentage': (values['bearish'] / total) * 100,
                    'count': values['count']
                })
            
            # Sort timeline by date
            sentiment_data['sentiment_timeline'].sort(key=lambda x: x['date'])
            
            # Calculate overall percentages
            total_articles = sum(sentiment_data['overall_sentiment'].values())
            if total_articles > 0:
                for key in sentiment_data['overall_sentiment']:
                    sentiment_data['overall_sentiment'][key] = round((sentiment_data['overall_sentiment'][key] / total_articles) * 100)
        
        return jsonify(sentiment_data)
        
    except Exception as e:
        app.logger.error(f"Error fetching market sentiment data: {str(e)}")
        return jsonify({"error": "Failed to retrieve market sentiment data"}), 500

@app.route('/api/trend-insight/unusual-patterns')
def get_unusual_patterns():
    """Get unusual trading patterns for TrendInsight dashboard."""
    if 'username' not in session:
        return jsonify({"error": "Authentication required"}), 401

    try:
        # This would normally analyze multiple data points from Alpha Vantage
        # For this demo, we'll create simulated patterns based on volume and price
        
        # Track API call for any Alpha Vantage calls we make
        track_api_call('alpha_vantage', 'unusual_patterns')
        
        # Get top gainers/losers as a starting point
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "TOP_GAINERS_LOSERS",
            "apikey": alpha_vantage_key
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        unusual_patterns = []
        
        # Process potential unusual patterns
        if 'top_gainers' in data:
            for stock in data['top_gainers'][:5]:  # Limit to top 5
                # Check if volume is significant
                try:
                    volume = int(stock.get('volume', '0'))
                    change_percent = float(stock.get('change_percentage', '0').rstrip('%'))
                    
                    if volume > 1000000 and change_percent > 5:  # Significant volume and change
                        unusual_patterns.append({
                            'symbol': stock.get('ticker', ''),
                            'name': stock.get('name', ''),
                            'pattern_type': 'volume_price_surge',
                            'description': f"Volume spike {round(volume/1000000, 1)}M with strong price movement",
                            'change_percentage': change_percent,
                            'direction': 'up'
                        })
                except (ValueError, TypeError):
                    continue
        
        if 'top_losers' in data:
            for stock in data['top_losers'][:5]:  # Limit to top 5
                try:
                    volume = int(stock.get('volume', '0'))
                    change_percent = float(stock.get('change_percentage', '0').rstrip('%'))
                    
                    if volume > 1000000 and abs(change_percent) > 5:  # Significant volume and change
                        unusual_patterns.append({
                            'symbol': stock.get('ticker', ''),
                            'name': stock.get('name', ''),
                            'pattern_type': 'volume_price_drop',
                            'description': f"Unusual selling volume with sharp price decline",
                            'change_percentage': change_percent,
                            'direction': 'down'
                        })
                except (ValueError, TypeError):
                    continue
        
        # For the most active, look for potential breakouts or breakdowns
        if 'most_actively_traded' in data:
            for stock in data['most_actively_traded'][:5]:
                try:
                    volume = int(stock.get('volume', '0'))
                    change_percent = float(stock.get('change_percentage', '0').rstrip('%'))
                    
                    if volume > 5000000:  # Very high volume
                        if change_percent > 0:
                            pattern_type = 'potential_breakout'
                            description = "High volume indicating potential breakout"
                            direction = 'up'
                        else:
                            pattern_type = 'potential_breakdown'
                            description = "High volume indicating potential breakdown"
                            direction = 'down'
                            
                        unusual_patterns.append({
                            'symbol': stock.get('ticker', ''),
                            'name': stock.get('name', ''),
                            'pattern_type': pattern_type,
                            'description': description,
                            'change_percentage': change_percent,
                            'direction': direction
                        })
                except (ValueError, TypeError):
                    continue
        
        # Return the unusual patterns
        return jsonify(unusual_patterns)
        
    except Exception as e:
        app.logger.error(f"Error detecting unusual patterns: {str(e)}")
        return jsonify({"error": "Failed to detect unusual patterns"}), 500

@app.route('/api/trend-insight/recommendations')
def get_ai_recommendations():
    """Get AI-powered stock recommendations for TrendInsight dashboard."""
    if 'username' not in session:
        return jsonify({"error": "Authentication required"}), 401

    try:
        username = session['username']
        
        # In a real implementation, this would analyze the user's portfolio,
        # cross-reference with market sentiment, technicals, and fundamentals
        # For this demo, we'll create simulated recommendations
        
        # Track API calls
        track_api_call('alpha_vantage', 'recommendations')
        
        # Get user's tracked stocks
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute("""
            SELECT s.symbol, s.name
            FROM stocks s
            JOIN user_stocks us ON s.id = us.stock_id
            JOIN users u ON us.user_id = u.id
            WHERE u.username = %s
        """, (username,))
        
        user_stocks = cursor.fetchall()
        conn.close()
        
        # If user doesn't have stocks, use some defaults
        if not user_stocks:
            user_stocks = [
                {'symbol': 'AAPL', 'name': 'Apple Inc.'},
                {'symbol': 'MSFT', 'name': 'Microsoft Corporation'},
                {'symbol': 'GOOGL', 'name': 'Alphabet Inc.'},
                {'symbol': 'AMZN', 'name': 'Amazon.com Inc.'},
                {'symbol': 'NVDA', 'name': 'NVIDIA Corporation'}
            ]
        
        # Generate recommendations based on user stocks and news sentiment
        recommendations = []
        
        # For each user stock, generate a recommendation based on available data
        for stock in user_stocks:
            symbol = stock['symbol']
            
            # Call Alpha Vantage API for News Sentiment specific to this stock
            url = "https://www.alphavantage.co/query"
            params = {
                "function": "NEWS_SENTIMENT",
                "tickers": symbol,
                "sort": "RELEVANCE",
                "limit": "10",
                "apikey": alpha_vantage_key
            }
            
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            # Analyze sentiment for recommendation
            sentiment_scores = []
            key_points = []
            
            if 'feed' in data:
                for article in data['feed']:
                    # Extract ticker-specific sentiment
                    for ticker_sentiment in article.get('ticker_sentiment', []):
                        if ticker_sentiment.get('ticker') == symbol:
                            sentiment_scores.append(float(ticker_sentiment.get('ticker_sentiment_score', 0)))
                            relevance = float(ticker_sentiment.get('relevance_score', 0))
                            if relevance > 0.8:  # Only include highly relevant points
                                key_points.append(article.get('title', ''))
            
            # Determine action based on sentiment
            if sentiment_scores:
                avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
                
                if avg_sentiment > 0.3:
                    action = "BUY"
                    rationale = "Strong positive sentiment"
                elif avg_sentiment < -0.3:
                    action = "SELL"
                    rationale = "Significant negative sentiment"
                else:
                    action = "HOLD"
                    rationale = "Neutral market sentiment"
                
                # Get a key point if available
                if key_points:
                    rationale = key_points[0][:50] + "..." if len(key_points[0]) > 50 else key_points[0]
            else:
                # No sentiment data available
                action = "HOLD"
                rationale = "Insufficient data for analysis"
            
            recommendations.append({
                'symbol': symbol,
                'name': stock['name'],
                'action': action,
                'rationale': rationale
            })
        
        # Return the recommendations
        return jsonify(recommendations)
        
    except Exception as e:
        app.logger.error(f"Error generating AI recommendations: {str(e)}")
        return jsonify({"error": "Failed to generate AI recommendations"}), 500

@app.route('/api/trend-insight/correlations')
def get_asset_correlations():
    """Get cross-asset correlations for TrendInsight dashboard."""
    if 'username' not in session:
        return jsonify({"error": "Authentication required"}), 401

    try:
        # In a real implementation, this would calculate actual correlations
        # between different asset classes using historical price data.
        # For this demo, we'll return simulated correlation data
        
        # Track API call
        track_api_call('alpha_vantage', 'correlations')
        
        # Define assets for correlation matrix
        assets = [
            'S&P 500', 'NASDAQ', 'Russell 2000', 'Gold', 'Silver', 
            'Oil', '10Y Treasury', 'USD Index', 'Bitcoin', 'Ethereum'
        ]
        
        # Create correlation matrix (simulated data)
        correlations = {
            'assets': assets,
            'matrix': [
                [1.00, 0.92, 0.85, 0.21, 0.18, 0.45, -0.18, -0.25, 0.38, 0.35],  # S&P 500
                [0.92, 1.00, 0.80, 0.15, 0.12, 0.32, -0.22, -0.28, 0.47, 0.42],  # NASDAQ
                [0.85, 0.80, 1.00, 0.25, 0.22, 0.40, -0.15, -0.20, 0.30, 0.28],  # Russell 2000
                [0.21, 0.15, 0.25, 1.00, 0.85, 0.24, 0.42, -0.54, 0.25, 0.22],   # Gold
                [0.18, 0.12, 0.22, 0.85, 1.00, 0.28, 0.38, -0.48, 0.20, 0.18],   # Silver
                [0.45, 0.32, 0.40, 0.24, 0.28, 1.00, -0.12, -0.38, 0.19, 0.15],  # Oil
                [-0.18, -0.22, -0.15, 0.42, 0.38, -0.12, 1.00, -0.56, -0.11, -0.15], # 10Y Treasury
                [-0.25, -0.28, -0.20, -0.54, -0.48, -0.38, -0.56, 1.00, -0.29, -0.32], # USD Index
                [0.38, 0.47, 0.30, 0.25, 0.20, 0.19, -0.11, -0.29, 1.00, 0.82],  # Bitcoin
                [0.35, 0.42, 0.28, 0.22, 0.18, 0.15, -0.15, -0.32, 0.82, 1.00]   # Ethereum
            ],
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return jsonify(correlations)
        
    except Exception as e:
        app.logger.error(f"Error calculating asset correlations: {str(e)}")
        return jsonify({"error": "Failed to calculate asset correlations"}), 500

@app.route('/health')
def health_check():
    """Health check endpoint that verifies database connectivity."""
    try:
        # Test database connection
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute('SELECT 1')
            result = cur.fetchone()
        conn.close()
        
        # Log successful health check
        app.logger.info('Health check passed: database connection successful')
        
        return jsonify({
            'status': 'healthy',
            'database': 'connected',
            'timestamp': datetime.now().isoformat()
        }), 200
    except Exception as e:
        # Log the error
        app.logger.error(f'Health check failed: {str(e)}')
        
        return jsonify({
            'status': 'unhealthy',
            'error': 'An internal error has occurred. Please try again later.',
            'timestamp': datetime.now().isoformat()
        }), 500

if __name__ == '__main__':
    # Determine if debug mode should be on. By default, it's off.
    # Set FLASK_DEBUG=1 (or "true"/"on") in your development environment.
    debug_mode = os.environ.get("FLASK_DEBUG", "0").lower() in ("1", "true", "on")
    
    # Optionally, read the port from an environment variable.
    port = int(os.environ.get("PORT", "5001"))
    
    # Run the Flask app with the appropriate settings.
    app.run(host='0.0.0.0', port=port, debug=debug_mode)
