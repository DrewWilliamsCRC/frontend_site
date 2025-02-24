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
    DATABASE_URL = os.environ.get("DEV_DATABASE_URL", os.environ.get("DATABASE_URL"))
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
                    details JSONB,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
                
                -- Index for efficient querying of recent usage
                CREATE INDEX IF NOT EXISTS idx_api_usage_timestamp 
                ON api_usage (api_name, timestamp);
                
                -- Index for JSON querying if needed
                CREATE INDEX IF NOT EXISTS idx_api_usage_details 
                ON api_usage USING GIN (details);
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
                INSERT INTO api_usage (api_name, endpoint, timestamp, details)
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

# Register the blueprint with a URL prefix (e.g., /admin)
app.register_blueprint(user_manager_bp, url_prefix="/admin")

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
        if cached_articles:
            print(f"Found {len(cached_articles)} cached articles")
            return jsonify({'articles': cached_articles})
        
        print("No cached articles found, fetching from Guardian API")
        
        # Fetch articles for each section
        for section in user_sections:
            print(f"\nFetching articles for section: {section}")
            
            params = {
                'api-key': guardian_api_key,
                'section': section,
                'show-fields': 'headline,shortUrl',
                'page-size': 5,
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
                        all_articles.append({
                            'title': article['fields']['headline'],
                            'url': article['fields']['shortUrl']
                        })
                else:
                    print(f"Unexpected response format for section {section}")
            else:
                print(f"Error response for section {section}: {response.text}")
        
        print(f"\nTotal articles collected: {len(all_articles)}")
        
        # Cache the articles
        cache_timeout = 300 if app.debug else 600  # 5 minutes in dev, 10 in prod
        cache.set(cache_key, all_articles, timeout=cache_timeout)
        print(f"Articles cached with timeout: {cache_timeout} seconds")
        
        # Track API usage
        track_api_call('guardian', 'news')
        
        return jsonify({'articles': all_articles})
        
    except Exception as e:
        print(f"Error in get_news: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': 'Failed to fetch news'}), 500

@app.route('/api/news/usage')
def news_api_usage():
    """View Gnews API usage statistics."""
    if 'user' not in session:
        return redirect(url_for('login'))
        
    now = datetime.now()
    
    # Get current day's API calls
    current_day = now.strftime('%Y-%m-%d')
    day_calls = cache.get('gnews_api_calls_day_' + current_day) or []
    
    # Calculate statistics
    day_limit = 100  # Gnews free tier limit
    
    # Format timestamp
    day_time = datetime.strptime(current_day, '%Y-%m-%d')
    
    stats = {
        'day': {
            'used': len(day_calls),
            'limit': day_limit,
            'remaining': day_limit - len(day_calls),
            'period': day_time.strftime('%H:%M:%S %m/%d/%Y')
        }
    }
    
    return render_template('news_api_usage.html', stats=stats)

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

@app.route('/stock-tracker')
def stock_tracker():
    """Stock Tracker page showing various market lists and stock data."""
    if 'user' not in session:
        return redirect(url_for('login'))

    try:
        # Get top gainers/losers data
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "TOP_GAINERS_LOSERS",
            "apikey": ALPHA_VANTAGE_API_KEY
        }

        # Track API call
        track_api_call('alpha_vantage', 'top_gainers_losers')

        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()

        # Check for API limit messages
        if "Note" in data:
            flash("API rate limit reached. Please try again later.", "warning")
            return render_template('stock_tracker.html', lists={}, selected_list='top_gainers')

        # Process and format the data
        processed_data = {}
        for list_type in ['top_gainers', 'top_losers', 'most_actively_traded']:
            if list_type in data:
                processed_data[list_type] = []
                for stock in data[list_type]:
                    try:
                        processed_stock = {
                            'ticker': stock.get('ticker', ''),
                            'name': stock.get('name', ''),
                            'price': float(stock.get('price', '0.0')),
                            'change_amount': float(stock.get('change_amount', '0.0')),
                            'change_percentage': float(stock.get('change_percentage', '0.0').rstrip('%')),
                            'volume': int(stock.get('volume', '0'))
                        }
                        processed_data[list_type].append(processed_stock)
                    except (ValueError, TypeError) as e:
                        app.logger.error(f"Error processing stock data: {e}")
                        continue

        selected_list = request.args.get('list', 'top_gainers')
        
        return render_template('stock_tracker.html', 
                             lists=processed_data,
                             selected_list=selected_list)

    except Exception as e:
        app.logger.error(f"Error fetching stock data: {str(e)}")
        flash("Error fetching stock data. Please try again later.", "danger")
        return render_template('stock_tracker.html', lists={}, selected_list='top_gainers')

@app.route('/api/market-data/<endpoint>/<option>')
def get_market_data(endpoint, option):
    """API endpoint to get various market data based on the endpoint and option selected."""
    if 'user' not in session:
        return jsonify({'error': 'Unauthorized'}), 401

    if not ALPHA_VANTAGE_API_KEY:
        return jsonify({
            'error': 'API key not configured',
            'data': []
        })

    try:
        url = "https://www.alphavantage.co/query"
        app.logger.debug(f"Processing request for endpoint: {endpoint}, option: {option}")
        
        # Track API call with detailed information
        track_details = {
            'endpoint': endpoint,
            'option': option,
            'user': session['user']
        }

        # Configure request based on endpoint
        params = {
            "apikey": ALPHA_VANTAGE_API_KEY
        }

        if endpoint == 'TOP_GAINERS_LOSERS':
            params["function"] = "TOP_GAINERS_LOSERS"
            track_api_call('alpha_vantage', 'top_gainers_losers', track_details)
            
        elif endpoint == 'SECTOR':
            params["function"] = "SECTOR"
            app.logger.debug("Making SECTOR request with params: %s", params)
            track_api_call('alpha_vantage', 'sector_performance', track_details)
            
        elif endpoint == 'MARKET_STATUS':
            params["function"] = "MARKET_STATUS"
            track_api_call('alpha_vantage', 'market_status', track_details)
            
        elif endpoint == 'IPO_CALENDAR':
            params["function"] = "IPO_CALENDAR"
            track_api_call('alpha_vantage', 'ipo_calendar', track_details)
            
        elif endpoint == 'EARNINGS_CALENDAR':
            params["function"] = "EARNINGS_CALENDAR"
            track_api_call('alpha_vantage', 'earnings_calendar', track_details)
            
        elif endpoint == 'CRYPTO_INTRADAY':
            params.update({
                "function": "CRYPTO_INTRADAY",
                "symbol": option,
                "market": "USD",
                "interval": "5min"
            })
            track_details['crypto_symbol'] = option
            track_api_call('alpha_vantage', 'crypto_intraday', track_details)
            
        else:
            return jsonify({
                'error': 'Invalid endpoint specified',
                'data': []
            })

        app.logger.debug(f"Making API request to: {url}")
        app.logger.debug(f"With parameters: {params}")
        
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        
        app.logger.debug(f"Response status code: {response.status_code}")
        app.logger.debug(f"Response content type: {response.headers.get('content-type', 'unknown')}")
        
        # For SECTOR endpoint, log the raw response
        if endpoint == 'SECTOR':
            app.logger.debug(f"SECTOR raw response: {response.text[:1000]}...")  # First 1000 chars

        # Add the request option to the response object for use in process_response
        response.request_option = option
        
        # Process response based on endpoint type
        if endpoint in ['IPO_CALENDAR', 'EARNINGS_CALENDAR']:
            result = process_response(response, endpoint)
        else:
            # Check for rate limit messages in JSON response
            try:
                data = response.json()
                if "Note" in data:
                    error_msg = data.get("Note", "Rate limit reached")
                    app.logger.debug(f"Rate limit hit: {error_msg}")
                    track_details['error'] = error_msg
                    track_api_call('alpha_vantage', f'{endpoint.lower()}_rate_limit', track_details)
                    return jsonify({
                        'error': 'API rate limit reached - Please try again later',
                        'data': []
                    })
            except ValueError as e:
                app.logger.error(f"JSON parsing error: {str(e)}")
                app.logger.error(f"Response content: {response.text[:500]}...")
                return jsonify({
                    'error': 'Invalid response format from API',
                    'data': []
                })
                
            result = process_response(response, endpoint)
            
        return jsonify(result)

    except requests.exceptions.RequestException as e:
        error_msg = str(e)
        app.logger.error(f"Request error: {error_msg}")
        track_details['error'] = error_msg
        track_api_call('alpha_vantage', f'{endpoint.lower()}_request_error', track_details)
        return jsonify({
            'error': 'Failed to fetch market data',
            'data': []
        })
    except Exception as e:
        error_msg = str(e)
        app.logger.error(f"Unexpected error: {error_msg}")
        track_details['error'] = error_msg
        track_api_call('alpha_vantage', f'{endpoint.lower()}_error', track_details)
        return jsonify({
            'error': 'An unexpected error occurred',
            'data': []
        })

@cache.memoize(timeout=300)  # Cache for 5 minutes
def fetch_sector_data():
    """Fetch and cache sector performance data."""
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "SECTOR",
        "apikey": ALPHA_VANTAGE_API_KEY
    }
    
    try:
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        app.logger.error(f"Error fetching sector data: {str(e)}")
        return None

def process_response(response, endpoint):
    app.logger.debug(f"Processing {endpoint} response")
    app.logger.debug(f"Response content type: {response.headers.get('Content-Type', 'unknown')}")
    
    try:
        if endpoint in ['IPO_CALENDAR', 'EARNINGS_CALENDAR']:
            raw_text = response.text.strip()
            app.logger.debug(f"Raw response text (first 100 chars): {raw_text[:100]}...")
            
            if not raw_text:
                return {
                    'error': f'No {endpoint.lower().replace("_", " ")} data available',
                    'details': 'The API returned an empty response'
                }
            
            if raw_text.count('\n') <= 1:
                return {
                    'error': f'No {endpoint.lower().replace("_", " ")} data available at this time',
                    'details': 'No records found in the current time period'
                }
            
            lines = raw_text.split('\n')
            headers = lines[0].split(',')
            data = []
            
            for line in lines[1:]:
                if line.strip():  # Skip empty lines
                    values = line.split(',')
                    item = dict(zip(headers, values))
                    data.append(item)
            
            app.logger.debug(f"Processed {len(data)} records")
            return {'data': data}
            
        else:  # JSON response
            json_data = response.json()
            app.logger.debug(f"JSON response keys: {list(json_data.keys())}")
            
            if not json_data:
                return {
                    'error': f'No {endpoint.lower().replace("_", " ")} data available',
                    'details': 'The API returned an empty response'
                }
            
            if endpoint == 'TOP_GAINERS_LOSERS':
                if response.request_option in json_data:
                    return {'data': json_data[response.request_option]}
                return {
                    'error': 'No data available for the selected option',
                    'details': f'No data found for {response.request_option}'
                }
                
            elif endpoint == 'SECTOR':
                # Try to get cached data first
                cached_data = fetch_sector_data()
                if cached_data:
                    json_data = cached_data
                
                # Debug log the entire response for SECTOR endpoint
                app.logger.debug(f"Full SECTOR response: {json_data}")
                
                if not json_data:
                    return {
                        'error': 'No sector performance data available',
                        'details': 'The sector performance API is currently unavailable'
                    }
                
                time_period_map = {
                    'real_time': 'Rank A: Real-Time Performance',
                    '1day': 'Rank B: 1 Day Performance',
                    '5day': 'Rank C: 5 Day Performance',
                    '1month': 'Rank D: 1 Month Performance',
                    '3month': 'Rank E: 3 Month Performance',
                    'ytd': 'Rank F: Year-to-Date (YTD) Performance',
                    '1year': 'Rank G: 1 Year Performance',
                    '3year': 'Rank H: 3 Year Performance',
                    '5year': 'Rank I: 5 Year Performance',
                    '10year': 'Rank J: 10 Year Performance'
                }
                
                period_key = time_period_map.get(response.request_option)
                app.logger.debug(f"Looking for sector data with key: {period_key}")
                app.logger.debug(f"Available keys in response: {list(json_data.keys())}")
                
                if not period_key:
                    return {
                        'error': 'Invalid time period selected',
                        'details': f'The time period "{response.request_option}" is not supported'
                    }
                
                if period_key not in json_data:
                    return {
                        'error': 'No sector performance data available for the selected time period',
                        'details': f'No data found for period: {response.request_option}'
                    }
                
                sector_data = []
                for sector, performance in json_data[period_key].items():
                    try:
                        # Remove any '%' symbol and convert to float
                        perf_value = float(performance.rstrip('%') if isinstance(performance, str) else performance)
                        sector_data.append({
                            'sector': sector,
                            'performance': perf_value
                        })
                    except (ValueError, AttributeError) as e:
                        app.logger.error(f"Error processing sector {sector} data: {e}")
                        continue
                
                if sector_data:
                    app.logger.debug(f"Processed {len(sector_data)} sectors")
                    return {'data': sector_data}
                else:
                    return {
                        'error': 'No valid sector performance data available',
                        'details': 'Could not process any sector performance values'
                    }
                
            elif endpoint == 'MARKET_STATUS':
                if 'markets' in json_data:
                    return {'data': json_data['markets']}
                return {
                    'error': 'No market status data available',
                    'details': 'The market status data is not available in the API response'
                }
                
            elif endpoint == 'CRYPTO_INTRADAY':
                if 'Time Series Crypto (5min)' in json_data:
                    time_series = json_data['Time Series Crypto (5min)']
                    formatted_data = [
                        {
                            'timestamp': timestamp,
                            'price': float(values['1. open']),
                            'volume': float(values['5. volume'])
                        }
                        for timestamp, values in list(time_series.items())[:12]  # Last hour of data
                    ]
                    return {'data': formatted_data}
                return {
                    'error': 'No cryptocurrency data available',
                    'details': 'No recent trading data found for the selected cryptocurrency'
                }
            
            return {
                'error': 'Unsupported endpoint type',
                'details': f'The endpoint "{endpoint}" is not properly configured'
            }
            
    except Exception as e:
        app.logger.error(f"Error processing {endpoint} response: {str(e)}")
        app.logger.error(f"Response content: {response.text[:500]}...")  # Log first 500 chars of response
        return {
            'error': 'An error occurred while processing the data',
            'details': 'Please contact support if the issue persists'
        }

if __name__ == '__main__':
    # Determine if debug mode should be on. By default, it's off.
    # Set FLASK_DEBUG=1 (or "true"/"on") in your development environment.
    debug_mode = os.environ.get("FLASK_DEBUG", "0").lower() in ("1", "true", "on")
    
    # Optionally, read the port from an environment variable.
    port = int(os.environ.get("PORT", "5001"))
    
    # Run the Flask app with the appropriate settings.
    app.run(host='0.0.0.0', port=port, debug=debug_mode)
