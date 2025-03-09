# Load environment variables from .env file for configuration management
from dotenv import load_dotenv
import os
import random
import time
import logging

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
from datetime import datetime, timedelta
import re
import json
from functools import lru_cache
import sys
import pandas as pd # type: ignore
import numpy as np # type: ignore

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
        print("Guardian API key not found in environment, returning mock news data")
        # Return mock news data instead of failing with 500 error
        mock_articles = [
            {
                'title': 'S&P 500 Hits New Record as Tech Stocks Rally',
                'url': 'https://example.com/sp500-record'
            },
            {
                'title': 'Federal Reserve Signals Potential Rate Cuts',
                'url': 'https://example.com/fed-rate-cuts'
            },
            {
                'title': 'Global Markets React to Economic Data',
                'url': 'https://example.com/global-markets'
            },
            {
                'title': 'Tech Giants Announce New AI Initiatives',
                'url': 'https://example.com/tech-ai'
            },
            {
                'title': 'Retail Sales Exceed Expectations in Q1',
                'url': 'https://example.com/retail-sales'
            },
            {
                'title': 'Energy Sector Faces Challenges Amid Price Volatility',
                'url': 'https://example.com/energy-sector'
            },
            {
                'title': 'Housing Market Shows Signs of Cooling',
                'url': 'https://example.com/housing-market'
            },
            {
                'title': 'New Regulations Impact Financial Services',
                'url': 'https://example.com/financial-regulations'
            }
        ]
        return jsonify({
            'articles': mock_articles,
            'source': 'Mock Data (Guardian API key not configured)'
        })

    try:
        # The rest of the function remains unchanged
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
        
        # Try to get cached articles - also use cache in development mode to avoid rate limits
        cached_articles = cache.get(cache_key)
        if cached_articles:
            print(f"Found {len(cached_articles)} cached articles")
            return jsonify({'articles': cached_articles})
        
        print("No cached articles found, fetching from Guardian API")
        
        # Excluded sections and title patterns
        excluded_sections = ['corrections-and-clarifications', 'for-the-record']
        excluded_patterns = ['corrections and clarifications', 'for the record']
        
        # Set page size based on environment
        page_size = 15 if app.debug else 5  # Reduced from 50 to 15 in development mode to avoid rate limits
        
        # Fetch articles for each section
        for section in user_sections:
            if section.lower() in excluded_sections:
                print(f"Skipping excluded section: {section}")
                continue
                
            # Check if we have a section-specific cache to reduce API calls
            section_cache_key = f"news_section_{section}"
            section_articles = cache.get(section_cache_key)
            
            if section_articles:
                print(f"Using cached articles for section {section}")
                all_articles.extend(section_articles)
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
            
            # Implement retry logic with exponential backoff
            max_retries = 3
            retry_count = 0
            retry_delay = 1  # Start with 1 second delay
            
            section_articles_list = []
            while retry_count < max_retries:
                try:
                    response = requests.get(url, params=params, timeout=10)
                    print(f"API response status: {response.status_code}")
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        if 'response' in data and 'results' in data['response']:
                            section_articles = data['response']['results']
                            print(f"Found {len(section_articles)} articles in section {section}")
                            
                            for article in section_articles:
                                # Skip articles with excluded patterns in the title
                                title = article['fields']['headline'].lower()
                                if any(pattern in title for pattern in excluded_patterns):
                                    print(f"Skipping article with excluded pattern: {title}")
                                    continue
                                    
                                article_data = {
                                    'title': article['fields']['headline'],
                                    'url': article['fields']['shortUrl']
                                }
                                section_articles_list.append(article_data)
                                all_articles.append(article_data)
                            
                            # Cache the section-specific articles
                            section_cache_timeout = 1800  # 30 minutes
                            cache.set(section_cache_key, section_articles_list, timeout=section_cache_timeout)
                            print(f"Cached {len(section_articles_list)} articles for section {section} for {section_cache_timeout} seconds")
                            
                            # Success - break out of retry loop
                            break
                        else:
                            print(f"Unexpected response format for section {section}")
                            break  # No need to retry for format errors
                            
                    elif response.status_code == 429:  # Rate limit exceeded
                        print(f"Rate limit exceeded for section {section}. Using mock data.")
                        
                        # Use section-specific mock data
                        mock_section_articles = get_mock_articles_for_section(section)
                        all_articles.extend(mock_section_articles)
                        
                        # Cache the mock data for this section to avoid further calls
                        cache.set(section_cache_key, mock_section_articles, timeout=1800)  # 30 minutes
                        print(f"Cached mock articles for section {section}")
                        
                        # No need to retry when rate limited
                        break
                        
                    else:
                        print(f"Error response for section {section}: {response.text}")
                        
                        # Only retry for connection errors, not for client/server errors
                        if response.status_code >= 500:  # Server error, worth retrying
                            retry_count += 1
                            if retry_count < max_retries:
                                print(f"Retrying in {retry_delay} seconds...")
                                time.sleep(retry_delay)
                                retry_delay *= 2  # Exponential backoff
                            continue
                        else:
                            # Client error, no point retrying
                            break
                
                except requests.exceptions.RequestException as e:
                    print(f"Request exception for section {section}: {str(e)}")
                    retry_count += 1
                    if retry_count < max_retries:
                        print(f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    continue
        
        print(f"\nTotal articles collected: {len(all_articles)}")
        
        # Cache in both production and development mode
        cache_timeout = 1800 if app.debug else 300  # 30 minutes in dev, 5 minutes in prod
        cache.set(cache_key, all_articles, timeout=cache_timeout)
        print(f"Articles cached with key {cache_key} for {cache_timeout} seconds")
        
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
    if 'user' not in session:
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
            "apikey": ALPHA_VANTAGE_API_KEY
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
    if 'user' not in session:
        return jsonify({"error": "Authentication required"}), 401

    try:
        # This would normally analyze multiple data points from Alpha Vantage
        # For this demo, we'll create simulated patterns based on volume and price
        
        # Track API call for any Alpha Vantage calls we make
        track_api_call('alpha_vantage', 'unusual_patterns')
        
        # Get sector parameter if provided
        sector = request.args.get('sector', None)
        
        # Get top gainers/losers as a starting point
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "TOP_GAINERS_LOSERS",
            "apikey": ALPHA_VANTAGE_API_KEY
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Define sector mapping for common stocks
        sector_mapping = {
            'AAPL': 'technology', 'MSFT': 'technology', 'GOOGL': 'technology', 'AMZN': 'technology', 'NVDA': 'technology',
            'JNJ': 'healthcare', 'PFE': 'healthcare', 'UNH': 'healthcare', 'MRK': 'healthcare', 'ABT': 'healthcare',
            'JPM': 'finance', 'BAC': 'finance', 'WFC': 'finance', 'C': 'finance', 'GS': 'finance',
            'XOM': 'energy', 'CVX': 'energy', 'COP': 'energy', 'SLB': 'energy', 'OXY': 'energy'
        }
        
        unusual_patterns = []
        
        # Process potential unusual patterns
        if 'top_gainers' in data:
            for stock in data['top_gainers'][:5]:  # Limit to top 5
                # Check if volume is significant
                try:
                    volume = int(stock.get('volume', '0'))
                    change_percent = float(stock.get('change_percentage', '0').rstrip('%'))
                    ticker = stock.get('ticker', '')
                    
                    # Assign a sector (use 'other' if not in our mapping)
                    stock_sector = sector_mapping.get(ticker, 'other')
                    
                    # Skip if filtering by sector and this stock is not in that sector
                    if sector and sector.lower() != 'all sectors' and stock_sector != sector.lower():
                        continue
                    
                    if volume > 1000000 and change_percent > 5:  # Significant volume and change
                        unusual_patterns.append({
                            'symbol': ticker,
                            'name': stock.get('name', ''),
                            'pattern_type': 'volume_price_surge',
                            'description': f"Volume spike {round(volume/1000000, 1)}M with strong price movement",
                            'change_percentage': change_percent,
                            'direction': 'up',
                            'sector': stock_sector
                        })
                except (ValueError, TypeError):
                    continue
        
        if 'top_losers' in data:
            for stock in data['top_losers'][:5]:  # Limit to top 5
                try:
                    volume = int(stock.get('volume', '0'))
                    change_percent = float(stock.get('change_percentage', '0').rstrip('%'))
                    ticker = stock.get('ticker', '')
                    
                    # Assign a sector (use 'other' if not in our mapping)
                    stock_sector = sector_mapping.get(ticker, 'other')
                    
                    # Skip if filtering by sector and this stock is not in that sector
                    if sector and sector.lower() != 'all sectors' and stock_sector != sector.lower():
                        continue
                    
                    if volume > 1000000 and abs(change_percent) > 5:  # Significant volume and change
                        unusual_patterns.append({
                            'symbol': ticker,
                            'name': stock.get('name', ''),
                            'pattern_type': 'volume_price_drop',
                            'description': f"Unusual selling volume with sharp price decline",
                            'change_percentage': change_percent,
                            'direction': 'down',
                            'sector': stock_sector
                        })
                except (ValueError, TypeError):
                    continue
        
        # For the most active, look for potential breakouts or breakdowns
        if 'most_actively_traded' in data:
            for stock in data['most_actively_traded'][:5]:
                try:
                    volume = int(stock.get('volume', '0'))
                    change_percent = float(stock.get('change_percentage', '0').rstrip('%'))
                    ticker = stock.get('ticker', '')
                    
                    # Assign a sector (use 'other' if not in our mapping)
                    stock_sector = sector_mapping.get(ticker, 'other')
                    
                    # Skip if filtering by sector and this stock is not in that sector
                    if sector and sector.lower() != 'all sectors' and stock_sector != sector.lower():
                        continue
                    
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
                            'symbol': ticker,
                            'name': stock.get('name', ''),
                            'pattern_type': pattern_type,
                            'description': description,
                            'change_percentage': change_percent,
                            'direction': direction,
                            'sector': stock_sector
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
    if 'user' not in session:
        return jsonify({"error": "Authentication required"}), 401

    try:
        # In a real implementation, this would analyze the user's portfolio,
        # cross-reference with market sentiment, technicals, and fundamentals
        # For this demo, we'll use default recommendations since we may not have a stocks table
        
        # Track API calls
        track_api_call('alpha_vantage', 'recommendations')
        
        # Get sector parameter if provided
        sector = request.args.get('sector', None)
        
        # Use default stocks since we may not have a stocks table
        # Filter by sector if provided
        all_stocks = [
            {'symbol': 'AAPL', 'name': 'Apple Inc.', 'sector': 'technology'},
            {'symbol': 'MSFT', 'name': 'Microsoft Corporation', 'sector': 'technology'},
            {'symbol': 'GOOGL', 'name': 'Alphabet Inc.', 'sector': 'technology'},
            {'symbol': 'AMZN', 'name': 'Amazon.com Inc.', 'sector': 'technology'},
            {'symbol': 'NVDA', 'name': 'NVIDIA Corporation', 'sector': 'technology'},
            {'symbol': 'JNJ', 'name': 'Johnson & Johnson', 'sector': 'healthcare'},
            {'symbol': 'PFE', 'name': 'Pfizer Inc.', 'sector': 'healthcare'},
            {'symbol': 'UNH', 'name': 'UnitedHealth Group', 'sector': 'healthcare'},
            {'symbol': 'JPM', 'name': 'JPMorgan Chase & Co.', 'sector': 'finance'},
            {'symbol': 'BAC', 'name': 'Bank of America Corp.', 'sector': 'finance'},
            {'symbol': 'WFC', 'name': 'Wells Fargo & Co.', 'sector': 'finance'},
            {'symbol': 'XOM', 'name': 'Exxon Mobil Corp.', 'sector': 'energy'},
            {'symbol': 'CVX', 'name': 'Chevron Corporation', 'sector': 'energy'},
            {'symbol': 'COP', 'name': 'ConocoPhillips', 'sector': 'energy'}
        ]
        
        # Filter stocks by sector if provided
        if sector and sector.lower() != 'all sectors':
            user_stocks = [stock for stock in all_stocks if stock['sector'] == sector.lower()]
        else:
            # Pick a few stocks from each sector if no filter
            user_stocks = [all_stocks[0], all_stocks[5], all_stocks[8], all_stocks[11]]
        
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
                "apikey": ALPHA_VANTAGE_API_KEY
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
        # Return default recommendations on error
        default_recommendations = [
            {
                'symbol': 'AAPL',
                'name': 'Apple Inc.',
                'action': 'BUY',
                'rationale': 'Strong Q2 earnings, positive news sentiment'
            },
            {
                'symbol': 'MSFT',
                'name': 'Microsoft Corporation',
                'action': 'BUY',
                'rationale': 'Cloud growth, AI integration momentum'
            },
            {
                'symbol': 'NFLX',
                'name': 'Netflix Inc.',
                'action': 'HOLD',
                'rationale': 'Subscriber growth slowing, competition'
            },
            {
                'symbol': 'TSLA',
                'name': 'Tesla Inc.',
                'action': 'SELL',
                'rationale': 'Production issues, negative news trend'
            }
        ]
        return jsonify(default_recommendations)

@app.route('/api/trend-insight/correlations')
def get_asset_correlations():
    """Get cross-asset correlations for TrendInsight dashboard."""
    if 'user' not in session:
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

@app.route('/api/trend-insight/insider-transactions')
def get_insider_transactions():
    """Get insider transactions data for TrendInsight dashboard."""
    if 'user' not in session:
        return jsonify({"error": "Authentication required"}), 401

    try:
        # Track API call
        track_api_call('alpha_vantage', 'insider_transactions')
        
        # Get symbol parameter if provided (don't default to any specific symbol)
        symbol = request.args.get('symbol', None)
        
        # Get sector parameter if provided
        sector = request.args.get('sector', None)
        
        # In a production environment, this would fetch real data from Alpha Vantage
        # Alpha Vantage API endpoint for insider transactions is:
        # https://www.alphavantage.co/query?function=INSIDER_TRANSACTIONS&symbol=IBM&apikey=demo
        
        # For demo purposes, we'll create simulated data with more entries
        all_transactions = [
            # Technology sector
            {
                'symbol': 'AAPL',
                'name': 'Timothy Cook',
                'position': 'CEO',
                'transaction_type': 'BUY',
                'quantity': 12500,
                'value': 2187500,
                'date': '2025-02-15',
                'sector': 'technology'
            },
            {
                'symbol': 'MSFT',
                'name': 'Satya Nadella',
                'position': 'CEO',
                'transaction_type': 'SELL',
                'quantity': 8000,
                'value': 3120000,
                'date': '2025-02-14',
                'sector': 'technology'
            },
            {
                'symbol': 'GOOGL',
                'name': 'Sundar Pichai',
                'position': 'CEO',
                'transaction_type': 'BUY',
                'quantity': 5000,
                'value': 7250000,
                'date': '2025-02-12',
                'sector': 'technology'
            },
            {
                'symbol': 'NVDA',
                'name': 'Jensen Huang',
                'position': 'CEO',
                'transaction_type': 'SELL',
                'quantity': 10000,
                'value': 8500000, 
                'date': '2025-02-10',
                'sector': 'technology'
            },
            # Healthcare sector
            {
                'symbol': 'JNJ',
                'name': 'Joaquin Duato',
                'position': 'CEO',
                'transaction_type': 'BUY',
                'quantity': 3500,
                'value': 577500,
                'date': '2025-02-10',
                'sector': 'healthcare'
            },
            {
                'symbol': 'PFE',
                'name': 'Albert Bourla',
                'position': 'CEO',
                'transaction_type': 'SELL',
                'quantity': 15000,
                'value': 525000,
                'date': '2025-02-09',
                'sector': 'healthcare'
            },
            {
                'symbol': 'UNH',
                'name': 'Andrew Witty',
                'position': 'CEO',
                'transaction_type': 'BUY',
                'quantity': 2500,
                'value': 1225000,
                'date': '2025-02-07',
                'sector': 'healthcare'
            },
            {
                'symbol': 'MRK',
                'name': 'Robert Davis',
                'position': 'CEO',
                'transaction_type': 'SELL',
                'quantity': 8000,
                'value': 720000,
                'date': '2025-02-05',
                'sector': 'healthcare'
            },
            # Finance sector
            {
                'symbol': 'JPM',
                'name': 'Jamie Dimon',
                'position': 'CEO',
                'transaction_type': 'BUY',
                'quantity': 10000,
                'value': 1875000,
                'date': '2025-02-08',
                'sector': 'finance'
            },
            {
                'symbol': 'BAC',
                'name': 'Brian Moynihan',
                'position': 'CEO',
                'transaction_type': 'SELL',
                'quantity': 25000,
                'value': 950000,
                'date': '2025-02-07',
                'sector': 'finance'
            },
            {
                'symbol': 'GS',
                'name': 'David Solomon',
                'position': 'CEO',
                'transaction_type': 'BUY', 
                'quantity': 5000,
                'value': 2150000,
                'date': '2025-02-06',
                'sector': 'finance'
            },
            {
                'symbol': 'MS',
                'name': 'James Gorman',
                'position': 'CEO',
                'transaction_type': 'SELL',
                'quantity': 12000,
                'value': 1140000,
                'date': '2025-02-04',
                'sector': 'finance'
            },
            # Energy sector
            {
                'symbol': 'XOM',
                'name': 'Darren Woods',
                'position': 'CEO',
                'transaction_type': 'BUY',
                'quantity': 7500,
                'value': 825000,
                'date': '2025-02-06',
                'sector': 'energy'
            },
            {
                'symbol': 'CVX',
                'name': 'Michael Wirth',
                'position': 'CEO',
                'transaction_type': 'SELL',
                'quantity': 6000,
                'value': 978000,
                'date': '2025-02-05',
                'sector': 'energy'
            },
            {
                'symbol': 'COP',
                'name': 'Ryan Lance',
                'position': 'CEO',
                'transaction_type': 'BUY',
                'quantity': 8000,
                'value': 960000,
                'date': '2025-02-03',
                'sector': 'energy'
            },
            {
                'symbol': 'SLB',
                'name': 'Olivier Le Peuch',
                'position': 'CEO',
                'transaction_type': 'SELL',
                'quantity': 9000,
                'value': 540000,
                'date': '2025-02-01',
                'sector': 'energy'
            }
        ]
        
        # Start with all transactions
        transactions = all_transactions
        
        # Filter by sector if specified
        if sector and sector.lower() != 'all sectors':
            transactions = [t for t in transactions if t['sector'] == sector.lower()]
            
        # Filter by symbol only if explicitly provided
        if symbol:
            transactions = [t for t in transactions if t['symbol'] == symbol.upper()]
            
        # Sort by date (newest first)
        transactions.sort(key=lambda x: x['date'], reverse=True)
        
        # Limit to 15 most recent transactions (increased from 10)
        transactions = transactions[:15]
        
        return jsonify(transactions)
        
    except Exception as e:
        app.logger.error(f"Error fetching insider transactions: {str(e)}")
        # Return empty list in case of error
        return jsonify([])

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
    """Stock tracker dashboard."""
    # Check if the user is logged in
    if 'user' not in session:
        return redirect(url_for('login'))
    
    # Get current user settings
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    # If user settings are needed:
    settings = {}
    # Get user settings
    try:
        cursor.execute("SELECT * FROM settings WHERE username = %s", (session['user'],))
        settings_result = cursor.fetchone()
        if settings_result:
            settings = settings_result
        else:
            # Use default settings
            settings = {
                'button_width': 200,
                'button_height': 200,
                'theme': 'light'
            }
    except Exception as e:
        app.logger.error(f"Error fetching user settings: {str(e)}")
        settings = {
            'button_width': 200,
            'button_height': 200,
            'theme': 'light'
        }
    
    # Get stocks data (if needed)
    # For now, just render the template
    conn.close()
    
    return render_template(
        'stock_tracker.html',
        page_title="Stock Tracker",
        user_settings=settings
    )

@app.route('/trend-insight')
def trend_insight():
    """TrendInsight dashboard for market intelligence."""
    # Check if the user is logged in
    if 'user' not in session:
        return redirect(url_for('login'))
    
    # Get current user settings
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    # If user settings are needed:
    settings = {}
    # Get user settings
    try:
        cursor.execute("SELECT * FROM settings WHERE username = %s", (session['user'],))
        settings_result = cursor.fetchone()
        if settings_result:
            settings = settings_result
        else:
            # Use default settings
            settings = {
                'button_width': 200,
                'button_height': 200,
                'theme': 'light'
            }
    except Exception as e:
        app.logger.error(f"Error fetching user settings: {str(e)}")
        settings = {
            'button_width': 200,
            'button_height': 200,
            'theme': 'light'
        }
    
    conn.close()
    
    return render_template(
        'trend_insight.html',
        page_title="TrendInsight Dashboard",
        user_settings=settings
    )

@app.route('/api/trend-insight/weather-impact')
def get_weather_market_impact():
    """Get weather data for major financial centers and potential market impacts."""
    if 'user' not in session:
        return jsonify({"error": "Authentication required"}), 401

    try:
        # Track API call
        track_api_call('openweathermap', 'weather_impact')
        
        # Get sector parameter if provided
        sector = request.args.get('sector', None)
        
        # Major financial centers with their coordinates
        financial_centers = {
            'New York': {'lat': 40.7128, 'lon': -74.0060},
            'London': {'lat': 51.5074, 'lon': -0.1278},
            'Tokyo': {'lat': 35.6762, 'lon': 139.6503},
            'Shanghai': {'lat': 31.2304, 'lon': 121.4737},
            'Hong Kong': {'lat': 22.3193, 'lon': 114.1694}
        }
        
        # Define sector-specific cities based on industry concentration
        energy_cities = ['Houston', 'Dallas', 'Calgary']
        agriculture_cities = ['Chicago', 'Kansas City', 'Minneapolis']
        technology_cities = ['San Francisco', 'Seattle', 'Boston']
        healthcare_cities = ['Boston', 'San Diego', 'Basel']
        finance_cities = ['New York', 'London', 'Hong Kong']
        
        # Filter cities by sector if specified
        if sector and sector.lower() != 'all sectors':
            if sector.lower() == 'energy':
                extra_cities = energy_cities
            elif sector.lower() == 'healthcare':
                extra_cities = healthcare_cities
            elif sector.lower() == 'technology':
                extra_cities = technology_cities
            elif sector.lower() == 'finance':
                extra_cities = finance_cities
            else:
                extra_cities = []
        else:
            # Default to showing all major financial centers
            extra_cities = []
        
        # For this demo, we'll return mock weather data
        # In production, you would call the OpenWeatherMap API for each city
        # API endpoint: https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_key}
        
        weather_data = []
        
        # Generate mock data for financial centers
        for city, coords in financial_centers.items():
            # Only include New York, London, and Hong Kong by default
            if city not in ['New York', 'London', 'Hong Kong'] and city not in extra_cities:
                continue
                
            # Generate mock weather data
            weather_type = random.choice(['Clear', 'Clouds', 'Rain', 'Snow', 'Thunderstorm'])
            temperature = round(random.uniform(0, 35), 1)  # Celsius
            
            # Determine market impact based on weather and city
            impact = {
                'Clear': {
                    'sentiment': 'neutral',
                    'description': 'Normal trading conditions expected.'
                },
                'Clouds': {
                    'sentiment': 'neutral',
                    'description': 'No significant weather impact on markets.'
                },
                'Rain': {
                    'sentiment': 'slightly_negative',
                    'description': 'Minor disruptions possible in local operations.'
                },
                'Snow': {
                    'sentiment': 'negative',
                    'description': 'Potential transportation delays and reduced trading volume.'
                },
                'Thunderstorm': {
                    'sentiment': 'very_negative',
                    'description': 'Significant disruptions possible; energy markets may see volatility.'
                }
            }
            
            # Sectors most affected by weather in this city
            affected_sectors = []
            if weather_type in ['Snow', 'Thunderstorm', 'Rain']:
                if city in ['New York', 'London', 'Hong Kong']:
                    affected_sectors.append('Finance')
                if city in ['Houston', 'Dallas', 'Calgary']:
                    affected_sectors.append('Energy')
                if city in ['Chicago', 'Kansas City', 'Minneapolis']:
                    affected_sectors.append('Agriculture')
            
            weather_data.append({
                'city': city,
                'weather': weather_type,
                'temperature': temperature,
                'impact': impact[weather_type],
                'affected_sectors': affected_sectors
            })
            
        # If we have sector-specific cities to include
        for city in extra_cities:
            if city not in [c['city'] for c in weather_data]:
                # Mock coordinates
                coords = {'lat': 0, 'lon': 0}
                
                # Generate mock weather data
                weather_type = random.choice(['Clear', 'Clouds', 'Rain', 'Snow', 'Thunderstorm'])
                temperature = round(random.uniform(0, 35), 1)  # Celsius
                
                # Determine market impact based on weather and city
                impact = {
                    'Clear': {
                        'sentiment': 'neutral',
                        'description': 'Normal trading conditions expected.'
                    },
                    'Clouds': {
                        'sentiment': 'neutral',
                        'description': 'No significant weather impact on markets.'
                    },
                    'Rain': {
                        'sentiment': 'slightly_negative',
                        'description': 'Minor disruptions possible in local operations.'
                    },
                    'Snow': {
                        'sentiment': 'negative',
                        'description': 'Potential transportation delays and reduced trading volume.'
                    },
                    'Thunderstorm': {
                        'sentiment': 'very_negative',
                        'description': 'Significant disruptions possible; energy markets may see volatility.'
                    }
                }
                
                # Sectors most affected by weather in this city
                affected_sectors = []
                if weather_type in ['Snow', 'Thunderstorm', 'Rain']:
                    if city in energy_cities:
                        affected_sectors.append('Energy')
                    if city in agriculture_cities:
                        affected_sectors.append('Agriculture')
                    if city in technology_cities:
                        affected_sectors.append('Technology')
                    if city in healthcare_cities:
                        affected_sectors.append('Healthcare')
                    if city in finance_cities:
                        affected_sectors.append('Finance')
                
                weather_data.append({
                    'city': city,
                    'weather': weather_type,
                    'temperature': temperature,
                    'impact': impact[weather_type],
                    'affected_sectors': affected_sectors
                })
        
        return jsonify(weather_data)
        
    except Exception as e:
        app.logger.error(f"Error fetching weather impact data: {str(e)}")
        return jsonify({"error": "Failed to fetch weather impact data"}), 500

def verify_alpha_vantage_key(api_key):
    """Verify that the Alpha Vantage API key is valid by making a test request."""
    if not api_key:
        app.logger.warning("No Alpha Vantage API key provided")
        return False
        
    # Debug the API key length and format
    app.logger.info(f"Verifying Alpha Vantage API key - Length: {len(api_key)}, Starts with: {api_key[:4]}...")
        
    try:
        # Make a minimal API call to verify the key
        url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=IBM&apikey={api_key}"
        app.logger.info(f"Making test request to Alpha Vantage: {url.replace(api_key, 'XXXXX')}")
        
        response = requests.get(url, timeout=10)
        app.logger.info(f"Alpha Vantage test response status code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            app.logger.info(f"Alpha Vantage response: {data}")
            
            # Check if we got an error message about invalid API key
            if 'Error Message' in data and 'apikey' in data.get('Error Message', '').lower():
                app.logger.error(f"Invalid Alpha Vantage API key: {data['Error Message']}")
                return False
            # If we get rate limited but the key is valid
            elif 'Note' in data and 'call frequency' in data.get('Note', '').lower():
                app.logger.warning(f"Alpha Vantage API rate limit reached: {data['Note']}")
                return True  # Key is valid, just rate limited
            # If we got actual data
            elif 'Global Quote' in data and data['Global Quote']:
                app.logger.info("Alpha Vantage API key verified successfully with data")
                return True
            else:
                app.logger.warning(f"Unexpected Alpha Vantage API response structure: {data}")
                # Consider the key valid if we got a 200 response, even if data structure is unexpected
                return True
        else:
            app.logger.error(f"Alpha Vantage API returned status code {response.status_code}")
            return False
    except Exception as e:
        app.logger.error(f"Error verifying Alpha Vantage API key: {str(e)}")
        return False

@app.route('/api/market-indices')
def get_market_indices():
    """Fetch market indices data from Yahoo Finance."""
    # Check for authentication
    is_authenticated = 'user' in session
    
    # If not authenticated, return demo data instead of 401
    if not is_authenticated:
        app.logger.info("Unauthenticated request to market-indices API, returning demo data")
        # Return demo data
        demo_data = {
            "indices": {
                'DJI': {
                    'symbol': 'DJI',
                    'apiSymbol': '^DJI',
                    'price': '42,500.35',
                    'change': '+125.69',
                    'percent_change': '+0.32',
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'source': 'demo'
                },
                'SPX': {
                    'symbol': 'SPX',
                    'apiSymbol': '^GSPC',
                    'price': '5,450.32',
                    'change': '+15.29',
                    'percent_change': '+0.31',
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'source': 'demo'
                },
                'IXIC': {
                    'symbol': 'IXIC',
                    'apiSymbol': '^IXIC',
                    'price': '17,700.42',
                    'change': '-3.01',
                    'percent_change': '-0.02',
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'source': 'demo'
                },
                'VIX': {
                    'symbol': 'VIX',
                    'apiSymbol': '^VIX',
                    'price': '14.2',
                    'change': '-0.29',
                    'percent_change': '-2.04',
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'source': 'demo'
                },
                'TNX': {
                    'symbol': 'TNX',
                    'apiSymbol': '^TNX',
                    'price': '4.32',
                    'change': '+0.02',
                    'percent_change': '+0.51',
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'source': 'demo'
                }
            },
            "demo": True
        }
        return jsonify(demo_data)

    try:
        # Track API call
        track_api_call('yahoo_finance', 'market_indices')
        
        # Market symbols to fetch - Yahoo Finance format
        symbols = {
            'DJI': '^DJI',    # Dow Jones Industrial Average
            'SPX': '^GSPC',   # S&P 500
            'IXIC': '^IXIC',  # NASDAQ Composite
            'VIX': '^VIX',    # CBOE Volatility Index
            'TNX': '^TNX'     # 10-Year Treasury Note Yield
        }
        
        requested_symbol = request.args.get('symbol')
        
        # If a specific symbol is requested, only fetch that one
        if requested_symbol and requested_symbol in symbols:
            symbols_to_fetch = {requested_symbol: symbols[requested_symbol]}
        else:
            symbols_to_fetch = symbols
        
        # Generate results
        results = {}
        success_count = 0
        
        for symbol_key, yahoo_symbol in symbols_to_fetch.items():
            try:
                # Yahoo Finance API endpoint
                url = f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_symbol}?interval=1d&range=1d"
                app.logger.info(f"Fetching market data for {symbol_key} from Yahoo Finance: {url}")
                
                response = requests.get(url, timeout=10, headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1'
                })
                
                if response.status_code == 200:
                    data = response.json()
                    app.logger.debug(f"Yahoo Finance response for {symbol_key}: {str(data)[:200]}...")
                    
                    # Extract the relevant data from Yahoo Finance response
                    if 'chart' in data and 'result' in data['chart'] and data['chart']['result']:
                        chart_data = data['chart']['result'][0]
                        
                        # Get the latest quote
                        meta = chart_data['meta']
                        quote = chart_data.get('indicators', {}).get('quote', [{}])[0]
                        
                        # Get the latest price
                        # regularMarketPrice is the current price
                        current_price = meta.get('regularMarketPrice', 0)
                        
                        # Previous close
                        previous_close = meta.get('chartPreviousClose', 0)
                        
                        # Calculate change
                        change = current_price - previous_close
                        percent_change = (change / previous_close) * 100 if previous_close else 0
                        
                        results[symbol_key] = {
                            'symbol': symbol_key,
                            'apiSymbol': yahoo_symbol,
                            'price': str(current_price),
                            'change': str(change),
                            'percentChange': str(round(percent_change, 2)),
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'source': 'yahoo_finance'  # Mark as coming from Yahoo Finance
                        }
                        app.logger.info(f"Successfully fetched data for {symbol_key} from Yahoo Finance")
                        success_count += 1
                    else:
                        # Handle missing data in response
                        app.logger.warning(f"Invalid data structure from Yahoo Finance for {symbol_key}")
                        results[symbol_key] = {
                            'symbol': symbol_key,
                            'error': 'No data available from Yahoo Finance',
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        }
                else:
                    # Handle non-200 response
                    app.logger.error(f"Yahoo Finance API error for {symbol_key}: Status {response.status_code}")
                    results[symbol_key] = {
                        'symbol': symbol_key,
                        'error': f'API returned status code {response.status_code}',
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }
                
                # Add delay to avoid rate limiting
                time.sleep(0.5)
                
            except Exception as e:
                app.logger.error(f"Error fetching data for {symbol_key}: {str(e)}")
                results[symbol_key] = {
                    'symbol': symbol_key,
                    'error': f'Failed to fetch data: {str(e)}',
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                # Fall back to simulation for this symbol only
                if symbol_key == 'DJI':  # Dow Jones around 42,000-43,000
                    base_price = random.uniform(42000, 43000)
                elif symbol_key == 'SPX':  # S&P 500 around 5,400-5,500
                    base_price = random.uniform(5400, 5500)
                elif symbol_key == 'IXIC':  # NASDAQ around 17,600-17,800
                    base_price = random.uniform(17600, 17800)
                elif symbol_key == 'VIX':  # VIX usually between 12-18
                    base_price = random.uniform(12, 18)
                elif symbol_key == 'TNX':  # 10Y Treasury yield around 4.2-4.5%
                    base_price = random.uniform(4.2, 4.5)
                else:
                    continue  # Skip unknown symbols
                    
                # Random change between -1% and +1% of base price
                change_percent = random.uniform(-1, 1)
                change_amount = base_price * change_percent / 100
                
                results[symbol_key] = {
                    'symbol': symbol_key,
                    'apiSymbol': yahoo_symbol,
                    'price': f"{base_price:.2f}",
                    'change': f"{change_amount:.2f}",
                    'percentChange': f"{change_percent:.2f}",
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'source': 'simulation'  # Mark as simulated data
                }
                app.logger.warning(f"Using simulated data for {symbol_key} after API error")
        
        # New: Format the response to match our expected structure
        formatted_results = {
            "indices": results,
            "demo": False,
            "partial": success_count > 0 and success_count < len(symbols_to_fetch)
        }
        
        # If we couldn't fetch any data at all, log error but still return what we have
        if success_count == 0:
            app.logger.error("Failed to fetch any market data from Yahoo Finance - returning simulation data")
        else:
            app.logger.info(f"Successfully fetched {success_count}/{len(symbols_to_fetch)} market indices")
        
        return jsonify(formatted_results)
        
    except Exception as e:
        app.logger.error(f"Error fetching market indices: {str(e)}")
        return jsonify({"error": "Failed to fetch market indices data"}), 500

@app.route('/ai-insights')
def ai_insights():
    """Render AI insights dashboard page."""
    if 'user' not in session:
        return redirect(url_for('login'))
    
    return render_template('ai_insights_dashboard.html')

@app.route('/api/ai-insights')
def get_ai_insights():
    """API endpoint for AI market insights data."""
    if 'user' not in session:
        app.logger.info("No user in session, returning demo data for unauthenticated access")
        # Return demo data instead of 401 error
        return generate_demo_ai_insights()
    
    # Get time period from request parameters (default to '1d')
    period = request.args.get('period', '1d')
    app.logger.info(f"Requested period: {period}")
    
    try:
        # Path to AI experiments directory
        ai_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ai_experiments')
        app.logger.info(f"AI directory path: {ai_dir}")
        
        # Add AI directory to path
        sys.path.append(ai_dir)
        
        try:
            # Import our AI modules
            from alpha_vantage_pipeline import AlphaVantageAPI, MARKET_INDICES
            app.logger.info("Successfully imported AI modules")
        except ImportError as e:
            app.logger.error(f"Import error: {str(e)}")
            return jsonify({
                "error": "Failed to import AI modules",
                "details": str(e)
            }), 500
        
        # Create Alpha Vantage API instance - use default behavior which loads from environment
        api = AlphaVantageAPI()
        app.logger.info("AlphaVantageAPI instance created")
        
        # Try to get current market data
        app.logger.info("Starting to fetch market data")
        market_data = {}
        indices = {}
        
        try:
            for symbol_key, symbol in MARKET_INDICES.items():
                # Get quote data
                app.logger.info(f"Fetching quote for {symbol}")
                quote = api.call_api('GLOBAL_QUOTE', symbol=symbol)
                
                if quote and 'Global Quote' in quote:
                    quote_data = quote['Global Quote']
                    indices[symbol_key] = {
                        'price': quote_data.get('05. price', '0.00'),
                        'change': quote_data.get('09. change', '0.00'),
                        'changePercent': quote_data.get('10. change percent', '0.00%').replace('%', ''),
                        'high': quote_data.get('03. high', '0.00'),
                        'low': quote_data.get('04. low', '0.00'),
                        'volume': quote_data.get('06. volume', '0')
                    }
                    app.logger.info(f"Successfully fetched quote for {symbol}")
                else:
                    app.logger.warning(f"No quote data returned for {symbol}")
                
                time.sleep(0.5)  # Rate limit protection
        except Exception as e:
            app.logger.error(f"Error fetching market data: {str(e)}")
            # If we failed to fetch live data, fall back to demo data
            return generate_demo_ai_insights()
        
        # If we at least got market data but not the rest of the AI features,
        # return a hybrid response with live market data and demo AI data
        if indices:
            demo_data = generate_demo_ai_insights()
            demo_data.json['indices'] = indices
            return demo_data
            
        # If we failed to get any data, return demo data
        return generate_demo_ai_insights()
            
    except Exception as e:
        app.logger.error(f"Error in AI insights API: {str(e)}")
        return jsonify({
            "error": "Server error",
            "details": str(e)
        }), 500

def generate_demo_ai_insights():
    """Generate demo data for AI insights"""
    app.logger.info("Generating demo AI insights data")
    
    # Demo indices data
    indices = {
        'SPX': {
            'price': '4,800.23',
            'change': '28.32',
            'changePercent': '0.59',
            'high': '4,812.56',
            'low': '4,768.51',
            'volume': '3,840,500,000'
        },
        'DJI': {
            'price': '38,563.80',
            'change': '125.69',
            'changePercent': '0.33',
            'high': '38,620.74',
            'low': '38,345.11',
            'volume': '385,230,000'
        },
        'IXIC': {
            'price': '15,360.29',
            'change': '183.02',
            'changePercent': '1.21',
            'high': '15,385.75',
            'low': '15,177.27',
            'volume': '5,462,830,000'
        },
        'VIX': {
            'price': '16.32',
            'change': '-1.25',
            'changePercent': '-7.12',
            'high': '17.85',
            'low': '16.21',
            'volume': '0'
        },
        'TNX': {
            'price': '4.15',
            'change': '0.04',
            'changePercent': '0.97',
            'high': '4.16',
            'low': '4.11',
            'volume': '0'
        }
    }
    
    # Demo AI prediction data
    prediction_confidence = 72  # 0-100 scale, above 50 is bullish
    model_metrics = {
        'ensemble': {'accuracy': 0.68, 'precision': 0.71, 'recall': 0.65, 'f1': 0.68},
        'random_forest': {'accuracy': 0.66, 'precision': 0.69, 'recall': 0.63, 'f1': 0.66},
        'gradient_boosting': {'accuracy': 0.67, 'precision': 0.72, 'recall': 0.61, 'f1': 0.67},
        'neural_network': {'accuracy': 0.64, 'precision': 0.67, 'recall': 0.60, 'f1': 0.63}
    }
    
    # Demo prediction history (1: up, 0: down)
    prediction_history = {
        'dates': [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(10, 0, -1)],
        'actual': [1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
        'predicted': [1, 1, 1, 0, 1, 0, 0, 0, 1, 1]
    }
    
    # Demo feature importance
    feature_importance = [
        {'name': 'RSI (14)', 'value': 0.18},
        {'name': 'Price vs 200-day MA', 'value': 0.15},
        {'name': 'MACD Histogram', 'value': 0.12},
        {'name': 'Volatility (21-day)', 'value': 0.10},
        {'name': 'Price vs 50-day MA', 'value': 0.08},
        {'name': 'Bollinger Width', 'value': 0.07},
        {'name': 'Monthly Return', 'value': 0.06},
        {'name': 'Weekly Return', 'value': 0.05}
    ]
    
    # Demo return predictions
    return_prediction = {
        'SPX': {'predicted': 1.85, 'confidence': 0.73, 'rmse': 2.3, 'r2': 0.58},
        'DJI': {'predicted': 1.72, 'confidence': 0.68, 'rmse': 2.5, 'r2': 0.55},
        'IXIC': {'predicted': 2.18, 'confidence': 0.64, 'rmse': 2.8, 'r2': 0.51}
    }
    
    # Demo news sentiment
    news_sentiment = {
        'overall': 0.35,  # -1 to 1 scale
        'topSources': [
            {'name': 'Bloomberg', 'sentiment': 0.42},
            {'name': 'CNBC', 'sentiment': 0.38},
            {'name': 'Reuters', 'sentiment': 0.25},
            {'name': 'Wall Street Journal', 'sentiment': 0.15},
            {'name': 'Financial Times', 'sentiment': -0.12}
        ],
        'recentArticles': [
            {
                'title': 'Fed signals potential rate cuts later this year',
                'source': 'Bloomberg',
                'date': '2023-06-15',
                'sentiment': 0.58,
                'url': '#'
            },
            {
                'title': 'Tech stocks rally as inflation concerns ease',
                'source': 'CNBC',
                'date': '2023-06-14',
                'sentiment': 0.65,
                'url': '#'
            },
            {
                'title': 'Market volatility increases amid geopolitical tensions',
                'source': 'Financial Times',
                'date': '2023-06-13',
                'sentiment': -0.32,
                'url': '#'
            },
            {
                'title': 'Treasury yields climb after latest economic data',
                'source': 'Wall Street Journal',
                'date': '2023-06-12',
                'sentiment': -0.18,
                'url': '#'
            }
        ]
    }
    
    # Demo portfolio optimization
    portfolio_optimization = {
        'max_sharpe': {
            'weights': {'AAPL': 0.25, 'MSFT': 0.20, 'AMZN': 0.15, 'GOOGL': 0.10, 'NVDA': 0.15, 'BRK.B': 0.10, 'JNJ': 0.05},
            'stats': {
                'expectedReturn': 0.152,
                'volatility': 0.185,
                'sharpeRatio': 0.821,
                'maxDrawdown': 0.255
            }
        },
        'min_vol': {
            'weights': {'AAPL': 0.15, 'MSFT': 0.10, 'AMZN': 0.05, 'GOOGL': 0.05, 'NVDA': 0.05, 'BRK.B': 0.25, 'JNJ': 0.35},
            'stats': {
                'expectedReturn': 0.089,
                'volatility': 0.112,
                'sharpeRatio': 0.794,
                'maxDrawdown': 0.147
            }
        },
        'risk_parity': {
            'weights': {'AAPL': 0.18, 'MSFT': 0.17, 'AMZN': 0.12, 'GOOGL': 0.13, 'NVDA': 0.10, 'BRK.B': 0.15, 'JNJ': 0.15},
            'stats': {
                'expectedReturn': 0.113,
                'volatility': 0.145,
                'sharpeRatio': 0.779,
                'maxDrawdown': 0.198
            }
        }
    }
    
    # Demo economic indicators
    economic_indicators = [
        {
            'name': 'Inflation Rate (CPI)',
            'value': '2.9%',
            'change': '-0.2%',
            'status': 'positive',
            'trend': 'down',
            'category': 'Inflation',
            'description': 'Consumer Price Index, year-over-year change'
        },
        {
            'name': 'Core Inflation',
            'value': '3.2%',
            'change': '-0.1%',
            'status': 'warning',
            'trend': 'down',
            'category': 'Inflation',
            'description': 'CPI excluding food and energy'
        },
        {
            'name': 'Unemployment Rate',
            'value': '3.8%',
            'change': '+0.1%',
            'status': 'positive',
            'trend': 'up',
            'category': 'Employment',
            'description': 'Percentage of labor force that is jobless'
        },
        {
            'name': 'Non-Farm Payrolls',
            'value': '+236K',
            'change': '-30K',
            'status': 'positive',
            'trend': 'down',
            'category': 'Employment',
            'description': 'Jobs added excluding farm workers and some other categories'
        },
        {
            'name': 'GDP Growth Rate',
            'value': '2.4%',
            'change': '+0.3%',
            'status': 'positive',
            'trend': 'up',
            'category': 'GDP',
            'description': 'Annualized quarterly growth rate of Gross Domestic Product'
        },
        {
            'name': 'Fed Funds Rate',
            'value': '5.25-5.50%',
            'change': '0.00%',
            'status': 'neutral',
            'trend': 'unchanged',
            'category': 'Interest Rates',
            'description': 'Target interest rate range set by the Federal Reserve'
        },
        {
            'name': 'Retail Sales MoM',
            'value': '0.7%',
            'change': '+0.5%',
            'status': 'positive',
            'trend': 'up',
            'category': 'Consumer',
            'description': 'Month-over-month change in retail and food service sales'
        },
        {
            'name': 'Consumer Sentiment',
            'value': '67.5',
            'change': '+3.3',
            'status': 'positive',
            'trend': 'up',
            'category': 'Consumer',
            'description': 'University of Michigan Consumer Sentiment Index'
        }
    ]
    
    # Demo alerts
    alerts = [
        {
            'id': '1001',
            'name': 'S&P 500 Below 200-day MA',
            'condition': 'SPX price falls below 200-day moving average',
            'status': 'active',
            'lastTriggered': None,
            'icon': 'chart-line'
        },
        {
            'id': '1002',
            'name': 'VIX Spike Alert',
            'condition': 'VIX rises above 25',
            'status': 'triggered',
            'lastTriggered': '2023-05-18',
            'icon': 'bolt'
        },
        {
            'id': '1003',
            'name': 'AAPL RSI Oversold',
            'condition': 'AAPL RSI(14) falls below 30',
            'status': 'active',
            'lastTriggered': '2023-03-12',
            'icon': 'apple'
        }
    ]
    
    # Combine all data
    response_data = {
        'indices': indices,
        'predictionConfidence': prediction_confidence,
        'modelMetrics': model_metrics,
        'predictionHistory': prediction_history,
        'featureImportance': feature_importance,
        'returnPrediction': return_prediction,
        'newsSentiment': news_sentiment,
        'portfolioOptimization': portfolio_optimization,
        'economicIndicators': economic_indicators,
        'alerts': alerts,
        'lastUpdated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'status': 'Demo data'
    }
    
    return jsonify(response_data)

def calculate_market_metrics(processed_data, period='1d'):
    """Calculate market metrics based on processed data."""
    try:
        metrics = {
            "momentum": {"value": "Mixed", "score": 5.0, "status": "neutral", "description": "Mixed signals in recent market action"},
            "volatility": {"value": "Moderate", "score": 15.0, "status": "neutral", "description": "Volatility near historical average"},
            "breadth": {"value": "Mixed", "score": 50, "status": "neutral", "description": "Equal numbers of advancing and declining stocks"},
            "sentiment": {"value": "Neutral", "score": 50, "status": "neutral", "description": "Balanced sentiment indicators"},
            "technical": {"value": "Neutral", "score": 5.0, "status": "neutral", "description": "Equal bullish and bearish indicators"},
            "aiConfidence": {"value": "Moderate", "score": 50, "status": "neutral", "description": "AI models show moderate confidence"}
        }
        
        # Add AI experiments directory to path
        ai_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ai_experiments')
        sys.path.append(ai_dir)
        
        # Import Alpha Vantage API
        from alpha_vantage_pipeline import AlphaVantageAPI
        
        # Create Alpha Vantage API instance - use default behavior which loads from environment
        api = AlphaVantageAPI()
        
        # Fetch market news sentiment
        news_data = api.get_market_news(limit=20)
        
        # Calculate average sentiment score from news
        news_sentiment_score = 0
        if 'feed' in news_data and news_data['feed']:
            sentiment_scores = []
            for item in news_data['feed']:
                if 'overall_sentiment_score' in item:
                    sentiment_scores.append(float(item['overall_sentiment_score']))
            
            if sentiment_scores:
                avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
                # Convert to 0-100 scale (Alpha Vantage sentiment is approximately -1 to 1)
                news_sentiment_score = (avg_sentiment + 1) * 50
        
        # Map period to the appropriate timeframe for calculations
        period_map = {
            '1d': 1,      # 1 day look-back
            '1w': 5,      # 5 trading days (1 week)
            '1m': 21,     # 21 trading days (1 month)
            '3m': 63,     # 63 trading days (3 months)
            '1y': 252     # 252 trading days (1 year)
        }
        
        # Get the number of days for calculations based on the selected period
        days = period_map.get(period, 1)
        
        # Calculate technical indicators if data is available
        if 'SPX' in processed_data:
            df = processed_data['SPX']
            
            # Calculate momentum (based on recent returns for the selected period)
            if len(df) >= days:
                recent_returns = df['close'].pct_change(days).dropna()
                if len(recent_returns) > 0:
                    momentum_value = recent_returns.iloc[-1] * 100
                    
                    # Adjust thresholds based on the period (longer periods have naturally larger returns)
                    period_adjustments = {
                        '1d': {'strong': 1.5, 'positive': 0.5, 'negative': -0.5, 'strong_negative': -1.5},
                        '1w': {'strong': 2.5, 'positive': 1.0, 'negative': -1.0, 'strong_negative': -2.5},
                        '1m': {'strong': 5.0, 'positive': 2.0, 'negative': -2.0, 'strong_negative': -5.0},
                        '3m': {'strong': 8.0, 'positive': 3.0, 'negative': -3.0, 'strong_negative': -8.0},
                        '1y': {'strong': 15.0, 'positive': 8.0, 'negative': -8.0, 'strong_negative': -15.0}
                    }
                    
                    thresholds = period_adjustments.get(period, period_adjustments['1d'])
                    
                    period_desc = {
                        '1d': 'day',
                        '1w': 'week',
                        '1m': 'month',
                        '3m': '3 months',
                        '1y': 'year'
                    }.get(period, 'period')
                    
                    if momentum_value > thresholds['strong']:
                        metrics["momentum"] = {"value": "Strong", "score": 8.0, "status": "positive", 
                                              "description": f"Strong upward momentum: {momentum_value:.1f}% over the past {period_desc}"}
                    elif momentum_value > thresholds['positive']:
                        metrics["momentum"] = {"value": "Positive", "score": 7.0, "status": "positive", 
                                              "description": f"Positive momentum: {momentum_value:.1f}% over the past {period_desc}"}
                    elif momentum_value > 0:
                        metrics["momentum"] = {"value": "Mild", "score": 6.0, "status": "positive", 
                                              "description": f"Mild upward momentum: {momentum_value:.1f}% over the past {period_desc}"}
                    elif momentum_value > thresholds['negative']:
                        metrics["momentum"] = {"value": "Weak", "score": 4.0, "status": "neutral", 
                                              "description": f"Weak momentum: {momentum_value:.1f}% over the past {period_desc}"}
                    elif momentum_value > thresholds['strong_negative']:
                        metrics["momentum"] = {"value": "Negative", "score": 3.0, "status": "negative", 
                                              "description": f"Negative momentum: {momentum_value:.1f}% over the past {period_desc}"}
                    else:
                        metrics["momentum"] = {"value": "Strong Negative", "score": 2.0, "status": "negative", 
                                              "description": f"Strong downward momentum: {momentum_value:.1f}% over the past {period_desc}"}
            
            # Calculate volatility for the given period
            volatility_window = min(days, 21)  # Use days for shorter periods, cap at 21 for longer ones
            if len(df) >= volatility_window:
                # For longer periods, calculate realized volatility over the period
                if period in ['3m', '1y']:
                    recent_volatility = df['close'].pct_change().rolling(volatility_window).std().dropna() * 100 * np.sqrt(252) # type: ignore
                    if len(recent_volatility) > 0:
                        # For longer periods, calculate average volatility over the period
                        recent_window = min(days, len(recent_volatility))
                        vol_value = recent_volatility.iloc[-recent_window:].mean()
                else:
                    # For shorter periods, use the most recent volatility calculation
                    recent_volatility = df['close'].pct_change().rolling(volatility_window).std().dropna() * 100 * np.sqrt(252) # type: ignore
                    if len(recent_volatility) > 0:
                        vol_value = recent_volatility.iloc[-1]
                
                if 'vol_value' in locals():
                    period_desc = {
                        '1d': 'current',
                        '1w': 'weekly',
                        '1m': 'monthly',
                        '3m': 'quarterly',
                        '1y': 'yearly'
                    }.get(period, '')
                    
                    if vol_value < 10:
                        metrics["volatility"] = {"value": "Low", "score": vol_value, "status": "positive", 
                                                "description": f"Low {period_desc} volatility: {vol_value:.1f}% annualized"}
                    elif vol_value < 15:
                        metrics["volatility"] = {"value": "Moderate", "score": vol_value, "status": "neutral", 
                                                "description": f"Moderate {period_desc} volatility: {vol_value:.1f}% annualized"}
                    elif vol_value < 25:
                        metrics["volatility"] = {"value": "Elevated", "score": vol_value, "status": "neutral", 
                                                "description": f"Elevated {period_desc} volatility: {vol_value:.1f}% annualized"}
                    else:
                        metrics["volatility"] = {"value": "High", "score": vol_value, "status": "negative", 
                                                "description": f"High {period_desc} volatility: {vol_value:.1f}% annualized"}
            
            # Calculate technical score based on the selected period
            if 'rsi_14' in df.columns and 'macd_hist' in df.columns:
                # Use lookback for different periods
                lookback = min(days, len(df) - 1)  # Ensure we don't go beyond data length
                
                # For different periods, we'll check technical indicators over time
                if period in ['3m', '1y']:
                    # For longer periods, check how often indicators were bullish
                    bullish_days = 0
                    total_days = 0
                    
                    for i in range(min(lookback, len(df) - 1)):
                        row = df.iloc[-(i+1)]
                        day_bullish = 0
                        day_signals = 0
                        
                        # RSI
                        if 'rsi_14' in row:
                            day_signals += 1
                            if row['rsi_14'] > 50:
                                day_bullish += 1
                        
                        # MACD
                        if 'macd_hist' in row:
                            day_signals += 1
                            if row['macd_hist'] > 0:
                                day_bullish += 1
                        
                        # Moving Averages
                        if 'sma_50' in row and 'close' in row:
                            day_signals += 1
                            if row['close'] > row['sma_50']:
                                day_bullish += 1
                        
                        if 'sma_200' in row and 'close' in row:
                            day_signals += 1
                            if row['close'] > row['sma_200']:
                                day_bullish += 1
                        
                        # Bollinger Bands
                        if 'bb_upper' in row and 'bb_lower' in row and 'close' in row:
                            day_signals += 1
                            bb_position = (row['close'] - row['bb_lower']) / (row['bb_upper'] - row['bb_lower'])
                            if bb_position > 0.5:
                                day_bullish += 1
                        
                        if day_signals > 0:
                            bullish_days += (day_bullish / day_signals) > 0.5
                            total_days += 1
                    
                    if total_days > 0:
                        tech_score = (bullish_days / total_days) * 10
                        period_desc = '3 months' if period == '3m' else 'year'
                        
                        if tech_score >= 7.5:
                            metrics["technical"] = {"value": "Bullish", "score": tech_score, "status": "positive", 
                                                   "description": f"Bullish technical indicators for {bullish_days} of {total_days} days over the past {period_desc}"}
                        elif tech_score >= 6:
                            metrics["technical"] = {"value": "Mildly Bullish", "score": tech_score, "status": "positive", 
                                                   "description": f"Mildly bullish technical indicators over the past {period_desc}"}
                        elif tech_score > 4:
                            metrics["technical"] = {"value": "Neutral", "score": tech_score, "status": "neutral", 
                                                   "description": f"Mixed technical indicators over the past {period_desc}"}
                        elif tech_score >= 2.5:
                            metrics["technical"] = {"value": "Mildly Bearish", "score": tech_score, "status": "negative", 
                                                   "description": f"Mildly bearish technical indicators over the past {period_desc}"}
                        else:
                            metrics["technical"] = {"value": "Bearish", "score": tech_score, "status": "negative", 
                                                   "description": f"Bearish technical indicators for {total_days - bullish_days} of {total_days} days over the past {period_desc}"}
                else:
                    # For shorter periods, use the most recent data
                    last_row = df.iloc[-1]
                    
                    # Count bullish signals
                    bullish_count = 0
                    total_signals = 0
                    
                    # RSI
                    if 'rsi_14' in last_row:
                        total_signals += 1
                        if last_row['rsi_14'] > 50:
                            bullish_count += 1
                    
                    # MACD
                    if 'macd_hist' in last_row:
                        total_signals += 1
                        if last_row['macd_hist'] > 0:
                            bullish_count += 1
                    
                    # Moving Averages
                    if 'sma_50' in last_row and 'close' in last_row:
                        total_signals += 1
                        if last_row['close'] > last_row['sma_50']:
                            bullish_count += 1
                    
                    if 'sma_200' in last_row and 'close' in last_row:
                        total_signals += 1
                        if last_row['close'] > last_row['sma_200']:
                            bullish_count += 1
                    
                    # Bollinger Bands
                    if 'bb_upper' in last_row and 'bb_lower' in last_row and 'close' in last_row:
                        total_signals += 1
                        bb_position = (last_row['close'] - last_row['bb_lower']) / (last_row['bb_upper'] - last_row['bb_lower'])
                        if bb_position > 0.5:
                            bullish_count += 1
                    
                    if total_signals > 0:
                        tech_score = (bullish_count / total_signals) * 10
                        period_desc = {
                            '1d': 'current',
                            '1w': 'recent',
                            '1m': 'monthly'
                        }.get(period, '')
                        
                        if tech_score >= 7.5:
                            metrics["technical"] = {"value": "Bullish", "score": tech_score, "status": "positive", 
                                                   "description": f"{bullish_count} of {total_signals} {period_desc} indicators are bullish"}
                        elif tech_score >= 6:
                            metrics["technical"] = {"value": "Mildly Bullish", "score": tech_score, "status": "positive", 
                                                   "description": f"{bullish_count} of {total_signals} {period_desc} indicators are bullish"}
                        elif tech_score > 4:
                            metrics["technical"] = {"value": "Neutral", "score": tech_score, "status": "neutral", 
                                                   "description": f"{bullish_count} of {total_signals} {period_desc} indicators are bullish"}
                        elif tech_score >= 2.5:
                            metrics["technical"] = {"value": "Mildly Bearish", "score": tech_score, "status": "negative", 
                                                   "description": f"{bullish_count} of {total_signals} {period_desc} indicators are bullish"}
                        else:
                            metrics["technical"] = {"value": "Bearish", "score": tech_score, "status": "negative", 
                                                   "description": f"{bullish_count} of {total_signals} {period_desc} indicators are bullish"}
        
        # Use S&P, VIX relationship and news sentiment for combined sentiment calculation
        if 'SPX' in processed_data and 'VIX' in processed_data:
            spx_df = processed_data['SPX']
            vix_df = processed_data['VIX']
            
            lookback_days = period_map.get(period, 5)
            
            if len(spx_df) > lookback_days and len(vix_df) > lookback_days:
                # Calculate returns for the selected period
                spx_return = spx_df['close'].pct_change(lookback_days).iloc[-1] * 100
                vix_change = vix_df['close'].pct_change(lookback_days).iloc[-1] * 100
                
                # Calculate sentiment score (negative correlation expected)
                sentiment_score = 50  # Neutral baseline
                
                # Adjust thresholds based on the period
                period_adjustments = {
                    '1d': {'strong': 1.5, 'positive': 0.5, 'negative': -0.5, 'strong_negative': -1.5, 'vix_thresholds': [-5, -2, 0, 2, 5]},
                    '1w': {'strong': 2.5, 'positive': 1.0, 'negative': -1.0, 'strong_negative': -2.5, 'vix_thresholds': [-10, -5, 0, 5, 10]},
                    '1m': {'strong': 5.0, 'positive': 2.0, 'negative': -2.0, 'strong_negative': -5.0, 'vix_thresholds': [-15, -8, 0, 8, 15]},
                    '3m': {'strong': 8.0, 'positive': 3.0, 'negative': -3.0, 'strong_negative': -8.0, 'vix_thresholds': [-20, -10, 0, 10, 20]},
                    '1y': {'strong': 15.0, 'positive': 8.0, 'negative': -8.0, 'strong_negative': -15.0, 'vix_thresholds': [-30, -15, 0, 15, 30]}
                }
                
                thresholds = period_adjustments.get(period, period_adjustments['1d'])
                
                # Adjust based on SPX return
                if spx_return > thresholds['strong']:
                    sentiment_score += 15
                elif spx_return > thresholds['positive']:
                    sentiment_score += 10
                elif spx_return > 0:
                    sentiment_score += 5
                elif spx_return > thresholds['negative']:
                    sentiment_score -= 5
                elif spx_return > thresholds['strong_negative']:
                    sentiment_score -= 10
                else:
                    sentiment_score -= 15
                
                # Adjust based on VIX change (inversely)
                vix_thresholds = thresholds['vix_thresholds']
                if vix_change < vix_thresholds[0]:
                    sentiment_score += 15
                elif vix_change < vix_thresholds[1]:
                    sentiment_score += 10
                elif vix_change < vix_thresholds[2]:
                    sentiment_score += 5
                elif vix_change < vix_thresholds[3]:
                    sentiment_score -= 5
                elif vix_change < vix_thresholds[4]:
                    sentiment_score -= 10
                else:
                    sentiment_score -= 15
                
                # Add news sentiment if available (with higher weight)
                if news_sentiment_score > 0:
                    # Weighted average: 60% price/VIX signals, 40% news sentiment
                    sentiment_score = sentiment_score * 0.6 + news_sentiment_score * 0.4
                
                # Cap between 0 and 100
                sentiment_score = max(0, min(100, sentiment_score))
                
                # Assign sentiment category
                if sentiment_score >= 80:
                    news_desc = " with extremely positive news sentiment" if news_sentiment_score > 75 else ""
                    metrics["sentiment"] = {"value": "Extremely Bullish", "score": sentiment_score, "status": "positive", 
                                           "description": f"Market sentiment is extremely optimistic{news_desc}"}
                elif sentiment_score >= 65:
                    news_desc = " with positive news sentiment" if news_sentiment_score > 60 else ""
                    metrics["sentiment"] = {"value": "Bullish", "score": sentiment_score, "status": "positive", 
                                           "description": f"Market sentiment shows optimism{news_desc}"}
                elif sentiment_score >= 55:
                    news_desc = " with mildly positive news" if news_sentiment_score > 55 else ""
                    metrics["sentiment"] = {"value": "Mildly Bullish", "score": sentiment_score, "status": "positive", 
                                           "description": f"Market sentiment is cautiously optimistic{news_desc}"}
                elif sentiment_score >= 45:
                    metrics["sentiment"] = {"value": "Neutral", "score": sentiment_score, "status": "neutral", 
                                           "description": "Market sentiment is balanced"}
                elif sentiment_score >= 35:
                    news_desc = " with mildly negative news" if news_sentiment_score < 45 else ""
                    metrics["sentiment"] = {"value": "Mildly Bearish", "score": sentiment_score, "status": "negative", 
                                           "description": f"Market sentiment is cautiously pessimistic{news_desc}"}
                elif sentiment_score >= 20:
                    news_desc = " with negative news sentiment" if news_sentiment_score < 40 else ""
                    metrics["sentiment"] = {"value": "Bearish", "score": sentiment_score, "status": "negative", 
                                           "description": f"Market sentiment shows pessimism{news_desc}"}
                else:
                    news_desc = " with extremely negative news sentiment" if news_sentiment_score < 25 else ""
                    metrics["sentiment"] = {"value": "Extremely Bearish", "score": sentiment_score, "status": "negative", 
                                           "description": f"Market sentiment is extremely pessimistic{news_desc}"}
        
        # Calculate AI confidence based on real vs. demo data
        if 'SPX' in processed_data and len(processed_data) >= 3:
            # Higher confidence if we have more market data
            confidence_score = min(len(processed_data) * 15, 75)
            
            # Add news sentiment component to confidence
            if news_sentiment_score > 0:
                # Higher confidence with news data
                confidence_score = min(confidence_score + 15, 85)
                
            # Adjust value based on score
            if confidence_score >= 80:
                metrics["aiConfidence"] = {"value": "Very High", "score": confidence_score, "status": "positive", 
                                          "description": "AI models have high confidence with comprehensive market data"}
            elif confidence_score >= 65:
                metrics["aiConfidence"] = {"value": "High", "score": confidence_score, "status": "positive", 
                                          "description": "AI models show good confidence with sufficient data"}
            elif confidence_score >= 50:
                metrics["aiConfidence"] = {"value": "Moderate", "score": confidence_score, "status": "neutral", 
                                          "description": "AI models show moderate confidence with available data"}
            elif confidence_score >= 30:
                metrics["aiConfidence"] = {"value": "Low", "score": confidence_score, "status": "negative", 
                                          "description": "AI models have limited confidence due to data constraints"}
            else:
                metrics["aiConfidence"] = {"value": "Very Low", "score": confidence_score, "status": "negative", 
                                          "description": "AI models have very low confidence with limited data"}
        
        return metrics
    
    except Exception as e:
        app.logger.error(f"Error calculating market metrics: {str(e)}")
        return {
            "momentum": {"value": "Error", "score": 5.0, "status": "neutral", "description": "Error calculating momentum"},
            "volatility": {"value": "Error", "score": 15.0, "status": "neutral", "description": "Error calculating volatility"},
            "breadth": {"value": "Error", "score": 50, "status": "neutral", "description": "Error calculating market breadth"},
            "sentiment": {"value": "Error", "score": 50, "status": "neutral", "description": "Error calculating sentiment"},
            "technical": {"value": "Error", "score": 5.0, "status": "neutral", "description": "Error calculating technical indicators"},
            "aiConfidence": {"value": "Error", "score": 50, "status": "neutral", "description": "Error calculating AI confidence"}
        }

@app.route('/api/portfolio-optimization', methods=['POST'])
def portfolio_optimization():
    """API endpoint for portfolio optimization."""
    if 'user' not in session:
        return jsonify({"error": "Authentication required"}), 401
    
    # Get request data
    req_data = request.get_json()
    
    if not req_data:
        return jsonify({"error": "No request data provided"}), 400
    
    # Extract parameters
    symbols = req_data.get('symbols')
    risk_tolerance = req_data.get('risk_tolerance', 'moderate')
    optimization_method = req_data.get('method', 'mpt')  # 'mpt' or 'risk_parity'
    historical_days = req_data.get('historical_days', 365 * 2)
    custom_risk_budget = req_data.get('custom_risk_budget')
    
    # Validate symbols
    if not symbols or not isinstance(symbols, list) or len(symbols) < 2:
        return jsonify({
            "error": "At least two valid symbols are required for portfolio optimization"
        }), 400
    
    # Path to AI experiments directory
    ai_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ai_experiments')
    app.logger.info(f"AI directory path: {ai_dir}")
    
    # Add AI directory to path
    if ai_dir not in sys.path:
        sys.path.append(ai_dir)
    
    try:
        # Import our portfolio optimizer module
        from portfolio_optimizer import MPTOptimizer, RiskParityOptimizer
        app.logger.info("Successfully imported portfolio optimizer module")
    except ImportError as e:
        app.logger.error(f"Import error: {str(e)}")
        return jsonify({
            "error": "Failed to import portfolio optimizer module"
        }), 500
    
    try:
        # Create optimizer based on method
        if optimization_method.lower() == 'risk_parity':
            optimizer = RiskParityOptimizer(symbols, historical_days=historical_days)
            
            # Custom risk budget optimization for risk parity
            if custom_risk_budget and isinstance(custom_risk_budget, dict):
                # Validate custom risk budget
                if not all(symbol in symbols for symbol in custom_risk_budget) or \
                   not all(isinstance(weight, (int, float)) for weight in custom_risk_budget.values()):
                    return jsonify({
                        "error": "Invalid custom risk budget provided"
                    }), 400
                
                # Complete missing symbols with equal weights
                total_specified_weight = sum(custom_risk_budget.values())
                remaining_weight = 1.0 - total_specified_weight
                unspecified_symbols = [s for s in symbols if s not in custom_risk_budget]
                
                if unspecified_symbols and remaining_weight > 0:
                    weight_per_symbol = remaining_weight / len(unspecified_symbols)
                    for symbol in unspecified_symbols:
                        custom_risk_budget[symbol] = weight_per_symbol
                
                app.logger.info(f"Using custom risk budget: {custom_risk_budget}")
                result = optimizer.optimize_with_custom_risk(custom_risk_budget)
            else:
                # Standard risk parity optimization
                result = optimizer.optimize(risk_tolerance)
        else:
            # Default to MPT optimization
            optimizer = MPTOptimizer(symbols, historical_days=historical_days)
            result = optimizer.optimize(risk_tolerance)
        
        # Add additional metadata
        result['optimization_method'] = optimization_method
        result['historical_days'] = historical_days
        result['request_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Include raw asset data (last 30 days) for charting
        if optimizer.data is not None and not optimizer.data.empty:
            # Get last 30 days of data
            last_30_days = optimizer.data.iloc[-30:].copy()
            # Convert index to string for JSON serialization
            last_30_days.index = last_30_days.index.strftime('%Y-%m-%d')
            # Convert to dictionary for JSON response
            result['asset_data'] = {
                'dates': last_30_days.index.tolist(),
                'prices': {symbol: last_30_days[symbol].tolist() for symbol in symbols if symbol in last_30_days.columns}
            }
            
            # Calculate correlation matrix
            if optimizer.returns is not None and not optimizer.returns.empty:
                correlation_matrix = optimizer.returns.corr().round(3)
                result['correlation_matrix'] = correlation_matrix.to_dict()
        
        return jsonify(result), 200
        
    except Exception as e:
        app.logger.error(f"An unexpected error occurred: {str(e)}")
        return jsonify({
            "error": "An internal error has occurred"
        }), 500
        app.logger.error(f"Error during portfolio optimization: {str(e)}")
        import traceback
        app.logger.error(traceback.format_exc())
        
        return jsonify({
            "error": "Failed to optimize portfolio",
            "details": str(e)
        }), 500

@app.route('/api/alerts', methods=['GET'])
def get_alerts():
    """API endpoint to get all alert rules."""
    if 'user' not in session:
        return jsonify({"error": "Authentication required"}), 401
    
    # Path to AI experiments directory
    ai_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ai_experiments')
    
    # Add AI directory to path if needed
    if ai_dir not in sys.path:
        sys.path.append(ai_dir)
    
    try:
        # Import our alert system module
        from alert_system import AlertManager
        app.logger.info("Successfully imported alert system module")
    except ImportError as e:
        app.logger.error(f"Import error: {str(e)}")
        return jsonify({
            "error": "Failed to import alert system module"
        }), 500
    
    # Get user_id from session
    user_id = session['user'].get('id')
    
    try:
        # Get alert rules from database
        cursor = get_db_connection().cursor(dictionary=True)
        cursor.execute(
            "SELECT * FROM alert_rules WHERE user_id = %s ORDER BY created_at DESC",
            (user_id,)
        )
        alert_rules = cursor.fetchall()
        cursor.close()
        
        # Format response
        response = {
            "alert_rules": []
        }
        
        for rule in alert_rules:
            # Parse rule parameters from JSON
            rule_params = json.loads(rule['rule_params'])
            
            # Add formatted rule to response
            response["alert_rules"].append({
                "id": rule['id'],
                "name": rule['name'],
                "type": rule['rule_type'],
                "enabled": rule['enabled'],
                "params": rule_params,
                "last_triggered": rule['last_triggered'],
                "created_at": rule['created_at'].isoformat() if rule['created_at'] else None
            })
        
        return jsonify(response), 200
        
    except Exception as e:
        app.logger.error(f"Error getting alert rules: {str(e)}")
        import traceback
        app.logger.error(traceback.format_exc())
        
        return jsonify({
            "error": "Failed to retrieve alert rules"
        }), 500


@app.route('/api/alerts', methods=['POST'])
def create_alert():
    """API endpoint to create a new alert rule."""
    if 'user' not in session:
        return jsonify({"error": "Authentication required"}), 401
    
    # Get request data
    req_data = request.get_json()
    
    if not req_data:
        return jsonify({"error": "No request data provided"}), 400
    
    # Extract parameters
    name = req_data.get('name')
    rule_type = req_data.get('type')
    rule_params = req_data.get('params', {})
    enabled = req_data.get('enabled', True)
    
    # Validate parameters
    if not name or not rule_type:
        return jsonify({
            "error": "Missing required parameters: name and type are required"
        }), 400
    
    # Validate rule_type
    valid_rule_types = ['price', 'prediction', 'portfolio']
    if rule_type not in valid_rule_types:
        return jsonify({
            "error": f"Invalid rule type: {rule_type}. Must be one of: {', '.join(valid_rule_types)}"
        }), 400
    
    # Validate rule_params based on rule_type
    if rule_type == 'price':
        required_params = ['symbol', 'threshold', 'condition']
        if not all(param in rule_params for param in required_params):
            return jsonify({
                "error": f"Missing required parameters for price alert: {', '.join(required_params)}"
            }), 400
    elif rule_type == 'prediction':
        required_params = ['symbol', 'metric', 'threshold', 'condition']
        if not all(param in rule_params for param in required_params):
            return jsonify({
                "error": f"Missing required parameters for prediction alert: {', '.join(required_params)}"
            }), 400
    elif rule_type == 'portfolio':
        required_params = ['portfolio_id', 'metric', 'threshold', 'condition']
        if not all(param in rule_params for param in required_params):
            return jsonify({
                "error": f"Missing required parameters for portfolio alert: {', '.join(required_params)}"
            }), 400
    
    # Path to AI experiments directory
    ai_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ai_experiments')
    
    # Add AI directory to path if needed
    if ai_dir not in sys.path:
        sys.path.append(ai_dir)
    
    try:
        # Import our alert system module
        from alert_system import AlertManager, PriceAlertRule, PredictionAlertRule, PortfolioAlertRule
        app.logger.info("Successfully imported alert system module")
    except ImportError as e:
        app.logger.error(f"Import error: {str(e)}")
        return jsonify({
            "error": "Failed to import alert system module",
            "details": str(e)
        }), 500
    
    # Get user_id from session
    user_id = session['user'].get('id')
    
    try:
        # Insert alert rule into database
        cursor = get_db_connection().cursor()
        cursor.execute(
            """
            INSERT INTO alert_rules 
            (user_id, name, rule_type, rule_params, enabled, created_at) 
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (user_id, name, rule_type, json.dumps(rule_params), enabled, datetime.now())
        )
        
        # Get the ID of the inserted rule
        rule_id = cursor.lastrowid
        get_db_connection().commit()
        cursor.close()
        
        # Return the created rule
        return jsonify({
            "id": rule_id,
            "name": name,
            "type": rule_type,
            "params": rule_params,
            "enabled": enabled,
            "created_at": datetime.now().isoformat()
        }), 201
        
    except Exception as e:
        app.logger.error(f"Error creating alert rule: {str(e)}")
        import traceback
        app.logger.error(traceback.format_exc())
        
        return jsonify({
            "error": "Failed to create alert rule"
        }), 500


@app.route('/api/alerts/<int:rule_id>', methods=['DELETE'])
def delete_alert(rule_id):
    """API endpoint to delete an alert rule."""
    if 'user' not in session:
        return jsonify({"error": "Authentication required"}), 401
    
    # Get user_id from session
    user_id = session['user'].get('id')
    
    try:
        # Check if the rule exists and belongs to the user
        cursor = get_db_connection().cursor(dictionary=True)
        cursor.execute(
            "SELECT id FROM alert_rules WHERE id = %s AND user_id = %s",
            (rule_id, user_id)
        )
        rule = cursor.fetchone()
        
        if not rule:
            cursor.close()
            return jsonify({
                "error": "Alert rule not found or you don't have permission to delete it"
            }), 404
        
        # Delete the rule
        cursor.execute(
            "DELETE FROM alert_rules WHERE id = %s",
            (rule_id,)
        )
        get_db_connection().commit()
        cursor.close()
        
        return jsonify({
            "success": True,
            "message": f"Alert rule {rule_id} deleted successfully"
        }), 200
        
    except Exception as e:
        app.logger.error(f"Error deleting alert rule: {str(e)}")
        import traceback
        app.logger.error(traceback.format_exc())
        
        return jsonify({
            "error": "Failed to delete alert rule"
        }), 500

@app.route('/ai-dashboard')
def ai_dashboard():
    """
    Render the advanced AI financial dashboard with market insights,
    portfolio optimization, alerts, economic indicators, and news sentiment analysis.
    """
    # Allow unauthenticated access but show a banner
    is_authenticated = 'user' in session
    
    # Add debug logging
    app.logger.info("Rendering AI dashboard template")
    app.logger.info(f"Is authenticated: {is_authenticated}")
    app.logger.info(f"Static folder path: {app.static_folder}")
    
    # Return the rendered template
    return render_template('ai_dashboard.html', is_authenticated=is_authenticated, debug_mode=True)

@app.route('/market-indices-standalone')
def market_indices_standalone():
    """
    Render a standalone page that only displays market indices.
    This is a simplified test page.
    """
    return render_template('market_indices_standalone.html')

@app.route('/api/ai-status')
def ai_status():
    """
    Check the status of AI systems and return availability information.
    This endpoint is used by the frontend to monitor AI service health.
    """
    try:
        # In a real-world scenario, we would check actual AI service health
        # For now, we'll return a mock status based on the server being up
        
        # Check if any required API keys are missing
        required_keys = ["ALPHA_VANTAGE_API_KEY"]
        missing_keys = [key for key in required_keys if not os.environ.get(key)]
        
        # Get authentication status
        is_authenticated = 'user' in session
        
        if missing_keys:
            app.logger.warning(f"AI status check: Missing required API keys: {', '.join(missing_keys)}")
            status = "degraded"
            message = f"Missing API keys: {', '.join(missing_keys)}"
        else:
            status = "active"
            message = "All AI systems operational"
        
        return jsonify({
            "status": status,
            "message": message,
            "authenticated": is_authenticated,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        # Log the detailed exception message
        app.logger.error(f"Exception occurred in AI status check: {str(e)}")
        return jsonify({
            "status": "inactive",
            "message": "AI systems are currently experiencing issues. Please try again later.",
            "last_checked": datetime.now().isoformat(),
            "services": {
                "prediction_engine": "offline",
                "market_data": "offline",
                "news_sentiment": "offline",
                "portfolio_optimization": "offline"
            }
        })

@app.route('/api/economic-indicators')
def economic_indicators():
    """
    Provide economic indicators data for the dashboard.
    This endpoint returns key economic indicators with their latest values.
    """
    try:
        # In a production environment, this would fetch real data from sources like:
        # - Federal Reserve Economic Data (FRED)
        # - Bureau of Labor Statistics
        # - Trading Economics API
        
        # For now, we'll return sample data
        indicators = [
            {
                "name": "GDP Growth Rate",
                "value": 0.0251,
                "previous": 0.0212,
                "forecast": 0.0230,
                "category": "Growth",
                "importance": 3,
                "release_date": (datetime.now() - timedelta(days=25)).isoformat()
            },
            {
                "name": "Inflation Rate",
                "value": 0.0312,
                "previous": 0.0345,
                "forecast": 0.0325,
                "category": "Prices",
                "importance": 3,
                "release_date": (datetime.now() - timedelta(days=10)).isoformat()
            },
            {
                "name": "Unemployment Rate",
                "value": 0.0375,
                "previous": 0.0389,
                "forecast": 0.0370,
                "category": "Labor",
                "importance": 3,
                "release_date": (datetime.now() - timedelta(days=5)).isoformat()
            },
            {
                "name": "Interest Rate",
                "value": 0.0525,
                "previous": 0.0525,
                "forecast": 0.0500,
                "category": "Central Bank",
                "importance": 3,
                "release_date": (datetime.now() - timedelta(days=15)).isoformat()
            },
            {
                "name": "Consumer Confidence",
                "value": 102.5,
                "previous": 101.8,
                "forecast": 102.0,
                "category": "Sentiment",
                "importance": 2,
                "release_date": (datetime.now() - timedelta(days=8)).isoformat()
            },
            {
                "name": "Retail Sales MoM",
                "value": 0.0041,
                "previous": 0.0028,
                "forecast": 0.0035,
                "category": "Consumption",
                "importance": 2,
                "release_date": (datetime.now() - timedelta(days=12)).isoformat()
            },
            {
                "name": "Housing Starts",
                "value": 1.425,
                "previous": 1.392,
                "forecast": 1.410,
                "category": "Real Estate",
                "importance": 2,
                "release_date": (datetime.now() - timedelta(days=9)).isoformat()
            },
            {
                "name": "Manufacturing PMI",
                "value": 51.2,
                "previous": 49.8,
                "forecast": 50.5,
                "category": "Manufacturing",
                "importance": 2,
                "release_date": (datetime.now() - timedelta(days=7)).isoformat()
            },
            {
                "name": "Services PMI",
                "value": 53.6,
                "previous": 52.9,
                "forecast": 53.0,
                "category": "Services",
                "importance": 2,
                "release_date": (datetime.now() - timedelta(days=7)).isoformat()
            },
            {
                "name": "Balance of Trade",
                "value": -68.2,
                "previous": -70.5,
                "forecast": -69.0,
                "category": "Trade",
                "importance": 2,
                "release_date": (datetime.now() - timedelta(days=20)).isoformat()
            },
            {
                "name": "Government Debt to GDP",
                "value": 1.28,
                "previous": 1.26,
                "forecast": 1.29,
                "category": "Government",
                "importance": 1,
                "release_date": (datetime.now() - timedelta(days=90)).isoformat()
            },
            {
                "name": "Industrial Production MoM",
                "value": 0.0029,
                "previous": -0.0016,
                "forecast": 0.0025,
                "category": "Production",
                "importance": 1,
                "release_date": (datetime.now() - timedelta(days=14)).isoformat()
            }
        ]
        
        return jsonify({
            "indicators": indicators,
            "last_updated": datetime.now().isoformat()
        })
        
    except Exception as e:
        # Log the detailed exception message
        app.logger.error(f"Exception occurred while retrieving economic indicators: {str(e)}")
        return jsonify({
            "error": "Failed to retrieve economic indicators. Please try again later."
        }), 500

@app.route('/api/news-sentiment')
def news_sentiment():
    """
    Provide news articles with sentiment analysis.
    This endpoint returns recent news articles with sentiment scores.
    """
    try:
        # In a production environment, this would fetch real news and analyze sentiment using:
        # - News APIs (Guardian, NewsAPI, etc.)
        # - Natural Language Processing for sentiment analysis
        
        # For now, we'll return sample data
        articles = [
            {
                "title": "Tech Stocks Rally on Strong Earnings Reports",
                "summary": "Major technology companies reported better-than-expected quarterly earnings, leading to a rally in tech stocks. Investors are optimistic about the sector's growth prospects.",
                "url": "https://example.com/tech-stocks-rally",
                "source": "Financial Times",
                "published_at": (datetime.now() - timedelta(hours=4)).isoformat(),
                "sentiment": "positive",
                "sentiment_score": 0.78,
                "entities": [
                    {"name": "Tech Stocks", "type": "FINANCIAL_INSTRUMENT"},
                    {"name": "Earnings", "type": "FINANCIAL_CONCEPT"},
                    {"name": "Investors", "type": "ENTITY"}
                ]
            },
            {
                "title": "Federal Reserve Signals Potential Rate Cut in Coming Months",
                "summary": "The Federal Reserve has indicated it may begin cutting interest rates in the coming months as inflation shows signs of cooling. Markets responded positively to the news.",
                "url": "https://example.com/fed-rate-cut-signal",
                "source": "Wall Street Journal",
                "published_at": (datetime.now() - timedelta(hours=7)).isoformat(),
                "sentiment": "positive",
                "sentiment_score": 0.65,
                "entities": [
                    {"name": "Federal Reserve", "type": "ORGANIZATION"},
                    {"name": "Interest Rates", "type": "FINANCIAL_CONCEPT"},
                    {"name": "Inflation", "type": "ECONOMIC_INDICATOR"}
                ]
            },
            {
                "title": "Oil Prices Drop Amid Supply Concerns",
                "summary": "Oil prices fell sharply today due to concerns about oversupply in the global market. OPEC+ members are considering increasing production despite weak demand forecasts.",
                "url": "https://example.com/oil-prices-drop",
                "source": "Reuters",
                "published_at": (datetime.now() - timedelta(hours=10)).isoformat(),
                "sentiment": "negative",
                "sentiment_score": -0.55,
                "entities": [
                    {"name": "Oil Prices", "type": "COMMODITY"},
                    {"name": "OPEC+", "type": "ORGANIZATION"},
                    {"name": "Supply", "type": "ECONOMIC_CONCEPT"}
                ]
            },
            {
                "title": "Retail Sales Growth Slows in Q2",
                "summary": "Retail sales growth slowed in the second quarter as consumers cut back on discretionary spending. Analysts cite inflation and economic uncertainty as key factors.",
                "url": "https://example.com/retail-sales-slow",
                "source": "Bloomberg",
                "published_at": (datetime.now() - timedelta(hours=13)).isoformat(),
                "sentiment": "negative",
                "sentiment_score": -0.42,
                "entities": [
                    {"name": "Retail Sales", "type": "ECONOMIC_INDICATOR"},
                    {"name": "Q2", "type": "TIME_PERIOD"},
                    {"name": "Consumers", "type": "ENTITY"}
                ]
            },
            {
                "title": "New AI Regulations May Impact Tech Sector",
                "summary": "Proposed regulations for artificial intelligence technologies could impact the tech sector, as companies may face new compliance requirements and limitations.",
                "url": "https://example.com/ai-regulations",
                "source": "CNBC",
                "published_at": (datetime.now() - timedelta(hours=16)).isoformat(),
                "sentiment": "neutral",
                "sentiment_score": -0.05,
                "entities": [
                    {"name": "AI Regulations", "type": "REGULATION"},
                    {"name": "Tech Sector", "type": "INDUSTRY"},
                    {"name": "Compliance", "type": "BUSINESS_CONCEPT"}
                ]
            },
            {
                "title": "Housing Market Shows Signs of Stabilization",
                "summary": "After months of declining prices, the housing market is showing signs of stabilization. Mortgage rates have decreased slightly, leading to increased buyer interest.",
                "url": "https://example.com/housing-market-stabilizes",
                "source": "Market Watch",
                "published_at": (datetime.now() - timedelta(hours=21)).isoformat(),
                "sentiment": "positive",
                "sentiment_score": 0.38,
                "entities": [
                    {"name": "Housing Market", "type": "MARKET"},
                    {"name": "Mortgage Rates", "type": "FINANCIAL_CONCEPT"},
                    {"name": "Prices", "type": "ECONOMIC_CONCEPT"}
                ]
            }
        ]
        
        # Calculate sentiment summary
        positive_count = sum(1 for article in articles if article["sentiment"] == "positive")
        negative_count = sum(1 for article in articles if article["sentiment"] == "negative")
        neutral_count = sum(1 for article in articles if article["sentiment"] == "neutral")
        
        avg_sentiment = sum(article["sentiment_score"] for article in articles) / len(articles)
        
        sentiment_summary = {
            "positive_count": positive_count,
            "negative_count": negative_count,
            "neutral_count": neutral_count,
            "average_score": avg_sentiment
        }
        
        return jsonify({
            "articles": articles,
            "sentiment_summary": sentiment_summary,
            "last_updated": datetime.now().isoformat()
        })
        
    except Exception as e:
        app.logger.error(f"Failed to retrieve news sentiment: {str(e)}")
        return jsonify({
            "error": "An internal error has occurred. Please try again later."
        }), 500

@app.route('/api/analyze')
def analyze_news_data():
    """
    Analyzes news data using pandas to provide statistics and insights.
    This endpoint demonstrates pandas functionality in the container.
    
    Returns:
        JSON response with news analysis data
    """
    try:
        # Create sample data instead of fetching from API (to avoid rate limiting)
        sample_data = [
            {"title": "AI breakthrough in medical research", "section": "technology", "date": "2025-03-01", "author": "Jane Smith"},
            {"title": "New climate policy announced", "section": "environment", "date": "2025-03-02", "author": "John Doe"},
            {"title": "Quantum computing reaches milestone", "section": "science", "date": "2025-03-03", "author": "Alice Johnson"},
            {"title": "Tech giants face new regulations", "section": "technology", "date": "2025-03-04", "author": "Bob Brown"},
            {"title": "Renewable energy surpasses coal", "section": "environment", "date": "2025-03-05", "author": "Carol White"},
            {"title": "Mars rover discovers water evidence", "section": "science", "date": "2025-03-06", "author": "David Green"},
            {"title": "AI ethics guidelines published", "section": "technology", "date": "2025-03-07", "author": "Eve Black"},
            {"title": "Endangered species recovery plan", "section": "environment", "date": "2025-03-08", "author": "Frank Blue"},
            {"title": "New particle discovered at CERN", "section": "science", "date": "2025-03-09", "author": "Grace Gray"},
            {"title": "Cybersecurity threats increasing", "section": "technology", "date": "2025-03-10", "author": "Henry Red"},
            {"title": "Ocean plastic reduction initiative", "section": "environment", "date": "2025-03-11", "author": "Irene Yellow"},
            {"title": "Space telescope reveals distant galaxies", "section": "science", "date": "2025-03-12", "author": "Jack Purple"},
            {"title": "AI generated content guidelines", "section": "technology", "date": "2025-03-13", "author": "Kate Orange"},
            {"title": "Climate summit reaches agreement", "section": "environment", "date": "2025-03-14", "author": "Leo Brown"},
            {"title": "Breakthrough in fusion energy", "section": "science", "date": "2025-03-15", "author": "Mia Silver"}
        ]
        
        # Convert to pandas dataframe for analysis
        df = pd.DataFrame(sample_data)
        
        # Extract publication year and month
        df['publication_date'] = pd.to_datetime(df['date'])
        df['year'] = df['publication_date'].dt.year
        df['month'] = df['publication_date'].dt.month
        
        # Perform basic analysis
        total_articles = len(df)
        articles_by_section = df.groupby('section').size().to_dict()
        
        # Convert tuple keys to strings for JSON serialization
        articles_by_month_data = df.groupby(['year', 'month']).size()
        articles_by_month = {f"{year}-{month:02d}": count for (year, month), count in articles_by_month_data.items()}
        
        # Calculate average title length
        df['title_length'] = df['title'].apply(len)
        avg_title_length = df['title_length'].mean()
        
        # Find most common words in titles
        all_title_words = ' '.join(df['title']).lower()
        word_counts = {}
        for word in all_title_words.split():
            # Remove punctuation
            word = ''.join(c for c in word if c.isalnum())
            if len(word) > 3:  # Skip short words
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Get top 10 most common words
        common_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Author analysis
        articles_by_author = df.groupby('author').size().to_dict()
        
        # Create a simple time series analysis
        df['day'] = df['publication_date'].dt.day
        time_series = df.groupby('day').size().to_dict()
        
        # Demonstrate some pandas operations
        # Calculate rolling average of title length
        df = df.sort_values('publication_date')
        df['rolling_avg_length'] = df['title_length'].rolling(window=3, min_periods=1).mean()
        
        # Create a correlation matrix of numeric columns
        correlation = df[['title_length', 'day', 'month']].corr().to_dict()
        
        # Demonstrate pandas capabilities with a simple DataFrame description
        description = df.describe().to_dict()
        
        return jsonify({
            'status': 'success',
            'data': {
                'total_articles': total_articles,
                'articles_by_section': articles_by_section,
                'articles_by_month': articles_by_month,
                'average_title_length': avg_title_length,
                'common_title_words': common_words,
                'articles_by_author': articles_by_author,
                'time_series': time_series,
                'correlation_matrix': correlation,
                'dataframe_description': description,
                'pandas_version': pd.__version__,
                'numpy_version': np.__version__,
                'analysis_timestamp': datetime.now().isoformat()
            }
        })
    except Exception as e:
        print(f"Error in analyze_news_data: {str(e)}")
        log_error(f"Error in analyze_news_data: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': 'An internal error has occurred while analyzing news data.'
        }), 500

def log_error(message):
    logging.error(message)

def get_mock_articles_for_section(section):
    """
    Return mock articles for a specific section when the API is rate limited
    
    Args:
        section (str): The news section to get mock articles for
        
    Returns:
        list: A list of mock article dictionaries relevant to the given section
    """
    # Base mock articles that could appear in any section
    base_articles = [
        {
            'title': 'Breaking News: Major Developments Expected',
            'url': 'https://example.com/breaking-news'
        },
        {
            'title': 'Analysis: What Recent Events Mean for the Future',
            'url': 'https://example.com/analysis-future'
        }
    ]
    
    # Section-specific mock articles
    section_specific = {
        'news': [
            {
                'title': 'Global Headlines: Top Stories from Around the World',
                'url': 'https://example.com/global-headlines'
            },
            {
                'title': 'Latest Updates on Major News Events',
                'url': 'https://example.com/latest-updates'
            }
        ],
        'world': [
            {
                'title': 'International Relations: New Diplomatic Efforts Underway',
                'url': 'https://example.com/international-relations'
            },
            {
                'title': 'Global Climate Initiatives Face Challenges',
                'url': 'https://example.com/climate-initiatives'
            }
        ],
        'business': [
            {
                'title': 'Markets Respond to Economic Data Release',
                'url': 'https://example.com/markets-respond'
            },
            {
                'title': 'Corporate Earnings Exceed Expectations',
                'url': 'https://example.com/corporate-earnings'
            },
            {
                'title': 'New Regulations Impact Financial Sector',
                'url': 'https://example.com/financial-regulations'
            }
        ],
        'technology': [
            {
                'title': 'Tech Giants Announce New Innovations',
                'url': 'https://example.com/tech-innovations'
            },
            {
                'title': 'AI Developments Transform Industries',
                'url': 'https://example.com/ai-developments'
            }
        ],
        'science': [
            {
                'title': 'Breakthrough Research in Quantum Computing',
                'url': 'https://example.com/quantum-computing'
            },
            {
                'title': 'Space Exploration: New Discoveries Announced',
                'url': 'https://example.com/space-exploration'
            }
        ],
        'sport': [
            {
                'title': 'Championship Results: Unexpected Outcomes',
                'url': 'https://example.com/championship-results'
            },
            {
                'title': 'Analysis of Recent Sporting Events',
                'url': 'https://example.com/sporting-analysis'
            }
        ],
        'environment': [
            {
                'title': 'Climate Change: New Studies Released',
                'url': 'https://example.com/climate-studies'
            },
            {
                'title': 'Sustainability Initiatives Gain Momentum',
                'url': 'https://example.com/sustainability'
            }
        ],
        'uk-news': [
            {
                'title': 'UK Policy Changes: What You Need to Know',
                'url': 'https://example.com/uk-policy'
            },
            {
                'title': 'Regional Developments Across the UK',
                'url': 'https://example.com/uk-regional'
            }
        ],
        'culture': [
            {
                'title': 'Arts and Entertainment: Weekend Highlights',
                'url': 'https://example.com/weekend-arts'
            },
            {
                'title': 'Cultural Events Drawing Record Attendance',
                'url': 'https://example.com/cultural-events'
            }
        ]
    }
    
    # Combine base articles with section-specific ones
    result = base_articles.copy()
    
    # Add section-specific articles if available, otherwise use generic ones
    if section in section_specific:
        result.extend(section_specific[section])
    else:
        # For unknown sections, add some generic articles
        result.extend([
            {
                'title': f'Top Stories in {section.capitalize()}',
                'url': f'https://example.com/{section}-top-stories'
            },
            {
                'title': f'Latest Developments in {section.capitalize()}',
                'url': f'https://example.com/{section}-latest'
            }
        ])
    
    # Add a note that these are mock articles due to rate limiting
    for article in result:
        article['title'] = f"{article['title']} [Mock - API Rate Limited]"
    
    return result

def fetch_news_for_section(section, page_size=50):
    """
    Fetches news articles for a specific section from the Guardian API.
    
    Args:
        section (str): The section to fetch articles for
        page_size (int): Number of articles to retrieve
        
    Returns:
        list: List of article dictionaries, or empty list on error
    """
    api_key = os.environ.get("GUARDIAN_API_KEY")
    if not api_key:
        print("Guardian API key not found")
        return []
    
    # Check if we have cached data for this section
    cache_key = f"news_section_helper_{section}"
    cached_data = cache.get(cache_key)
    if cached_data:
        print(f"Using cached data for section {section}")
        return cached_data
    
    try:
        print(f"Fetching articles for section: {section}")
        url = "https://content.guardianapis.com/search"
        params = {
            'api-key': api_key,
            'section': section,
            'show-fields': 'headline,shortUrl',
            'page-size': page_size,
            'order-by': 'newest'
        }
        
        print(f"Making API request to: {url}")
        print(f"With parameters: {params}")
        
        # Implement retry logic with exponential backoff
        max_retries = 2
        retry_count = 0
        retry_delay = 1
        
        while retry_count < max_retries:
            try:
                response = requests.get(url, params=params, timeout=10)
                print(f"API response status: {response.status_code}")
                
                if response.status_code == 200:
                    data = response.json()
                    if 'response' in data and 'results' in data['response']:
                        results = data['response']['results']
                        print(f"Found {len(results)} articles in section {section}")
                        
                        # Cache the results for 30 minutes
                        cache.set(cache_key, results, timeout=1800)
                        
                        return results
                    else:
                        print(f"Unexpected response structure: {data}")
                        return []
                        
                elif response.status_code == 429:  # Rate limit exceeded
                    print(f"Rate limit exceeded for section {section}. Using mock data.")
                    
                    # Generate mock articles for this section
                    mock_articles = get_mock_articles_for_section(section)
                    
                    # Transform to match Guardian API structure
                    formatted_mock = []
                    for article in mock_articles:
                        formatted_mock.append({
                            'webTitle': article['title'],
                            'fields': {
                                'headline': article['title'],
                                'shortUrl': article['url']
                            }
                        })
                    
                    # Cache the mock data
                    cache.set(cache_key, formatted_mock, timeout=1800)
                    
                    return formatted_mock
                    
                else:
                    print(f"Error response for section {section}: {response.text}")
                    
                    # Only retry server errors
                    if response.status_code >= 500:
                        retry_count += 1
                        if retry_count < max_retries:
                            print(f"Retrying in {retry_delay} seconds...")
                            time.sleep(retry_delay)
                            retry_delay *= 2
                        continue
                    else:
                        return []
                        
            except requests.exceptions.RequestException as e:
                print(f"Request exception for section {section}: {str(e)}")
                retry_count += 1
                if retry_count < max_retries:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                continue
                
        return []  # Return empty list if all retries fail
        
    except Exception as e:
        print(f"Error fetching news for section {section}: {str(e)}")
        return []

# Add this import near other similar imports
from ai_experiments.transformer_pipeline import generate_market_predictions_for_dashboard as generate_transformer_predictions

# Add this route for transformer predictions API
@app.route('/api/market/transformer-predictions')
@limiter.limit("20 per minute")
def transformer_predictions_api():
    """API endpoint for transformer model market predictions."""
    try:
        predictions_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 
            'ai_experiments/data/transformer_predictions.json'
        )
        
        # Check if predictions file exists and is fresh (less than 12 hours old)
        if os.path.exists(predictions_file):
            file_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(predictions_file))
            if file_age < timedelta(hours=12):
                with open(predictions_file, 'r') as f:
                    return jsonify(json.load(f))
        
        # Generate new predictions if file doesn't exist or is too old
        try:
            # Use a shorter timeout for web requests
            predictions = generate_transformer_predictions()
            return jsonify({
                'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'predictions': predictions
            })
        except Exception as e:
            app.logger.error(f"Error generating transformer predictions: {str(e)}")
            
            # Fallback to file if it exists, even if it's old
            if os.path.exists(predictions_file):
                with open(predictions_file, 'r') as f:
                    return jsonify(json.load(f))
            
            # Generate mock data if no file exists
            return jsonify({
                'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'predictions': generate_mock_transformer_predictions()
            })
    except Exception as e:
        app.logger.error(f"Error in transformer predictions API: {str(e)}")
        return jsonify({
            'error': str(e),
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'predictions': generate_mock_transformer_predictions()
        })

def generate_mock_transformer_predictions():
    """Generate mock transformer predictions for fallback."""
    mock_data = {}
    symbols = {
        'GSPC': '^GSPC', # S&P 500
        'DJI': '^DJI',   # Dow Jones
        'IXIC': '^IXIC'  # NASDAQ
    }
    
    for symbol_key, symbol in symbols.items():
        # Generate random prediction
        current_price = random.uniform(1000, 5000)
        direction = random.choice(['up', 'down'])
        magnitude = random.uniform(0.1, 3.0)
        factor = 1 + (magnitude / 100) if direction == 'up' else 1 - (magnitude / 100)
        predicted_price = current_price * factor
        
        # Generate dates
        today = datetime.now()
        prediction_date = today + timedelta(days=1)
        
        mock_data[symbol_key] = {
            'symbol': symbol,
            'latest_date': today.strftime('%Y-%m-%d'),
            'latest_close': float(current_price),
            'prediction_dates': [
                prediction_date.strftime('%Y-%m-%d')
            ],
            'predicted_prices': [float(predicted_price)],
            'direction': direction,
            'magnitude': float(magnitude),
            'confidence': random.uniform(0.6, 0.9),
            'model_type': 'transformer'
        }
    
    return mock_data

# Add this import near other similar imports
from ai_experiments.alternative_data_sources import (
    get_entity_sentiment, 
    get_reddit_sentiment,
    get_retail_satellite_data,
    get_agricultural_satellite_data
)

# Add this as a new route
@app.route('/api/alternative-data/news-sentiment')
@limiter.limit("20 per minute")
def news_sentiment_api():
    """API endpoint for news sentiment from web scraping."""
    try:
        # Try to get data from cache with 6-hour expiry
        entity_sentiment = get_entity_sentiment(max_age_hours=6)
        
        # Return sentiment data
        return jsonify({
            'generated_at': datetime.now().isoformat(),
            'entities': entity_sentiment
        })
    except Exception as e:
        app.logger.error(f"Error getting news sentiment data: {str(e)}")
        return jsonify({
            'error': str(e),
            'generated_at': datetime.now().isoformat(),
            'entities': {}
        })

@app.route('/api/alternative-data/reddit-sentiment')
@limiter.limit("20 per minute")
def reddit_sentiment_api():
    """API endpoint for Reddit sentiment analysis."""
    try:
        # Get Reddit client credentials from environment variables
        reddit_client_id = os.environ.get('REDDIT_CLIENT_ID')
        reddit_client_secret = os.environ.get('REDDIT_CLIENT_SECRET')
        reddit_user_agent = os.environ.get('REDDIT_USER_AGENT', 'web:financial-dashboard:v1.0 (by /u/your_username)')
        
        # Get Reddit sentiment data with 6-hour expiry
        reddit_data = get_reddit_sentiment(
            reddit_client_id=reddit_client_id,
            reddit_client_secret=reddit_client_secret,
            reddit_user_agent=reddit_user_agent,
            max_age_hours=6
        )
        
        return jsonify(reddit_data)
    except Exception as e:
        app.logger.error(f"Error getting Reddit sentiment data: {str(e)}")
        return jsonify({
            'error': str(e),
            'generated_at': datetime.now().isoformat(),
            'entities': {},
            'subreddits': {}
        })

@app.route('/api/alternative-data/retail-satellite')
@limiter.limit("20 per minute")
def retail_satellite_api():
    """API endpoint for retail satellite imagery analysis."""
    try:
        # Get satellite API key from environment variables
        satellite_api_key = os.environ.get('SATELLITE_API_KEY')
        
        # Use mock data by default
        use_mock = request.args.get('use_mock', 'true').lower() == 'true'
        
        # Get retail satellite data
        retail_data = get_retail_satellite_data(
            api_key=satellite_api_key,
            use_mock=use_mock
        )
        
        return jsonify(retail_data)
    except Exception as e:
        app.logger.error(f"Error getting retail satellite data: {str(e)}")
        return jsonify({
            'error': str(e),
            'generated_at': datetime.now().isoformat(),
            'locations': {}
        })

@app.route('/api/alternative-data/agricultural-satellite')
@limiter.limit("20 per minute")
def agricultural_satellite_api():
    """API endpoint for agricultural satellite imagery analysis."""
    try:
        # Get satellite API key from environment variables
        satellite_api_key = os.environ.get('SATELLITE_API_KEY')
        
        # Use mock data by default
        use_mock = request.args.get('use_mock', 'true').lower() == 'true'
        
        # Get agricultural satellite data
        agricultural_data = get_agricultural_satellite_data(
            api_key=satellite_api_key,
            use_mock=use_mock
        )
        
        return jsonify(agricultural_data)
    except Exception as e:
        app.logger.error(f"Error getting agricultural satellite data: {str(e)}")
        return jsonify({
            'error': str(e),
            'generated_at': datetime.now().isoformat(),
            'regions': {}
        })

@app.route('/api/alternative-data/summary')
@limiter.limit("20 per minute")
def alternative_data_summary_api():
    """API endpoint for a summary of all alternative data sources."""
    try:
        # Get data from all sources
        entity_sentiment = get_entity_sentiment(max_age_hours=6)
        
        # Get Reddit client credentials
        reddit_client_id = os.environ.get('REDDIT_CLIENT_ID')
        reddit_client_secret = os.environ.get('REDDIT_CLIENT_SECRET')
        reddit_user_agent = os.environ.get('REDDIT_USER_AGENT', 'web:financial-dashboard:v1.0 (by /u/your_username)')
        
        # Get Reddit sentiment data
        reddit_data = get_reddit_sentiment(
            reddit_client_id=reddit_client_id,
            reddit_client_secret=reddit_client_secret,
            reddit_user_agent=reddit_user_agent,
            max_age_hours=6
        )
        
        # Get satellite API key
        satellite_api_key = os.environ.get('SATELLITE_API_KEY')
        
        # Use mock data for satellite
        retail_data = get_retail_satellite_data(
            api_key=satellite_api_key,
            use_mock=True
        )
        
        agricultural_data = get_agricultural_satellite_data(
            api_key=satellite_api_key,
            use_mock=True
        )
        
        # Create summary data
        summary = {
            'generated_at': datetime.now().isoformat(),
            'sentiment_analysis': {
                'news': {
                    'source_count': len(entity_sentiment) if entity_sentiment else 0,
                    'top_positive': _get_top_sentiment_entities(entity_sentiment, 'positive', 3),
                    'top_negative': _get_top_sentiment_entities(entity_sentiment, 'negative', 3)
                },
                'social': {
                    'analyzed_posts': reddit_data.get('analyzed_posts', 0),
                    'analyzed_comments': reddit_data.get('analyzed_comments', 0),
                    'top_entities': _get_top_reddit_entities(reddit_data, 5)
                }
            },
            'satellite_data': {
                'retail': {
                    'locations_count': len(retail_data.get('locations', {})),
                    'high_traffic_locations': _get_high_traffic_retail(retail_data, 3),
                    'stock_impact': _get_retail_stock_impact(retail_data, 3)
                },
                'agricultural': {
                    'regions_count': len(agricultural_data.get('regions', {})),
                    'yield_changes': _get_agricultural_yield_changes(agricultural_data, 3),
                    'price_impact': _get_agricultural_price_impact(agricultural_data, 3)
                }
            }
        }
        
        return jsonify(summary)
    except Exception as e:
        app.logger.error(f"Error getting alternative data summary: {str(e)}")
        return jsonify({
            'error': str(e),
            'generated_at': datetime.now().isoformat()
        })

# Helper functions for alternative data summary
def _get_top_sentiment_entities(entity_sentiment, sentiment_type, count=3):
    """Get top entities by sentiment."""
    if not entity_sentiment:
        return []
    
    # For positive sentiment, sort by avg_sentiment desc
    # For negative sentiment, sort by avg_sentiment asc
    entities = []
    for symbol, data in entity_sentiment.items():
        if data.get('article_count', 0) > 0:
            entities.append({
                'symbol': symbol,
                'avg_sentiment': data.get('avg_sentiment', 0),
                'article_count': data.get('article_count', 0)
            })
    
    if sentiment_type == 'positive':
        sorted_entities = sorted(entities, key=lambda x: x['avg_sentiment'], reverse=True)
    else:
        sorted_entities = sorted(entities, key=lambda x: x['avg_sentiment'])
    
    return sorted_entities[:count]

def _get_top_reddit_entities(reddit_data, count=5):
    """Get top entities from Reddit data by mention count."""
    if not reddit_data or 'entities' not in reddit_data:
        return []
    
    entities = []
    for symbol, data in reddit_data.get('entities', {}).items():
        if data.get('mentions', 0) > 0:
            entities.append({
                'symbol': symbol,
                'mentions': data.get('mentions', 0),
                'avg_sentiment': data.get('avg_sentiment', 0)
            })
    
    sorted_entities = sorted(entities, key=lambda x: x['mentions'], reverse=True)
    return sorted_entities[:count]

def _get_high_traffic_retail(retail_data, count=3):
    """Get retail locations with highest traffic."""
    if not retail_data or 'locations' not in retail_data:
        return []
    
    locations = []
    for location_id, data in retail_data.get('locations', {}).items():
        occupancy = data.get('analysis', {}).get('occupied_percentage', 0)
        locations.append({
            'name': data.get('name', ''),
            'ticker': data.get('ticker', ''),
            'occupancy': occupancy,
            'category': data.get('analysis', {}).get('traffic_category', '')
        })
    
    sorted_locations = sorted(locations, key=lambda x: x['occupancy'], reverse=True)
    return sorted_locations[:count]

def _get_retail_stock_impact(retail_data, count=3):
    """Get retail stock impact data."""
    if not retail_data or 'locations' not in retail_data:
        return []
    
    impacts = []
    for location_id, data in retail_data.get('locations', {}).items():
        if 'stock_impact' in data:
            impacts.append({
                'name': data.get('name', ''),
                'ticker': data.get('stock_impact', {}).get('ticker', ''),
                'traffic_change': data.get('stock_impact', {}).get('traffic_change', 0),
                'estimated_impact': data.get('stock_impact', {}).get('estimated_impact', 0)
            })
    
    # Sort by absolute estimated impact
    sorted_impacts = sorted(impacts, key=lambda x: abs(x['estimated_impact']), reverse=True)
    return sorted_impacts[:count]

def _get_agricultural_yield_changes(agricultural_data, count=3):
    """Get agricultural regions with largest yield changes."""
    if not agricultural_data or 'regions' not in agricultural_data:
        return []
    
    regions = []
    for region_id, data in agricultural_data.get('regions', {}).items():
        yield_change = data.get('analysis', {}).get('yield_change', 0)
        regions.append({
            'name': data.get('name', ''),
            'crop': data.get('crop', ''),
            'ticker': data.get('ticker', ''),
            'yield_change': yield_change,
            'crop_health': data.get('analysis', {}).get('crop_health', '')
        })
    
    # Sort by absolute yield change
    sorted_regions = sorted(regions, key=lambda x: abs(x['yield_change']), reverse=True)
    return sorted_regions[:count]

def _get_agricultural_price_impact(agricultural_data, count=3):
    """Get agricultural price impact data."""
    if not agricultural_data or 'regions' not in agricultural_data:
        return []
    
    impacts = []
    for region_id, data in agricultural_data.get('regions', {}).items():
        if 'price_impact' in data:
            impacts.append({
                'name': data.get('name', ''),
                'crop': data.get('crop', ''),
                'ticker': data.get('ticker', ''),
                'price_change': data.get('price_impact', {}).get('price_change_percent', 0),
                'direction': data.get('price_impact', {}).get('price_direction', ''),
                'confidence': data.get('price_impact', {}).get('confidence', 0)
            })
    
    # Sort by absolute price change
    sorted_impacts = sorted(impacts, key=lambda x: abs(x['price_change']), reverse=True)
    return sorted_impacts[:count]

if __name__ == '__main__':
    # Determine if debug mode should be on. By default, it's off.
    # Set FLASK_DEBUG=1 (or "true"/"on") in your development environment.
    debug_mode = os.environ.get("FLASK_DEBUG", "0").lower() in ("1", "true", "on")
    
    # Optionally, read the port from an environment variable.
    port = int(os.environ.get("PORT", "5001"))
    
    # Run the Flask app with the appropriate settings.
    app.run(host='0.0.0.0', port=port, debug=debug_mode)
