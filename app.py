import os
import sqlite3
import random
import requests
import datetime

from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'CHANGE_ME_TO_SOMETHING_SECURE'  # Replace with your own secret key.

DB_NAME = 'data/users.db'

# --- Hard-coded API Keys ---
OWM_API_KEY = "1fef9413c0c77c739ef23d222e05db76"
ALPHA_VANTAGE_KEY = "C60CDZU1P1BU27RK"


def init_db():
    """Initialize the DB, ensuring 'users' table has city_name & stock_symbol columns."""
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
        # Add stock_symbol column if not present
        try:
            conn.execute("ALTER TABLE users ADD COLUMN stock_symbol TEXT")
        except sqlite3.OperationalError:
            pass

    print("Database initialized or updated.")


init_db()


# -----------------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------------

def get_user_settings(username):
    """
    Retrieves the city_name and stock_symbol for the given user.
    Returns (city_name, stock_symbol) or defaults ("New York", "AAPL") if blank.
    """
    default_city = "New York"
    default_stock = "AAPL"
    with sqlite3.connect(DB_NAME) as conn:
        cur = conn.cursor()
        cur.execute("SELECT city_name, stock_symbol FROM users WHERE username=?", (username,))
        row = cur.fetchone()
    if row:
        user_city, user_stock = row
        user_city = user_city if user_city else default_city
        user_stock = user_stock if user_stock else default_stock
        return (user_city, user_stock)
    else:
        return (default_city, default_stock)


def get_coordinates_for_city(city_name):
    """
    Uses OpenWeatherMap Geocoding API to get (lat, lon) for a city (hard-coded OWM_API_KEY).
    Returns (lat, lon) or (None, None) on failure.
    """
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


def get_weekly_forecast(lat, lon):
    """
    Calls OWM One Call API for daily forecast (up to 7 days) given lat/lon.
    Returns the first 5 of those days as a list of daily forecast dicts, each with:
      - date_str (string like "Jan 25")
      - icon_url
      - description
      - temp_min
      - temp_max
    or an empty list on failure.
    """
    url = (
        f"http://api.openweathermap.org/data/3.0/onecall"
        f"?lat={lat}&lon={lon}&exclude=current,minutely,hourly,alerts"
        f"&units=imperial&appid={OWM_API_KEY}"
    )
    forecast_list = []
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        if "daily" in data:
            # Slice to first 5 days
            daily_data = data["daily"][:5]
            for day in daily_data:
                dt = day["dt"]  # Unix timestamp
                date_str = datetime.datetime.utcfromtimestamp(dt).strftime("%b %d")
                # Weather icon & desc
                icon_code = day["weather"][0]["icon"]
                icon_url = f"https://openweathermap.org/img/wn/{icon_code}@2x.png"
                description = day["weather"][0].get("description", "")
                # Temps
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


def get_stock_price(symbol):
    """
    Fetches the current stock price using Alpha Vantage (GLOBAL_QUOTE).
    Hard-coded ALPHA_VANTAGE_KEY = 'C60CDZU1P1BU27RK'.
    Returns a float (e.g. 124.56) or None on failure.
    """
    base_url = "https://www.alphavantage.co/query"
    params = {
        "function": "GLOBAL_QUOTE",
        "symbol": symbol,
        "apikey": ALPHA_VANTAGE_KEY
    }
    try:
        r = requests.get(base_url, params=params, timeout=5)
        r.raise_for_status()
        data = r.json()
        # Data structure is typically: { "Global Quote": { "01. symbol": "AAPL", "05. price": "130.48", ... }}
        global_quote = data.get("Global Quote", {})
        price_str = global_quote.get("05. price")
        if price_str:
            return round(float(price_str), 2)
    except Exception as e:
        print(f"[ERROR] Stock fetch for {symbol} failed: {e}")
    return None


# ADDED: New helper to get historical stock data
def get_stock_history(symbol, days=30):
    """
    Fetch daily historical stock data for the given symbol from Alpha Vantage.
    Returns a dict with:
       {
         'dates': [date1, date2, ...],
         'prices': [price1, price2, ...]
       }
    for the last `days` trading days (if available).
    """
    base_url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": symbol,
        "apikey": ALPHA_VANTAGE_KEY,
        "outputsize": "compact",  # ~100 recent data points
    }
    try:
        r = requests.get(base_url, params=params, timeout=5)
        r.raise_for_status()
        data = r.json()

        time_series = data.get("Time Series (Daily)", {})
        rows = []
        for date_str, daily_data in time_series.items():
            close_price = float(daily_data["4. close"])
            rows.append((date_str, close_price))

        # Sort rows by date ascending
        rows.sort(key=lambda x: x[0])

        # If you only want the most recent `days`, slice from the end
        if len(rows) > days:
            rows = rows[-days:]

        dates = [r[0] for r in rows]
        prices = [r[1] for r in rows]
        return {"dates": dates, "prices": prices}

    except Exception as e:
        print(f"[ERROR] Historical stock fetch for {symbol} failed: {e}")
        return {"dates": [], "prices": []}


# -----------------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------------

@app.route('/')
def home():
    """
    Home page, requires user login.
    Displays:
      - random quote
      - random joke
      - random dog picture
      - stock quote (Alpha Vantage)
      - 7-day weather forecast (OWM One Call)
      - stock price chart (Chart.js)
    """
    if 'user' not in session:
        return redirect(url_for('login'))

    username = session['user']
    city_name, stock_symbol = get_user_settings(username)

    # Prepare variables for template
    quote_text = quote_author = None
    joke_setup = joke_punchline = None
    dog_image_url = None
    stock_price = None
    daily_forecasts = []

    # 1) Random Quote
    try:
        r = requests.get("https://type.fit/api/quotes", timeout=5)
        r.raise_for_status()
        all_quotes = r.json()
        if all_quotes:
            chosen_quote = random.choice(all_quotes)
            quote_text = chosen_quote.get("text", "")
            quote_author = chosen_quote.get("author", "Unknown")
    except Exception as e:
        print(f"[ERROR] Quote fetch failed: {e}")

    # 2) Random Joke
    try:
        joke_r = requests.get("https://official-joke-api.appspot.com/random_joke", timeout=5)
        joke_r.raise_for_status()
        jdata = joke_r.json()
        joke_setup = jdata.get("setup")
        joke_punchline = jdata.get("punchline")
    except Exception as e:
        print(f"[ERROR] Joke fetch failed: {e}")

    # 3) Random Dog Picture
    try:
        dog_r = requests.get("https://dog.ceo/api/breeds/image/random", timeout=5)
        dog_r.raise_for_status()
        ddata = dog_r.json()
        dog_image_url = ddata.get("message")
    except Exception as e:
        print(f"[ERROR] Dog fetch failed: {e}")

    # 4) Current Stock Quote
    stock_price = get_stock_price(stock_symbol)

    # 4b) Historical Data for Chart.js
    stock_history = get_stock_history(stock_symbol, days=30)

    # 5) 7-day Weather Forecast
    lat, lon = get_coordinates_for_city(city_name)
    if lat is not None and lon is not None:
        daily_forecasts = get_weekly_forecast(lat, lon)
    else:
        print(f"[WARN] Could not get lat/lon for city '{city_name}'. Weather unavailable.")

    return render_template(
        "index.html",
        user=username,
        city_name=city_name,

        quote_text=quote_text,
        quote_author=quote_author,
        joke_setup=joke_setup,
        joke_punchline=joke_punchline,
        dog_image_url=dog_image_url,

        stock_symbol=stock_symbol,
        stock_price=stock_price,
        # Pass the new historical data:
        stock_history=stock_history,

        daily_forecasts=daily_forecasts
    )


@app.route('/settings', methods=['GET', 'POST'])
def settings():
    """User can set their city_name & stock_symbol."""
    if 'user' not in session:
        return redirect(url_for('login'))

    username = session['user']
    if request.method == 'POST':
        new_city = request.form.get('city_name', '').strip()
        new_stock = request.form.get('stock_symbol', '').strip()

        with sqlite3.connect(DB_NAME) as conn:
            cur = conn.cursor()
            cur.execute("""
                UPDATE users
                SET city_name = ?, stock_symbol = ?
                WHERE username = ?
            """, (new_city, new_stock, username))
            conn.commit()

        flash("Settings updated!", "success")
        return redirect(url_for('home'))
    else:
        # Show current settings
        city_name, stock_symbol = get_user_settings(username)
        return render_template('settings.html', city_name=city_name, stock_symbol=stock_symbol)


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
                session['user'] = username
                flash("Login successful!", "success")
                return redirect(url_for('home'))
        flash("Invalid credentials. Please try again.", "danger")
    return render_template('login.html')


@app.route('/logout')
def logout():
    """Logout route."""
    session.pop('user', None)
    flash("Logged out successfully.", "info")
    return redirect(url_for('login'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    """Registration route."""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if not username or not password:
            flash("Username and password required.", "danger")
            return redirect(url_for('register'))

        pw_hash = generate_password_hash(password)
        try:
            with sqlite3.connect(DB_NAME) as conn:
                cur = conn.cursor()
                cur.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)",
                            (username, pw_hash))
                conn.commit()
            flash("Registration successful. You can now log in.", "success")
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash("Username already exists. Choose another.", "danger")
    return render_template('register.html')


if __name__ == '__main__':
    # Run on port 5001 instead of 5000
    app.run(debug=True, host='0.0.0.0', port=5001)