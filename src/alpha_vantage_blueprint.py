from flask import Blueprint, render_template, jsonify, request, session, redirect, url_for, current_app, flash
import os
import requests
from datetime import datetime, timedelta
from functools import wraps
import logging
import json
from hashlib import md5
import time
# Removed direct import of cache to avoid circular imports

# Define the blueprint
alpha_vantage_bp = Blueprint('alpha_vantage', __name__, url_prefix='/alpha_vantage')

# Get API key from environment
ALPHA_VANTAGE_API_KEY = os.environ.get("ALPHA_VANTAGE_API_KEY")

# Cache durations for different types of data
CACHE_DURATIONS = {
    'TIME_SERIES_INTRADAY': 60 * 5,          # 5 minutes
    'TIME_SERIES_DAILY': 60 * 60,            # 1 hour
    'TIME_SERIES_WEEKLY': 60 * 60 * 24,        # 24 hours
    'TIME_SERIES_MONTHLY': 60 * 60 * 24,       # 24 hours
    'GLOBAL_QUOTE': 60 * 5,                  # 5 minutes
    'SYMBOL_SEARCH': 60 * 60 * 24,            # 24 hours
    'OVERVIEW': 60 * 60 * 24,                # 24 hours
    'INCOME_STATEMENT': 86400,               # 1 day
    'BALANCE_SHEET': 86400,                  # 1 day
    'CASH_FLOW': 86400,                      # 1 day
    'EARNINGS': 86400,                       # 1 day
    'LISTING_STATUS': 86400,                 # 1 day
    'EARNINGS_CALENDAR': 86400,              # 1 day
    'IPO_CALENDAR': 86400,                   # 1 day
    'COMPANY_OVERVIEW': 86400,               # 1 day
    'OPTIONS': 86400,                        # 1 day
    'NEWS_SENTIMENT': 86400,                 # 1 day
    'TOP_GAINERS_LOSERS': 86400,             # 1 day
    'INSIDER_TRANSACTIONS': 86400,          # 1 day
    'FX_INTRADAY': 86400,                    # 1 day
    'FX_DAILY': 86400,                       # 1 day
    'FX_WEEKLY': 86400,                       # 1 day
    'FX_MONTHLY': 86400,                      # 1 day
    'CURRENCY_EXCHANGE_RATE': 86400,          # 1 day
    'CRYPTO_INTRADAY': 86400,                 # 1 day
    'DIGITAL_CURRENCY_DAILY': 86400,          # 1 day
    'DIGITAL_CURRENCY_WEEKLY': 86400,         # 1 day
    'DIGITAL_CURRENCY_MONTHLY': 86400,        # 1 day
    'WTI': 86400,                              # 1 day
    'BRENT': 86400,                             # 1 day
    'NATURAL_GAS': 86400,                      # 1 day
    'COPPER': 86400,                            # 1 day
    'ALUMINUM': 86400,                          # 1 day
    'WHEAT': 86400,                              # 1 day
    'CORN': 86400,                               # 1 day
    'COTTON': 86400,                             # 1 day
    'SUGAR': 86400,                              # 1 day
    'COFFEE': 86400,                             # 1 day
    'REAL_GDP': 86400,                           # 1 day
    'REAL_GDP_PER_CAPITA': 86400,                # 1 day
    'TREASURY_YIELD': 86400,                     # 1 day
    'FEDERAL_FUNDS_RATE': 86400,                  # 1 day
    'CPI': 86400,                                 # 1 day
    'INFLATION': 86400,                           # 1 day
    'RETAIL_SALES': 86400,                         # 1 day
    'DURABLES': 86400,                              # 1 day
    'UNEMPLOYMENT': 86400,                           # 1 day
    'NONFARM_PAYROLL': 86400,                         # 1 day
    'SMA': 86400,                                    # 1 day
    'EMA': 86400,                                     # 1 day
    'WMA': 86400,                                     # 1 day
    'DEMA': 86400,                                     # 1 day
    'TEMA': 86400,                                     # 1 day
    'TRIMA': 86400,                                     # 1 day
    'KAMA': 86400,                                     # 1 day
    'MAMA': 86400,                                     # 1 day
    'T3': 86400,                                         # 1 day
    'MACD': 86400,                                         # 1 day
    'STOCH': 86400,                                         # 1 day
    'RSI': 86400,                                             # 1 day
    'ADX': 86400,                                                 # 1 day
    'CCI': 86400,                                                 # 1 day
    'AROON': 86400,                                                 # 1 day
    'BBANDS': 86400,                                                 # 1 day
    'AD': 86400,                                                     # 1 day
    'OBV': 86400,                                                     # 1 day
    'default': 60 * 15  # 15 minutes default
}

# Enhanced API documentation with examples
API_DOCUMENTATION = {
    'TIME_SERIES_INTRADAY': {
        'description': 'Returns intraday time series of the equity specified, covering extended trading hours where applicable.',
        'examples': [
            {'params': {'symbol': 'MSFT', 'interval': '5min'}, 'description': 'Get 5-minute intraday data for Microsoft'},
            {'params': {'symbol': 'AAPL', 'interval': '15min', 'outputsize': 'full'}, 'description': 'Get full 15-minute intraday data for Apple'}
        ],
        'notes': 'Data is delayed by 15 minutes for free API keys. Premium API keys provide real-time intraday data.'
    },
    'TIME_SERIES_DAILY': {
        'description': 'Returns daily time series of the equity specified, covering 20+ years of historical data.',
        'examples': [
            {'params': {'symbol': 'IBM'}, 'description': 'Get daily data for IBM (default compact)'},
            {'params': {'symbol': 'MSFT', 'outputsize': 'full'}, 'description': 'Get full daily data for Microsoft (20+ years)'}
        ],
        'notes': 'The "compact" outputsize (default) returns the latest 100 data points, while "full" returns up to 20+ years of historical data.'
    },
    'GLOBAL_QUOTE': {
        'description': 'Returns the latest price and volume information for a security of your choice.',
        'examples': [
            {'params': {'symbol': 'AAPL'}, 'description': 'Get the latest price information for Apple'}
        ],
        'notes': 'This endpoint provides a lightweight response with just the latest price data, ideal for quick price checks.'
    },
    'SYMBOL_SEARCH': {
        'description': 'Returns best-matching symbols and market information based on keywords of your choice.',
        'examples': [
            {'params': {'keywords': 'microsoft'}, 'description': 'Search for Microsoft related securities'},
            {'params': {'keywords': 'BA'}, 'description': 'Search for securities matching "BA" (Boeing, etc.)'}
        ],
        'notes': 'Search results include symbols from multiple exchanges globally.'
    }
}

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@alpha_vantage_bp.route('/browser') # type: ignore
@login_required
def browser():
    """Main page for the Alpha Vantage API browser."""
    return render_template('alpha_vantage_browser.html')

def get_cache_key(function_name, params):
    """Generate a unique cache key for an API request."""
    # Sort params to ensure consistent keys regardless of param order
    sorted_params = dict(sorted(params.items()))
    # Create a string representation and hash it
    param_str = json.dumps(sorted_params)
    return f"alphavantage_{function_name}_{md5(param_str.encode()).hexdigest()}"

def get_cache_duration(function_name):
    """Get cache duration for the given function"""
    # Direct match
    if function_name in CACHE_DURATIONS:
        return CACHE_DURATIONS[function_name]
        
    # Return default cache duration
    return CACHE_DURATIONS['default']

@alpha_vantage_bp.route('/api/<category>/<function_name>', methods=['GET']) # type: ignore
@login_required
def api_call(category, function_name):
    """Make request to Alpha Vantage API with caching"""
    # Check if function exists
    if category not in API_FUNCTIONS or function_name not in API_FUNCTIONS[category]:
        return jsonify({
            'error': 'Invalid function',
            'message': f'Function {function_name} not found in category {category}'
        }), 400

    # Get parameters from request
    params = {k: v for k, v in request.args.items() if k != 'apikey'}
    
    # Add required API key
    alpha_vantage_api_key = current_app.config.get('ALPHA_VANTAGE_API_KEY')
    if not alpha_vantage_api_key:
        return jsonify({
            'error': 'API key not configured',
            'message': 'Alpha Vantage API key is not configured in the application.'
        }), 500
    
    params['apikey'] = alpha_vantage_api_key
    
    # Check for required parameters
    function_params = API_FUNCTIONS[category][function_name]['params']
    missing_params = [param for param, details in function_params.items() 
                     if details.get('required', False) and param not in params]
    
    if missing_params:
        return jsonify({
            'error': 'Missing parameters',
            'message': f'Required parameters missing: {", ".join(missing_params)}'
        }), 400

    # Generate cache key
    cache_key = get_cache_key(function_name, params)
    
    # Check cache first
    cached_result = current_app.cache.get(cache_key)
    if cached_result:
        # Add cache metadata to response
        if isinstance(cached_result, dict):
            cached_result['_cached'] = True
            cached_result['_cache_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        return jsonify(cached_result)
    
    # If not in cache, make API request
    url = 'https://www.alphavantage.co/query'
    params['function'] = function_name
    
    try:
        response = requests.get(url, params=params, timeout=10)
        
        # Check if rate limit reached
        data = response.json()
        if 'Note' in data and 'Thank you for using Alpha Vantage' in data['Note']:
            return jsonify({
                'error': 'Rate limit exceeded',
                'message': data['Note'],
                'retry_after': 60,
                'status': 429
            }), 429
            
        # Store in cache if successful
        if response.status_code == 200:
            cache_duration = get_cache_duration(function_name)
            current_app.cache.set(cache_key, data, timeout=cache_duration)
            
        return jsonify(data), response.status_code
        
    except requests.exceptions.RequestException as e:
        current_app.logger.error(f"Request failed: {e}")
        return jsonify({
            'error': 'Request failed',
            'message': 'An internal error has occurred. Please try again later.',
            'status': 500
        }), 500

# API Categories and their functions
API_FUNCTIONS = {
    'stock': {
        'TIME_SERIES_INTRADAY': {
            'params': ['symbol', 'interval', 'adjusted', 'extended_hours', 'month', 'outputsize'],
            'description': 'Intraday time series of the equity specified'
        },
        'TIME_SERIES_DAILY': {
            'params': ['symbol', 'outputsize'],
            'description': 'Daily time series of the equity specified'
        },
        'TIME_SERIES_DAILY_ADJUSTED': {
            'params': ['symbol', 'outputsize'],
            'description': 'Daily time series with split/dividend-adjusted data'
        },
        'TIME_SERIES_WEEKLY': {
            'params': ['symbol'],
            'description': 'Weekly time series of the equity specified'
        },
        'TIME_SERIES_WEEKLY_ADJUSTED': {
            'params': ['symbol'],
            'description': 'Weekly time series with split/dividend-adjusted data'
        },
        'TIME_SERIES_MONTHLY': {
            'params': ['symbol'],
            'description': 'Monthly time series of the equity specified'
        },
        'TIME_SERIES_MONTHLY_ADJUSTED': {
            'params': ['symbol'],
            'description': 'Monthly time series with split/dividend-adjusted data'
        },
        'GLOBAL_QUOTE': {
            'params': ['symbol'],
            'description': 'Latest price and volume information'
        },
        'SYMBOL_SEARCH': {
            'params': ['keywords'],
            'description': 'Search for best-matching symbols based on keywords'
        },
        'MARKET_STATUS': {
            'params': [],
            'description': 'Current market status for global exchanges'
        }
    },
    'options': {
        'OPTIONS': {
            'params': ['symbol', 'date', 'strike', 'option_type'],
            'description': 'Options data for a specific equity'
        }
    },
    'intelligence': {
        'NEWS_SENTIMENT': {
            'params': ['tickers', 'topics', 'time_from', 'time_to', 'sort', 'limit'],
            'description': 'News and sentiment data for symbols or topics'
        },
        'TOP_GAINERS_LOSERS': {
            'params': [],
            'description': 'Top gaining and losing US stocks'
        },
        'INSIDER_TRANSACTIONS': {
            'params': ['symbol'],
            'description': 'Insider transactions for a specific company'
        }
    },
    'fundamental': {
        'OVERVIEW': {
            'params': ['symbol'],
            'description': 'Company information, financial ratios, and other key metrics'
        },
        'INCOME_STATEMENT': {
            'params': ['symbol'],
            'description': 'Annual and quarterly income statements'
        },
        'BALANCE_SHEET': {
            'params': ['symbol'],
            'description': 'Annual and quarterly balance sheets'
        },
        'CASH_FLOW': {
            'params': ['symbol'],
            'description': 'Annual and quarterly cash flows'
        },
        'EARNINGS': {
            'params': ['symbol'],
            'description': 'Annual and quarterly earnings (EPS)'
        },
        'LISTING_STATUS': {
            'params': ['date'],
            'description': 'Active and delisted US stocks'
        },
        'EARNINGS_CALENDAR': {
            'params': ['symbol', 'horizon'],
            'description': 'Earnings calendar for public companies'
        },
        'IPO_CALENDAR': {
            'params': [],
            'description': 'Upcoming IPO calendar'
        },
        'COMPANY_OVERVIEW': {
            'params': ['symbol'],
            'description': 'Detailed company information'
        }
    },
    'forex': {
        'FX_INTRADAY': {
            'params': ['from_symbol', 'to_symbol', 'interval'],
            'description': 'Intraday time series of forex rates'
        },
        'FX_DAILY': {
            'params': ['from_symbol', 'to_symbol'],
            'description': 'Daily time series of forex rates'
        },
        'FX_WEEKLY': {
            'params': ['from_symbol', 'to_symbol'],
            'description': 'Weekly time series of forex rates'
        },
        'FX_MONTHLY': {
            'params': ['from_symbol', 'to_symbol'],
            'description': 'Monthly time series of forex rates'
        },
        'CURRENCY_EXCHANGE_RATE': {
            'params': ['from_currency', 'to_currency'],
            'description': 'Realtime exchange rate for any pair of currencies'
        }
    },
    'crypto': {
        'CRYPTO_INTRADAY': {
            'params': ['symbol', 'market', 'interval'],
            'description': 'Intraday time series of cryptocurrency'
        },
        'DIGITAL_CURRENCY_DAILY': {
            'params': ['symbol', 'market'],
            'description': 'Daily time series of cryptocurrency'
        },
        'DIGITAL_CURRENCY_WEEKLY': {
            'params': ['symbol', 'market'],
            'description': 'Weekly time series of cryptocurrency'
        },
        'DIGITAL_CURRENCY_MONTHLY': {
            'params': ['symbol', 'market'],
            'description': 'Monthly time series of cryptocurrency'
        }
    },
    'commodities': {
        'WTI': {
            'params': ['interval'],
            'description': 'Crude Oil (WTI) prices'
        },
        'BRENT': {
            'params': ['interval'],
            'description': 'Crude Oil (Brent) prices'
        },
        'NATURAL_GAS': {
            'params': ['interval'],
            'description': 'Natural Gas prices'
        },
        'COPPER': {
            'params': ['interval'],
            'description': 'Copper prices'
        },
        'ALUMINUM': {
            'params': ['interval'],
            'description': 'Aluminum prices'
        },
        'WHEAT': {
            'params': ['interval'],
            'description': 'Wheat prices'
        },
        'CORN': {
            'params': ['interval'],
            'description': 'Corn prices'
        },
        'COTTON': {
            'params': ['interval'],
            'description': 'Cotton prices'
        },
        'SUGAR': {
            'params': ['interval'],
            'description': 'Sugar prices'
        },
        'COFFEE': {
            'params': ['interval'],
            'description': 'Coffee prices'
        }
    },
    'economic': {
        'REAL_GDP': {
            'params': ['interval'],
            'description': 'Real Gross Domestic Product (GDP)'
        },
        'REAL_GDP_PER_CAPITA': {
            'params': [],
            'description': 'Real GDP per Capita'
        },
        'TREASURY_YIELD': {
            'params': ['interval', 'maturity'],
            'description': 'Treasury Yield'
        },
        'FEDERAL_FUNDS_RATE': {
            'params': ['interval'],
            'description': 'Federal Funds Rate'
        },
        'CPI': {
            'params': ['interval'],
            'description': 'Consumer Price Index (CPI)'
        },
        'INFLATION': {
            'params': ['interval'],
            'description': 'Inflation Rates'
        },
        'RETAIL_SALES': {
            'params': ['interval'],
            'description': 'Retail Sales'
        },
        'DURABLES': {
            'params': ['interval'],
            'description': 'Durable Goods Orders'
        },
        'UNEMPLOYMENT': {
            'params': ['interval'],
            'description': 'Unemployment Rate'
        },
        'NONFARM_PAYROLL': {
            'params': ['interval'],
            'description': 'Nonfarm Payroll'
        }
    },
    'technical': {
        'SMA': {
            'params': ['symbol', 'interval', 'time_period', 'series_type'],
            'description': 'Simple Moving Average (SMA)'
        },
        'EMA': {
            'params': ['symbol', 'interval', 'time_period', 'series_type'],
            'description': 'Exponential Moving Average (EMA)'
        },
        'WMA': {
            'params': ['symbol', 'interval', 'time_period', 'series_type'],
            'description': 'Weighted Moving Average (WMA)'
        },
        'DEMA': {
            'params': ['symbol', 'interval', 'time_period', 'series_type'],
            'description': 'Double Exponential Moving Average (DEMA)'
        },
        'TEMA': {
            'params': ['symbol', 'interval', 'time_period', 'series_type'],
            'description': 'Triple Exponential Moving Average (TEMA)'
        },
        'TRIMA': {
            'params': ['symbol', 'interval', 'time_period', 'series_type'],
            'description': 'Triangular Moving Average (TRIMA)'
        },
        'KAMA': {
            'params': ['symbol', 'interval', 'time_period', 'series_type'],
            'description': 'Kaufman Adaptive Moving Average (KAMA)'
        },
        'MAMA': {
            'params': ['symbol', 'interval', 'series_type', 'fastlimit', 'slowlimit'],
            'description': 'MESA Adaptive Moving Average (MAMA)'
        },
        'T3': {
            'params': ['symbol', 'interval', 'time_period', 'series_type'],
            'description': 'Triple Exponential Moving Average (T3)'
        },
        'MACD': {
            'params': ['symbol', 'interval', 'series_type', 'fastperiod', 'slowperiod', 'signalperiod'],
            'description': 'Moving Average Convergence/Divergence (MACD)'
        },
        'STOCH': {
            'params': ['symbol', 'interval', 'fastkperiod', 'slowkperiod', 'slowdperiod'],
            'description': 'Stochastic Oscillator'
        },
        'RSI': {
            'params': ['symbol', 'interval', 'time_period', 'series_type'],
            'description': 'Relative Strength Index (RSI)'
        },
        'ADX': {
            'params': ['symbol', 'interval', 'time_period'],
            'description': 'Average Directional Movement Index (ADX)'
        },
        'CCI': {
            'params': ['symbol', 'interval', 'time_period'],
            'description': 'Commodity Channel Index (CCI)'
        },
        'AROON': {
            'params': ['symbol', 'interval', 'time_period'],
            'description': 'Aroon Indicator'
        },
        'BBANDS': {
            'params': ['symbol', 'interval', 'time_period', 'series_type', 'nbdevup', 'nbdevdn'],
            'description': 'Bollinger Bands'
        },
        'AD': {
            'params': ['symbol', 'interval'],
            'description': 'Chaikin A/D Line'
        },
        'OBV': {
            'params': ['symbol', 'interval'],
            'description': 'On Balance Volume (OBV)'
        }
    }
}

@alpha_vantage_bp.route('/api/functions') # type: ignore
@login_required
def get_functions():
    """Return the list of available API functions with documentation"""
    # Enhance API function definitions with additional documentation if available
    enhanced_functions = {}
    for category, functions in API_FUNCTIONS.items():
        enhanced_functions[category] = {}
        for func_name, details in functions.items():
            enhanced_functions[category][func_name] = details.copy()
            # Add enhanced documentation if available
            if func_name in API_DOCUMENTATION:
                enhanced_functions[category][func_name]['enhanced_docs'] = API_DOCUMENTATION[func_name]
    
    return jsonify(enhanced_functions)

@alpha_vantage_bp.route('/docs/<function_name>') # type: ignore
@login_required
def function_docs(function_name):
    """Return detailed documentation for a specific function"""
    # Find the function in API_FUNCTIONS
    for category, functions in API_FUNCTIONS.items():
        if function_name in functions:
            docs = functions[function_name].copy()
            # Add enhanced documentation if available
            if function_name in API_DOCUMENTATION:
                docs['enhanced_docs'] = API_DOCUMENTATION[function_name]
            return jsonify(docs)
    
    # If function not found
    return jsonify({
        'error': 'Function not found',
        'message': f'Documentation for {function_name} not available'
    }), 404 