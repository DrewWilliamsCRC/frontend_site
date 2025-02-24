from flask import Blueprint, render_template, jsonify, request, session, redirect, url_for
import os
import requests
from datetime import datetime
from functools import wraps
import logging

alpha_vantage_bp = Blueprint('alpha_vantage', __name__)

# Get API key from environment
ALPHA_VANTAGE_API_KEY = os.environ.get("ALPHA_VANTAGE_API_KEY")

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@alpha_vantage_bp.route('/browser')
@login_required
def browser():
    """Main page for the Alpha Vantage API browser."""
    return render_template('alpha_vantage_browser.html')

@alpha_vantage_bp.route('/api/<category>/<function_name>', methods=['GET'])
@login_required
def api_call(category, function_name):
    """Generic endpoint to handle Alpha Vantage API calls."""
    if not ALPHA_VANTAGE_API_KEY:
        return jsonify({
            'error': 'API key not configured',
            'data': None
        })

    # Get parameters from request
    params = request.args.to_dict()
    params['apikey'] = ALPHA_VANTAGE_API_KEY
    params['function'] = function_name

    try:
        url = "https://www.alphavantage.co/query"
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        # Check for rate limit messages
        if "Note" in data:
            return jsonify({
                'error': 'API rate limit reached - Please try again later',
                'data': None
            })

        return jsonify({
            'error': None,
            'data': data
        })

    except requests.exceptions.RequestException as e:
        # Log the actual error for debugging
        logging.error(f"Alpha Vantage API request error: {str(e)}")
        return jsonify({
            'error': 'Failed to fetch data from the API. Please try again later.',
            'data': None
        })
    except ValueError as e:
        # JSON parsing error
        logging.error(f"JSON parsing error in Alpha Vantage response: {str(e)}")
        return jsonify({
            'error': 'Invalid response format from the API.',
            'data': None
        })
    except Exception as e:
        # Log unexpected errors but don't expose details to client
        logging.error(f"Unexpected error in Alpha Vantage API call: {str(e)}")
        return jsonify({
            'error': 'An unexpected error occurred. Please try again later.',
            'data': None
        })

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

@alpha_vantage_bp.route('/api/functions')
@login_required
def get_functions():
    """Return the list of available API functions and their parameters."""
    return jsonify(API_FUNCTIONS) 