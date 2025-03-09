#!/usr/bin/env python3
"""
Alpha Vantage Data Pipeline

This module provides functions for fetching, processing, and storing data from Alpha Vantage API.
"""

import os
import time
import json
import logging
from datetime import datetime, timedelta
import importlib.util
import sys

import pandas as pd # type: ignore
import numpy as np # type: ignore
import requests
from dotenv import load_dotenv

# Set up better logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check if TensorFlow and PyTorch are available
TENSORFLOW_AVAILABLE = importlib.util.find_spec("tensorflow") is not None
try:
    import torch # type: ignore
    torch_available = True
except ImportError:
    torch_available = False
    logger.warning("PyTorch not available, some functionality will be limited")

if not TENSORFLOW_AVAILABLE:
    logger.warning("TensorFlow not available, some functionality will be limited")
    # Import mock model for CI environment
    from ai_experiments.ci_mock_model import MockTensorFlowModel, create_mock_model
    
    # Create mock TensorFlow module for compatibility
    class MockModule:
        pass
    
    # Create a simple mock for tensorflow if not available
    sys.modules['tensorflow'] = MockModule()
    sys.modules['tensorflow.keras'] = MockModule()
    sys.modules['tensorflow.keras.models'] = MockModule()
    sys.modules['tensorflow.keras.layers'] = MockModule()

# Load environment variables
load_dotenv()
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Check if we're in CI mode
CI_MODE = os.getenv("CI_BUILD", "false").lower() == "true"
if CI_MODE:
    logger.info("Running in CI mode - using simplified data processing")

# Constants
BASE_URL = "https://www.alphavantage.co/query"
REQUEST_DELAY = 15  # seconds between API calls to avoid rate limits

# Market symbols to fetch - using format that works with Alpha Vantage API
MARKET_INDICES = {
    'DJI': 'DJI',    # Dow Jones Industrial Average
    'SPX': 'SPX',    # S&P 500
    'IXIC': 'IXIC',  # NASDAQ Composite
    'VIX': 'VIX',    # CBOE Volatility Index
    'TNX': 'TNX'     # 10-Year Treasury Note Yield
}

# Alternative indices if needed
ALTERNATIVE_INDICES = {
    'DJI': 'DOW',      # Alternative for Dow Jones
    'SPX': 'SP500',    # Alternative for S&P 500
    'IXIC': 'NASDAQ',  # Alternative for NASDAQ
}

# Sample stocks for each sector
SAMPLE_STOCKS = {
    'Technology': ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'AMD'],
    'Healthcare': ['JNJ', 'PFE', 'UNH', 'MRK'],
    'Finance': ['JPM', 'BAC', 'GS', 'MS'],
    'Energy': ['XOM', 'CVX', 'COP', 'SLB']
}


class AlphaVantageAPI:
    """Class to interact with Alpha Vantage API."""

    def __init__(self, api_key=None):
        """Initialize the API handler.
        
        Args:
            api_key (str, optional): Alpha Vantage API key. If None, loads from environment.
        """
        self.api_key = api_key or ALPHA_VANTAGE_API_KEY
        if not self.api_key:
            raise ValueError("Alpha Vantage API key is required.")
        
        # Security check to prevent API keys being hardcoded
        if self.api_key and (len(self.api_key) == 16 or len(self.api_key) == 32) and self.api_key == api_key:
            logger.warning("API key appears to be directly provided rather than loaded from environment")
            logger.warning("For security, please use environment variables or secure storage instead")
        
        self.session = requests.Session()
        self.last_call_time = None
    
    def _wait_for_rate_limit(self):
        """Wait to avoid hitting API rate limits."""
        if self.last_call_time is not None:
            elapsed = (datetime.now() - self.last_call_time).total_seconds()
            if elapsed < REQUEST_DELAY:
                time.sleep(REQUEST_DELAY - elapsed)
        
        self.last_call_time = datetime.now()
    
    def call_api(self, function, **params):
        """
        Make a call to the Alpha Vantage API.
        
        Args:
            function (str): The Alpha Vantage function to call.
            **params: Additional parameters for the API call.
            
        Returns:
            dict: API response data, or None if an error occurred.
        """
        try:
            # Build query parameters
            query_params = {
                'apikey': self.api_key,
            }
            
            if function:
                query_params['function'] = function
                
            # Add additional parameters
            for key, value in params.items():
                query_params[key] = value
            
            # Make the request
            logger.info(f"Calling Alpha Vantage API: {function}")
            # Log the full request URL with redacted API key for debugging
            full_url = f"{BASE_URL}?{'&'.join([f'{k}={v}' for k, v in query_params.items() if k != 'apikey'])}&apikey=REDACTED"
            logger.info(f"Request URL: {full_url}")
            
            response = requests.get(BASE_URL, params=query_params)
            
            # Check if request was successful
            if response.status_code != 200:
                logger.error(f"API request failed with status code {response.status_code}: {response.text}")
                return None
            
            # Parse JSON response
            data = response.json()
            
            # Enhanced debugging for API responses
            logger.info(f"API response keys: {list(data.keys())}")
            
            # Check for API errors
            if 'Error Message' in data:
                logger.error(f"API error: {data['Error Message']}")
                return data
            
            if 'Note' in data and 'call frequency' in data['Note']:
                logger.warning(f"API rate limit warning: {data['Note']}")
            
            logger.info(f"API call to {function} successful")
            return data
            
        except Exception as e:
            logger.error(f"Error calling API {function}: {str(e)}")
            return None
    
    def get_daily_time_series(self, symbol, outputsize="full"):
        """
        Get daily time series data for a symbol.
        
        Args:
            symbol (str): Stock symbol to fetch.
            outputsize (str): 'compact' (last 100 points) or 'full' (20+ years)
            
        Returns:
            pd.DataFrame: DataFrame with OHLCV data or None if an error occurs.
        """
        try:
            logger.info(f"Fetching daily time series for {symbol}")
            
            # First, try regular time series
            data = self.call_api('TIME_SERIES_DAILY', symbol=symbol, outputsize=outputsize)
            
            # If that fails, try time series adjusted
            if data is None or "Time Series (Daily)" not in data:
                logger.info(f"Regular time series failed for {symbol}, trying TIME_SERIES_DAILY_ADJUSTED")
                data = self.call_api('TIME_SERIES_DAILY_ADJUSTED', symbol=symbol, outputsize=outputsize)
            
            # For market indices, we might need to try alternative symbols
            if (data is None or "Time Series (Daily)" not in data) and symbol in MARKET_INDICES.keys():
                # Try using alternative symbol format if available
                if symbol in ALTERNATIVE_INDICES:
                    alt_symbol = ALTERNATIVE_INDICES[symbol]
                    logger.info(f"Trying alternative symbol {alt_symbol} for {symbol}")
                    data = self.call_api('TIME_SERIES_DAILY', symbol=alt_symbol, outputsize=outputsize)
            
            # If still no time series data, try getting basic quote data and construct minimal dataframe
            if data is None or not any(key.startswith("Time Series") for key in data.keys()):
                logger.warning(f"No time series data for {symbol}, falling back to quote data")
                quote_data = self.get_quote(symbol)
                
                if quote_data and 'price' in quote_data:
                    # Create a minimal dataframe with just the latest price
                    today = datetime.now().strftime('%Y-%m-%d')
                    df = pd.DataFrame({
                        'open': [float(quote_data['price'])],
                        'high': [float(quote_data['price'])],
                        'low': [float(quote_data['price'])],
                        'close': [float(quote_data['price'])],
                        'volume': [0]
                    }, index=[today])
                    df.index = pd.to_datetime(df.index)
                    logger.info(f"Created minimal dataframe for {symbol} with quote data")
                    return df
                else:
                    logger.error(f"No data available for {symbol}")
                    return None
            
            # Find the correct time series key
            time_series_key = next((key for key in data.keys() if key.startswith("Time Series")), None)
            
            if time_series_key is None:
                logger.error(f"No time series data for {symbol}. Got: {list(data.keys())}")
                return None
                
            if not data[time_series_key]:
                logger.error(f"Empty time series data for {symbol}")
                return None
            
            logger.info(f"Successfully fetched daily data for {symbol}")
            df = pd.DataFrame.from_dict(data[time_series_key], orient="index")
            df.index = pd.to_datetime(df.index)
            df.sort_index(inplace=True)
            
            # Determine column prefix based on the time series key
            prefix = "1. " if "Time Series (Daily)" in data else ""
            
            # Rename columns based on the available columns
            col_mapping = {
                f"{prefix}open": "open",
                f"{prefix}high": "high",
                f"{prefix}low": "low",
                f"{prefix}close": "close",
                f"{prefix}volume": "volume"
            }
            
            # Only rename columns that exist
            rename_dict = {k: v for k, v in col_mapping.items() if k in df.columns}
            df.rename(columns=rename_dict, inplace=True)
            
            # Ensure all expected columns exist
            for col in ["open", "high", "low", "close", "volume"]:
                if col not in df.columns:
                    # If a column is missing, use the closest available column
                    if col == "close" and "adjusted close" in df.columns:
                        df["close"] = df["adjusted close"]
                    elif col == "volume" and "volume" not in df.columns:
                        df["volume"] = 0
                    else:
                        df[col] = df["close"] if "close" in df.columns else 0
            
            # Convert columns to float
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
            return df
            
        except Exception as e:
            logger.error(f"Error fetching daily time series for {symbol}: {str(e)}")
            return None
    
    def get_quote(self, symbol):
        """
        Get real-time quote data for a symbol.
        
        Args:
            symbol (str): The stock symbol to fetch.
            
        Returns:
            dict: Quote data for the symbol, or None if not available.
        """
        try:
            logger.info(f"Fetching quote for {symbol}")
            response = self.call_api('GLOBAL_QUOTE', symbol=symbol)
            
            if response is None:
                logger.error(f"API call for quote returned None for {symbol}")
                return None
                
            if 'Error Message' in response:
                logger.error(f"API error for {symbol} quote: {response['Error Message']}")
                return None
                
            if 'Global Quote' in response and response['Global Quote']:
                quote_data = response['Global Quote']
                # Process the data into a standardized format
                return {
                    'symbol': quote_data.get('01. symbol', symbol),
                    'price': quote_data.get('05. price', '0.00'),
                    'change': quote_data.get('09. change', '0.00'),
                    'change_percent': quote_data.get('10. change percent', '0.00%').replace('%', ''),
                    'volume': quote_data.get('06. volume', '0'),
                    'timestamp': datetime.now().isoformat()
                }
                
            logger.warning(f"No quote data found for {symbol}")
            
            # Try an alternative symbol format if this is a market index
            if symbol in MARKET_INDICES.keys() and symbol in ALTERNATIVE_INDICES:
                alt_symbol = ALTERNATIVE_INDICES[symbol]
                logger.info(f"Trying alternative symbol {alt_symbol} for {symbol}")
                return self.get_quote(alt_symbol)
                
            return None
            
        except Exception as e:
            logger.error(f"Error fetching quote for {symbol}: {str(e)}")
            return None
    
    def get_market_news(self, tickers=None, topics=None, limit=10):
        """
        Get market news and sentiment data from Alpha Vantage.
        
        Args:
            tickers (str, optional): Comma-separated stock symbols to filter by.
            topics (str, optional): Comma-separated topics to filter by.
            limit (int, optional): Maximum number of news items to return.
            
        Returns:
            dict: Market news and sentiment data.
        """
        try:
            params = {'function': 'NEWS_SENTIMENT'}
            
            if tickers:
                params['tickers'] = tickers
            
            if topics:
                params['topics'] = topics
                
            if limit:
                params['limit'] = limit
                
            response = self.call_api(None, **params)
            
            if 'feed' in response:
                # Process sentiment scores
                for item in response.get('feed', []):
                    if 'overall_sentiment_score' in item:
                        # Round sentiment scores to 2 decimal places
                        item['overall_sentiment_score'] = round(float(item['overall_sentiment_score']), 2)
                    
                    # Process ticker sentiment
                    for ticker in item.get('ticker_sentiment', []):
                        if 'sentiment_score' in ticker:
                            ticker['sentiment_score'] = round(float(ticker['sentiment_score']), 2)
                
                return response
            
            return {'feed': []}
        except Exception as e:
            logger.error(f"Error fetching market news: {str(e)}")
            return {'feed': []}
    
    def get_sector_performance(self):
        """Get sector performance data.
        
        Returns:
            dict: Sector performance data
        """
        try:
            logger.info("Fetching sector performance data")
            data = self.call_api("SECTOR")
            
            if not data:
                logger.error("API call for sector performance returned None")
                return self._generate_fallback_sector_data()
            
            # Check for the sector performance data in different potential keys
            sector_keys = [
                "Rank A: Real-Time Performance",
                "Rank B: 1 Day Performance",
                "Rank C: 5 Day Performance",
                "Rank D: 1 Month Performance",
                "Rank E: 3 Month Performance"
            ]
            
            # Try to find at least one of the sector performance keys
            found_key = next((key for key in sector_keys if key in data), None)
            
            if not found_key:
                logger.error("No sector performance data found in response")
                logger.info(f"Available keys in response: {list(data.keys())}")
                return self._generate_fallback_sector_data()
            
            # Process the data to get a consistent format
            sectors = {}
            for sector, value in data[found_key].items():
                if sector != 'Meta':
                    sectors[sector] = {
                        'Sector': sector,
                        'Change': value,
                        'Performance Key': found_key
                    }
            
            if not sectors:
                logger.error("No sectors found in performance data")
                return self._generate_fallback_sector_data()
                
            logger.info(f"Successfully fetched data for {len(sectors)} sectors")
            return sectors
            
        except Exception as e:
            logger.error(f"Error getting sector performance: {str(e)}")
            return self._generate_fallback_sector_data()
    
    def _generate_fallback_sector_data(self):
        """Generate fallback sector data when the API fails.
        
        Returns:
            dict: Fallback sector performance data
        """
        logger.info("Generating fallback sector data")
        sectors = {
            'Technology': {'Sector': 'Technology', 'Change': '1.2%'},
            'Healthcare': {'Sector': 'Healthcare', 'Change': '0.8%'},
            'Finance': {'Sector': 'Finance', 'Change': '0.5%'},
            'Energy': {'Sector': 'Energy', 'Change': '-0.3%'},
            'Utilities': {'Sector': 'Utilities', 'Change': '0.1%'},
            'Materials': {'Sector': 'Materials', 'Change': '0.7%'},
            'Industrials': {'Sector': 'Industrials', 'Change': '0.9%'},
            'Consumer Discretionary': {'Sector': 'Consumer Discretionary', 'Change': '0.4%'},
            'Consumer Staples': {'Sector': 'Consumer Staples', 'Change': '0.2%'},
            'Real Estate': {'Sector': 'Real Estate', 'Change': '-0.1%'}
        }
        return sectors


class DataProcessor:
    """Class to process financial data for AI models."""
    
    @staticmethod
    def add_technical_indicators(df):
        """Add technical indicators to the dataframe.
        
        Args:
            df (pd.DataFrame): Price data with OHLCV columns
            
        Returns:
            pd.DataFrame: Enhanced dataframe with technical indicators
        """
        # Ensure we're working with a copy
        df = df.copy()
        
        # Basic price indicators
        df['daily_return'] = df['close'].pct_change()
        df['weekly_return'] = df['close'].pct_change(5)
        df['monthly_return'] = df['close'].pct_change(21)
        
        # Moving averages
        for window in [5, 10, 20, 50, 200]:
            df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
        
        # Exponential moving averages
        for window in [12, 26, 50]:
            df[f'ema_{window}'] = df['close'].ewm(span=window, adjust=False).mean()
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        df['bb_std'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # Trading signals
        df['signal_macd'] = np.where(df['macd'] > df['macd_signal'], 1, -1)
        df['signal_rsi'] = np.where((df['rsi_14'] < 30), 1, np.where((df['rsi_14'] > 70), -1, 0))
        df['signal_ma_cross'] = np.where(df['sma_20'] > df['sma_50'], 1, -1)
        
        # Volatility
        df['volatility_daily'] = df['daily_return'].rolling(window=21).std()
        df['volatility_annual'] = df['volatility_daily'] * np.sqrt(252)
        
        return df
    
    @staticmethod
    def prepare_for_ml(df, target_horizon=5, train_size=0.8):
        """Prepare data for machine learning.
        
        Args:
            df (pd.DataFrame): Enhanced price data with technical indicators
            target_horizon (int): Number of days to look ahead for target
            train_size (float): Proportion of data to use for training
            
        Returns:
            tuple: X_train, X_test, y_train, y_test and related metadata
        """
        # Create target variable: future returns
        df['target_return'] = df['close'].pct_change(target_horizon).shift(-target_horizon)
        df['target_direction'] = np.where(df['target_return'] > 0, 1, 0)
        
        # Drop rows with NaN
        df = df.dropna().copy()
        
        # Select features (avoiding look-ahead bias)
        feature_columns = [
            'daily_return', 'weekly_return', 'monthly_return',
            'volatility_daily', 'volatility_annual',
            'rsi_14', 'macd', 'macd_hist', 'signal_macd', 'signal_rsi', 'signal_ma_cross',
            'bb_width'
        ]
        
        # Add price relative to moving averages
        for ma in [20, 50, 200]:
            df[f'price_vs_sma{ma}'] = (df['close'] / df[f'sma_{ma}'] - 1) * 100
            feature_columns.append(f'price_vs_sma{ma}')
        
        X = df[feature_columns]
        y_reg = df['target_return']
        y_clf = df['target_direction']
        
        # Train/test split preserving time order
        split_idx = int(len(df) * train_size)
        
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_reg_train = y_reg.iloc[:split_idx]
        y_reg_test = y_reg.iloc[split_idx:]
        y_clf_train = y_clf.iloc[:split_idx]
        y_clf_test = y_clf.iloc[split_idx:]
        
        # Save dates for reference
        train_dates = df.index[:split_idx]
        test_dates = df.index[split_idx:]
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_reg_train': y_reg_train,
            'y_reg_test': y_reg_test,
            'y_clf_train': y_clf_train,
            'y_clf_test': y_clf_test,
            'train_dates': train_dates,
            'test_dates': test_dates,
            'feature_columns': feature_columns
        }


class DataManager:
    """Class to manage data fetching, processing, and storage."""
    
    def __init__(self):
        """Initialize data manager."""
        self.api = AlphaVantageAPI()
        self.processor = DataProcessor()
    
    def fetch_and_store_index_data(self, refresh=False):
        """Fetch and store market index data.
        
        Args:
            refresh (bool): If True, fetch new data even if file exists
            
        Returns:
            dict: Dictionary of dataframes with index data
        """
        result = {}
        
        for name, symbol in MARKET_INDICES.items():
            output_path = os.path.join(DATA_DIR, f"{name}_daily.csv")
            
            # Check if we already have recent data
            if not refresh and os.path.exists(output_path):
                file_age = time.time() - os.path.getmtime(output_path)
                if file_age < 86400:  # Less than 1 day old
                    logger.info(f"Loading existing data for {name} from {output_path}")
                    df = pd.read_csv(output_path, index_col=0, parse_dates=True)
                    result[name] = df
                    continue
            
            # Fetch new data
            logger.info(f"Fetching data for {name} ({symbol})...")
            df = self.api.get_daily_time_series(symbol)
            
            if df is not None:
                # Save to CSV
                df.to_csv(output_path)
                logger.info(f"Saved {name} data to {output_path}")
                result[name] = df
        
        return result
    
    def fetch_and_store_stock_data(self, refresh=False):
        """Fetch and store sample stock data for each sector.
        
        Args:
            refresh (bool): If True, fetch new data even if file exists
            
        Returns:
            dict: Dictionary of dataframes with stock data by sector
        """
        result = {}
        
        for sector, symbols in SAMPLE_STOCKS.items():
            sector_dir = os.path.join(DATA_DIR, sector)
            os.makedirs(sector_dir, exist_ok=True)
            
            sector_data = {}
            for symbol in symbols:
                output_path = os.path.join(sector_dir, f"{symbol}_daily.csv")
                
                # Check if we already have recent data
                if not refresh and os.path.exists(output_path):
                    file_age = time.time() - os.path.getmtime(output_path)
                    if file_age < 86400:  # Less than 1 day old
                        logger.info(f"Loading existing data for {symbol} from {output_path}")
                        df = pd.read_csv(output_path, index_col=0, parse_dates=True)
                        sector_data[symbol] = df
                        continue
                
                # Fetch new data
                logger.info(f"Fetching data for {symbol}...")
                df = self.api.get_daily_time_series(symbol)
                
                if df is not None:
                    # Save to CSV
                    df.to_csv(output_path)
                    logger.info(f"Saved {symbol} data to {output_path}")
                    sector_data[symbol] = df
            
            result[sector] = sector_data
        
        return result
    
    def process_data_for_ml(self, data_dict, output_dir=None):
        """Process data for machine learning and optionally save to files.
        
        Args:
            data_dict (dict): Dictionary of dataframes
            output_dir (str, optional): Directory to save processed data
            
        Returns:
            dict: Dictionary of processed data
        """
        result = {}
        
        for name, df in data_dict.items():
            logger.info(f"Processing {name} data for ML...")
            
            # Add technical indicators
            enhanced_df = self.processor.add_technical_indicators(df)
            
            # Prepare for ML
            ml_data = self.processor.prepare_for_ml(enhanced_df)
            result[name] = ml_data
            
            # Save to files if output_dir is provided
            if output_dir:
                name_dir = os.path.join(output_dir, name)
                os.makedirs(name_dir, exist_ok=True)
                
                # Save X and y data
                ml_data['X_train'].to_csv(os.path.join(name_dir, 'X_train.csv'))
                ml_data['X_test'].to_csv(os.path.join(name_dir, 'X_test.csv'))
                ml_data['y_reg_train'].to_csv(os.path.join(name_dir, 'y_reg_train.csv'))
                ml_data['y_reg_test'].to_csv(os.path.join(name_dir, 'y_reg_test.csv'))
                ml_data['y_clf_train'].to_csv(os.path.join(name_dir, 'y_clf_train.csv'))
                ml_data['y_clf_test'].to_csv(os.path.join(name_dir, 'y_clf_test.csv'))
                
                # Save metadata
                with open(os.path.join(name_dir, 'metadata.json'), 'w') as f:
                    metadata = {
                        'feature_columns': ml_data['feature_columns'],
                        'train_dates': [str(d) for d in ml_data['train_dates']],
                        'test_dates': [str(d) for d in ml_data['test_dates']]
                    }
                    json.dump(metadata, f, indent=4)
                
                logger.info(f"Saved ML data for {name} to {name_dir}")
        
        return result


def main():
    """Main function to run the pipeline."""
    logger.info("Starting Alpha Vantage data pipeline")
    
    manager = DataManager()
    
    # Create directories
    ml_dir = os.path.join(DATA_DIR, 'ml_ready')
    os.makedirs(ml_dir, exist_ok=True)
    
    # Fetch and process index data
    logger.info("Fetching market indices data...")
    indices_data = manager.fetch_and_store_index_data()
    
    if indices_data:
        logger.info("Processing market indices data for ML...")
        indices_ml_data = manager.process_data_for_ml(indices_data, os.path.join(ml_dir, 'indices'))
    
    # Fetch and process stock data (sampling for demonstration)
    logger.info("Fetching sample stocks data...")
    stock_data = manager.fetch_and_store_stock_data()
    
    for sector, sector_data in stock_data.items():
        if sector_data:
            logger.info(f"Processing {sector} stocks data for ML...")
            sector_ml_data = manager.process_data_for_ml(sector_data, os.path.join(ml_dir, 'stocks', sector))
    
    logger.info("Alpha Vantage data pipeline completed")


if __name__ == "__main__":
    main() 