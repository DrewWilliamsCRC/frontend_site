#!/usr/bin/env python3
"""
Economic Data Manager Module

This module provides classes and functions for fetching and analyzing 
economic indicators from FRED and other sources.
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
import time

import pandas as pd # type: ignore
import numpy as np # type: ignore
import requests
from fredapi import Fred # type: ignore
import yfinance as yf # type: ignore
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('economic_data_manager')

# Load environment variables
load_dotenv()
FRED_API_KEY = os.getenv('FRED_API_KEY')

# Key economic indicators with their series IDs
ECONOMIC_INDICATORS = {
    'GDP': {
        'id': 'GDP',
        'name': 'Gross Domestic Product',
        'category': 'output',
        'frequency': 'quarterly',
        'units': 'billions_usd'
    },
    'GDPC1': {
        'id': 'GDPC1',
        'name': 'Real Gross Domestic Product',
        'category': 'output',
        'frequency': 'quarterly',
        'units': 'billions_2017_usd'
    },
    'UNRATE': {
        'id': 'UNRATE',
        'name': 'Unemployment Rate',
        'category': 'labor',
        'frequency': 'monthly',
        'units': 'percent'
    },
    'CPIAUCSL': {
        'id': 'CPIAUCSL',
        'name': 'Consumer Price Index (All Urban Consumers)',
        'category': 'prices',
        'frequency': 'monthly',
        'units': 'index'
    },
    'FEDFUNDS': {
        'id': 'FEDFUNDS',
        'name': 'Federal Funds Effective Rate',
        'category': 'monetary',
        'frequency': 'monthly',
        'units': 'percent'
    },
    'DFF': {
        'id': 'DFF',
        'name': 'Federal Funds Effective Rate (Daily)',
        'category': 'monetary',
        'frequency': 'daily',
        'units': 'percent'
    },
    'MORTGAGE30US': {
        'id': 'MORTGAGE30US',
        'name': '30-Year Fixed Rate Mortgage Average',
        'category': 'finance',
        'frequency': 'weekly',
        'units': 'percent'
    },
    'UMCSENT': {
        'id': 'UMCSENT',
        'name': 'University of Michigan: Consumer Sentiment',
        'category': 'sentiment',
        'frequency': 'monthly',
        'units': 'index'
    },
    'DTWEXBGS': {
        'id': 'DTWEXBGS',
        'name': 'Trade Weighted U.S. Dollar Index',
        'category': 'international',
        'frequency': 'daily',
        'units': 'index'
    },
    'PAYEMS': {
        'id': 'PAYEMS',
        'name': 'Total Nonfarm Payrolls',
        'category': 'labor',
        'frequency': 'monthly',
        'units': 'thousands'
    },
    'INDPRO': {
        'id': 'INDPRO',
        'name': 'Industrial Production Index',
        'category': 'output',
        'frequency': 'monthly',
        'units': 'index'
    },
    'TDSP': {
        'id': 'TDSP',
        'name': 'Household Debt Service Payments as a Percent of Disposable Personal Income',
        'category': 'households',
        'frequency': 'quarterly',
        'units': 'percent'
    }
}

# Economic indicator categories for easier access
ECONOMIC_CATEGORIES = {
    'growth': ['GDP', 'GDPC1', 'INDPRO'],
    'inflation': ['CPIAUCSL'],
    'labor': ['UNRATE', 'PAYEMS'],
    'monetary': ['FEDFUNDS', 'DFF'],
    'sentiment': ['UMCSENT'],
    'finance': ['MORTGAGE30US', 'TDSP'],
    'international': ['DTWEXBGS']
}


class EconomicDataManager:
    """Class for fetching and analyzing economic data from FRED API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize with FRED API key.
        
        Args:
            api_key (str, optional): FRED API key, defaults to environment variable
        """
        self.api_key = api_key or FRED_API_KEY
        if not self.api_key:
            logger.warning("FRED API key not provided, this source will be unavailable")
            self.fred = None
        else:
            try:
                self.fred = Fred(api_key=self.api_key)
                logger.info("FRED API client initialized successfully")
            except Exception as e:
                logger.error(f"Error initializing FRED API client: {str(e)}")
                self.fred = None
        
        # Cache for economic data to avoid repeated API calls
        self.data_cache = {}
        self.last_update = {}
    
    def get_economic_indicator(self, series_id: str, start_date: Optional[str] = None, 
                              end_date: Optional[str] = None, frequency: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Fetch economic indicator data from FRED.
        
        Args:
            series_id (str): FRED series ID
            start_date (str, optional): Start date in 'YYYY-MM-DD' format
            end_date (str, optional): End date in 'YYYY-MM-DD' format
            frequency (str, optional): Frequency of data ('d', 'm', 'q', 'a')
            
        Returns:
            pd.DataFrame: DataFrame with economic indicator data or None if an error occurs
        """
        if not self.fred:
            logger.error("FRED API client not initialized")
            return None
        
        # Default to last 5 years if no dates provided
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if not start_date:
            start_date = (datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d')
        
        # Check if data is in cache and recent enough
        cache_key = f"{series_id}_{start_date}_{end_date}_{frequency}"
        if cache_key in self.data_cache:
            last_update = self.last_update.get(cache_key)
            if last_update and (datetime.now() - last_update).total_seconds() < 86400:  # Cache for 24 hours
                logger.info(f"Using cached data for {series_id}")
                return self.data_cache[cache_key]
        
        try:
            logger.info(f"Fetching economic data for {series_id}")
            data = self.fred.get_series(
                series_id, 
                start_date=start_date, 
                end_date=end_date,
                frequency=frequency
            )
            
            if data is None or len(data) == 0:
                logger.warning(f"No data returned for {series_id}")
                return None
            
            # Convert to DataFrame for easier manipulation
            df = pd.DataFrame({series_id: data})
            df.index.name = 'date'
            
            # Add metadata
            if series_id in ECONOMIC_INDICATORS:
                for key, value in ECONOMIC_INDICATORS[series_id].items():
                    if key != 'id':
                        df[f'meta_{key}'] = value
            
            # Calculate percentage changes
            if len(df) > 1:
                # Different periods for different frequencies
                periods = {
                    'daily': [1, 5, 20, 60],  # 1-day, 1-week, 1-month, 3-month
                    'weekly': [1, 4, 12, 26],  # 1-week, 1-month, 3-month, 6-month
                    'monthly': [1, 3, 6, 12],  # 1-month, 3-month, 6-month, 1-year
                    'quarterly': [1, 2, 4, 8]  # 1-quarter, 2-quarter, 1-year, 2-year
                }
                
                # Get frequency from metadata or guess from index
                if series_id in ECONOMIC_INDICATORS:
                    freq = ECONOMIC_INDICATORS[series_id]['frequency']
                else:
                    # Try to guess frequency from index
                    index_diff = (df.index[1] - df.index[0]).days
                    if index_diff <= 3:
                        freq = 'daily'
                    elif index_diff <= 9:
                        freq = 'weekly'
                    elif index_diff <= 45:
                        freq = 'monthly'
                    else:
                        freq = 'quarterly'
                
                # Calculate percent changes
                for period in periods.get(freq, [1, 3, 6, 12]):
                    if period < len(df):
                        df[f'pct_change_{period}'] = df[series_id].pct_change(periods=period)
            
            # Cache the data
            self.data_cache[cache_key] = df
            self.last_update[cache_key] = datetime.now()
            
            logger.info(f"Successfully fetched economic data for {series_id}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching economic data for {series_id}: {str(e)}")
            return None
    
    def get_economic_category(self, category: str, start_date: Optional[str] = None,
                             end_date: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Fetch multiple economic indicators for a category.
        
        Args:
            category (str): Category of economic indicators
            start_date (str, optional): Start date in 'YYYY-MM-DD' format
            end_date (str, optional): End date in 'YYYY-MM-DD' format
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary of indicator series IDs and their data
        """
        if category not in ECONOMIC_CATEGORIES:
            logger.error(f"Unknown economic category: {category}")
            return {}
        
        results = {}
        for series_id in ECONOMIC_CATEGORIES[category]:
            data = self.get_economic_indicator(series_id, start_date, end_date)
            if data is not None:
                results[series_id] = data
                time.sleep(0.5)  # Rate limit protection
        
        logger.info(f"Fetched {len(results)} indicators for category {category}")
        return results
    
    def get_all_key_indicators(self, start_date: Optional[str] = None,
                              end_date: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Fetch all key economic indicators.
        
        Args:
            start_date (str, optional): Start date in 'YYYY-MM-DD' format
            end_date (str, optional): End date in 'YYYY-MM-DD' format
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary of indicator series IDs and their data
        """
        results = {}
        
        # Create a list of the most important indicators from each category
        key_indicators = ['GDP', 'UNRATE', 'CPIAUCSL', 'FEDFUNDS', 'MORTGAGE30US', 'UMCSENT']
        
        for series_id in key_indicators:
            data = self.get_economic_indicator(series_id, start_date, end_date)
            if data is not None:
                results[series_id] = data
                time.sleep(0.5)  # Rate limit protection
        
        logger.info(f"Fetched {len(results)} key economic indicators")
        return results
    
    def get_economic_summary(self) -> Dict[str, Any]:
        """
        Get a summary of current economic conditions.
        
        Returns:
            Dict: Summary of key economic indicators and their recent trends
        """
        # First get the latest data for key indicators
        indicators = ['GDP', 'UNRATE', 'CPIAUCSL', 'FEDFUNDS']
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d')
        
        data = {}
        for indicator in indicators:
            df = self.get_economic_indicator(indicator, start_date, end_date)
            if df is not None:
                data[indicator] = df
                time.sleep(0.5)  # Rate limit protection
        
        if not data:
            logger.error("Could not fetch any economic data for summary")
            return {
                "error": "Could not fetch economic data",
                "timestamp": datetime.now().isoformat()
            }
        
        # Create summary
        summary = {
            "timestamp": datetime.now().isoformat(),
            "indicators": {}
        }
        
        # Process each indicator
        for indicator, df in data.items():
            if df is None or df.empty:
                continue
                
            latest_value = df[indicator].iloc[-1]
            latest_date = df.index[-1].strftime('%Y-%m-%d')
            
            # Get trend over different periods
            trends = {}
            for period_col in [col for col in df.columns if col.startswith('pct_change_')]:
                period = period_col.split('_')[-1]
                if not pd.isna(df[period_col].iloc[-1]):
                    trends[f"{period}_period"] = df[period_col].iloc[-1]
            
            # Find historical context (percentile)
            percentile = df[indicator].rank(pct=True).iloc[-1]
            
            # Determine status based on indicators
            status = self._determine_indicator_status(indicator, latest_value, trends)
            
            summary["indicators"][indicator] = {
                "name": ECONOMIC_INDICATORS.get(indicator, {}).get("name", indicator),
                "value": float(latest_value),
                "date": latest_date,
                "trends": {k: float(v) for k, v in trends.items()},
                "percentile": float(percentile),
                "status": status
            }
        
        # Add overall assessment
        summary["overall_assessment"] = self._get_overall_assessment(summary["indicators"])
        
        return summary
    
    def _determine_indicator_status(self, indicator: str, value: float, 
                                   trends: Dict[str, float]) -> str:
        """
        Determine the status of an economic indicator.
        
        Args:
            indicator (str): Indicator series ID
            value (float): Latest value of the indicator
            trends (Dict): Dictionary of trends over different periods
            
        Returns:
            str: Status ('positive', 'negative', 'neutral', or 'mixed')
        """
        # Default status is neutral
        status = "neutral"
        
        if indicator == "UNRATE":
            # For unemployment rate, lower is better
            if trends.get("6_period", 0) < -0.05:
                status = "positive"  # Unemployment decreasing
            elif trends.get("6_period", 0) > 0.05:
                status = "negative"  # Unemployment increasing
        
        elif indicator == "CPIAUCSL":
            # For inflation, moderate is good, too high or too low is bad
            annual_change = trends.get("12_period", 0)
            if annual_change is not None:
                if 0.01 <= annual_change <= 0.03:
                    status = "positive"  # Healthy inflation
                elif annual_change > 0.05:
                    status = "negative"  # High inflation
                elif annual_change < 0:
                    status = "negative"  # Deflation
        
        elif indicator == "GDP" or indicator == "GDPC1":
            # For GDP, growth is good
            if trends.get("1_period", 0) > 0.005:
                status = "positive"  # Good growth
            elif trends.get("1_period", 0) < 0:
                status = "negative"  # Contraction
        
        elif indicator == "FEDFUNDS":
            # For Fed Funds rate, context matters based on inflation and growth
            # This is a simplified approach
            if value < 0.005:
                status = "mixed"  # Zero lower bound may indicate economic stress
            elif 0.02 <= value <= 0.045:
                status = "neutral"  # Normal range
            elif value > 0.05:
                status = "mixed"  # High rates may slow growth
        
        return status
    
    def _get_overall_assessment(self, indicators: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Create an overall assessment of economic conditions.
        
        Args:
            indicators (Dict): Dictionary of indicator summaries
            
        Returns:
            Dict: Overall economic assessment
        """
        status_counts = {"positive": 0, "negative": 0, "neutral": 0, "mixed": 0}
        
        for indicator, data in indicators.items():
            status = data.get("status")
            if status in status_counts:
                status_counts[status] += 1
        
        # Determine overall status
        if status_counts["positive"] > status_counts["negative"]:
            overall_status = "positive"
            description = "Economic indicators showing strength"
        elif status_counts["negative"] > status_counts["positive"]:
            overall_status = "negative"
            description = "Economic indicators showing weakness"
        elif status_counts["mixed"] > max(status_counts["positive"], status_counts["negative"]):
            overall_status = "mixed"
            description = "Economic indicators showing mixed signals"
        else:
            overall_status = "neutral"
            description = "Economic indicators in neutral territory"
        
        # Add specific details based on indicators
        details = []
        
        if "GDP" in indicators and indicators["GDP"]["trends"].get("1_period") is not None:
            gdp_growth = indicators["GDP"]["trends"].get("1_period")
            if gdp_growth > 0:
                details.append(f"GDP growing at {gdp_growth:.1%}")
            else:
                details.append(f"GDP contracting at {-gdp_growth:.1%}")
        
        if "UNRATE" in indicators:
            details.append(f"Unemployment rate at {indicators['UNRATE']['value']:.1f}%")
        
        if "CPIAUCSL" in indicators and indicators["CPIAUCSL"]["trends"].get("12_period") is not None:
            inflation = indicators["CPIAUCSL"]["trends"].get("12_period")
            details.append(f"Inflation running at {inflation:.1%} year-over-year")
        
        if "FEDFUNDS" in indicators:
            details.append(f"Federal Funds rate at {indicators['FEDFUNDS']['value']:.2f}%")
        
        return {
            "status": overall_status,
            "description": description,
            "details": details
        }


class MarketEconomicCorrelator:
    """Class for analyzing correlations between economic indicators and market performance."""
    
    def __init__(self, econ_manager: EconomicDataManager):
        """
        Initialize with EconomicDataManager instance.
        
        Args:
            econ_manager (EconomicDataManager): Instance of EconomicDataManager
        """
        self.econ_manager = econ_manager
    
    def get_market_correlations(self, symbol: str, indicators: List[str], 
                               lookback_years: int = 5) -> Dict[str, Any]:
        """
        Calculate correlations between market performance and economic indicators.
        
        Args:
            symbol (str): Market symbol (e.g. "^GSPC" for S&P 500)
            indicators (List[str]): List of economic indicator IDs
            lookback_years (int): Number of years to look back
            
        Returns:
            Dict: Correlation data between market and economic indicators
        """
        start_date = (datetime.now() - timedelta(days=365*lookback_years)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        try:
            # Fetch market data
            logger.info(f"Fetching market data for {symbol}")
            market_data = yf.download(symbol, start=start_date, end=end_date)
            
            if market_data.empty:
                logger.warning(f"No market data returned for {symbol}")
                return {"error": f"No market data for {symbol}"}
            
            # Calculate market returns
            market_data['return'] = market_data['Close'].pct_change()
            market_monthly = market_data['Close'].resample('M').last().pct_change().dropna()
            market_quarterly = market_data['Close'].resample('Q').last().pct_change().dropna()
            
            # Fetch economic indicators
            indicator_data = {}
            for indicator in indicators:
                data = self.econ_manager.get_economic_indicator(indicator, start_date, end_date)
                if data is not None:
                    indicator_data[indicator] = data
                    time.sleep(0.5)  # Rate limit protection
            
            if not indicator_data:
                logger.warning("Could not fetch any economic indicators")
                return {"error": "Could not fetch economic indicators"}
            
            # Calculate correlations
            correlations = {}
            lagged_correlations = {}
            granger_results = {}
            
            for indicator, data in indicator_data.items():
                if data is None or data.empty:
                    continue
                
                # Resample indicator to monthly frequency
                indicator_monthly = data[indicator].resample('M').last().dropna()
                
                # Create joint DataFrame for correlation calculation
                if not indicator_monthly.empty and not market_monthly.empty:
                    joint_df = pd.DataFrame({
                        'market': market_monthly,
                        'indicator': indicator_monthly
                    }).dropna()
                    
                    if len(joint_df) > 3:
                        # Calculate correlation
                        correlation = joint_df['market'].corr(joint_df['indicator'])
                        correlations[indicator] = correlation
                        
                        # Calculate lagged correlations (indicator leading market)
                        lagged = {}
                        for lag in [1, 3, 6, 12]:
                            if len(joint_df) > lag:
                                joint_df['indicator_lag'] = joint_df['indicator'].shift(lag)
                                lagged_corr = joint_df['market'].corr(joint_df['indicator_lag'])
                                if not pd.isna(lagged_corr):
                                    lagged[f"{lag}_months"] = lagged_corr
                        
                        lagged_correlations[indicator] = lagged
            
            return {
                "symbol": symbol,
                "indicators": indicators,
                "lookback_years": lookback_years,
                "correlations": {k: float(v) for k, v in correlations.items()},
                "lagged_correlations": lagged_correlations,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error calculating market correlations: {str(e)}")
            return {"error": str(e)}


if __name__ == "__main__":
    # Example usage
    econ_manager = EconomicDataManager()
    
    # Get unemployment data
    unemployment = econ_manager.get_economic_indicator('UNRATE')
    if unemployment is not None:
        print(f"Latest unemployment rate: {unemployment['UNRATE'].iloc[-1]}%")
    
    # Get economic summary
    summary = econ_manager.get_economic_summary()
    print("\nEconomic Summary:")
    print(f"Overall: {summary['overall_assessment']['status']} - {summary['overall_assessment']['description']}")
    
    for indicator, data in summary['indicators'].items():
        print(f"{data['name']}: {data['value']} ({data['status']})")
    
    # Calculate market correlations
    correlator = MarketEconomicCorrelator(econ_manager)
    correlations = correlator.get_market_correlations('^GSPC', ['UNRATE', 'CPIAUCSL', 'FEDFUNDS'], lookback_years=5)
    
    print("\nMarket Correlations with Economic Indicators:")
    for indicator, corr in correlations.get('correlations', {}).items():
        print(f"{indicator}: {corr:.3f}") 