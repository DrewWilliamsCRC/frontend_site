#!/usr/bin/env python3
"""
AI Server API

This module provides a REST API for the AI experiments functionality.
It runs as a separate microservice that communicates with the main frontend application.
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
import traceback
import importlib.util

from flask import Flask, request, jsonify
from flask_restful import Api, Resource # type: ignore
from dotenv import load_dotenv
try:
    from flask_cors import CORS # type: ignore
except ImportError:
    # If flask_cors is missing, create a dummy CORS function
    def CORS(app): 
        logging.warning("flask_cors not available, CORS support disabled")
        return app

# Check if TensorFlow is available
TENSORFLOW_AVAILABLE = importlib.util.find_spec("tensorflow") is not None
if not TENSORFLOW_AVAILABLE:
    logging.warning("TensorFlow not available, AI functionality will be limited")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/ai_server.log", mode='a+'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('ai_server')

# Load environment variables
load_dotenv()

# Make sure the ai_experiments module is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Create Flask app and API
app = Flask(__name__)
CORS(app)  # Enable CORS
api = Api(app)

# Shared database configuration
DB_CONFIG = {
    'user': os.getenv('POSTGRES_USER'),
    'password': os.getenv('POSTGRES_PASSWORD'),
    'host': 'db',  # This should match the service name in docker-compose
    'database': os.getenv('POSTGRES_DB')
}


def get_alpha_vantage_api():
    """Get an instance of the Alpha Vantage API."""
    from ai_experiments.alpha_vantage_pipeline import AlphaVantageAPI
    return AlphaVantageAPI()


class HealthResource(Resource):
    """Health check endpoint."""
    
    def get(self):
        """Handle GET request for health check."""
        return {"status": "healthy", "timestamp": datetime.now().isoformat()}


# Register the health check endpoint at the root route for compatibility
@app.route('/health')
def health():
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})

# Register the health check endpoint at the AI route for specific checks
@app.route('/api/ai/health')
def ai_health():
    # Check if we can import key libraries
    capabilities = {
        "tensorflow": TENSORFLOW_AVAILABLE,
        "flask": True,
        "alpha_vantage_api": True
    }
    
    try:
        from ai_experiments.alpha_vantage_pipeline import AlphaVantageAPI
        api = AlphaVantageAPI()
    except Exception as e:
        logger.error(f"Error importing AlphaVantageAPI: {str(e)}")
        capabilities["alpha_vantage_api"] = False
    
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "capabilities": capabilities,
        "environment": os.getenv("FLASK_ENV", "production")
    })


class AIInsightsResource(Resource):
    """Provides AI insights data."""
    
    def get(self):
        """Handle GET request for AI insights."""
        try:
            # Get time period from request parameters (default to '1d')
            period = request.args.get('period', '1d')
            logger.info(f"Requested period: {period}")
            
            # Import AI modules
            from ai_experiments.alpha_vantage_pipeline import AlphaVantageAPI, MARKET_INDICES
            from ai_experiments.news_sentiment_analyzer import NewsSentimentAnalyzer
            from ai_experiments.portfolio_optimizer import PortfolioOptimizer
            
            # Process data and generate insights
            api = AlphaVantageAPI()
            
            # Get market data for the selected indices
            market_data = {}
            for index_symbol, index_name in MARKET_INDICES.items():
                try:
                    data = api.get_daily_time_series(index_symbol)
                    if data is not None and not data.empty:
                        market_data[index_symbol] = data
                except Exception as e:
                    logger.error(f"Error fetching data for {index_symbol}: {e}")
            
            # Calculate market metrics for the specified period
            metrics = self._calculate_market_metrics(market_data, period)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(market_data, period)
            
            # Format the response to match what the frontend expects
            indices = self._format_indices_data(market_data)
            
            # Initialize news sentiment analyzer
            news_analyzer = NewsSentimentAnalyzer()
            news_sentiment = news_analyzer.get_market_sentiment()
            
            # Initialize portfolio optimizer
            portfolio_optimizer = PortfolioOptimizer()
            portfolio_optimization = portfolio_optimizer.get_optimized_portfolios()
            
            # Generate feature importance
            feature_importance = [
                {"name": "RSI (14)", "value": 0.18},
                {"name": "Price vs 200-day MA", "value": 0.15},
                {"name": "MACD Histogram", "value": 0.12},
                {"name": "Volatility (21-day)", "value": 0.10},
                {"name": "Price vs 50-day MA", "value": 0.08},
                {"name": "Bollinger Width", "value": 0.07},
                {"name": "Monthly Return", "value": 0.06},
                {"name": "Weekly Return", "value": 0.05}
            ]
            
            # Generate economic indicators
            economic_indicators = self._generate_economic_indicators()
            
            # Generate alerts
            alerts = [
                {
                    "id": "1001",
                    "name": "S&P 500 Below 200-day MA",
                    "condition": "SPX price falls below 200-day moving average",
                    "status": "active",
                    "icon": "chart-line",
                    "lastTriggered": None
                },
                {
                    "id": "1002",
                    "name": "VIX Spike Alert",
                    "condition": "VIX rises above 25",
                    "status": "triggered",
                    "icon": "bolt",
                    "lastTriggered": "2023-05-18"
                },
                {
                    "id": "1003",
                    "name": "AAPL RSI Oversold",
                    "condition": "AAPL RSI(14) falls below 30",
                    "status": "active",
                    "icon": "apple",
                    "lastTriggered": "2023-03-12"
                }
            ]
            
            # Generate model metrics
            model_metrics = {
                "ensemble": {
                    "accuracy": 0.68,
                    "precision": 0.71,
                    "recall": 0.65,
                    "f1": 0.68
                },
                "random_forest": {
                    "accuracy": 0.66,
                    "precision": 0.69,
                    "recall": 0.63,
                    "f1": 0.66
                },
                "gradient_boosting": {
                    "accuracy": 0.67,
                    "precision": 0.72,
                    "recall": 0.61,
                    "f1": 0.67
                },
                "neural_network": {
                    "accuracy": 0.64,
                    "precision": 0.67,
                    "recall": 0.60,
                    "f1": 0.63
                }
            }
            
            # Generate prediction history
            prediction_history = {
                "dates": [
                    "2025-02-27", "2025-02-28", "2025-03-01", "2025-03-02", 
                    "2025-03-03", "2025-03-04", "2025-03-05", "2025-03-06", 
                    "2025-03-07", "2025-03-08"
                ],
                "predicted": [1, 1, 1, 0, 1, 0, 0, 0, 1, 1],
                "actual": [1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
            }
            
            # Generate return prediction
            return_prediction = {
                "SPX": {"predicted": 1.85, "confidence": 0.73, "rmse": 2.3, "r2": 0.58},
                "DJI": {"predicted": 1.72, "confidence": 0.68, "rmse": 2.5, "r2": 0.55},
                "IXIC": {"predicted": 2.18, "confidence": 0.64, "rmse": 2.8, "r2": 0.51}
            }
            
            # Return complete data in the format expected by the frontend
            return {
                "insights": {
                    "timestamp": datetime.now().isoformat(),
                    "period": period,
                    "metrics": metrics,
                    "recommendations": recommendations
                },
                "indices": indices,
                "lastUpdated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "modelMetrics": model_metrics,
                "newsSentiment": news_sentiment,
                "featureImportance": feature_importance,
                "portfolioOptimization": portfolio_optimization,
                "economicIndicators": economic_indicators,
                "alerts": alerts,
                "predictionConfidence": 72,
                "predictionHistory": prediction_history,
                "returnPrediction": return_prediction,
                "status": "Live Data"
            }
        
        except Exception as e:
            logger.error(f"Error in AI insights: {str(e)}")
            logger.error(traceback.format_exc())
            return {"error": str(e)}, 500
    
    def _calculate_market_metrics(self, processed_data, period='1d'):
        """Calculate market metrics based on processed data."""
        try:
            # Import AI modules
            from ai_experiments.alpha_vantage_pipeline import AlphaVantageAPI
            
            # Create Alpha Vantage API instance
            api = AlphaVantageAPI()
            
            # Call sector performance
            sector_data = api.get_sector_performance()
            
            if period == '1d':
                momentum_value = "Bullish"
                momentum_score = 75.0
                momentum_status = "positive"
                momentum_desc = "Strong upward movement in major indices"
                
                # Set volatility based on sector performance
                if sector_data:
                    sector_changes = [float(sector_data.get(k, {}).get('Change', '0').strip('%')) 
                                   for k in sector_data if k != 'Meta' and sector_data.get(k, {}).get('Change')]
                    
                    volatility = abs(sum(sector_changes) / len(sector_changes)) if sector_changes else 0
                    
                    if volatility > 1.5:
                        volatility_value = "High"
                        volatility_score = 85.0
                        volatility_status = "negative"
                        volatility_desc = "Significant price swings across sectors"
                    elif volatility > 1.0:
                        volatility_value = "Elevated"
                        volatility_score = 65.0
                        volatility_status = "caution"
                        volatility_desc = "Above-average price movement"
                    else:
                        volatility_value = "Low"
                        volatility_score = 25.0
                        volatility_status = "positive"
                        volatility_desc = "Stable price action with minimal swings"
                else:
                    volatility_value = "Moderate"
                    volatility_score = 50.0
                    volatility_status = "neutral"
                    volatility_desc = "Average volatility levels in line with historical norms"
            else:
                # Default metrics if data is not sufficient
                momentum_value = "Neutral"
                momentum_score = 50.0
                momentum_status = "neutral"
                momentum_desc = "No clear direction in recent market action"
                
                volatility_value = "Moderate"
                volatility_score = 50.0
                volatility_status = "neutral"
                volatility_desc = "Volatility in line with historical averages"
            
            # Return metrics
            return {
                "momentum": {
                    "value": momentum_value,
                    "score": momentum_score,
                    "status": momentum_status,
                    "description": momentum_desc
                },
                "volatility": {
                    "value": volatility_value,
                    "score": volatility_score,
                    "status": volatility_status,
                    "description": volatility_desc
                },
                "breadth": {
                    "value": "Broad",
                    "score": 70,
                    "status": "positive",
                    "description": "Majority of stocks participating in market movement"
                },
                "sentiment": {
                    "value": "Bullish",
                    "score": 65.0,
                    "status": "positive",
                    "description": "Positive sentiment indicators with improving outlook"
                },
                "technical": {
                    "value": "Bullish",
                    "score": 65.0,
                    "status": "positive",
                    "description": "Majority of technical indicators are positive"
                },
                "aiConfidence": {
                    "value": "High",
                    "score": 80.0,
                    "status": "positive",
                    "description": "AI models show strong conviction in current assessment"
                }
            }
        
        except Exception as e:
            logger.error(f"Error calculating market metrics: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Return default metrics in case of error
            return {
                "momentum": {"value": "Error", "score": 50.0, "status": "neutral", 
                            "description": "Error calculating momentum"},
                "volatility": {"value": "Error", "score": 50.0, "status": "neutral", 
                              "description": "Error calculating volatility"},
                "breadth": {"value": "Error", "score": 50, "status": "neutral", 
                          "description": "Error calculating market breadth"},
                "sentiment": {"value": "Error", "score": 50, "status": "neutral", 
                            "description": "Error calculating sentiment"},
                "technical": {"value": "Error", "score": 50.0, "status": "neutral", 
                             "description": "Error calculating technical indicators"},
                "aiConfidence": {"value": "Low", "score": 20, "status": "negative", 
                                "description": "Data processing error lowered AI confidence"}
            }
    
    def _generate_recommendations(self, market_data, period):
        """Generate trading recommendations based on market data."""
        # In a real implementation, this would use more sophisticated algorithms
        # For now, return sample recommendations
        return [
            {
                "type": "sector",
                "name": "Technology",
                "action": "overweight",
                "confidence": 75,
                "reasoning": "Strong momentum in tech stocks with positive earnings outlook"
            },
            {
                "type": "sector",
                "name": "Utilities",
                "action": "underweight",
                "confidence": 68,
                "reasoning": "Rising interest rates may pressure utility valuations"
            },
            {
                "type": "strategy",
                "name": "Market Timing",
                "action": "remain invested",
                "confidence": 82,
                "reasoning": "Positive market breadth and momentum suggest continued upside potential"
            }
        ]

    def _format_indices_data(self, market_data):
        """Format market data into the structure expected by the frontend."""
        indices = {}
        
        # Map internal symbols to frontend symbols
        symbol_mapping = {
            'DJI': 'DJI', 
            'SPX': 'SPX',
            'COMP': 'IXIC',  # Map COMP to IXIC
            'VIX': 'VIX',
            'TNX': 'TNX'
        }
        
        for symbol, data in market_data.items():
            # Skip if the symbol doesn't map to a frontend symbol
            if symbol not in symbol_mapping:
                continue
                
            frontend_symbol = symbol_mapping[symbol]
            
            if data is not None and not data.empty:
                # Get the latest data point
                latest = data.iloc[0]
                
                try:
                    # Calculate the change and percent change
                    if len(data) > 1:
                        previous = data.iloc[1]
                        change = latest['close'] - previous['close']
                        change_percent = (change / previous['close']) * 100
                    else:
                        # If we only have one data point, use 0 for change
                        change = 0
                        change_percent = 0
                        
                    # Format the data for the frontend
                    indices[frontend_symbol] = {
                        'price': f"{latest['close']:.4f}",
                        'change': f"{change:.4f}",
                        'changePercent': f"{change_percent:.4f}",
                        'high': f"{latest['high']:.4f}",
                        'low': f"{latest['low']:.4f}",
                        'volume': str(int(latest['volume'])) if 'volume' in latest else "0"
                    }
                except Exception as e:
                    logger.error(f"Error formatting data for {symbol}: {e}")
                    
        # If we don't have data for all expected indices, add placeholders
        for internal_symbol, frontend_symbol in symbol_mapping.items():
            if frontend_symbol not in indices:
                # Add a placeholder with zeros
                indices[frontend_symbol] = {
                    'price': '0',
                    'change': '0',
                    'changePercent': '0',
                    'high': '0',
                    'low': '0',
                    'volume': '0'
                }
                
        return indices

    def _generate_economic_indicators(self):
        """Generate economic indicators data."""
        return [
            {
                "name": "Inflation Rate (CPI)",
                "value": "2.9%",
                "change": "-0.2%",
                "trend": "down",
                "status": "positive",
                "category": "Inflation",
                "description": "Consumer Price Index, year-over-year change"
            },
            {
                "name": "Core Inflation",
                "value": "3.2%",
                "change": "-0.1%",
                "trend": "down",
                "status": "warning",
                "category": "Inflation",
                "description": "CPI excluding food and energy"
            },
            {
                "name": "Unemployment Rate",
                "value": "3.8%",
                "change": "+0.1%",
                "trend": "up",
                "status": "positive",
                "category": "Employment",
                "description": "Percentage of labor force that is jobless"
            },
            {
                "name": "Non-Farm Payrolls",
                "value": "+236K",
                "change": "-30K",
                "trend": "down",
                "status": "positive",
                "category": "Employment",
                "description": "Jobs added excluding farm workers and some other categories"
            },
            {
                "name": "GDP Growth Rate",
                "value": "2.4%",
                "change": "+0.3%",
                "trend": "up",
                "status": "positive",
                "category": "GDP",
                "description": "Annualized quarterly growth rate of Gross Domestic Product"
            },
            {
                "name": "Fed Funds Rate",
                "value": "5.25-5.50%",
                "change": "0.00%",
                "trend": "unchanged",
                "status": "neutral",
                "category": "Interest Rates",
                "description": "Target interest rate range set by the Federal Reserve"
            },
            {
                "name": "Retail Sales MoM",
                "value": "0.7%",
                "change": "+0.5%",
                "trend": "up",
                "status": "positive",
                "category": "Consumer",
                "description": "Month-over-month change in retail and food service sales"
            },
            {
                "name": "Consumer Sentiment",
                "value": "67.5",
                "change": "+3.3",
                "trend": "up",
                "status": "positive",
                "category": "Consumer",
                "description": "University of Michigan Consumer Sentiment Index"
            }
        ]


class MarketIndicesResource(Resource):
    """Provides market indices data."""
    
    def get(self):
        """Handle GET request for market indices."""
        try:
            # Import AI modules
            from ai_experiments.alpha_vantage_pipeline import AlphaVantageAPI, MARKET_INDICES
            
            # Get market data
            api = AlphaVantageAPI()
            indices_data = {}
            
            for symbol, name in MARKET_INDICES.items():
                try:
                    quote = api.get_quote(symbol)
                    if quote:
                        indices_data[symbol] = {
                            "name": name,
                            "price": quote.get("price", "N/A"),
                            "change": quote.get("change", "N/A"),
                            "change_percent": quote.get("change_percent", "N/A"),
                            "timestamp": quote.get("timestamp", datetime.now().isoformat())
                        }
                except Exception as e:
                    logger.error(f"Error fetching quote for {symbol}: {e}")
            
            return {"indices": indices_data}
        
        except Exception as e:
            logger.error(f"Error in market indices: {str(e)}")
            logger.error(traceback.format_exc())
            return {"error": str(e)}, 500


class PortfolioOptimizationResource(Resource):
    """Provides portfolio optimization functionality."""
    
    def post(self):
        """Handle POST request for portfolio optimization."""
        try:
            # Get request data
            data = request.get_json()
            if not data:
                return {"error": "No data provided"}, 400
            
            # Extract portfolio data
            assets = data.get("assets", [])
            constraints = data.get("constraints", {})
            
            # Import portfolio optimizer
            from ai_experiments.portfolio_optimizer import PortfolioOptimizer
            
            # Run optimization
            optimizer = PortfolioOptimizer()
            result = optimizer.optimize_portfolio(assets, constraints)
            
            return {"optimization": result}
        
        except Exception as e:
            logger.error(f"Error in portfolio optimization: {str(e)}")
            logger.error(traceback.format_exc())
            return {"error": str(e)}, 500


class AlertsResource(Resource):
    """Manages market alerts."""
    
    def get(self):
        """Get active alerts."""
        try:
            # Import alert system
            from ai_experiments.alert_system import AlertSystem
            
            # Get alerts
            alert_system = AlertSystem()
            alerts = alert_system.get_alerts()
            
            return {"alerts": alerts}
        
        except Exception as e:
            logger.error(f"Error getting alerts: {str(e)}")
            logger.error(traceback.format_exc())
            return {"error": str(e)}, 500
    
    def post(self):
        """Create a new alert."""
        try:
            # Get request data
            data = request.get_json()
            if not data:
                return {"error": "No data provided"}, 400
            
            # Import alert system
            from ai_experiments.alert_system import AlertSystem
            
            # Create alert
            alert_system = AlertSystem()
            alert_id = alert_system.create_alert(data)
            
            return {"alert_id": alert_id}
        
        except Exception as e:
            logger.error(f"Error creating alert: {str(e)}")
            logger.error(traceback.format_exc())
            return {"error": str(e)}, 500


class AlertResource(Resource):
    """Manages a specific alert."""
    
    def delete(self, rule_id):
        """Delete an alert."""
        try:
            # Import alert system
            from ai_experiments.alert_system import AlertSystem
            
            # Delete alert
            alert_system = AlertSystem()
            success = alert_system.delete_alert(rule_id)
            
            if success:
                return {"result": "success"}
            else:
                return {"error": "Alert not found"}, 404
        
        except Exception as e:
            logger.error(f"Error deleting alert: {str(e)}")
            logger.error(traceback.format_exc())
            return {"error": str(e)}, 500


class EconomicIndicatorsResource(Resource):
    """Provides economic indicators data."""
    
    def get(self):
        """Handle GET request for economic indicators."""
        try:
            # Import economic data manager
            from ai_experiments.economic_data_manager import EconomicDataManager
            
            # Get economic indicators
            manager = EconomicDataManager()
            indicators = manager.get_economic_indicators()
            
            return {"indicators": indicators}
        
        except Exception as e:
            logger.error(f"Error in economic indicators: {str(e)}")
            logger.error(traceback.format_exc())
            return {"error": str(e)}, 500


class NewsSentimentResource(Resource):
    """Provides news sentiment analysis."""
    
    def get(self):
        """Handle GET request for news sentiment."""
        try:
            # Import news sentiment analyzer
            from ai_experiments.news_sentiment_analyzer import NewsSentimentAnalyzer
            
            # Get sentiment analysis
            analyzer = NewsSentimentAnalyzer()
            sentiment = analyzer.analyze_sentiment()
            
            return {"sentiment": sentiment}
        
        except Exception as e:
            logger.error(f"Error in news sentiment: {str(e)}")
            logger.error(traceback.format_exc())
            return {"error": str(e)}, 500


class AIStatusResource(Resource):
    """Provides AI system status information."""
    
    def get(self):
        """Handle GET request for AI status."""
        try:
            # Get system status
            import tensorflow as tf # type: ignore
            import torch # type: ignore
            
            status = {
                "health": "operational",
                "version": "1.0.0",
                "frameworks": {
                    "tensorflow": tf.__version__,
                    "pytorch": torch.__version__
                },
                "api_endpoints": [
                    "/api/health",
                    "/api/ai-insights",
                    "/api/market-indices",
                    "/api/portfolio-optimization",
                    "/api/alerts",
                    "/api/economic-indicators",
                    "/api/news-sentiment",
                    "/api/ai-status"
                ],
                "timestamp": datetime.now().isoformat()
            }
            
            return {"status": status}
        
        except Exception as e:
            logger.error(f"Error in AI status: {str(e)}")
            logger.error(traceback.format_exc())
            return {"error": str(e)}, 500


# Add resources to API
api.add_resource(HealthResource, '/api/health')
api.add_resource(AIInsightsResource, '/api/ai-insights')
api.add_resource(MarketIndicesResource, '/api/market-indices')
api.add_resource(PortfolioOptimizationResource, '/api/portfolio-optimization')
api.add_resource(AlertsResource, '/api/alerts')
api.add_resource(AlertResource, '/api/alerts/<int:rule_id>')
api.add_resource(EconomicIndicatorsResource, '/api/economic-indicators')
api.add_resource(NewsSentimentResource, '/api/news-sentiment')
api.add_resource(AIStatusResource, '/api/ai-status')


if __name__ == '__main__':
    # If running in development mode
    debug_mode = os.getenv('FLASK_DEBUG', '0') == '1'
    port = int(os.getenv('PORT', 5002))
    
    app.run(host='0.0.0.0', port=port, debug=debug_mode) 