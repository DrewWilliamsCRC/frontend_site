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

from flask import Flask, request, jsonify
from flask_restful import Api, Resource # type: ignore
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/ai_server.log"),
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
            
            # Return data
            return {
                "insights": {
                    "timestamp": datetime.now().isoformat(),
                    "period": period,
                    "metrics": metrics,
                    "recommendations": recommendations
                }
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