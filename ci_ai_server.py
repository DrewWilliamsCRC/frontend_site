#!/usr/bin/env python3
"""
Simplified AI Server for CI Testing

This is a minimal version of the AI server used only for CI testing
to ensure the container can start without errors.
"""

import os
import sys
import logging
from flask import Flask, jsonify

# Set up logging for CI - only to stdout
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('ci_ai_server')
logger.info('Starting CI AI Server')

# Create Flask app
app = Flask(__name__)

@app.route('/health')
def health():
    """Health check endpoint."""
    logger.info('Health check called')
    return jsonify({
        "status": "healthy",
        "environment": "CI",
        "message": "This is a simplified CI version of the AI server"
    })

@app.route('/api/ai-insights')
def insights():
    """Dummy insights endpoint for CI testing."""
    logger.info('AI insights endpoint called')
    return jsonify({
        "message": "This is a CI test server",
        "insights": {
            "metrics": [
                {"name": "Momentum", "value": 70, "status": "positive"}
            ],
            "recommendations": [
                {"text": "CI test recommendation", "confidence": 0.8}
            ]
        },
        "indices": []
    })

@app.route('/api/market-indices')
def indices():
    """Dummy market indices endpoint for CI testing."""
    logger.info('Market indices endpoint called')
    return jsonify({
        "indices": []
    })

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5002))
    logger.info(f'Starting server on port {port}')
    app.run(host='0.0.0.0', port=port, debug=True) 