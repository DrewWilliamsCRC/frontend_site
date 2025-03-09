#!/usr/bin/env python3
"""
Simplified AI Server for CI Testing

This is a minimal version of the AI server used only for CI testing
to ensure the container can start without errors.
"""

import os
import logging
from flask import Flask, jsonify

# Set up logging only to stdout - no files
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('ci_ai_server')
logger.info('Starting CI AI Server')

# Create Flask app
app = Flask(__name__)

@app.route('/health')
def health():
    """Simple health check endpoint."""
    logger.info('Health check called')
    return jsonify({
        "status": "healthy",
        "environment": "CI"
    })

@app.route('/api/ai-insights')
def insights():
    """Dummy API endpoint."""
    logger.info('AI insights endpoint called')
    return jsonify({
        "message": "CI test server",
        "insights": {
            "metrics": [
                {"name": "Test Metric", "value": 100, "status": "positive"}
            ],
            "recommendations": [
                {"text": "CI test recommendation", "confidence": 1.0}
            ]
        },
        "indices": []
    })

@app.route('/api/market-indices')
def indices():
    """Dummy market indices endpoint."""
    logger.info('Market indices endpoint called')
    return jsonify({
        "indices": []
    })

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5002))
    logger.info(f'Starting server on port {port}')
    app.run(host='0.0.0.0', port=port) 