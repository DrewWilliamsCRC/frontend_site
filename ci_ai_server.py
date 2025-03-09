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

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('ci_ai_server')

# Create Flask app
app = Flask(__name__)

@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "environment": "CI",
        "message": "This is a simplified CI version of the AI server"
    })

@app.route('/api/ai-insights')
def insights():
    """Dummy insights endpoint for CI testing."""
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
    return jsonify({
        "indices": []
    })

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5002))
    app.run(host='0.0.0.0', port=port, debug=True) 