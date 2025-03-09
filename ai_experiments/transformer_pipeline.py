#!/usr/bin/env python3
"""
Transformer Pipeline for Market Prediction

This module provides functions for training and using transformer models
for market prediction, with integration into the existing AI dashboard.
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
import argparse
from typing import Dict, List, Tuple, Optional, Union, Any

import numpy as np # type: ignore
import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore

# Make sure the directory containing this file is in the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import project modules
from alpha_vantage_pipeline import (
    fetch_market_data, clean_and_process_data, compute_technical_indicators,
    DATA_DIR, MARKET_INDICES
)
from models.market_predictor import MarketPredictor

# Try to import the transformer model, but provide a fallback if not available
try:
    from models.deep_learning_models import train_market_transformer
    DEEP_LEARNING_AVAILABLE = True
except ImportError:
    DEEP_LEARNING_AVAILABLE = False
    print("Deep learning models not available. Using mock implementations for transformer predictions.")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('transformer_pipeline')

# Constants
MODELS_DIR = os.path.join(current_dir, "models")
VISUALIZATIONS_DIR = os.path.join(current_dir, "visualizations")
os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)

def prepare_data_for_transformer(symbol: str, days: int = 500) -> pd.DataFrame:
    """
    Prepare market data for transformer model training.
    
    Args:
        symbol (str): Market symbol to prepare data for
        days (int): Number of days of historical data to use
        
    Returns:
        pd.DataFrame: Processed data ready for transformer model
    """
    # Fetch market data
    data = fetch_market_data(symbol, days=days)
    if data is None:
        logger.error(f"Failed to fetch data for {symbol}")
        return None
    
    # Clean and process the data
    data = clean_and_process_data(data)
    if data is None:
        logger.error(f"Failed to process data for {symbol}")
        return None
    
    # Add technical indicators
    data = compute_technical_indicators(data)
    if data is None:
        logger.error(f"Failed to compute technical indicators for {symbol}")
        return None
    
    logger.info(f"Prepared data for {symbol}, shape: {data.shape}")
    return data

def generate_market_predictions_for_dashboard(market_indices=None, prediction_days=5):
    """
    Generate market predictions for the dashboard using transformer models.
    
    Args:
        market_indices (dict): Dictionary of market indices to predict (default: MARKET_INDICES)
        prediction_days (int): Number of days to predict
        
    Returns:
        dict: Dictionary of predictions for each market index
    """
    if market_indices is None:
        market_indices = MARKET_INDICES
    
    predictions = {}
    
    # Check if deep learning models are available
    if not DEEP_LEARNING_AVAILABLE:
        logger.warning("Deep learning models not available. Using mock predictions.")
        return generate_mock_predictions(market_indices, prediction_days)
    
    # Try to use MarketPredictor with transformer model
    try:
        for index_name, symbol in market_indices.items():
            try:
                logger.info(f"Generating predictions for {index_name} ({symbol})")
                
                # Prepare data
                data = prepare_data_for_transformer(symbol, days=500)
                if data is None:
                    continue
                
                # Create predictor with transformer model
                predictor = MarketPredictor(model_type='transformer', market_index=index_name)
                
                # Generate predictions
                prediction = predictor.predict(data, prediction_days=prediction_days)
                if prediction:
                    predictions[index_name] = prediction
                    logger.info(f"Generated prediction for {index_name}: {prediction['direction']} with {prediction['confidence']:.2f} confidence")
                else:
                    logger.warning(f"Failed to generate prediction for {index_name}")
            except Exception as e:
                logger.error(f"Error predicting {index_name}: {e}")
        
        # If no predictions were generated, use mock predictions
        if not predictions:
            logger.warning("No predictions generated. Using mock predictions.")
            predictions = generate_mock_predictions(market_indices, prediction_days)
        
        return predictions
        
    except Exception as e:
        logger.error(f"Error in generate_market_predictions: {e}")
        return generate_mock_predictions(market_indices, prediction_days)

def generate_mock_predictions(market_indices, prediction_days=5):
    """
    Generate mock predictions when the transformer model is not available.
    
    Args:
        market_indices (dict): Dictionary of market indices to predict
        prediction_days (int): Number of days to predict
        
    Returns:
        dict: Dictionary of mock predictions for each market index
    """
    logger.info("Generating mock predictions")
    mock_data = {}
    
    for index_name, symbol in market_indices.items():
        # Generate random direction, magnitude, and confidence
        direction = 'up' if np.random.random() > 0.4 else 'down'
        magnitude = np.random.uniform(0.5, 2.5)
        confidence = np.random.uniform(0.6, 0.85)
        
        # Use realistic values for current_price
        index_prices = {
            'DJI': 38500.0,
            'SPX': 5100.0,
            'IXIC': 16200.0,
            'VIX': 15.0,
            'TNX': 4.2
        }
        current_price = index_prices.get(index_name, 100.0) * (1 + np.random.uniform(-0.01, 0.01))
        
        # Calculate predicted price
        symbol_key = symbol.replace('^', '')  # Remove ^ for indices
        factor = 1 + (magnitude / 100) if direction == 'up' else 1 - (magnitude / 100)
        predicted_price = current_price * factor
        
        # Generate dates
        today = datetime.now()
        prediction_date = today + timedelta(days=1)
        
        mock_data[symbol_key] = {
            'symbol': symbol,
            'latest_date': today.strftime('%Y-%m-%d'),
            'latest_close': float(current_price),
            'prediction_dates': [
                (today + timedelta(days=i+1)).strftime('%Y-%m-%d') for i in range(prediction_days)
            ],
            'predicted_prices': [float(current_price * (1 + (i+1) * ((magnitude/100) if direction == 'up' else -(magnitude/100)))) for i in range(prediction_days)],
            'direction': direction,
            'magnitude': float(magnitude),
            'confidence': float(confidence),
            'model_type': 'transformer (mocked)'
        }
    
    return mock_data

def train_transformer_models(market_indices=None, epochs=50, batch_size=32, seq_length=60):
    """
    Train transformer models for market indices.
    
    Args:
        market_indices (dict): Dictionary of market indices to train models for
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        seq_length (int): Sequence length for time series
        
    Returns:
        dict: Dictionary of training results
    """
    if not DEEP_LEARNING_AVAILABLE:
        logger.warning("Deep learning models not available. Cannot train models.")
        return None
    
    if market_indices is None:
        market_indices = MARKET_INDICES
    
    results = {}
    
    for index_name, symbol in market_indices.items():
        try:
            logger.info(f"Training transformer model for {index_name} ({symbol})")
            
            # Prepare data
            data = prepare_data_for_transformer(symbol, days=1000)
            if data is None:
                logger.error(f"Failed to prepare data for {index_name}")
                continue
            
            # Train model
            model, history = train_market_transformer(
                data,
                market_index=index_name,
                epochs=epochs,
                batch_size=batch_size,
                seq_length=seq_length
            )
            
            if model:
                logger.info(f"Successfully trained model for {index_name}")
                results[index_name] = {"status": "success"}
            else:
                logger.warning(f"Failed to train model for {index_name}")
                results[index_name] = {"status": "failed"}
                
        except Exception as e:
            logger.error(f"Error training model for {index_name}: {e}")
            results[index_name] = {"status": "error", "message": str(e)}
    
    return results

def main():
    """Command-line interface for the transformer pipeline."""
    parser = argparse.ArgumentParser(description="Train and use transformer models for market prediction")
    parser.add_argument("--train", action="store_true", help="Train transformer models")
    parser.add_argument("--predict", action="store_true", help="Generate predictions")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--seq-length", type=int, default=60, help="Sequence length for time series")
    parser.add_argument("--prediction-days", type=int, default=5, help="Number of days to predict")
    parser.add_argument("--output", type=str, help="Output file for predictions (JSON)")
    
    args = parser.parse_args()
    
    if args.train:
        logger.info("Training transformer models")
        results = train_transformer_models(
            epochs=args.epochs,
            batch_size=args.batch_size,
            seq_length=args.seq_length
        )
        print(json.dumps(results, indent=2))
    
    if args.predict:
        logger.info("Generating predictions")
        predictions = generate_market_predictions_for_dashboard(
            prediction_days=args.prediction_days
        )
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(predictions, f, indent=2)
            logger.info(f"Saved predictions to {args.output}")
        else:
            print(json.dumps(predictions, indent=2))

if __name__ == "__main__":
    main() 