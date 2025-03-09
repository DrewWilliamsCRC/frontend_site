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
from models.deep_learning_models import train_market_transformer

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
        symbol (str): Market symbol
        days (int): Number of days of historical data
        
    Returns:
        pd.DataFrame: Prepared data
    """
    # Fetch and process market data
    logger.info(f"Preparing data for {symbol}")
    
    try:
        # First try to load from cached file
        data_file = os.path.join(DATA_DIR, f"{symbol}_daily.csv")
        if os.path.exists(data_file):
            data = pd.read_csv(data_file, index_col=0, parse_dates=True)
            logger.info(f"Loaded data from {data_file}")
        else:
            # Fetch from Alpha Vantage if file doesn't exist
            data = fetch_market_data(symbol, interval='daily')
            logger.info(f"Fetched data from Alpha Vantage API")
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {str(e)}")
        raise
    
    # Ensure we have enough data
    if len(data) < days:
        logger.warning(f"Only {len(data)} days of data available for {symbol}")
    
    # Clean and process data
    data = clean_and_process_data(data)
    
    # Compute technical indicators
    data = compute_technical_indicators(data)
    
    # Drop rows with NaN values
    data = data.dropna()
    
    # Limit to the requested number of days
    data = data.iloc[-days:]
    
    logger.info(f"Prepared {len(data)} days of data with {len(data.columns)} features")
    
    return data

def train_transformer_model(symbol: str, target_column: str = 'close', 
                           seq_len: int = 20, pred_len: int = 5, 
                           epochs: int = 50) -> Tuple[Any, Any, Dict[str, List[float]]]:
    """
    Train a transformer model for market prediction.
    
    Args:
        symbol (str): Market symbol
        target_column (str): Target column for prediction
        seq_len (int): Sequence length for input
        pred_len (int): Prediction length for output
        epochs (int): Number of training epochs
        
    Returns:
        Tuple: (model, scaler, training_history)
    """
    # Prepare data
    data = prepare_data_for_transformer(symbol)
    
    # Train transformer model
    logger.info(f"Training transformer model for {symbol}")
    transformer, scaler, history = train_market_transformer(
        data, 
        target_column=target_column,
        sequence_length=seq_len,
        prediction_length=pred_len,
        epochs=epochs
    )
    
    # Save model
    model_path = os.path.join(MODELS_DIR, f"transformer_{symbol}.pt")
    transformer.save(model_path)
    
    logger.info(f"Transformer model for {symbol} trained and saved to {model_path}")
    
    return transformer, scaler, history

def plot_training_history(history: Dict[str, List[float]], symbol: str) -> str:
    """
    Plot training history and save to file.
    
    Args:
        history (Dict[str, List[float]]): Training history
        symbol (str): Market symbol
        
    Returns:
        str: Path to saved plot file
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title(f'Transformer Model Training History ({symbol})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    plot_path = os.path.join(VISUALIZATIONS_DIR, f"transformer_training_{symbol}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_path

def plot_prediction_performance(data: pd.DataFrame, 
                               predictions: np.ndarray, 
                               symbol: str,
                               target_column: str = 'close',
                               seq_len: int = 20,
                               pred_len: int = 5) -> str:
    """
    Plot prediction performance against actual values.
    
    Args:
        data (pd.DataFrame): Input data
        predictions (np.ndarray): Model predictions
        symbol (str): Market symbol
        target_column (str): Target column name
        seq_len (int): Sequence length
        pred_len (int): Prediction length
        
    Returns:
        str: Path to saved plot file
    """
    # Extract actual values
    actual = data[target_column].iloc[seq_len:].values
    
    # Reshape predictions if needed
    if len(predictions.shape) > 1 and predictions.shape[1] > 1:
        # Find the index of the target column
        if target_column in data.columns:
            target_idx = data.columns.get_loc(target_column)
            predictions = predictions[:, :, target_idx]
        else:
            # If we can't find the target column, just use the first feature
            predictions = predictions[:, :, 0]
    
    # Reshape predictions to match actual
    pred_flat = predictions.reshape(-1)[:len(actual)]
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    # Plot actual values
    plt.plot(actual, label='Actual', color='blue', alpha=0.7)
    
    # Plot predictions with shift to account for sequence length
    plt.plot(np.arange(seq_len, seq_len + len(pred_flat)), 
             pred_flat, label='Predicted', color='red', linestyle='--')
    
    # Add title and labels
    plt.title(f'Transformer Model Predictions vs. Actual ({symbol} - {target_column})')
    plt.xlabel('Time Steps')
    plt.ylabel(f'{target_column.capitalize()} Value')
    plt.legend()
    plt.grid(True)
    
    # Add info text
    plt.figtext(0.02, 0.02, 
                f"Model: Transformer\nSequence Length: {seq_len}\n"
                f"Prediction Length: {pred_len}\nDate: {datetime.now().strftime('%Y-%m-%d')}",
                fontsize=9)
    
    # Save plot
    plot_path = os.path.join(VISUALIZATIONS_DIR, f"transformer_prediction_{symbol}_{target_column}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_path

def generate_market_predictions_for_dashboard(symbols: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Generate market predictions using transformer model for the dashboard.
    
    Args:
        symbols (Optional[List[str]]): List of market symbols to predict
        
    Returns:
        Dict[str, Any]: Prediction results formatted for dashboard
    """
    # Use default market indices if symbols not provided
    if symbols is None:
        symbols = list(MARKET_INDICES.values())
    
    results = {}
    
    for symbol in symbols:
        # Clean symbol name for file operations
        clean_symbol = symbol.replace('^', '')
        
        try:
            # Prepare data
            data = prepare_data_for_transformer(symbol)
            
            # Check if we have a saved model
            model_path = os.path.join(MODELS_DIR, f"transformer_{clean_symbol}.pt")
            
            if os.path.exists(model_path):
                # Load existing model
                from models.deep_learning_models import MarketPredictionTransformer
                transformer = MarketPredictionTransformer.load(model_path)
                logger.info(f"Loaded existing transformer model for {symbol}")
            else:
                # Train new model
                transformer, _, _ = train_transformer_model(
                    symbol, 
                    target_column='close',
                    seq_len=20,
                    pred_len=5,
                    epochs=30  # Reduced epochs for quick results
                )
                logger.info(f"Trained new transformer model for {symbol}")
            
            # Make predictions on the latest data
            input_data = data.values
            if hasattr(transformer, 'scaler') and transformer.scaler is not None:
                input_data = transformer.scaler.transform(input_data)
            
            predictions = transformer.predict(input_data)
            
            # Format results for dashboard
            latest_date = data.index[-1]
            prediction_dates = [latest_date + timedelta(days=i+1) for i in range(5)]
            
            # Extract close price predictions
            close_idx = data.columns.get_loc('close')
            close_predictions = predictions[:, :, close_idx].reshape(-1)[-5:]
            
            # Calculate confidence based on model loss history (if available)
            confidence = 0.8  # Default confidence
            
            # Calculate direction and magnitude
            latest_close = data['close'].iloc[-1]
            direction = 'up' if close_predictions[0] > latest_close else 'down'
            magnitude = abs(close_predictions[0] - latest_close) / latest_close * 100
            
            # Store results
            results[clean_symbol] = {
                'symbol': symbol,
                'latest_date': latest_date.strftime('%Y-%m-%d'),
                'latest_close': float(latest_close),
                'prediction_dates': [d.strftime('%Y-%m-%d') for d in prediction_dates],
                'predicted_prices': close_predictions.tolist(),
                'direction': direction,
                'magnitude': float(magnitude),
                'confidence': float(confidence),
                'model_type': 'transformer'
            }
            
            logger.info(f"Generated predictions for {symbol}: {direction.upper()} {magnitude:.2f}%")
            
        except Exception as e:
            logger.error(f"Error generating predictions for {symbol}: {str(e)}")
            continue
    
    # Save results to file for dashboard to load
    results_file = os.path.join(DATA_DIR, "transformer_predictions.json")
    with open(results_file, 'w') as f:
        json.dump({
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'predictions': results
        }, f, indent=2)
    
    logger.info(f"Saved prediction results to {results_file}")
    
    return results

def main():
    """Main function to run transformer pipeline."""
    parser = argparse.ArgumentParser(description='Transformer Pipeline for Market Prediction')
    parser.add_argument('--symbols', nargs='+', help='Market symbols to analyze', 
                        default=['^GSPC', '^DJI', '^IXIC'])
    parser.add_argument('--days', type=int, default=500, help='Days of historical data')
    parser.add_argument('--seq-len', type=int, default=20, help='Sequence length')
    parser.add_argument('--pred-len', type=int, default=5, help='Prediction length')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--dashboard', action='store_true', help='Generate dashboard predictions')
    
    args = parser.parse_args()
    
    if args.dashboard:
        # Generate predictions for dashboard
        generate_market_predictions_for_dashboard(args.symbols)
        return
    
    # Process each symbol
    for symbol in args.symbols:
        try:
            # Train model and get predictions
            transformer, scaler, history = train_transformer_model(
                symbol,
                seq_len=args.seq_len,
                pred_len=args.pred_len,
                epochs=args.epochs
            )
            
            # Plot training history
            history_plot = plot_training_history(history, symbol)
            logger.info(f"Saved training history plot to {history_plot}")
            
            # Prepare data for prediction plot
            data = prepare_data_for_transformer(symbol, days=args.days)
            
            # Make predictions
            input_data = data.values
            if hasattr(scaler, 'transform'):
                input_data = scaler.transform(input_data)
            
            predictions = transformer.predict(input_data)
            
            # Plot prediction performance
            pred_plot = plot_prediction_performance(
                data, 
                predictions, 
                symbol,
                seq_len=args.seq_len,
                pred_len=args.pred_len
            )
            logger.info(f"Saved prediction performance plot to {pred_plot}")
            
        except Exception as e:
            logger.error(f"Error processing {symbol}: {str(e)}")
            continue

if __name__ == "__main__":
    main() 