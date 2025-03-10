#!/usr/bin/env python3
"""
Market Predictor ML Model

This module implements a machine learning model for predicting market movements
based on Alpha Vantage data. It includes both regression models for price prediction
and classification models for direction prediction.
"""

import os
import sys
import json
import logging
import pickle
from datetime import datetime, timedelta

import numpy as np # type: ignore
import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier # type: ignore
from sklearn.linear_model import LogisticRegression, Ridge # type: ignore
from sklearn.svm import SVC, SVR # type: ignore
from sklearn.neural_network import MLPClassifier, MLPRegressor # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
from sklearn.metrics import ( # type: ignore
    accuracy_score, precision_score, recall_score, f1_score, 
    classification_report, confusion_matrix, mean_absolute_error, 
    mean_squared_error, r2_score
)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit # type: ignore

# Add parent directory to path so we can import from alpha_vantage_pipeline
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ai_experiments.alpha_vantage_pipeline import DataProcessor, DataManager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('market_predictor')

# Constants - use pre-existing directories
MODELS_DIR = "/app/ai_experiments/models"
DATA_DIR = "/app/ai_experiments/data"
ML_READY_DIR = os.path.join(DATA_DIR, "ml_ready")


class ModelManager:
    """Class to manage multiple ML models for market prediction."""
    
    def __init__(self, index_name='SPX'):
        """Initialize the model manager.
        
        Args:
            index_name (str): Name of the market index to model
        """
        self.index_name = index_name
        self.models_dir = os.path.join(MODELS_DIR, index_name)
        if not os.path.exists(self.models_dir):
            logger.warning(f"Models directory {self.models_dir} does not exist, model saving will be disabled")
        
        # Initialize scalers and models
        self.scaler = StandardScaler()
        self.dir_models = {}  # Direction prediction models
        self.ret_models = {}  # Return prediction models
        
        # Load data if available
        self.data_loaded = False
        self.ml_data = None
        self._load_data()
    
    def _load_data(self):
        """Load preprocessed ML data for the index."""
        index_ml_dir = os.path.join(ML_READY_DIR, 'indices', self.index_name)
        
        if not os.path.exists(index_ml_dir):
            logger.warning(f"No preprocessed data found for {self.index_name}")
            return
        
        try:
            # Load feature and target data
            X_train = pd.read_csv(os.path.join(index_ml_dir, 'X_train.csv'), index_col=0)
            X_test = pd.read_csv(os.path.join(index_ml_dir, 'X_test.csv'), index_col=0)
            y_reg_train = pd.read_csv(os.path.join(index_ml_dir, 'y_reg_train.csv'), index_col=0, squeeze=True)
            y_reg_test = pd.read_csv(os.path.join(index_ml_dir, 'y_reg_test.csv'), index_col=0, squeeze=True)
            y_clf_train = pd.read_csv(os.path.join(index_ml_dir, 'y_clf_train.csv'), index_col=0, squeeze=True)
            y_clf_test = pd.read_csv(os.path.join(index_ml_dir, 'y_clf_test.csv'), index_col=0, squeeze=True)
            
            # Load metadata
            with open(os.path.join(index_ml_dir, 'metadata.json'), 'r') as f:
                metadata = json.load(f)
            
            # Store data
            self.ml_data = {
                'X_train': X_train,
                'X_test': X_test,
                'y_reg_train': y_reg_train,
                'y_reg_test': y_reg_test,
                'y_clf_train': y_clf_train,
                'y_clf_test': y_clf_test,
                'feature_columns': metadata['feature_columns']
            }
            
            # Scale features
            self.scaler.fit(X_train)
            self.ml_data['X_train_scaled'] = self.scaler.transform(X_train)
            self.ml_data['X_test_scaled'] = self.scaler.transform(X_test)
            
            self.data_loaded = True
            logger.info(f"Loaded ML data for {self.index_name}")
            
        except Exception as e:
            logger.error(f"Error loading ML data for {self.index_name}: {e}")
    
    def train_direction_models(self):
        """Train classification models for market direction prediction."""
        if not self.data_loaded:
            logger.error("No data loaded. Cannot train models.")
            return
        
        logger.info(f"Training direction prediction models for {self.index_name}")
        
        X_train = self.ml_data['X_train_scaled']
        y_train = self.ml_data['y_clf_train']
        
        # Random Forest Classifier
        logger.info("Training Random Forest Classifier...")
        rf_model = RandomForestClassifier(
            n_estimators=100, 
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )
        rf_model.fit(X_train, y_train)
        self.dir_models['random_forest'] = rf_model
        
        # Save model
        self._save_model(rf_model, 'random_forest_dir')
        
        # Gradient Boosting Classifier
        logger.info("Training Gradient Boosting Classifier...")
        gb_model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        gb_model.fit(X_train, y_train)
        self.dir_models['gradient_boosting'] = gb_model
        
        # Save model
        self._save_model(gb_model, 'gradient_boosting_dir')
        
        # Logistic Regression
        logger.info("Training Logistic Regression...")
        lr_model = LogisticRegression(
            C=1.0,
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        )
        lr_model.fit(X_train, y_train)
        self.dir_models['logistic_regression'] = lr_model
        
        # Save model
        self._save_model(lr_model, 'logistic_regression_dir')
        
        # SVM Classifier
        logger.info("Training SVM Classifier...")
        svm_model = SVC(
            C=1.0,
            kernel='rbf',
            probability=True,
            random_state=42
        )
        svm_model.fit(X_train, y_train)
        self.dir_models['svm'] = svm_model
        
        # Save model
        self._save_model(svm_model, 'svm_dir')
        
        # Neural Network Classifier
        logger.info("Training Neural Network Classifier...")
        nn_model = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            alpha=0.0001,
            max_iter=1000,
            random_state=42
        )
        nn_model.fit(X_train, y_train)
        self.dir_models['neural_network'] = nn_model
        
        # Save model
        self._save_model(nn_model, 'neural_network_dir')
        
        logger.info("Finished training direction models")
    
    def train_return_models(self):
        """Train regression models for market return prediction."""
        if not self.data_loaded:
            logger.error("No data loaded. Cannot train models.")
            return
        
        logger.info(f"Training return prediction models for {self.index_name}")
        
        X_train = self.ml_data['X_train_scaled']
        y_train = self.ml_data['y_reg_train']
        
        # Random Forest Regressor
        logger.info("Training Random Forest Regressor...")
        rf_model = RandomForestRegressor(
            n_estimators=100, 
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )
        rf_model.fit(X_train, y_train)
        self.ret_models['random_forest'] = rf_model
        
        # Save model
        self._save_model(rf_model, 'random_forest_ret')
        
        # Ridge Regression
        logger.info("Training Ridge Regression...")
        ridge_model = Ridge(
            alpha=1.0,
            random_state=42
        )
        ridge_model.fit(X_train, y_train)
        self.ret_models['ridge'] = ridge_model
        
        # Save model
        self._save_model(ridge_model, 'ridge_ret')
        
        # SVR
        logger.info("Training SVR...")
        svr_model = SVR(
            C=1.0,
            kernel='rbf'
        )
        svr_model.fit(X_train, y_train)
        self.ret_models['svr'] = svr_model
        
        # Save model
        self._save_model(svr_model, 'svr_ret')
        
        # Neural Network Regressor
        logger.info("Training Neural Network Regressor...")
        nn_model = MLPRegressor(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            alpha=0.0001,
            max_iter=1000,
            random_state=42
        )
        nn_model.fit(X_train, y_train)
        self.ret_models['neural_network'] = nn_model
        
        # Save model
        self._save_model(nn_model, 'neural_network_ret')
        
        logger.info("Finished training return models")
    
    def evaluate_direction_models(self):
        """Evaluate direction prediction models on test data."""
        if not self.data_loaded or not self.dir_models:
            logger.error("No data or models loaded. Cannot evaluate.")
            return
        
        logger.info(f"Evaluating direction prediction models for {self.index_name}")
        
        X_test = self.ml_data['X_test_scaled']
        y_test = self.ml_data['y_clf_test']
        
        results = {}
        
        for name, model in self.dir_models.items():
            logger.info(f"Evaluating {name} model...")
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
            
            logger.info(f"  Accuracy: {accuracy:.4f}")
            logger.info(f"  Precision: {precision:.4f}")
            logger.info(f"  Recall: {recall:.4f}")
            logger.info(f"  F1 Score: {f1:.4f}")
            
            # Create confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix - {name}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.savefig(os.path.join(self.models_dir, f'{name}_dir_cm.png'))
            plt.close()
            
            # Feature importance (for tree-based models)
            if hasattr(model, 'feature_importances_'):
                feature_imp = pd.DataFrame({
                    'Feature': self.ml_data['feature_columns'],
                    'Importance': model.feature_importances_
                }).sort_values(by='Importance', ascending=False)
                
                plt.figure(figsize=(10, 8))
                sns.barplot(x='Importance', y='Feature', data=feature_imp.head(15))
                plt.title(f'Top 15 Feature Importance - {name}')
                plt.tight_layout()
                plt.savefig(os.path.join(self.models_dir, f'{name}_dir_features.png'))
                plt.close()
        
        # Summary of results
        summary_df = pd.DataFrame(results).T
        summary_df.to_csv(os.path.join(self.models_dir, 'direction_model_results.csv'))
        
        return results
    
    def evaluate_return_models(self):
        """Evaluate return prediction models on test data."""
        if not self.data_loaded or not self.ret_models:
            logger.error("No data or models loaded. Cannot evaluate.")
            return
        
        logger.info(f"Evaluating return prediction models for {self.index_name}")
        
        X_test = self.ml_data['X_test_scaled']
        y_test = self.ml_data['y_reg_test']
        
        results = {}
        
        for name, model in self.ret_models.items():
            logger.info(f"Evaluating {name} model...")
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'r2': r2
            }
            
            logger.info(f"  MAE: {mae:.4f}")
            logger.info(f"  MSE: {mse:.4f}")
            logger.info(f"  RMSE: {rmse:.4f}")
            logger.info(f"  R²: {r2:.4f}")
            
            # Plot actual vs predicted
            plt.figure(figsize=(12, 6))
            plt.plot(y_test.values, label='Actual')
            plt.plot(y_pred, label='Predicted')
            plt.title(f'Actual vs Predicted Returns - {name}')
            plt.ylabel('5-Day Return (%)')
            plt.xlabel('Test Sample')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(self.models_dir, f'{name}_ret_prediction.png'))
            plt.close()
            
            # Feature importance (for tree-based models)
            if hasattr(model, 'feature_importances_'):
                feature_imp = pd.DataFrame({
                    'Feature': self.ml_data['feature_columns'],
                    'Importance': model.feature_importances_
                }).sort_values(by='Importance', ascending=False)
                
                plt.figure(figsize=(10, 8))
                sns.barplot(x='Importance', y='Feature', data=feature_imp.head(15))
                plt.title(f'Top 15 Feature Importance - {name}')
                plt.tight_layout()
                plt.savefig(os.path.join(self.models_dir, f'{name}_ret_features.png'))
                plt.close()
        
        # Summary of results
        summary_df = pd.DataFrame(results).T
        summary_df.to_csv(os.path.join(self.models_dir, 'return_model_results.csv'))
        
        return results
    
    def _save_model(self, model, name):
        """Save model to disk.
        
        Args:
            model: Trained model
            name (str): Name to save the model under
        """
        if not self.models_dir:
            logger.warning("No models directory available, skipping model save")
            return
            
        try:
            model_path = os.path.join(self.models_dir, f'{name}.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            logger.info(f"Saved model to {model_path}")
        except Exception as e:
            logger.warning(f"Could not save model to {name}: {e}")
    
    def load_model(self, model_type, name):
        """Load a model from disk.
        
        Args:
            model_type (str): 'dir' for direction or 'ret' for return
            name (str): Name of the model
            
        Returns:
            The loaded model
        """
        model_path = os.path.join(self.models_dir, f'{name}_{model_type}.pkl')
        
        if not os.path.exists(model_path):
            logger.error(f"Model {model_path} not found")
            return None
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        logger.info(f"Loaded model from {model_path}")
        
        if model_type == 'dir':
            self.dir_models[name] = model
        else:
            self.ret_models[name] = model
        
        return model
    
    def predict(self, features_df):
        """Make predictions using all trained models.
        
        Args:
            features_df (pd.DataFrame): Features for prediction
            
        Returns:
            dict: Predictions from all models
        """
        if not self.dir_models and not self.ret_models:
            logger.error("No models loaded. Cannot make predictions.")
            return None
        
        # Check if features contain all required columns
        missing_cols = set(self.ml_data['feature_columns']) - set(features_df.columns)
        if missing_cols:
            logger.error(f"Missing features for prediction: {missing_cols}")
            return None
        
        # Select and scale features
        X = features_df[self.ml_data['feature_columns']]
        X_scaled = self.scaler.transform(X)
        
        results = {
            'direction': {},
            'return': {}
        }
        
        # Direction predictions
        for name, model in self.dir_models.items():
            dir_pred = model.predict(X_scaled)
            dir_prob = model.predict_proba(X_scaled)[:, 1]  # Probability of positive class
            
            results['direction'][name] = {
                'prediction': dir_pred.tolist(),
                'probability': dir_prob.tolist()
            }
        
        # Return predictions
        for name, model in self.ret_models.items():
            ret_pred = model.predict(X_scaled)
            
            results['return'][name] = {
                'prediction': ret_pred.tolist()
            }
        
        return results
    
    def ensemble_predict(self, features_df):
        """Make ensemble predictions by combining all models.
        
        Args:
            features_df (pd.DataFrame): Features for prediction
            
        Returns:
            dict: Ensemble predictions
        """
        individual_preds = self.predict(features_df)
        
        if not individual_preds:
            return None
        
        # Direction ensemble (majority vote)
        dir_votes = np.zeros(len(features_df))
        
        for name, result in individual_preds['direction'].items():
            dir_votes += np.array(result['prediction'])
        
        # If more models predict up (1) than down (0), then predict up
        dir_ensemble = (dir_votes > len(individual_preds['direction']) / 2).astype(int)
        
        # Return ensemble (average)
        ret_preds = []
        
        for name, result in individual_preds['return'].items():
            ret_preds.append(np.array(result['prediction']))
        
        ret_ensemble = np.mean(np.array(ret_preds), axis=0)
        
        return {
            'direction_ensemble': dir_ensemble.tolist(),
            'return_ensemble': ret_ensemble.tolist(),
            'individual': individual_preds
        }


class TradingStrategy:
    """Class to implement and backtest trading strategies based on ML predictions."""
    
    def __init__(self, index_name='SPX'):
        """Initialize the strategy.
        
        Args:
            index_name (str): Name of the market index to model
        """
        self.index_name = index_name
        self.model_manager = ModelManager(index_name)
    
    def backtest_strategy(self, strategy_type='ensemble', lookback=5, investment=10000):
        """Backtest a trading strategy.
        
        Args:
            strategy_type (str): Type of strategy ('ensemble', 'random_forest', etc.)
            lookback (int): Number of days to look back for signals
            investment (float): Initial investment amount
            
        Returns:
            dict: Backtest results
        """
        if not self.model_manager.data_loaded:
            logger.error("No data loaded. Cannot backtest.")
            return None
        
        # Load or train models if needed
        if not self.model_manager.dir_models:
            try:
                self.model_manager.load_model('dir', 'random_forest')
                self.model_manager.load_model('dir', 'gradient_boosting')
                self.model_manager.load_model('dir', 'logistic_regression')
            except:
                logger.info("Training direction models...")
                self.model_manager.train_direction_models()
        
        # Get test data
        X_test = self.model_manager.ml_data['X_test']
        y_test = self.model_manager.ml_data['y_reg_test']  # Actual returns
        
        # Make predictions
        if strategy_type == 'ensemble':
            predictions = self.model_manager.ensemble_predict(X_test)
            signals = np.array(predictions['direction_ensemble'])
        else:
            # Use a specific model
            model = self.model_manager.dir_models.get(strategy_type)
            if not model:
                logger.error(f"Model {strategy_type} not found")
                return None
            
            X_test_scaled = self.model_manager.ml_data['X_test_scaled']
            signals = model.predict(X_test_scaled)
        
        # Initialize backtest variables
        cash = investment
        shares = 0
        position = 0
        trades = []
        equity_curve = [investment]
        
        # Get price data
        price_data_path = os.path.join(DATA_DIR, f"{self.index_name}_daily.csv")
        
        if not os.path.exists(price_data_path):
            logger.error(f"Price data not found: {price_data_path}")
            return None
        
        prices_df = pd.read_csv(price_data_path, index_col=0, parse_dates=True)
        prices_df = prices_df.loc[y_test.index]  # Align with test dates
        
        # Run backtest
        for i in range(len(signals)):
            date = y_test.index[i]
            signal = signals[i]
            price = prices_df.loc[date, 'close']
            
            # Always hold lookback days
            if i > 0 and i % lookback == 0:
                # Close previous position
                if position != 0:
                    cash += shares * price
                    trades.append({
                        'date': date,
                        'action': 'SELL' if position > 0 else 'COVER',
                        'price': price,
                        'shares': abs(shares),
                        'cash': cash
                    })
                    position = 0
                    shares = 0
                
                # Open new position based on signal
                if signal == 1:  # Buy signal
                    shares = cash / price
                    cash = 0
                    position = 1
                    trades.append({
                        'date': date,
                        'action': 'BUY',
                        'price': price,
                        'shares': shares,
                        'cash': cash
                    })
                elif signal == 0:  # Sell/short signal
                    shares = -1 * (cash / price)  # Short position
                    position = -1
                    trades.append({
                        'date': date,
                        'action': 'SHORT',
                        'price': price,
                        'shares': abs(shares),
                        'cash': cash
                    })
            
            # Calculate portfolio value
            portfolio_value = cash
            if position != 0:
                portfolio_value += shares * price
            
            equity_curve.append(portfolio_value)
        
        # Close final position
        if position != 0:
            final_price = prices_df.iloc[-1]['close']
            cash += shares * final_price
            trades.append({
                'date': prices_df.index[-1],
                'action': 'SELL' if position > 0 else 'COVER',
                'price': final_price,
                'shares': abs(shares),
                'cash': cash
            })
        
        # Calculate metrics
        final_value = equity_curve[-1]
        total_return = (final_value - investment) / investment * 100
        
        # Calculate benchmark return (buy and hold)
        initial_price = prices_df.iloc[0]['close']
        final_price = prices_df.iloc[-1]['close']
        benchmark_return = (final_price - initial_price) / initial_price * 100
        
        equity_curve = pd.Series(equity_curve, index=[prices_df.index[0]] + list(prices_df.index))
        
        # Plot equity curve
        plt.figure(figsize=(12, 6))
        plt.plot(equity_curve, label='Strategy')
        
        # Plot buy & hold for comparison
        benchmark_curve = investment * (1 + prices_df['close'].pct_change().cumsum())
        benchmark_curve.iloc[0] = investment  # Set initial value
        plt.plot(benchmark_curve, label='Buy & Hold')
        
        plt.title(f'Equity Curve - {strategy_type} Strategy')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value ($)')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.model_manager.models_dir, f'{strategy_type}_equity_curve.png'))
        plt.close()
        
        # Return results
        results = {
            'initial_investment': investment,
            'final_value': final_value,
            'total_return': total_return,
            'benchmark_return': benchmark_return,
            'outperformance': total_return - benchmark_return,
            'num_trades': len(trades),
            'trades': trades,
            'equity_curve': equity_curve.to_dict()
        }
        
        # Save results
        with open(os.path.join(self.model_manager.models_dir, f'{strategy_type}_backtest.json'), 'w') as f:
            # Convert dates to strings for JSON serialization
            results_json = results.copy()
            results_json['trades'] = [{**trade, 'date': trade['date'].strftime('%Y-%m-%d')} 
                                      for trade in trades]
            results_json['equity_curve'] = {date.strftime('%Y-%m-%d'): value 
                                            for date, value in results['equity_curve'].items()}
            json.dump(results_json, f, indent=4)
        
        return results


def main():
    """Main function to train and evaluate models."""
    logger.info("Starting market predictor model training and evaluation")
    
    # Process command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Train and evaluate market prediction models')
    parser.add_argument('--index', type=str, default='SPX', help='Market index to model')
    parser.add_argument('--train', action='store_true', help='Train models')
    parser.add_argument('--eval', action='store_true', help='Evaluate models')
    parser.add_argument('--backtest', action='store_true', help='Backtest trading strategy')
    args = parser.parse_args()
    
    # Initialize model manager
    model_manager = ModelManager(args.index)
    
    # Train models if requested
    if args.train:
        logger.info(f"Training models for {args.index}")
        model_manager.train_direction_models()
        model_manager.train_return_models()
    
    # Evaluate models if requested
    if args.eval:
        logger.info(f"Evaluating models for {args.index}")
        model_manager.evaluate_direction_models()
        model_manager.evaluate_return_models()
    
    # Backtest strategy if requested
    if args.backtest:
        logger.info(f"Backtesting strategy for {args.index}")
        strategy = TradingStrategy(args.index)
        results = strategy.backtest_strategy()
        
        logger.info(f"Strategy return: {results['total_return']:.2f}%")
        logger.info(f"Benchmark return: {results['benchmark_return']:.2f}%")
        logger.info(f"Outperformance: {results['outperformance']:.2f}%")
    
    logger.info("Market predictor tasks completed")


if __name__ == "__main__":
    main() 