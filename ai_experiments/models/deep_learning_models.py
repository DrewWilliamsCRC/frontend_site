#!/usr/bin/env python3
"""
Deep Learning Models for Financial Time Series

This module provides implementations of various deep learning models
specifically designed for financial time series prediction.
"""

import os
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

import numpy as np # type: ignore
import pandas as pd # type: ignore
from sklearn.preprocessing import StandardScaler, MinMaxScaler # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score # type: ignore

# TensorFlow/Keras
import tensorflow as tf # type: ignore
from tensorflow.keras.models import Sequential, Model # type: ignore
from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout, BatchNormalization, Input, Concatenate # type: ignore
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, GlobalAveragePooling1D, Bidirectional # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.regularizers import l1_l2 # type: ignore

# PyTorch
import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.optim as optim # type: ignore
from torch.utils.data import Dataset, DataLoader, TensorDataset # type: ignore
import math

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('deep_learning_models')

# Directory to save model checkpoints
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "saved_models")
os.makedirs(MODELS_DIR, exist_ok=True)


# Utility Functions
def prepare_sequence_data(data: pd.DataFrame, target_col: str, feature_cols: List[str], 
                         sequence_length: int, forecast_horizon: int = 1, 
                         train_ratio: float = 0.8) -> Tuple:
    """
    Prepare sequence data for time series models.
    
    Args:
        data (pd.DataFrame): Historical price data
        target_col (str): Column to predict (e.g., 'close')
        feature_cols (List[str]): Features to use for prediction
        sequence_length (int): Number of timesteps to use for each prediction
        forecast_horizon (int): How many steps ahead to predict
        train_ratio (float): Ratio of data to use for training
        
    Returns:
        Tuple: (X_train, y_train, X_test, y_test, scalers)
    """
    # Create feature and target arrays
    X = data[feature_cols].values
    y = data[target_col].values
    
    # Scale the data
    feature_scaler = StandardScaler()
    X_scaled = feature_scaler.fit_transform(X)
    
    target_scaler = StandardScaler()
    y_scaled = target_scaler.fit_transform(y.reshape(-1, 1)).flatten()
    
    # Create sequences
    X_sequences = []
    y_values = []
    
    for i in range(len(X_scaled) - sequence_length - forecast_horizon + 1):
        X_sequences.append(X_scaled[i:i+sequence_length])
        y_values.append(y_scaled[i+sequence_length+forecast_horizon-1])
    
    X_sequences = np.array(X_sequences)
    y_values = np.array(y_values)
    
    # Split into train and test sets
    train_size = int(len(X_sequences) * train_ratio)
    X_train, X_test = X_sequences[:train_size], X_sequences[train_size:]
    y_train, y_test = y_values[:train_size], y_values[train_size:]
    
    scalers = {
        'feature': feature_scaler,
        'target': target_scaler
    }
    
    return X_train, y_train, X_test, y_test, scalers


def evaluate_regression_model(y_true: np.ndarray, y_pred: np.ndarray, 
                            scaler: Optional[StandardScaler] = None) -> Dict[str, float]:
    """
    Evaluate a regression model with various metrics.
    
    Args:
        y_true (np.ndarray): True values
        y_pred (np.ndarray): Predicted values
        scaler (StandardScaler, optional): Scaler to inverse transform values
        
    Returns:
        Dict: Evaluation metrics
    """
    # If scaler is provided, inverse transform the values
    if scaler is not None:
        y_true = scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()
        y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    
    # Calculate metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Calculate directional accuracy
    y_true_dir = np.sign(np.diff(y_true))
    y_pred_dir = np.sign(np.diff(y_pred))
    dir_acc = np.mean(y_true_dir == y_pred_dir)
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'dir_accuracy': dir_acc
    }


def evaluate_classification_model(y_true: np.ndarray, y_pred: np.ndarray, 
                                y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Evaluate a classification model with various metrics.
    
    Args:
        y_true (np.ndarray): True classes
        y_pred (np.ndarray): Predicted classes
        y_prob (np.ndarray, optional): Probability scores for positive class
        
    Returns:
        Dict: Evaluation metrics
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    return metrics


# TensorFlow/Keras Models
class LSTMModel:
    """LSTM model for time series prediction with TensorFlow/Keras."""
    
    def __init__(self, sequence_length: int, n_features: int, n_units: List[int] = [50, 50],
                output_dim: int = 1, dropout_rate: float = 0.2):
        """
        Initialize LSTM model.
        
        Args:
            sequence_length (int): Length of input sequences
            n_features (int): Number of features per timestep
            n_units (List[int]): Number of units in each LSTM layer
            output_dim (int): Dimension of output (1 for regression)
            dropout_rate (float): Dropout rate for regularization
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.n_units = n_units
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.model = self._build_model()
        
    def _build_model(self) -> Model:
        """Build and compile the LSTM model."""
        model = Sequential()
        
        # First LSTM layer with return sequences for stacking
        model.add(LSTM(self.n_units[0], activation='relu', 
                      return_sequences=len(self.n_units) > 1,
                      input_shape=(self.sequence_length, self.n_features)))
        model.add(Dropout(self.dropout_rate))
        
        # Add additional LSTM layers if specified
        for i in range(1, len(self.n_units)):
            return_sequences = i < len(self.n_units) - 1
            model.add(LSTM(self.n_units[i], activation='relu', return_sequences=return_sequences))
            model.add(Dropout(self.dropout_rate))
        
        # Output layer
        model.add(Dense(self.output_dim))
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
             X_val: np.ndarray, y_val: np.ndarray,
             epochs: int = 100, batch_size: int = 32) -> Dict[str, List[float]]:
        """
        Train the LSTM model.
        
        Args:
            X_train (np.ndarray): Training sequences
            y_train (np.ndarray): Training targets
            X_val (np.ndarray): Validation sequences
            y_val (np.ndarray): Validation targets
            epochs (int): Maximum number of training epochs
            batch_size (int): Batch size for training
            
        Returns:
            Dict: Training history
        """
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)
        ]
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return {
            'loss': history.history['loss'],
            'val_loss': history.history['val_loss']
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with the trained model.
        
        Args:
            X (np.ndarray): Input sequences
            
        Returns:
            np.ndarray: Predictions
        """
        return self.model.predict(X)
    
    def save_model(self, filepath: str) -> None:
        """Save the model to disk."""
        self.model.save(filepath)
    
    @classmethod
    def load_model(cls, filepath: str) -> 'LSTMModel':
        """Load a saved model from disk."""
        model = tf.keras.models.load_model(filepath)
        
        # Extract model parameters from the loaded model
        sequence_length, n_features = model.input_shape[1:]
        output_dim = model.output_shape[1]
        
        # Create a new instance with the same parameters
        instance = cls(sequence_length, n_features, output_dim=output_dim)
        instance.model = model
        
        return instance


class BiLSTMAttentionModel:
    """Bidirectional LSTM with attention mechanism for time series prediction."""
    
    def __init__(self, sequence_length: int, n_features: int, n_units: int = 64,
                output_dim: int = 1, dropout_rate: float = 0.2):
        """
        Initialize BiLSTM model with attention.
        
        Args:
            sequence_length (int): Length of input sequences
            n_features (int): Number of features per timestep
            n_units (int): Number of units in LSTM layers
            output_dim (int): Dimension of output (1 for regression)
            dropout_rate (float): Dropout rate for regularization
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.n_units = n_units
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.model = self._build_model()
        
    def _attention_layer(self, inputs, time_steps):
        """
        Attention mechanism to focus on relevant parts of the sequence.
        
        Args:
            inputs: Output from BiLSTM layer
            time_steps: Number of time steps
            
        Returns:
            Attention-weighted representation
        """
        # Attention weights
        a = Dense(time_steps, activation='softmax')(inputs)
        a_probs = tf.expand_dims(a, -1)
        
        # Apply attention weights
        output_attention = inputs * a_probs
        
        # Sum over time dimension
        return tf.reduce_sum(output_attention, 1)
    
    def _build_model(self) -> Model:
        """Build and compile the BiLSTM model with attention."""
        # Input layer
        input_layer = Input(shape=(self.sequence_length, self.n_features))
        
        # Bidirectional LSTM layer
        lstm_layer = Bidirectional(LSTM(self.n_units, return_sequences=True))(input_layer)
        lstm_layer = Dropout(self.dropout_rate)(lstm_layer)
        
        # Attention mechanism
        attention_layer = self._attention_layer(lstm_layer, self.sequence_length)
        
        # Fully connected layers
        dense_layer = Dense(32, activation='relu')(attention_layer)
        dense_layer = Dropout(self.dropout_rate)(dense_layer)
        
        # Output layer
        output_layer = Dense(self.output_dim)(dense_layer)
        
        # Create and compile model
        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
             X_val: np.ndarray, y_val: np.ndarray,
             epochs: int = 100, batch_size: int = 32) -> Dict[str, List[float]]:
        """
        Train the BiLSTM model.
        
        Args:
            X_train (np.ndarray): Training sequences
            y_train (np.ndarray): Training targets
            X_val (np.ndarray): Validation sequences
            y_val (np.ndarray): Validation targets
            epochs (int): Maximum number of training epochs
            batch_size (int): Batch size for training
            
        Returns:
            Dict: Training history
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_checkpoint_path = os.path.join(MODELS_DIR, f"bilstm_attention_{timestamp}.h5")
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001),
            ModelCheckpoint(model_checkpoint_path, monitor='val_loss', save_best_only=True)
        ]
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return {
            'loss': history.history['loss'],
            'val_loss': history.history['val_loss']
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with the trained model.
        
        Args:
            X (np.ndarray): Input sequences
            
        Returns:
            np.ndarray: Predictions
        """
        return self.model.predict(X)
    
    def save_model(self, filepath: str) -> None:
        """Save the model to disk."""
        self.model.save(filepath)
    
    @classmethod
    def load_model(cls, filepath: str) -> 'BiLSTMAttentionModel':
        """Load a saved model from disk."""
        model = tf.keras.models.load_model(filepath)
        
        # Extract model parameters from the loaded model
        sequence_length, n_features = model.input_shape[1:]
        output_dim = model.output_shape[1]
        
        # Create a new instance with the same parameters
        instance = cls(sequence_length, n_features, output_dim=output_dim)
        instance.model = model
        
        return instance


class TransformerModel:
    """Transformer model for time series prediction using TensorFlow/Keras."""
    
    def __init__(self, sequence_length: int, n_features: int, d_model: int = 64,
                n_heads: int = 4, n_layers: int = 2, output_dim: int = 1,
                dropout_rate: float = 0.1):
        """
        Initialize Transformer model.
        
        Args:
            sequence_length (int): Length of input sequences
            n_features (int): Number of features per timestep
            d_model (int): Dimension of the model (embedding dimension)
            n_heads (int): Number of attention heads
            n_layers (int): Number of transformer layers
            output_dim (int): Dimension of output (1 for regression)
            dropout_rate (float): Dropout rate for regularization
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.model = self._build_model()
        
    def _positional_encoding(self, position, d_model):
        """
        Create positional encoding for transformer.
        
        Args:
            position (int): Maximum sequence length
            d_model (int): Dimension of the model
            
        Returns:
            tf.Tensor: Positional encoding
        """
        angles = tf.range(position, dtype=tf.float32)[:, tf.newaxis] * tf.range(d_model, dtype=tf.float32)[tf.newaxis, :]
        angles = 1 / tf.pow(10000.0, (2 * (angles // 2)) / tf.cast(d_model, tf.float32))
        
        # Apply sin to even indices, cos to odd indices
        sines = tf.math.sin(angles[:, 0::2])
        cosines = tf.math.cos(angles[:, 1::2])
        
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        
        return tf.cast(pos_encoding, tf.float32)
    
    def _transformer_encoder(self, inputs, mask=None):
        """
        Create transformer encoder.
        
        Args:
            inputs: Input tensor
            mask: Optional mask tensor
            
        Returns:
            tf.Tensor: Encoded output
        """
        # Add positional encoding
        pos_encoding = self._positional_encoding(self.sequence_length, self.d_model)
        inputs = inputs + pos_encoding[:, :self.sequence_length, :]
        
        # Add dropout
        outputs = tf.keras.layers.Dropout(rate=self.dropout_rate)(inputs)
        
        # Stack transformer layers
        for i in range(self.n_layers):
            # Multi-head attention
            attention = tf.keras.layers.MultiHeadAttention(
                num_heads=self.n_heads, key_dim=self.d_model // self.n_heads)(outputs, outputs, outputs, mask)
            attention = tf.keras.layers.Dropout(rate=self.dropout_rate)(attention)
            
            # Add & normalize (first residual connection)
            outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(outputs + attention)
            
            # Feed-forward network
            ffn = tf.keras.Sequential([
                tf.keras.layers.Dense(self.d_model * 4, activation='relu'),
                tf.keras.layers.Dense(self.d_model)
            ])
            ffn_output = ffn(outputs)
            ffn_output = tf.keras.layers.Dropout(rate=self.dropout_rate)(ffn_output)
            
            # Add & normalize (second residual connection)
            outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(outputs + ffn_output)
        
        return outputs
    
    def _build_model(self) -> Model:
        """Build and compile the Transformer model."""
        # Input layer
        inputs = Input(shape=(self.sequence_length, self.n_features))
        
        # Project input to d_model dimensions
        projected_inputs = Dense(self.d_model)(inputs)
        
        # Transformer encoder
        encoder_output = self._transformer_encoder(projected_inputs)
        
        # Global average pooling across sequence dimension
        pooled_output = GlobalAveragePooling1D()(encoder_output)
        
        # Final dense layers
        outputs = Dense(64, activation='relu')(pooled_output)
        outputs = Dropout(self.dropout_rate)(outputs)
        outputs = Dense(self.output_dim)(outputs)
        
        # Create and compile model
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.0005), loss='mse')
        
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
             X_val: np.ndarray, y_val: np.ndarray,
             epochs: int = 100, batch_size: int = 32) -> Dict[str, List[float]]:
        """
        Train the Transformer model.
        
        Args:
            X_train (np.ndarray): Training sequences
            y_train (np.ndarray): Training targets
            X_val (np.ndarray): Validation sequences
            y_val (np.ndarray): Validation targets
            epochs (int): Maximum number of training epochs
            batch_size (int): Batch size for training
            
        Returns:
            Dict: Training history
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_checkpoint_path = os.path.join(MODELS_DIR, f"transformer_{timestamp}.h5")
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.00001),
            ModelCheckpoint(model_checkpoint_path, monitor='val_loss', save_best_only=True)
        ]
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return {
            'loss': history.history['loss'],
            'val_loss': history.history['val_loss']
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with the trained model.
        
        Args:
            X (np.ndarray): Input sequences
            
        Returns:
            np.ndarray: Predictions
        """
        return self.model.predict(X)
    
    def save_model(self, filepath: str) -> None:
        """Save the model to disk."""
        self.model.save(filepath)
    
    @classmethod
    def load_model(cls, filepath: str) -> 'TransformerModel':
        """Load a saved model from disk."""
        model = tf.keras.models.load_model(filepath)
        
        # Extract model parameters from the loaded model
        sequence_length, n_features = model.input_shape[1:]
        output_dim = model.output_shape[1]
        
        # Create a new instance with the same parameters
        instance = cls(sequence_length, n_features, output_dim=output_dim)
        instance.model = model
        
        return instance


# PyTorch Models
class LSTMPyTorch(nn.Module):
    """LSTM model using PyTorch."""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, output_dim: int,
                dropout: float = 0.2):
        """
        Initialize LSTM model.
        
        Args:
            input_dim (int): Number of features
            hidden_dim (int): Size of hidden layers
            num_layers (int): Number of LSTM layers
            output_dim (int): Output dimension (1 for regression)
            dropout (float): Dropout probability
        """
        super(LSTMPyTorch, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        """Forward pass."""
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # LSTM forward
        out, _ = self.lstm(x, (h0, c0))
        
        # Take output from the last time step
        out = self.fc(out[:, -1, :])
        return out


class LSTMTrainer:
    """Trainer class for PyTorch LSTM models."""
    
    def __init__(self, model, device=None):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model
            device: Device to run on (CPU or GPU)
        """
        self.model = model
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def train(self, train_loader, val_loader, learning_rate=0.001, num_epochs=100):
        """
        Train the model.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            learning_rate (float): Learning rate
            num_epochs (int): Number of epochs
            
        Returns:
            Dict: Training history
        """
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
        
        best_val_loss = float('inf')
        best_model_state = None
        train_losses = []
        val_losses = []
        
        for epoch in range(num_epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * inputs.size(0)
            
            train_loss = train_loss / len(train_loader.dataset)
            train_losses.append(train_loss)
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item() * inputs.size(0)
            
            val_loss = val_loss / len(val_loader.dataset)
            val_losses.append(val_loss)
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Print progress
            print(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.model.state_dict().copy()
        
        # Load best model state
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        return {
            'train_loss': train_losses,
            'val_loss': val_losses
        }
    
    def predict(self, test_loader):
        """
        Make predictions with the trained model.
        
        Args:
            test_loader: DataLoader for test data
            
        Returns:
            torch.Tensor: Predictions
        """
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                predictions.append(outputs.cpu())
        
        return torch.cat(predictions, dim=0)
    
    def save_model(self, filepath):
        """Save the model to disk."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'input_dim': self.model.input_dim,
            'hidden_dim': self.model.hidden_dim,
            'num_layers': self.model.num_layers,
            'output_dim': self.model.output_dim
        }, filepath)
    
    @classmethod
    def load_model(cls, filepath, device=None):
        """Load a saved model from disk."""
        checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
        
        # Create model with saved parameters
        model = LSTMPyTorch(
            input_dim=checkpoint['input_dim'],
            hidden_dim=checkpoint['hidden_dim'],
            num_layers=checkpoint['num_layers'],
            output_dim=checkpoint['output_dim']
        )
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return cls(model, device)


# Custom dataset for PyTorch
class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for time series data."""
    
    def __init__(self, X, y):
        """
        Initialize dataset.
        
        Args:
            X: Input sequences
            y: Target values
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Model Factory
def create_model(model_type: str, **kwargs) -> Any:
    """
    Factory function to create models of different types.
    
    Args:
        model_type (str): Type of model to create
        **kwargs: Model-specific parameters
        
    Returns:
        Model instance
    """
    if model_type == 'lstm_keras':
        sequence_length = kwargs.get('sequence_length')
        n_features = kwargs.get('n_features')
        n_units = kwargs.get('n_units', [50, 50])
        output_dim = kwargs.get('output_dim', 1)
        dropout_rate = kwargs.get('dropout_rate', 0.2)
        
        return LSTMModel(sequence_length, n_features, n_units, output_dim, dropout_rate)
    
    elif model_type == 'bilstm_attention':
        sequence_length = kwargs.get('sequence_length')
        n_features = kwargs.get('n_features')
        n_units = kwargs.get('n_units', 64)
        output_dim = kwargs.get('output_dim', 1)
        dropout_rate = kwargs.get('dropout_rate', 0.2)
        
        return BiLSTMAttentionModel(sequence_length, n_features, n_units, output_dim, dropout_rate)
    
    elif model_type == 'transformer':
        sequence_length = kwargs.get('sequence_length')
        n_features = kwargs.get('n_features')
        d_model = kwargs.get('d_model', 64)
        n_heads = kwargs.get('n_heads', 4)
        n_layers = kwargs.get('n_layers', 2)
        output_dim = kwargs.get('output_dim', 1)
        dropout_rate = kwargs.get('dropout_rate', 0.1)
        
        return TransformerModel(sequence_length, n_features, d_model, n_heads, n_layers, output_dim, dropout_rate)
    
    elif model_type == 'lstm_pytorch':
        input_dim = kwargs.get('input_dim')
        hidden_dim = kwargs.get('hidden_dim', 64)
        num_layers = kwargs.get('num_layers', 2)
        output_dim = kwargs.get('output_dim', 1)
        dropout = kwargs.get('dropout', 0.2)
        
        model = LSTMPyTorch(input_dim, hidden_dim, num_layers, output_dim, dropout)
        return LSTMTrainer(model)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Example usage
    import pandas as pd # type: ignore
    import numpy as np # type: ignore
    from alpha_vantage_pipeline import AlphaVantageAPI
    
    # Fetch data
    api = AlphaVantageAPI()
    data = api.get_daily_time_series('AAPL')
    
    # Calculate technical indicators
    data['returns'] = data['close'].pct_change()
    data['ma_5'] = data['close'].rolling(window=5).mean()
    data['ma_20'] = data['close'].rolling(window=20).mean()
    data['rsi_14'] = data['returns'].apply(lambda x: max(0, x)).rolling(window=14).mean() / \
                     data['returns'].apply(lambda x: abs(x)).rolling(window=14).mean()
    data['volatility'] = data['returns'].rolling(window=20).std()
    data.dropna(inplace=True)
    
    # Prepare features and target
    feature_cols = ['open', 'high', 'low', 'close', 'volume', 'returns', 'ma_5', 'ma_20', 'rsi_14', 'volatility']
    target_col = 'close'
    sequence_length = 20
    
    # Create datasets
    X_train, y_train, X_val, y_val, scalers = prepare_sequence_data(
        data, target_col, feature_cols, sequence_length, forecast_horizon=1, train_ratio=0.8
    )
    
    # Create and train model
    model = create_model('lstm_keras', 
                        sequence_length=sequence_length, 
                        n_features=len(feature_cols),
                        n_units=[64, 32])
    
    history = model.train(X_train, y_train, X_val, y_val, epochs=50, batch_size=32)
    
    # Make predictions
    predictions = model.predict(X_val)
    
    # Evaluate model
    target_scaler = scalers['target']
    metrics = evaluate_regression_model(y_val, predictions, target_scaler)
    
    print("\nModel Evaluation:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Save model
    model.save_model(os.path.join(MODELS_DIR, "lstm_example.h5"))


class TimeSeriesTransformerDataset(Dataset):
    """Dataset class for time series data used by the transformer model."""
    
    def __init__(self, data: np.ndarray, seq_len: int, pred_len: int, 
                 target_idx: Optional[int] = None, transform: Optional[Callable] = None):
        """
        Initialize the dataset for time series transformer models.
        
        Args:
            data (np.ndarray): Input data array of shape (n_samples, n_features)
            seq_len (int): Sequence length (look-back window)
            pred_len (int): Prediction length (forecast horizon)
            target_idx (Optional[int]): Index of target variable (if None, use last column)
            transform (Optional[Callable]): Optional transform to apply to the data
        """
        self.data = torch.FloatTensor(data)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.transform = transform
        self.target_idx = -1 if target_idx is None else target_idx
        
        # Calculate valid indices
        self.indices = self._get_valid_indices()
        
    def _get_valid_indices(self) -> List[int]:
        """Get valid start indices that allow full sequence and prediction windows."""
        return list(range(len(self.data) - self.seq_len - self.pred_len + 1))
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get sample by index.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (input_sequence, target_sequence)
        """
        # Get start index for this sample
        start_idx = self.indices[idx]
        
        # Extract input sequence
        input_seq = self.data[start_idx:start_idx + self.seq_len]
        
        # Extract target sequence
        if self.target_idx == -1:  # Use the entire data for prediction
            target_seq = self.data[
                start_idx + self.seq_len:start_idx + self.seq_len + self.pred_len
            ]
        else:  # Use only the target column
            target_seq = self.data[
                start_idx + self.seq_len:start_idx + self.seq_len + self.pred_len, 
                self.target_idx
            ].unsqueeze(-1)
        
        # Apply transform if needed
        if self.transform:
            input_seq = self.transform(input_seq)
            
        return input_seq, target_seq


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer models to provide 
    temporal information about sequence position.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        """
        Initialize the positional encoding.
        
        Args:
            d_model (int): Embedding dimension
            max_len (int): Maximum sequence length
        """
        super(PositionalEncoding, self).__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        # Calculate sine and cosine positional encodings
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        # Register buffer (persistent state that's not a parameter)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input tensor.
        
        Args:
            x (torch.Tensor): Input tensor of shape (seq_len, batch_size, d_model)
        
        Returns:
            torch.Tensor: Output tensor with positional encoding added
        """
        return x + self.pe[:x.size(0), :]


class TimeSeriesTransformer(nn.Module):
    """
    Transformer model adapted for time series forecasting with 
    multi-headed attention mechanisms.
    """
    
    def __init__(self, 
                 input_dim: int, 
                 output_dim: int,
                 d_model: int = 64, 
                 nhead: int = 4, 
                 num_encoder_layers: int = 3,
                 num_decoder_layers: int = 3,
                 dim_feedforward: int = 256, 
                 dropout: float = 0.1, 
                 activation: str = 'relu'):
        """
        Initialize the time series transformer model.
        
        Args:
            input_dim (int): Number of input features
            output_dim (int): Number of output features
            d_model (int): Hidden dimension of the model
            nhead (int): Number of attention heads
            num_encoder_layers (int): Number of encoder layers
            num_decoder_layers (int): Number of decoder layers
            dim_feedforward (int): Dimension of feedforward network
            dropout (float): Dropout rate
            activation (str): Activation function
        """
        super(TimeSeriesTransformer, self).__init__()
        
        # Input embedding layer
        self.input_embedding = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder and decoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout, 
            activation=activation
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout, 
            activation=activation
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_encoder_layers
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, 
            num_layers=num_decoder_layers
        )
        
        # Output projection layer
        self.output_projection = nn.Linear(d_model, output_dim)
        
        # Initialize parameters
        self._init_parameters()
        
    def _init_parameters(self):
        """Initialize model parameters."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """
        Generate a square mask for the sequence to mask future positions.
        
        Args:
            sz (int): Sequence length
            
        Returns:
            torch.Tensor: Square mask tensor
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def forward(self, 
                src: torch.Tensor, 
                tgt: torch.Tensor, 
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            src (torch.Tensor): Source sequence [seq_len, batch_size, features]
            tgt (torch.Tensor): Target sequence [tgt_len, batch_size, features]
            src_mask (Optional[torch.Tensor]): Source sequence mask
            tgt_mask (Optional[torch.Tensor]): Target sequence mask
            
        Returns:
            torch.Tensor: Output predictions
        """
        # If src is [batch_size, seq_len, features], transpose to [seq_len, batch_size, features]
        if src.size(0) != src.size(1) and src.size(0) > src.size(2):
            src = src.transpose(0, 1)
            tgt = tgt.transpose(0, 1)
            
        # Generate masks if not provided
        if src_mask is None:
            src_mask = torch.zeros(src.size(0), src.size(0)).to(src.device)
        if tgt_mask is None:
            tgt_mask = self._generate_square_subsequent_mask(tgt.size(0)).to(tgt.device)
            
        # Embed input sequences
        src = self.input_embedding(src) * math.sqrt(self.input_embedding.out_features)
        tgt = self.input_embedding(tgt) * math.sqrt(self.input_embedding.out_features)
        
        # Add positional encoding
        src = self.pos_encoder(src)
        tgt = self.pos_encoder(tgt)
        
        # Encode and decode sequences
        memory = self.transformer_encoder(src, src_mask)
        output = self.transformer_decoder(tgt, memory, tgt_mask)
        
        # Project to output dimension
        output = self.output_projection(output)
        
        return output


class MarketPredictionTransformer:
    """
    Wrapper class for training and using the TimeSeriesTransformer for market prediction.
    """
    
    def __init__(self, 
                 input_dim: int, 
                 output_dim: int,
                 seq_len: int = 20, 
                 pred_len: int = 5, 
                 d_model: int = 64,
                 learning_rate: float = 0.001,
                 batch_size: int = 32,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the market prediction transformer.
        
        Args:
            input_dim (int): Number of input features
            output_dim (int): Number of output features
            seq_len (int): Input sequence length
            pred_len (int): Prediction sequence length
            d_model (int): Model hidden dimension
            learning_rate (float): Learning rate
            batch_size (int): Batch size
            device (str): Device to use ('cuda' or 'cpu')
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = d_model
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.device = device
        
        # Create model
        self.model = TimeSeriesTransformer(
            input_dim=input_dim,
            output_dim=output_dim,
            d_model=d_model
        ).to(device)
        
        # Setup optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        logger.info(f"Initialized MarketPredictionTransformer on {device}")
    
    def prepare_data(self, data: np.ndarray, target_idx: Optional[int] = None) -> Tuple[DataLoader, DataLoader]:
        """
        Prepare data loaders for training and validation.
        
        Args:
            data (np.ndarray): Input data array
            target_idx (Optional[int]): Target index
            
        Returns:
            Tuple[DataLoader, DataLoader]: Training and validation data loaders
        """
        # Split data into train and validation
        train_ratio = 0.8
        train_size = int(len(data) * train_ratio)
        
        train_data = data[:train_size]
        val_data = data[train_size - self.seq_len - self.pred_len + 1:]
        
        # Create datasets
        train_dataset = TimeSeriesTransformerDataset(
            data=train_data, 
            seq_len=self.seq_len, 
            pred_len=self.pred_len,
            target_idx=target_idx
        )
        
        val_dataset = TimeSeriesTransformerDataset(
            data=val_data, 
            seq_len=self.seq_len, 
            pred_len=self.pred_len,
            target_idx=target_idx
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False
        )
        
        return train_loader, val_loader
    
    def train(self, 
              train_loader: DataLoader, 
              val_loader: DataLoader, 
              epochs: int = 50,
              patience: int = 10,
              verbose: bool = True) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader): Validation data loader
            epochs (int): Number of epochs
            patience (int): Early stopping patience
            verbose (bool): Whether to print progress
            
        Returns:
            Dict[str, List[float]]: Training history
        """
        # Training history
        history = {
            'train_loss': [],
            'val_loss': []
        }
        
        # Early stopping variables
        best_val_loss = float('inf')
        no_improve_epochs = 0
        best_model_state = None
        
        logger.info(f"Starting training for {epochs} epochs")
        
        # Training loop
        for epoch in range(epochs):
            # Train for one epoch
            self.model.train()
            train_loss = 0
            
            for batch_idx, (src, tgt_true) in enumerate(train_loader):
                src, tgt_true = src.to(self.device), tgt_true.to(self.device)
                
                # Create initial decoder input (teacher forcing)
                tgt_input = torch.zeros_like(tgt_true)
                tgt_input = tgt_input.to(self.device)
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                output = self.model(src, tgt_input)
                
                # Calculate loss
                loss = self.criterion(output, tgt_true)
                
                # Backward pass
                loss.backward()
                
                # Update weights
                self.optimizer.step()
                
                # Accumulate loss
                train_loss += loss.item()
            
            # Calculate epoch average loss
            train_loss /= len(train_loader)
            history['train_loss'].append(train_loss)
            
            # Evaluate on validation set
            self.model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for src, tgt_true in val_loader:
                    src, tgt_true = src.to(self.device), tgt_true.to(self.device)
                    
                    # Create initial decoder input
                    tgt_input = torch.zeros_like(tgt_true)
                    tgt_input = tgt_input.to(self.device)
                    
                    # Forward pass
                    output = self.model(src, tgt_input)
                    
                    # Calculate loss
                    loss = self.criterion(output, tgt_true)
                    
                    # Accumulate loss
                    val_loss += loss.item()
            
            # Calculate epoch average validation loss
            val_loss /= len(val_loader)
            history['val_loss'].append(val_loss)
            
            if verbose:
                logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Check for improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve_epochs = 0
                best_model_state = self.model.state_dict()
            else:
                no_improve_epochs += 1
            
            # Early stopping
            if no_improve_epochs >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            
        logger.info(f"Training complete. Best validation loss: {best_val_loss:.4f}")
        
        return history
    
    def predict(self, data: np.ndarray, target_idx: Optional[int] = None) -> np.ndarray:
        """
        Generate predictions for input data.
        
        Args:
            data (np.ndarray): Input data
            target_idx (Optional[int]): Target index
            
        Returns:
            np.ndarray: Predictions
        """
        # Create dataset
        dataset = TimeSeriesTransformerDataset(
            data=data, 
            seq_len=self.seq_len, 
            pred_len=self.pred_len,
            target_idx=target_idx
        )
        
        # Create data loader
        data_loader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=False
        )
        
        # Switch to evaluation mode
        self.model.eval()
        
        # Generate predictions
        predictions = []
        
        with torch.no_grad():
            for src, _ in data_loader:
                src = src.to(self.device)
                
                # Create initial decoder input
                tgt_input = torch.zeros(self.pred_len, src.size(0), src.size(2)).to(self.device)
                
                # Forward pass
                output = self.model(src, tgt_input)
                
                # Store predictions
                predictions.append(output.cpu().numpy())
        
        # Concatenate predictions
        return np.concatenate(predictions, axis=1)
    
    def save(self, path: str):
        """
        Save model to disk.
        
        Args:
            path (str): Save path
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'seq_len': self.seq_len,
            'pred_len': self.pred_len,
            'd_model': self.d_model,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size
        }, path)
        
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str, device: Optional[str] = None):
        """
        Load model from disk.
        
        Args:
            path (str): Load path
            device (Optional[str]): Device to load model on
            
        Returns:
            MarketPredictionTransformer: Loaded model
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
        checkpoint = torch.load(path, map_location=device)
        
        # Create model instance
        model = cls(
            input_dim=checkpoint['input_dim'],
            output_dim=checkpoint['output_dim'],
            seq_len=checkpoint['seq_len'],
            pred_len=checkpoint['pred_len'],
            d_model=checkpoint['d_model'],
            learning_rate=checkpoint['learning_rate'],
            batch_size=checkpoint['batch_size'],
            device=device
        )
        
        # Load state dictionaries
        model.model.load_state_dict(checkpoint['model_state_dict'])
        model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        logger.info(f"Model loaded from {path}")
        
        return model


# Function to train and use the transformer model for market prediction
def train_market_transformer(data, target_column=None, sequence_length=20, prediction_length=5, epochs=50):
    """
    Train a transformer model for market prediction.
    
    Args:
        data (pd.DataFrame): Input data frame
        target_column (str): Target column name
        sequence_length (int): Input sequence length
        prediction_length (int): Prediction sequence length
        epochs (int): Number of training epochs
        
    Returns:
        MarketPredictionTransformer: Trained model
    """
    # Convert to numpy array
    if isinstance(data, pd.DataFrame):
        if target_column is not None:
            target_idx = data.columns.get_loc(target_column)
        else:
            target_idx = None
        data = data.values
    else:
        target_idx = None
    
    # Normalize data
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    
    # Determine dimensions
    input_dim = data.shape[1]
    output_dim = 1 if target_idx is not None else input_dim
    
    # Create model
    transformer = MarketPredictionTransformer(
        input_dim=input_dim,
        output_dim=output_dim,
        seq_len=sequence_length,
        pred_len=prediction_length
    )
    
    # Prepare data
    train_loader, val_loader = transformer.prepare_data(data, target_idx)
    
    # Train model
    history = transformer.train(train_loader, val_loader, epochs=epochs)
    
    return transformer, scaler, history 