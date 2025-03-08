# Alpha Vantage AI Experiments

This project leverages Alpha Vantage API data to build, train, and deploy machine learning models for financial market prediction. The system includes data fetching, processing, model training, and a visualization dashboard.

## Project Overview

The Alpha Vantage AI Experiments project consists of four main components:

1. **Data Pipeline**: Fetches and processes financial data from Alpha Vantage API
2. **Analysis Notebook**: Interactive exploration of financial data and model prototyping
3. **ML Models**: Training and evaluation of market prediction models
4. **Visualization Dashboard**: Web-based interface for viewing AI insights

## Getting Started

### Prerequisites

- Python 3.7+ 
- pip (Python package manager)
- Alpha Vantage API key (set in your `.env` file)

### Installation

1. Clone the repository (if you haven't already):
   ```bash
   git clone <repository-url>
   cd frontend_site
   git checkout ai-alpha-vantage-experiments
   ```

2. Install Python dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter requests python-dotenv
   ```

3. Set up your Alpha Vantage API key:
   ```
   ALPHA_VANTAGE_API_KEY=your_api_key_here
   ```

## Detailed Architecture

### 1. Data Pipeline Architecture

The data pipeline is the foundation of the entire system, responsible for acquiring, processing, and preparing market data for analysis and model training.

#### Data Acquisition Layer

The pipeline begins with the `AlphaVantageAPI` class which handles the following tasks:

- **API Authentication**: Validates and uses the API key for secure data access
- **Rate Limiting**: Implements exponential backoff strategy to respect Alpha Vantage's API rate limits
- **Request Handling**: Makes HTTP requests to various Alpha Vantage endpoints with proper error handling
- **Response Parsing**: Transforms JSON responses into structured data formats (pandas DataFrames)
- **Caching**: Implements local caching to reduce redundant API calls and improve performance

```python
# Example of how the API layer works
api = AlphaVantageAPI(api_key)
data = api.get_daily_time_series(symbol="^GSPC", outputsize="full")
```

#### Data Processing Layer

After acquisition, raw data is processed by the `DataProcessor` class which:

- **Data Cleaning**: Handles missing values, outliers, and incorrect data points
- **Feature Engineering**: Calculates technical indicators across multiple timeframes:
  - **Trend Indicators**: SMA, EMA, MACD (Multiple timeframes: 5, 10, 20, 50, 200 days)
  - **Momentum Indicators**: RSI, Stochastic Oscillator, Rate of Change (ROC)
  - **Volatility Indicators**: Bollinger Bands, ATR (Average True Range)
  - **Volume Indicators**: OBV (On-Balance Volume), Volume Rate of Change
- **Normalization**: Standardizes features to improve model training
- **Lag Features**: Creates time-shifted features to capture historical patterns
- **Cross-Market Features**: Adds correlations between different market indices

#### Data Management Layer

The `DataManager` class coordinates the entire pipeline:

- **Storage Strategy**: Implements efficient data storage using CSV files and/or database systems
- **Versioning**: Maintains data versions to ensure reproducibility of experiments
- **Data Merging**: Combines data from different sources (multiple indices, economic indicators)
- **Train/Test Split**: Prepares data for ML with proper time-series-aware splitting
- **Feature Selection**: Identifies most relevant features using statistical methods

```python
# Example of complete pipeline usage
manager = DataManager()
processed_data = manager.fetch_process_and_store_data(
    symbols=['SPX', 'DJI', 'IXIC'],
    start_date='2010-01-01',
    end_date='2023-12-31'
)
```

### 2. Machine Learning Model Architecture

Our system implements multiple models to capture different aspects of market behavior and to enable ensemble techniques.

#### Model Types

We implement two primary categories of models:

1. **Classification Models**: Predict market direction (up/down)
   - **Random Forest Classifier**: Robust ensemble method less prone to overfitting
   - **Gradient Boosting Classifier**: Sequential tree-based model with high accuracy
   - **Neural Network Classifier**: Multi-layer perceptron for capturing complex patterns
   - **Support Vector Classifier**: Effective for high-dimensional data
   - **Logistic Regression**: Baseline model with interpretable coefficients

2. **Regression Models**: Predict percentage returns over specified time horizons
   - **Random Forest Regressor**: Ensemble of decision trees for numerical prediction
   - **Gradient Boosting Regressor**: Boosting algorithm for numerical targets
   - **Neural Network Regressor**: Deep learning approach for return forecasting
   - **Ridge Regression**: Linear model with L2 regularization
   - **SVR (Support Vector Regression)**: Non-linear regression technique

#### Feature Selection and Importance Analysis

Each model incorporates feature importance analysis to identify the most predictive market indicators:

- **Random Forest Importance**: Measures decrease in impurity across all trees
- **Permutation Importance**: Evaluates impact of shuffling each feature on performance
- **SHAP Values**: Explains individual predictions using game theory principles
- **Recursive Feature Elimination**: Iteratively removes least important features

#### Hyperparameter Optimization

Models are fine-tuned using:

- **Grid Search**: Exhaustive search over specified parameter values
- **Random Search**: Monte Carlo sampling of parameter space
- **Bayesian Optimization**: Probabilistic model-based approach
- **Time Series Cross-Validation**: Forward-chaining validation to prevent lookahead bias

```python
# Example model training process
from models.market_predictor import ModelManager

model_manager = ModelManager()
model_manager.train_all_models(
    data=processed_data,
    target_column='direction',  # or 'return_5d' for regression
    test_size=0.2,
    random_state=42
)

# Hyperparameter tuning example
best_params = model_manager.optimize_hyperparameters(
    model_type='random_forest',
    param_grid={
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
)
```

#### Ensemble Methods

To improve prediction accuracy, we implement several ensemble techniques:

- **Voting Ensemble**: Combines predictions through majority voting (classification) or averaging (regression)
- **Stacking**: Trains meta-model on predictions from base models
- **Weighted Ensemble**: Assigns weights to models based on historical performance
- **Time-Dependent Ensemble**: Adjusts model weights based on recent accuracy

#### Model Evaluation

Performance is measured using specialized metrics for financial prediction:

- **Classification Metrics**: Accuracy, Precision, Recall, F1 Score, ROC-AUC
- **Regression Metrics**: RMSE, MAE, R², Directional Accuracy
- **Trading Metrics**: Return, Sharpe Ratio, Maximum Drawdown, Win Rate
- **Walk-Forward Analysis**: Progressive re-training on expanding windows

### 3. Visualization Dashboard Integration

The visualization dashboard serves as the interface between the AI system and users, providing intuitive access to model predictions and insights.

#### Data Flow Architecture

1. **Backend Processing**:
   - The Flask application triggers the `get_ai_insights()` function on request
   - This function requests current market data from Alpha Vantage API
   - Data is processed through the pipeline to generate features
   - Trained models are loaded and make predictions on current data
   - Prediction confidence and supporting metrics are calculated
   - Results are formatted as JSON response

2. **Frontend Rendering**:
   - The dashboard makes an AJAX request to `/api/ai-insights` endpoint
   - JSON data is parsed and stored in the `API_DATA` variable
   - Dashboard components are populated with prediction data
   - Charts and visualizations are rendered using Chart.js
   - UI elements are updated to reflect current market insights

#### Dashboard Components

Each visualization component has a specific purpose:

1. **Market Direction Gauge**:
   - Displays prediction confidence (0-100%) using a semi-circular gauge
   - Color gradient indicates bearish (red) to bullish (green) sentiment
   - Shows model accuracy, precision, recall, and F1 score
   - Updates dynamically when switching between models

2. **Return Prediction Chart**:
   - Plots historical model predictions against actual returns
   - Visualizes predicted 5-day return for selected market index
   - Includes confidence interval and model performance metrics
   - Supports switching between different market indices

3. **Feature Importance Visualization**:
   - Horizontal bar chart showing the relative importance of each feature
   - Helps users understand which factors are driving predictions
   - Features are ranked in descending order of importance
   - Values are normalized for easy comparison

4. **Market Metrics Grid**:
   - Displays calculated metrics for momentum, volatility, breadth, etc.
   - Each metric has a status indicator (red/yellow/green) and description
   - Values are based on technical analysis of current market data
   - Updates based on selected time period (1D, 1W, 1M, 3M, 1Y)

#### Real-time Updates and User Interaction

The dashboard supports:

1. **Model Switching**: Users can select different models to compare predictions
2. **Index Selection**: Supports viewing predictions for different market indices
3. **Time Period Adjustment**: Metrics can be viewed across different timeframes
4. **Periodic Refreshing**: Data is refreshed every 5 minutes to ensure currency
5. **Responsive Design**: Dashboard adapts to different screen sizes
6. **Theme Support**: Full compatibility with both light and dark mode themes

#### Data Fallback Mechanism

To ensure dashboard functionality even when API access is limited:

1. **Error Detection**: System detects API failures or rate limiting
2. **Demo Data**: Falls back to simulated data when needed
3. **Status Notification**: Users are informed when viewing simulated vs. real data
4. **Graceful Degradation**: Core functionality remains available even with limited data

## Technical Details

### Data Pipeline

The data pipeline manages:

1. **API Rate Limiting**: Ensures compliance with Alpha Vantage's rate limits
2. **Data Caching**: Minimizes API calls by reusing recent data
3. **Feature Engineering**: Calculates 15+ technical indicators including:
   - Moving Averages (SMA, EMA)
   - Momentum Oscillators (RSI, MACD)
   - Volatility Metrics (Bollinger Bands)
   - Price Patterns and Trends

### ML Models

The machine learning models focus on two prediction tasks:

1. **Direction Classification**: Predicting market movement direction (up/down)
2. **Return Regression**: Forecasting percentage returns over a specified horizon

Performance metrics tracked include:
- Accuracy, Precision, Recall, F1 (for classification)
- MAE, RMSE, R² (for regression)

### Visualization Dashboard

The dashboard uses:
- Chart.js for interactive data visualization
- Real-time model switching
- Responsive design for all devices
- Dark/light mode compatibility

## Future Enhancements

1. **Additional Data Sources**: Incorporate news sentiment, economic indicators, etc.
2. **Advanced Models**: Implement deep learning, time series forecasting, and reinforcement learning
3. **Portfolio Optimization**: Add portfolio allocation suggestions based on predictions
4. **Real-time Alerts**: Implement notification system for significant market predictions

## Troubleshooting

### Common Issues

1. **API Rate Limiting**: Alpha Vantage has strict rate limits. If you encounter errors, try:
   - Reducing the frequency of API calls
   - Setting up a paid API plan
   - Using the cached data option

2. **Model Performance**: If models show poor performance:
   - Try different feature combinations
   - Adjust the training/test split
   - Tune hyperparameters

3. **Missing Data**: If data is incomplete:
   - Check your API key is valid
   - Verify internet connectivity
   - Look for specific error messages in the logs

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Alpha Vantage for providing the financial data API
- scikit-learn for machine learning tools
- Chart.js for visualization capabilities 