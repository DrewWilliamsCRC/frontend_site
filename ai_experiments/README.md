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

## Project Components

### 1. Data Pipeline (`alpha_vantage_pipeline.py`)

This module provides a robust framework for fetching, processing, and storing financial data from Alpha Vantage API.

#### Key Classes:

- **AlphaVantageAPI**: Handles API requests, rate limiting, and data validation
- **DataProcessor**: Calculates technical indicators and prepares data for ML
- **DataManager**: Orchestrates data fetching, storage, and ML data preparation

#### Usage:

```python
# Run the full pipeline
python alpha_vantage_pipeline.py

# Or use components in your own script
from alpha_vantage_pipeline import DataManager
manager = DataManager()
indices_data = manager.fetch_and_store_index_data()
```

### 2. Analysis Notebook (`notebooks/alpha_vantage_analysis.ipynb`)

An interactive Jupyter notebook for exploring financial data, calculating technical indicators, and prototyping ML models.

#### Features:

- Data fetching and visualization
- Technical indicator calculation
- Feature engineering
- Model training and evaluation

#### Usage:

```bash
jupyter notebook notebooks/alpha_vantage_analysis.ipynb
```

### 3. ML Models (`models/market_predictor.py`)

This module implements machine learning models for market prediction, including both classification (direction) and regression (return) models.

#### Key Classes:

- **ModelManager**: Handles loading data, training models, and making predictions
- **TradingStrategy**: Implements and backtests trading strategies based on model predictions

#### Models Implemented:

- Random Forest
- Gradient Boosting
- Logistic Regression/Ridge Regression
- Support Vector Machines
- Neural Networks
- Ensemble methods

#### Usage:

```bash
# Train and evaluate all models
python models/market_predictor.py --train --eval --index SPX

# Backtest a trading strategy
python models/market_predictor.py --backtest --index SPX
```

### 4. Visualization Dashboard (`templates/ai_insights_dashboard.html`)

A web-based dashboard for visualizing AI predictions and market insights, integrated with the main Flask application.

#### Features:

- Market direction prediction with confidence metrics
- 5-day return forecasts
- Feature importance visualization
- Model performance metrics
- Technical indicator summaries

#### Access:

Navigate to `/ai-insights` in your browser after starting the Flask application:

```bash
./dev.sh up
```

## Data Flow

1. **Data Collection**: Alpha Vantage API provides financial time series data
2. **Data Processing**: Technical indicators are calculated and features engineered
3. **Model Training**: ML models are trained on historical data
4. **Prediction**: Models predict market direction and future returns
5. **Visualization**: Results are displayed in the web dashboard

## API Endpoints

The project adds the following API endpoints to the main application:

- **GET /ai-insights**: Renders the AI insights dashboard
- **GET /api/ai-insights**: Returns JSON data for AI predictions and metrics

## Directory Structure

```
ai_experiments/
├── README.md                  # This file
├── alpha_vantage_pipeline.py  # Data pipeline module
├── data/                      # Stored financial data (created at runtime)
│   ├── DJI_daily.csv
│   ├── SPX_daily.csv
│   └── ...
├── models/                    # ML model implementations
│   ├── market_predictor.py
│   └── SPX/                   # Trained models (created at runtime)
│       ├── random_forest_dir.pkl
│       └── ...
└── notebooks/                 # Jupyter notebooks
    └── alpha_vantage_analysis.ipynb
```

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