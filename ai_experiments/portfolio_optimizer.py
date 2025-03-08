#!/usr/bin/env python3
"""
Portfolio Optimizer Module

This module provides classes and functions for portfolio optimization
and asset allocation based on various strategies and models.
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np # type: ignore
import pandas as pd # type: ignore
from scipy.optimize import minimize # type: ignore
import yfinance as yf # type: ignore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('portfolio_optimizer')

class PortfolioOptimizer:
    """Base class for portfolio optimization."""
    
    def __init__(self, symbols: List[str], historical_days: int = 365*2):
        """
        Initialize with list of assets to optimize.
        
        Args:
            symbols (List[str]): List of asset symbols
            historical_days (int): Number of days of historical data to use
        """
        self.symbols = symbols
        self.historical_days = historical_days
        self.data = None
        self.returns = None
        self.cov_matrix = None
        self.mean_returns = None
    
    def fetch_data(self) -> pd.DataFrame:
        """
        Fetch historical price data for all symbols.
        
        Returns:
            pd.DataFrame: DataFrame with historical price data
        """
        try:
            start_date = (datetime.now() - timedelta(days=self.historical_days)).strftime('%Y-%m-%d')
            end_date = datetime.now().strftime('%Y-%m-%d')
            
            logger.info(f"Fetching historical data for {len(self.symbols)} symbols")
            data = yf.download(self.symbols, start=start_date, end=end_date)['Adj Close']
            
            # Handle single symbol case
            if isinstance(data, pd.Series):
                data = pd.DataFrame(data, columns=[self.symbols[0]])
            
            self.data = data
            logger.info(f"Successfully fetched data with shape {data.shape}")
            
            # Calculate returns and statistics
            self.calculate_returns()
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching historical data: {str(e)}")
            return pd.DataFrame()
    
    def calculate_returns(self) -> pd.DataFrame:
        """
        Calculate daily returns from price data.
        
        Returns:
            pd.DataFrame: DataFrame with daily returns
        """
        if self.data is None or self.data.empty:
            logger.error("No price data available to calculate returns")
            return pd.DataFrame()
        
        # Calculate percentage returns
        self.returns = self.data.pct_change().dropna()
        
        # Calculate mean returns and covariance matrix
        self.mean_returns = self.returns.mean()
        self.cov_matrix = self.returns.cov()
        
        return self.returns
    
    def optimize(self, risk_tolerance: str = 'moderate') -> Dict[str, Any]:
        """
        Optimize portfolio weights.
        
        Args:
            risk_tolerance (str): Risk tolerance level ('low', 'moderate', 'high')
            
        Returns:
            Dict: Optimized portfolio details
        """
        # This method should be implemented by subclasses
        raise NotImplementedError("Subclasses must implement optimize method")
    
    def calculate_portfolio_performance(self, weights: np.ndarray) -> Tuple[float, float, float]:
        """
        Calculate expected return and risk for a given set of weights.
        
        Args:
            weights (np.ndarray): Array of portfolio weights
            
        Returns:
            Tuple[float, float, float]: (expected return, volatility, Sharpe ratio)
        """
        if self.mean_returns is None or self.cov_matrix is None:
            logger.error("Returns data not available")
            return 0.0, 0.0, 0.0
        
        # Calculate expected portfolio return (annualized)
        portfolio_return = np.sum(self.mean_returns * weights) * 252
        
        # Calculate portfolio volatility (annualized)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix * 252, weights)))
        
        # Calculate Sharpe ratio (assuming risk-free rate of 0.02 or 2%)
        sharpe_ratio = (portfolio_return - 0.02) / portfolio_volatility
        
        return portfolio_return, portfolio_volatility, sharpe_ratio
    
    def get_allocation_results(self, weights: np.ndarray) -> Dict[str, Any]:
        """
        Format the optimization results.
        
        Args:
            weights (np.ndarray): Optimized portfolio weights
            
        Returns:
            Dict: Portfolio allocation details
        """
        if weights is None or len(weights) != len(self.symbols):
            logger.error("Invalid weights array")
            return {}
        
        # Calculate portfolio performance metrics
        expected_return, volatility, sharpe = self.calculate_portfolio_performance(weights)
        
        # Create result dictionary
        allocation = {symbol: weight for symbol, weight in zip(self.symbols, weights)}
        
        result = {
            'symbols': self.symbols,
            'weights': allocation,
            'expected_annual_return': float(expected_return),
            'annual_volatility': float(volatility),
            'sharpe_ratio': float(sharpe),
            'timestamp': datetime.now().isoformat()
        }
        
        return result


class MPTOptimizer(PortfolioOptimizer):
    """Modern Portfolio Theory optimizer."""
    
    def __init__(self, symbols: List[str], historical_days: int = 365*2):
        """
        Initialize MPT optimizer.
        
        Args:
            symbols (List[str]): List of asset symbols
            historical_days (int): Number of days of historical data to use
        """
        super().__init__(symbols, historical_days)
        self.risk_free_rate = 0.02  # Default risk-free rate of 2%
    
    def get_min_vol_portfolio(self) -> np.ndarray:
        """
        Optimize portfolio for minimum volatility.
        
        Returns:
            np.ndarray: Optimal weights for minimum volatility
        """
        if self.mean_returns is None or self.cov_matrix is None:
            logger.error("Returns data not available")
            return np.array([])
        
        n_assets = len(self.symbols)
        
        # Objective function: portfolio volatility to minimize
        def portfolio_volatility(weights):
            return np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix * 252, weights)))
        
        # Constraints and bounds
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  # Weights sum to 1
        bounds = tuple((0.0, 1.0) for _ in range(n_assets))  # No short selling
        
        # Initial guess: equal weight portfolio
        initial_weights = np.array([1.0/n_assets] * n_assets)
        
        # Optimize
        result = minimize(
            portfolio_volatility, 
            initial_weights, 
            method='SLSQP', 
            bounds=bounds, 
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if not result['success']:
            logger.warning(f"Optimization failed: {result['message']}")
            return initial_weights
        
        return result['x']
    
    def get_max_sharpe_portfolio(self) -> np.ndarray:
        """
        Optimize portfolio for maximum Sharpe ratio.
        
        Returns:
            np.ndarray: Optimal weights for maximum Sharpe ratio
        """
        if self.mean_returns is None or self.cov_matrix is None:
            logger.error("Returns data not available")
            return np.array([])
        
        n_assets = len(self.symbols)
        
        # Objective function: negative Sharpe ratio to minimize
        def neg_sharpe_ratio(weights):
            p_return, p_volatility, p_sharpe = self.calculate_portfolio_performance(weights)
            return -p_sharpe  # Minimize negative Sharpe = maximize Sharpe
        
        # Constraints and bounds
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  # Weights sum to 1
        bounds = tuple((0.0, 1.0) for _ in range(n_assets))  # No short selling
        
        # Initial guess: equal weight portfolio
        initial_weights = np.array([1.0/n_assets] * n_assets)
        
        # Optimize
        result = minimize(
            neg_sharpe_ratio, 
            initial_weights, 
            method='SLSQP', 
            bounds=bounds, 
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if not result['success']:
            logger.warning(f"Optimization failed: {result['message']}")
            return initial_weights
        
        return result['x']
    
    def get_efficient_return_portfolio(self, target_return: float) -> np.ndarray:
        """
        Optimize portfolio for minimum volatility given a target return.
        
        Args:
            target_return (float): Target annualized return
            
        Returns:
            np.ndarray: Optimal weights for efficient portfolio
        """
        if self.mean_returns is None or self.cov_matrix is None:
            logger.error("Returns data not available")
            return np.array([])
        
        n_assets = len(self.symbols)
        
        # Objective function: portfolio volatility to minimize
        def portfolio_volatility(weights):
            return np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix * 252, weights)))
        
        # Constraints: weights sum to 1 and portfolio return equals target
        def portfolio_return(weights):
            return np.sum(self.mean_returns * weights) * 252
        
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Weights sum to 1
            {'type': 'eq', 'fun': lambda x: portfolio_return(x) - target_return}  # Return equals target
        ]
        
        bounds = tuple((0.0, 1.0) for _ in range(n_assets))  # No short selling
        
        # Initial guess: equal weight portfolio
        initial_weights = np.array([1.0/n_assets] * n_assets)
        
        # Optimize
        result = minimize(
            portfolio_volatility, 
            initial_weights, 
            method='SLSQP', 
            bounds=bounds, 
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if not result['success']:
            logger.warning(f"Optimization failed: {result['message']}")
            return initial_weights
        
        return result['x']
    
    def optimize(self, risk_tolerance: str = 'moderate') -> Dict[str, Any]:
        """
        Optimize portfolio based on risk tolerance.
        
        Args:
            risk_tolerance (str): Risk tolerance level ('low', 'moderate', 'high')
            
        Returns:
            Dict: Optimized portfolio details
        """
        if self.data is None or self.data.empty:
            logger.info("No data available, fetching data first")
            self.fetch_data()
            
        if self.mean_returns is None or self.cov_matrix is None:
            logger.error("Returns data not available after fetch attempt")
            return {}
        
        # Map risk tolerance to optimization strategy
        if risk_tolerance.lower() == 'low':
            # For low risk tolerance, minimize volatility
            logger.info("Optimizing for minimum volatility (low risk)")
            weights = self.get_min_vol_portfolio()
            strategy = "Minimum Volatility"
            
        elif risk_tolerance.lower() == 'high':
            # For high risk tolerance, maximize Sharpe ratio
            logger.info("Optimizing for maximum Sharpe ratio (high risk)")
            weights = self.get_max_sharpe_portfolio()
            strategy = "Maximum Sharpe Ratio"
            
        else:  # 'moderate' as default
            # For moderate risk, target a reasonable return
            logger.info("Optimizing for balanced return/risk (moderate risk)")
            
            # First find max Sharpe portfolio to understand return range
            max_sharpe_weights = self.get_max_sharpe_portfolio()
            max_return, _, _ = self.calculate_portfolio_performance(max_sharpe_weights)
            
            # Target 70% of the max Sharpe portfolio's return
            target_return = max_return * 0.7
            weights = self.get_efficient_return_portfolio(target_return)
            strategy = f"Efficient Portfolio (Target Return: {target_return:.2%})"
        
        # Get allocation results
        result = self.get_allocation_results(weights)
        
        # Add strategy information
        result['strategy'] = strategy
        result['risk_tolerance'] = risk_tolerance
        
        return result


class RiskParityOptimizer(PortfolioOptimizer):
    """Risk Parity Portfolio optimizer."""
    
    def __init__(self, symbols: List[str], historical_days: int = 365*2):
        """
        Initialize Risk Parity optimizer.
        
        Args:
            symbols (List[str]): List of asset symbols
            historical_days (int): Number of days of historical data to use
        """
        super().__init__(symbols, historical_days)
    
    def get_risk_contributions(self, weights: np.ndarray) -> np.ndarray:
        """
        Calculate risk contribution of each asset.
        
        Args:
            weights (np.ndarray): Portfolio weights
            
        Returns:
            np.ndarray: Risk contribution of each asset
        """
        if self.cov_matrix is None:
            logger.error("Covariance matrix not available")
            return np.array([])
        
        # Portfolio volatility
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix * 252, weights)))
        
        # Marginal risk contribution
        marginal_contrib = np.dot(self.cov_matrix * 252, weights)
        
        # Risk contribution
        risk_contrib = np.multiply(marginal_contrib, weights) / portfolio_vol
        
        return risk_contrib
    
    def risk_budget_objective(self, weights: np.ndarray, target_risk: Optional[np.ndarray] = None) -> float:
        """
        Objective function for risk parity optimization.
        
        Args:
            weights (np.ndarray): Portfolio weights
            target_risk (np.ndarray, optional): Target risk budget (default: equal risk)
            
        Returns:
            float: Sum of squared differences between actual and target risk contributions
        """
        if self.cov_matrix is None:
            logger.error("Covariance matrix not available")
            return float('inf')
        
        n_assets = len(weights)
        
        # If target risk not specified, use equal risk allocation
        if target_risk is None:
            target_risk = np.ones(n_assets) / n_assets
        
        # Ensure weights are positive (required for risk parity)
        weights = np.clip(weights, 0.001, 1.0)
        
        # Normalize weights to sum to 1
        weights = weights / np.sum(weights)
        
        # Get risk contributions
        risk_contrib = self.get_risk_contributions(weights)
        
        # Calculate squared differences from target risk
        diff = np.sum(np.square(risk_contrib - target_risk))
        
        return diff
    
    def get_risk_parity_portfolio(self, target_risk: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Optimize for risk parity portfolio.
        
        Args:
            target_risk (np.ndarray, optional): Target risk budget (default: equal risk)
            
        Returns:
            np.ndarray: Optimal weights for risk parity
        """
        if self.cov_matrix is None:
            logger.error("Covariance matrix not available")
            return np.array([])
        
        n_assets = len(self.symbols)
        
        # If target risk not specified, use equal risk allocation
        if target_risk is None:
            target_risk = np.ones(n_assets) / n_assets
        
        # Initial guess: equal weight portfolio
        initial_weights = np.array([1.0/n_assets] * n_assets)
        
        # Constraints: weights sum to 1
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        
        # Bounds: ensure positive weights
        bounds = tuple((0.01, 1.0) for _ in range(n_assets))
        
        # Optimize
        result = minimize(
            lambda w: self.risk_budget_objective(w, target_risk),
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )
        
        if not result['success']:
            logger.warning(f"Risk parity optimization failed: {result['message']}")
            return initial_weights
        
        # Normalize weights to ensure they sum to 1
        weights = result['x']
        weights = weights / np.sum(weights)
        
        return weights
    
    def get_customized_risk_portfolio(self, asset_risk_weights: Dict[str, float]) -> np.ndarray:
        """
        Optimize for a customized risk budget.
        
        Args:
            asset_risk_weights (Dict[str, float]): Mapping of symbols to target risk weights
            
        Returns:
            np.ndarray: Optimal weights for customized risk budget
        """
        if self.cov_matrix is None:
            logger.error("Covariance matrix not available")
            return np.array([])
        
        # Convert dict to array matching symbol order
        if set(asset_risk_weights.keys()) != set(self.symbols):
            logger.error("Asset risk weights do not match portfolio symbols")
            return np.array([])
        
        target_risk = np.array([asset_risk_weights[symbol] for symbol in self.symbols])
        
        # Normalize target risk to sum to 1
        target_risk = target_risk / np.sum(target_risk)
        
        return self.get_risk_parity_portfolio(target_risk)
    
    def optimize(self, risk_tolerance: str = 'moderate') -> Dict[str, Any]:
        """
        Optimize portfolio based on risk tolerance.
        
        Args:
            risk_tolerance (str): Risk tolerance level ('low', 'moderate', 'high')
            
        Returns:
            Dict: Optimized portfolio details
        """
        if self.data is None or self.data.empty:
            logger.info("No data available, fetching data first")
            self.fetch_data()
            
        if self.mean_returns is None or self.cov_matrix is None:
            logger.error("Returns data not available after fetch attempt")
            return {}
        
        # Get risk parity weights (equal risk contribution)
        weights = self.get_risk_parity_portfolio()
        
        # Get allocation results
        result = self.get_allocation_results(weights)
        
        # Add strategy information
        result['strategy'] = "Risk Parity (Equal Risk Contribution)"
        result['risk_tolerance'] = risk_tolerance
        result['risk_contributions'] = {
            symbol: float(contrib) for symbol, contrib in 
            zip(self.symbols, self.get_risk_contributions(weights))
        }
        
        return result
    
    def optimize_with_custom_risk(self, asset_risk_weights: Dict[str, float]) -> Dict[str, Any]:
        """
        Optimize with custom risk budget.
        
        Args:
            asset_risk_weights (Dict[str, float]): Mapping of symbols to target risk weights
            
        Returns:
            Dict: Optimized portfolio details
        """
        if self.data is None or self.data.empty:
            logger.info("No data available, fetching data first")
            self.fetch_data()
            
        if self.mean_returns is None or self.cov_matrix is None:
            logger.error("Returns data not available after fetch attempt")
            return {}
        
        # Get customized risk weights
        weights = self.get_customized_risk_portfolio(asset_risk_weights)
        
        # Get allocation results
        result = self.get_allocation_results(weights)
        
        # Add strategy information
        result['strategy'] = "Risk Parity (Custom Risk Budget)"
        result['target_risk_budget'] = asset_risk_weights
        result['actual_risk_contributions'] = {
            symbol: float(contrib) for symbol, contrib in 
            zip(self.symbols, self.get_risk_contributions(weights))
        }
        
        return result


# Let's add some utility functions
def plot_efficient_frontier(mean_returns: pd.Series, cov_matrix: pd.DataFrame, 
                           num_portfolios: int = 1000, risk_free_rate: float = 0.02) -> Dict[str, Any]:
    """
    Calculate and visualize the efficient frontier.
    
    Args:
        mean_returns (pd.Series): Mean returns for each asset
        cov_matrix (pd.DataFrame): Covariance matrix of returns
        num_portfolios (int): Number of portfolios to simulate
        risk_free_rate (float): Risk-free interest rate
        
    Returns:
        Dict: Data for plotting efficient frontier
    """
    results = {'returns': [], 'volatility': [], 'sharpe': [], 'weights': []}
    n_assets = len(mean_returns)
    
    # Generate random portfolios
    np.random.seed(42)
    for _ in range(num_portfolios):
        # Generate random weights
        weights = np.random.random(n_assets)
        weights = weights / np.sum(weights)
        
        # Calculate portfolio return and volatility
        portfolio_return = np.sum(mean_returns * weights) * 252
        portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
        
        # Calculate Sharpe ratio
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std_dev
        
        # Store results
        results['returns'].append(portfolio_return)
        results['volatility'].append(portfolio_std_dev)
        results['sharpe'].append(sharpe_ratio)
        results['weights'].append(weights)
    
    return results


if __name__ == "__main__":
    # Example usage
    symbols = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'BRK-B']
    
    # Create and use MPT optimizer
    mpt_optimizer = MPTOptimizer(symbols)
    data = mpt_optimizer.fetch_data()
    
    print(f"Fetched data for {len(symbols)} symbols")
    print(f"Latest prices: \n{data.iloc[-1]}")
    
    # Optimize for different risk profiles
    low_risk = mpt_optimizer.optimize('low')
    moderate_risk = mpt_optimizer.optimize('moderate')
    high_risk = mpt_optimizer.optimize('high')
    
    print("\nLow Risk Portfolio (MPT):")
    print(f"Strategy: {low_risk['strategy']}")
    print(f"Expected Return: {low_risk['expected_annual_return']:.2%}")
    print(f"Volatility: {low_risk['annual_volatility']:.2%}")
    print(f"Sharpe Ratio: {low_risk['sharpe_ratio']:.2f}")
    print("Allocation:")
    for symbol, weight in low_risk['weights'].items():
        print(f"  {symbol}: {weight:.2%}")
    
    # Create and use Risk Parity optimizer
    rp_optimizer = RiskParityOptimizer(symbols)
    # Reuse the same data
    rp_optimizer.data = data.copy()
    rp_optimizer.calculate_returns()
    
    # Get risk parity portfolio
    risk_parity = rp_optimizer.optimize()
    
    print("\nRisk Parity Portfolio:")
    print(f"Strategy: {risk_parity['strategy']}")
    print(f"Expected Return: {risk_parity['expected_annual_return']:.2%}")
    print(f"Volatility: {risk_parity['annual_volatility']:.2%}")
    print(f"Sharpe Ratio: {risk_parity['sharpe_ratio']:.2f}")
    print("Allocation:")
    for symbol, weight in risk_parity['weights'].items():
        print(f"  {symbol}: {weight:.2%}")
    print("Risk Contributions:")
    for symbol, contrib in risk_parity['risk_contributions'].items():
        print(f"  {symbol}: {contrib:.2%}")
    
    # Custom risk budget example
    custom_risk = {
        'AAPL': 0.3,      # 30% risk contribution
        'MSFT': 0.2,      # 20% risk contribution
        'AMZN': 0.2,      # 20% risk contribution
        'GOOGL': 0.15,    # 15% risk contribution
        'BRK-B': 0.15     # 15% risk contribution
    }
    
    custom_rp = rp_optimizer.optimize_with_custom_risk(custom_risk)
    
    print("\nCustom Risk Budget Portfolio:")
    print(f"Strategy: {custom_rp['strategy']}")
    print(f"Expected Return: {custom_rp['expected_annual_return']:.2%}")
    print(f"Volatility: {custom_rp['annual_volatility']:.2%}")
    print(f"Sharpe Ratio: {custom_rp['sharpe_ratio']:.2f}")
    print("Allocation:")
    for symbol, weight in custom_rp['weights'].items():
        print(f"  {symbol}: {weight:.2%}")
    print("Target Risk Contributions:")
    for symbol, target in custom_rp['target_risk_budget'].items():
        print(f"  {symbol}: {target:.2%}")
    print("Actual Risk Contributions:")
    for symbol, contrib in custom_rp['actual_risk_contributions'].items():
        print(f"  {symbol}: {contrib:.2%}") 