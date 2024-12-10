# Simple Moving Average (SMA) strategies
Python class to perform and optimize SMA trading strategies.

# SMA Crossover Trading Strategy

This repository contains a Python implementation of a **Simple Moving Average (SMA) Crossover Trading Strategy**, designed to help traders identify market trends and automate trading decisions. The project includes tools for strategy backtesting, performance evaluation, and parameter optimization.

---

## Features

- **Data Fetching**: Retrieve historical stock price data using `yfinance`.
- **SMA Calculation**: Compute short-term and long-term SMAs.
- **Strategy Backtesting**: Evaluate strategy performance with historical data.
- **Performance Visualization**: Plot price trends, SMA crossovers, and cumulative returns.
- **Optimization**: Use `scipy.optimize.brute` to find optimal SMA lengths and leverage values for maximum returns.
- **Key Metrics**: Analyze returns, volatility, and maximum drawdown.

---

## Requirements

To run the project, install the required Python packages:

```bash
pip install yfinance pandas numpy matplotlib scipy
