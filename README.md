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
```

## Usage
1. Clone the Repository
```bash
git clone [https://github.com/yourusername/sma-crossover-strategy.git](https://github.com/Marcussena/python_algotrading.git)
cd sma-crossover-strategy
```
2. Import and Initialize the Class
```bash
from smastrategy import SMAStrategy

# Example: Microsoft (MSFT) stocks
strategy = SMAStrategy(
    ticker="MSFT",
    SMA1=100,
    SMA2=200,
    start_date="2013-01-01",
    end_date="2023-01-01",
    leverage=1.0
)
```
3. Fetch and Process Data
```bash
strategy.fetch_data()
strategy.perform_calculation()
strategy.perform_strategy()
```
4. Optimize Parameters
```bash
best_params, best_performance = strategy.optimize_strategy(
    SMA1_range=(50, 150, 10),
    SMA2_range=(200, 300, 10),
    leverage_range=(1.0, 2.0, 0.1)
)
print(f"Optimized Parameters: {best_params}, Best Performance: {best_performance}")
```
5. Visualize Results
```bash
strategy.plot_sma()
strategy.plot_performance()
```
## Results
The strategy achieved notable returns with optimized SMA lengths and leverage. However, it is essential to backtest strategies for different assets and market conditions to evaluate robustness.

## Contributing
Contributions are welcome! Feel free to fork the repository, report issues, or suggest enhancements.

## Author
Marcus Sena
https://www.linkedin.com/in/marcus-sena-660198150/
https://medium.com/@marcusmvls-vinicius


