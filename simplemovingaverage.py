import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brute

class SMAStrategy:
    """
    A class to implement a Simple Moving Average (SMA) trading strategy.
    """
    def __init__(self, ticker, SMA1, SMA2, start_date, end_date, leverage = 1.0, bench = "^GSPC"):

        """
        Initialize the SMA strategy with given parameters.

        Parameters:
        - ticker: The ticker symbol of the asset.
        - SMA1: The short-term SMA window size.
        - SMA2: The long-term SMA window size.
        - start_date: Start date for historical data.
        - end_date: End date for historical data.
        - leverage: The leverage to apply to the strategy.
        - bench: The benchmark ticker for performance comparison.
        """

        self.ticker = ticker
        self.SMA1 = SMA1
        self.SMA2 = SMA2
        self.start_date = start_date
        self.end_date = end_date
        self.asset_data = None
        self.bench_data = None
        self.metrics = []
        self.leverage = leverage
        self.bench = bench

    def fetch_data(self):

        """
        Fetch historical price data for the asset and benchmark using yfinance.
        """

        asset_data = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        self.asset_data = asset_data

        bench_data = yf.download(self.bench, start=self.start_date, end=self.end_date)
        self.bench_data = bench_data
    
    def perform_calculation(self):
        
        """
        Calculate log returns, short-term SMA, and long-term SMA for the asset.
        Compute cumulative returns for the benchmark.
        """

        data = self.asset_data
        SMA1 = self.SMA1
        SMA2 = self.SMA2
        data["log_returns"] = np.log(data['Close']/data['Close'].shift(1))
        data["SMA1"] = data['Close'].rolling(SMA1).mean()
        data["SMA2"] = data['Close'].rolling(SMA2).mean()

        benchmark_data = self.bench_data
        benchmark_data["log_returns"] = np.log(benchmark_data['Close']/benchmark_data['Close'].shift(1))
        benchmark_data["cum_returns"] = benchmark_data["log_returns"].cumsum().apply(np.exp)


        self.asset_data = data
        self.bench_data = benchmark_data

    def plot_sma(self):
        """
        Plot the close prices along with the short and long SMAs.
        """

        if self.asset_data is None:
            print("Data not available. Please fetch data first.")
            return

        data = self.asset_data

        plt.figure(figsize=(12,7))

        ax = plt.gca()  # Get current axis
        ax.set_facecolor('lightgray')

        plt.plot(data["Close"], color = (0.0, 0.2, 0.4), label = "Price", linewidth = 1.5)
        plt.plot(data["SMA1"], color = (0.133, 0.545, 0.133), label = "short SMA")
        plt.plot(data["SMA2"], color = (0.863, 0.078, 0.235), label = "long SMA")

        plt.title(f"{self.ticker} Price and {self.SMA1}-day & {self.SMA2}-day SMA")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(color='white', linestyle='-', linewidth=0.7)
        plt.show()


    def perform_strategy(self):

        """
        Implement the SMA crossover strategy and compute strategy performance.
        """

        data = self.asset_data.copy()
        data.dropna()
        data['position'] = np.where(data["SMA1"] > data["SMA2"], 1.0, -1.0)

        # backtesting
        data["strategy"] = self.leverage * data["position"].shift(1) * data["log_returns"]
        data.dropna()
        data["cum_strategy"] = data["strategy"].cumsum().apply(np.exp)
        data["cum_returns"] = data["log_returns"].cumsum().apply(np.exp)

        self.asset_data = data

    def plot_performance(self):

        """
        Plot the under/outperformance of the strategy compared to the buy and hold strategy
        """

        data = self.asset_data
        bench_data = self.bench_data

        plt.figure(figsize=(12,7))

        ax = plt.gca()  # Get current axis
        ax.set_facecolor('lightgray')

        plt.plot(data["cum_strategy"], color = "g", label = "strategy")
        plt.plot(data["cum_returns"], color = "b", label = "returns")
        plt.plot(bench_data["cum_returns"], color = "m", label = "benchmark")

        plt.title(f"Performance of {self.ticker} compared to SMA strategy")
        plt.xlabel("Date")
        plt.ylabel("Return")
        plt.legend()
        plt.grid(color='white', linestyle='-', linewidth=0.7)
        plt.show()

    def calculate_metrics(self):

        """
        calculates and stores useful metrics of the strategy
        """
        data = self.asset_data
        metrics = self.metrics
        gross_perf_strategy = (data["cum_strategy"].iloc[-1] - 1) * 100
        metrics.append(gross_perf_strategy)
        gross_perf_asset = (data["cum_returns"].iloc[-1] - 1) * 100
        metrics.append(gross_perf_asset)
        xperf = gross_perf_strategy - gross_perf_asset
        metrics.append(xperf)
        asset_volatility = (data["log_returns"].apply(np.exp) - 1).std() * 252 ** 0.5
        metrics.append(asset_volatility)
        strategy_volatility = (data["strategy"].apply(np.exp) - 1).std() * 252 ** 0.5
        metrics.append(strategy_volatility)
        max_drawdown_asset = (data["log_returns"].min() - data["log_returns"].max())/data["log_returns"].max()
        metrics.append(max_drawdown_asset)
        max_drawdown_strategy = (data["cum_strategy"].min() - data["cum_strategy"].max())/data["cum_strategy"].max()
        metrics.append(max_drawdown_strategy)

        return metrics
        
    
    def print_summary(self):
        """
        prints a summary with important metrics of the strategy
        """
        metrics = self.calculate_metrics()

        print(f'''Summary:\n
        Strategy returns: {round(metrics[0], 3)}%\n
        Asset returns: {round(metrics[1],3)}%\n
        Excess returns: {round(metrics[2],3)}%\n
        Asset Volatility: {round(metrics[3],2) * 100}%\n
        Strategy Volatility: {round(metrics[4],2) * 100}%\n
        Asset Max Drawdown: {round(metrics[5],2) * 100}%\n
        Strategy Max Drawdown: {round(metrics[6],2) * 100}%''')
        

    def update_strategy(self, params):

        """
        Updates SMA values and leverage, and recalculates the strategy's performance.
        The negative signal is for the minimization function of the optimization method.
    
        Parameters:
        - params: A tuple containing (SMA1, SMA2, leverage).
        """

        data = self.asset_data.copy()

        self.SMA1 = int(params[0])
        data['SMA1'] = data['Close'].rolling(self.SMA1).mean()

        self.SMA2 = int(params[1])
        data['SMA2'] = data['Close'].rolling(self.SMA2).mean()

        self.leverage = params[2]

        self.asset_data = data
        self.perform_strategy()
        
        # return the negative of the cumulative return to find its minimum
        return -self.asset_data["cum_strategy"].iloc[-1]

    def optimize_strategy(self, SMA1_range, SMA2_range, lev_range):

        """
        Optimizes the strategy by finding the best combination of SMA1, SMA2, and leverage
        to maximize performance.
    
        Parameters:
        - SMA1_range: A tuple defining the range of values for SMA1 (start, stop, step).
        - SMA2_range: A tuple defining the range of values for SMA2 (start, stop, step).
        - leverage_range: A tuple defining the range of leverage values (start, stop, step).
    
        Returns:
        - A tuple containing the best SMA1, SMA2, and leverage values and the optimized strategy performance.
        """

        best_params = brute(self.update_strategy, (SMA1_range, SMA2_range, lev_range), finish=None)
        return best_params, -self.update_strategy(best_params)

        

if __name__ == '__main__':
    amzn_strategy = SMAStrategy("MSFT", 100, 200, "2013-01-01", "2023-01-01", leverage=1.9)
    amzn_strategy.fetch_data()
    amzn_strategy.perform_calculation()
    amzn_strategy.perform_strategy()
    best_params, best_performance = amzn_strategy.optimize_strategy((100, 136, 4), (200, 300, 4), (1.0, 2.0, 0.1))
    print(f'''Optimization results\n
    Best short SMA: {best_params[0]}\n
    Best long SMA: {best_params[1]}\n
    Best leverages: {best_params[2]}\n
    Best performance: {best_performance}''')
    amzn_strategy.plot_performance()
    amzn_strategy.print_summary()

    
