import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def compute_portfolio_metrics(df, permnos_data):
    """
    Compute portfolio metrics like expected returns and covariance matrix.

    Parameters:
        df (pd.DataFrame): DataFrame of financial data.
        permnos_data (pd.DataFrame): DataFrame with PERMNOs and tickers.

    Returns:
        tuple: Returns DataFrame, expected returns, and covariance matrix.
    """
    try:
        print("Computing portfolio metrics...")
        df['date'] = pd.to_datetime(df['date'])
        returns_df = df.pivot(index='date', columns='permno', values='return').dropna()

        permno_to_ticker = dict(zip(permnos_data['permno'], permnos_data['ticker']))
        valid_permnos = [permno for permno in returns_df.columns if permno in permno_to_ticker]
        valid_tickers = [permno_to_ticker[permno] for permno in valid_permnos]

        returns_df = returns_df[valid_permnos]
        returns_df.columns = valid_tickers

        expected_returns = returns_df.mean()
        cov_matrix = returns_df.cov()

        print("Portfolio metrics computed successfully.")
        return returns_df, expected_returns, cov_matrix
    except Exception as e:
        print(f"Error computing portfolio metrics: {e}")
        return None, None, None

def plot_efficient_frontier(expected_returns, cov_matrix):
    """
    Plot the efficient frontier for the given returns and covariance matrix.
    """
    print("Plotting Efficient Frontier...")
    num_portfolios = 10000
    results = np.zeros((3, num_portfolios))
    weights_record = []

    for i in range(num_portfolios):
        weights = np.random.random(len(expected_returns))
        weights /= np.sum(weights)
        weights_record.append(weights)

        portfolio_return = np.dot(weights, expected_returns)
        portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

        results[0, i] = portfolio_return
        results[1, i] = portfolio_std_dev
        results[2, i] = results[0, i] / results[1, i]

    plt.figure(figsize=(10, 6))
    plt.scatter(results[1, :], results[0, :], c=results[2, :], cmap="YlGnBu", marker="o")
    plt.title("Efficient Frontier")
    plt.xlabel("Portfolio Risk (Standard Deviation)")
    plt.ylabel("Portfolio Return")
    plt.colorbar(label="Sharpe Ratio")
    plt.grid()
    plt.show()
    print("Efficient Frontier plotted successfully.")

def plot_sharpe_ratio_distribution(expected_returns, cov_matrix):
    """
    Plot the distribution of Sharpe ratios for random portfolios.
    """
    print("Plotting Sharpe Ratio Distribution...")
    num_portfolios = 10000
    sharpe_ratios = []

    for _ in range(num_portfolios):
        weights = np.random.random(len(expected_returns))
        weights /= np.sum(weights)

        portfolio_return = np.dot(weights, expected_returns)
        portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = portfolio_return / portfolio_std_dev
        sharpe_ratios.append(sharpe_ratio)

    plt.figure(figsize=(10, 6))
    plt.hist(sharpe_ratios, bins=50, color="skyblue", alpha=0.7)
    plt.title("Sharpe Ratio Distribution")
    plt.xlabel("Sharpe Ratio")
    plt.ylabel("Frequency")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()
    print("Sharpe Ratio Distribution plotted successfully.")
