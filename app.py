import streamlit as st
import pandas as pd
from components.wrds_connection import connect_to_wrds, get_permnos_by_tickers, query_financial_data
from components.ai_interaction import create_agent
from components.visualizations import compute_portfolio_metrics, plot_efficient_frontier, plot_sharpe_ratio_distribution
import matplotlib.pyplot as plt

# Helper function to display a bar chart for portfolio returns
def display_portfolio_chart(portfolio_summary):
    fig, ax = plt.subplots(figsize=(10, 6))
    x_labels = portfolio_summary["permno"].astype(str)
    y_values = portfolio_summary["return"]
    ax.bar(x_labels, y_values, color="skyblue")
    ax.set_xlabel("Stock (PERMNO)", fontsize=12)
    ax.set_ylabel("Return", fontsize=12)
    ax.set_title("Portfolio Returns by Stock", fontsize=14)
    ax.tick_params(axis="x", rotation=45)
    st.pyplot(fig)

# Main Program
def main():
    st.title("Advanced Portfolio Optimization with AI")

    # Initialize session state variables
    if "conn" not in st.session_state:
        st.session_state["conn"] = None
    if "data_fetched" not in st.session_state:
        st.session_state["data_fetched"] = False
    if "agent_initialized" not in st.session_state:
        st.session_state["agent_initialized"] = False
    if "agent" not in st.session_state:
        st.session_state["agent"] = None
    if "portfolio_summary" not in st.session_state:
        st.session_state["portfolio_summary"] = None

    # Section 1: Connect to WRDS
    st.subheader("Connect to WRDS")
    wrds_username = st.text_input("WRDS Username:", key="username")
    wrds_password = st.text_input("WRDS Password:", type="password", key="password")

    if st.button("Connect to WRDS"):
        conn = connect_to_wrds(wrds_username, wrds_password)
        if conn:
            st.session_state["conn"] = conn
            st.success("Successfully connected to WRDS!")
        else:
            st.error("Failed to connect to WRDS. Please check your credentials.")

    # Section 2: Fetch Data
    if st.session_state["conn"]:
        st.subheader("Fetch and Analyze Financial Data")
        tickers = st.text_area("Enter stock tickers (comma-separated):", value="AAPL, MSFT, TSLA, AMZN, GOOGL")
        start_date = st.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
        end_date = st.date_input("End Date", value=pd.to_datetime("2025-01-01"))

        if st.button("Fetch Data"):
            tickers = [ticker.strip() for ticker in tickers.split(",")]
            permnos_data = get_permnos_by_tickers(st.session_state["conn"], tickers)
            if permnos_data.empty:
                st.error("No valid PERMNOs found for the provided tickers.")
                return

            permnos = permnos_data["permno"].tolist()
            df = query_financial_data(st.session_state["conn"], start_date, end_date, permnos)
            if df is None or df.empty:
                st.error("No data fetched from WRDS.")
                return

            returns_df, expected_returns, cov_matrix = compute_portfolio_metrics(df, permnos_data)
            if returns_df is None:
                st.error("Failed to compute portfolio metrics.")
                return

            st.session_state["data_fetched"] = True
            st.session_state["portfolio_summary"] = returns_df
            st.session_state["expected_returns"] = expected_returns
            st.session_state["cov_matrix"] = cov_matrix
            st.success("Data and metrics computed successfully!")

    # Section 3: Visualization Tools
    if st.session_state["data_fetched"]:
        st.subheader("Visualization Tools")
        if st.button("Plot Efficient Frontier"):
            try:
                plot_efficient_frontier(st.session_state["expected_returns"], st.session_state["cov_matrix"])
            except Exception as e:
                st.error(f"Error plotting Efficient Frontier: {e}")

        if st.button("Plot Sharpe Ratio Distribution"):
            try:
                plot_sharpe_ratio_distribution(st.session_state["expected_returns"], st.session_state["cov_matrix"])
            except Exception as e:
                st.error(f"Error plotting Sharpe Ratio Distribution: {e}")

    # Section 4: AI Agent Interaction
    if st.session_state["data_fetched"]:
        st.subheader("Ask the AI Agent")

        if not st.session_state["agent_initialized"]:
            thinking_placeholder = st.empty()
            agent, custom_tools = create_agent(
                st.session_state["portfolio_summary"],
                st.session_state["portfolio_summary"],
                st.session_state["expected_returns"],
                st.session_state["cov_matrix"],
                thinking_placeholder,
            )
            if agent:
                st.session_state["agent"] = agent
                st.session_state["agent_initialized"] = True
                st.success("AI Agent initialized successfully!")

        user_question = st.text_input("Enter your question about portfolio optimization:")
        if st.button("Submit Question"):
            if st.session_state["agent"]:
                try:
                    with st.spinner("Agent is thinking..."):
                        response = st.session_state["agent"].run(user_question)
                        st.success("Agent Response:")
                        st.write(response)
                except Exception as e:
                    st.error(f"Error processing query: {e}")
            else:
                st.error("Agent is not initialized.")

if __name__ == "__main__":
    main()
