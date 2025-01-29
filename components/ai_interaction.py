from langchain_openai import OpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.callbacks.base import BaseCallbackHandler
import streamlit as st
from components.visualizations import plot_efficient_frontier, plot_sharpe_ratio_distribution

# Callback handler for Streamlit integration
class StreamlitCallbackHandler(BaseCallbackHandler):
    def __init__(self, placeholder):
        self.placeholder = placeholder
        self.logs = ""

    def on_chain_start(self, serialized, inputs, **kwargs):
        self.logs += f"Starting chain...\nInputs: {inputs}\n\n"
        self.placeholder.text(self.logs)

    def on_tool_start(self, tool_name, tool_input, **kwargs):
        self.logs += f"Using tool: {tool_name}\nInput: {tool_input}\n\n"
        self.placeholder.text(self.logs)

    def on_text(self, text, **kwargs):
        self.logs += f"{text}\n"
        self.placeholder.text(self.logs)

    def on_chain_end(self, outputs, **kwargs):
        self.logs += f"Chain finished.\nOutputs: {outputs}\n\n"
        self.placeholder.text(self.logs)

# Step 1: Define helper functions for visualization tools
def efficient_frontier_tool(expected_returns, cov_matrix):
    """
    Plot the efficient frontier for the given returns and covariance matrix.
    """
    try:
        plot_efficient_frontier(expected_returns, cov_matrix)
        return "Efficient Frontier plotted successfully."
    except Exception as e:
        return f"Error plotting Efficient Frontier: {e}"

def sharpe_ratio_tool(expected_returns, cov_matrix):
    """
    Plot the Sharpe ratio distribution for the given returns and covariance matrix.
    """
    try:
        plot_sharpe_ratio_distribution(expected_returns, cov_matrix)
        return "Sharpe Ratio Distribution plotted successfully."
    except Exception as e:
        return f"Error plotting Sharpe Ratio Distribution: {e}"

# Step 2: Create the agent and define tools
def create_agent(df, returns_df, expected_returns, cov_matrix, streamlit_placeholder):
    """
    Create an AI agent using LangChain for portfolio optimization and querying.

    Parameters:
        df (pd.DataFrame): Portfolio DataFrame.
        returns_df (pd.DataFrame): Returns DataFrame.
        expected_returns (pd.Series): Expected returns.
        cov_matrix (pd.DataFrame): Covariance matrix.
        streamlit_placeholder: Streamlit placeholder for logs.

    Returns:
        agent: LangChain agent.
        custom_tools: Dictionary of custom visualization tools.
    """
    try:
        # Initialize OpenAI LLM
        llm = OpenAI(temperature=0, openai_api_key="")
        callback_handler = StreamlitCallbackHandler(streamlit_placeholder)

        # Custom visualization tools
        def tool_efficient_frontier():
            return efficient_frontier_tool(expected_returns, cov_matrix)

        def tool_sharpe_ratio():
            return sharpe_ratio_tool(expected_returns, cov_matrix)

        # Tool dictionary
        custom_tools = {
            "Efficient Frontier": tool_efficient_frontier,
            "Sharpe Ratio Distribution": tool_sharpe_ratio,
        }

        # Step 3: Initialize the LangChain agent
        agent = create_pandas_dataframe_agent(
            llm,
            df,
            verbose=True,
            allow_dangerous_code=True,
            callbacks=[callback_handler],
        )

        print("AI Agent created successfully.")
        return agent, custom_tools
    except Exception as e:
        st.error(f"Error creating AI agent: {e}")
        return None, {}
