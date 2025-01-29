import streamlit as st
from langchain_openai import OpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
import os
from dotenv import load_dotenv
import pandas as pd  # Import pandas for creating a sample DataFrame

# Load the API key from environment variables or .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Debugging: Show the API key in the terminal (but not in Streamlit for security)
print(f"Loaded API Key: {OPENAI_API_KEY}")

# Streamlit App
st.title("Portfolio Optimization Agent Test")

# Display the API key for debugging purposes (you can remove this for production)
if OPENAI_API_KEY:
    st.success("API Key loaded successfully!")
else:
    st.error("OpenAI API key is missing! Check your .env file or environment variables.")

# Create a sample DataFrame for testing
data = {
    'date': ['2021-01-01', '2021-01-02', '2021-01-03'],
    'permno': [12345, 12345, 12345],
    'return': [0.01, 0.02, 0.015]
}
df = pd.DataFrame(data)

# Dummy variables for testing
returns_df = df[['date', 'return']]
expected_returns = df['return'].mean()
cov_matrix = df[['return']].cov()

def create_agent_ui():
    """
    Create an AI agent and display responses in the UI.
    """
    try:
        # Initialize the OpenAI LLM
        llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)

        # Create the agent
        agent = create_pandas_dataframe_agent(llm, df, verbose=True, allow_dangerous_code=True)

        # User input for questions
        query = st.text_input("Ask a question about the DataFrame:")
        if st.button("Submit Question"):
            if query:
                # Get the agent response
                response = agent.run(query)
                st.write("Agent Response:", response)
            else:
                st.error("Please enter a query.")
    except Exception as e:
        st.error(f"Error creating agent: {e}")
        print(f"Error creating agent: {e}")

# Run the UI
if OPENAI_API_KEY:
    create_agent_ui()
else:
    st.error("Cannot create agent because the API key is missing.")
