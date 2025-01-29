import wrds
import pandas as pd

def connect_to_wrds(wrds_username, wrds_password):
    """
    Connect to the WRDS database using the provided username and password.

    Parameters:
        wrds_username (str): WRDS username
        wrds_password (str): WRDS password

    Returns:
        conn: WRDS connection object if successful, otherwise None
    """
    try:
        print("Connecting to WRDS...")
        conn = wrds.Connection(wrds_username=wrds_username, password=wrds_password)
        print("Connected successfully!")
        return conn
    except Exception as e:
        print(f"Error connecting to WRDS: {e}")
        return None

def get_permnos_by_tickers(conn, tickers):
    """
    Fetch PERMNOs for the given tickers.

    Parameters:
        conn: WRDS connection object
        tickers (list): List of stock tickers

    Returns:
        pd.DataFrame: DataFrame containing PERMNOs, tickers, and company names
    """
    try:
        print(f"Fetching PERMNOs for tickers: {tickers}...")
        query = f"""
            SELECT DISTINCT permno, ticker, comnam
            FROM crsp.stocknames
            WHERE ticker IN ({','.join([f"'{ticker}'" for ticker in tickers])})
        """
        result = conn.raw_sql(query)
        if result.empty:
            print("No matching PERMNOs found for the given tickers.")
            return pd.DataFrame()  # Return an empty DataFrame if no matches

        # Deduplicate and keep the most recent records
        deduplicated = result.drop_duplicates(subset=['ticker'], keep='last')
        print(f"Filtered PERMNOs:\n{deduplicated}")
        return deduplicated
    except Exception as e:
        print(f"Error fetching PERMNOs: {e}")
        return pd.DataFrame()

def query_financial_data(conn, start_date, end_date, permnos):
    """
    Query financial data for the given PERMNOs and date range.

    Parameters:
        conn: WRDS connection object
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
        permnos (list): List of PERMNOs

    Returns:
        pd.DataFrame: DataFrame containing financial data
    """
    try:
        print(f"Querying WRDS for data from {start_date} to {end_date} for PERMNOs: {permnos}...")
        query = f"""
            SELECT permno, date, prc AS price, vol AS volume, ret AS return
            FROM crsp.dsf
            WHERE date BETWEEN '{start_date}' AND '{end_date}'
            AND permno IN ({','.join(map(str, permnos))})
        """
        df = conn.raw_sql(query)
        print(f"Data retrieved successfully! {len(df)} rows fetched.")
        return df
    except Exception as e:
        print(f"Error querying data: {e}")
        return None
