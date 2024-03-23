import pandas as pd
from src.ingest_data import IngestData
import logging



logging.basicConfig(level=logging.INFO)

def data_ingestion() -> pd.DataFrame:
    """
    Function to ingest data from a specified file path.

    Returns
    -------
    pd.DataFrame
        The DataFrame containing the ingested data.
    """
    try:
        # Define the file path
        file_path = r"https://raw.githubusercontent.com/marcellindjambo/Boston-House-Price-Prediction/main/data/housing_data.csv"
        
        # Create an instance of IngestData with the specified file path
        data_ingestion = IngestData(file_path)
        
        # Get the data from the file path and store it in a DataFrame
        df = data_ingestion.get_data()
        
        # Log a success message
        logging.info("Data ingestion completed successfully.")
        
        # Return the DataFrame containing the ingested data
        return df 
    except Exception as e:
        # Log an error message with the exception information
        logging.error(f"Error ingesting data. Error: {e}")
