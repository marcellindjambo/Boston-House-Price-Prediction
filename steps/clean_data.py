from typing import Union
from src.data_cleaning import DivideData, Standardization, DataCleaning
import pandas as pd
import logging
logging.basicConfig(level=logging.INFO)

def Data_Cleaning_Step(df: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
    """
    This function performs a step in the data cleaning pipeline. It takes a pandas DataFrame as input,
    applies the data cleaning, standardization and division into training and testing sets, and returns
    the resulting training and testing sets for features and labels.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        tuple: A tuple containing the training and testing sets for features and labels.

    Raises:
        Exception: If any error occurs during the data cleaning step.
    """
    try:
        # Assign the input data to a variable
        data_cleaning = DataCleaning()  
        df_cleaned = data_cleaning.handle_data(df) 
        
        standardization = Standardization()  
        df_standardized = standardization.handle_data(df_cleaned) 

        divide_data = DivideData()  
        X_train, X_test, y_train, y_test = divide_data.handle_data(df_standardized)  

        logging.info("Data cleaning step completed successfully")
        return X_train, X_test, y_train, y_test

    except Exception as e:
        logging.error("Error in Data Cleaning Step: " + str(e)) 
        raise e  # Re-raise the exception
