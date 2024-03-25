from steps.ingest_df import data_ingestion
from steps.clean_data import Data_Cleaning_Step
from steps.model_train import train_evaluate_and_save_all_models

from datetime import datetime
import logging

# Configuration du journal
logging.basicConfig(level=logging.INFO)

def ml_pipeline():
    """
    This function is the main pipeline for the machine learning process.
    It performs the following steps:
    1. Logs the start time of the pipeline.
    2. Ingests the data.
    3. Cleans the data.
    4. Trains and evaluates the models and saves the best ones.
    5. Logs the end time of the pipeline.
    """
    # Log the start time of the pipeline
    debut_pipeline = datetime.now()
    logging.info(f"Start of pipeline : {debut_pipeline}")
    
    # Ingest the data
    # This function reads the dataset from a CSV file and returns a DataFrame.
    data = data_ingestion()
    
    # Clean the data
    # This function performs data cleaning steps such as handling missing values,
    # encoding categorical variables, and splitting the data into training and test sets.
    X_train, X_test, y_train, y_test = Data_Cleaning_Step(data)
    
    # Train and evaluate the models and save the best ones
    # This function trains multiple models, evaluates them, and saves the best ones.
    trained_model = train_evaluate_and_save_all_models(X_train, X_test, y_train, y_test)
    
    # Log the end time of the pipeline
    fin_pipeline = datetime.now()
    logging.info(f"End of pipeline : {fin_pipeline}")




if __name__ == "__main__":
    ml_pipeline()
