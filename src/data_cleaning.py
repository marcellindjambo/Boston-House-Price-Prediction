from typing import Union
import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder

logging.basicConfig(level=logging.INFO)

from abc import ABC, abstractmethod

# Define strategies for data cleaning

class DataCleaningStrategy(ABC):
    @abstractmethod
    def handle_data(self, df: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        """
        Abstract method for handling data.

        Args:
            df (pd.DataFrame): The input DataFrame.

        Returns:
            Union[pd.DataFrame, pd.Series]: Processed DataFrame or Series.
        """
        pass


class DataCleaning(DataCleaningStrategy):
    def handle_data(self, df: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        """
        Clean the input DataFrame.

        Args:
            df (pd.DataFrame): The input DataFrame.

        Returns:
            Union[pd.DataFrame, pd.Series]: Cleaned DataFrame or Series.
        """
        try:
            # Convert each column to numeric type
            df = df.apply(pd.to_numeric)
            logging.info("Convert into numeric completed successfully ")
            return df
        except Exception as e:
            logging.error("Error in Cleaning: " + str(e))
            raise e

class Standardization(DataCleaningStrategy):
    def handle_data(self, df: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        """
        Standardize the input DataFrame.

        Args:
            df (pd.DataFrame): The input DataFrame.

        Returns:
            Union[pd.DataFrame, pd.Series]: Standardized DataFrame or Series.
        """
        try:
            # Extract features and targets
            X = df.drop(columns=['MedianHomeValue'])
            y = df['MedianHomeValue']
            
            # Encode categorical feature if necessary
            if 'RadialHighwayAccessIndex' in X.columns:
                le = LabelEncoder()
                X['RadialHighwayAccessIndex'] = le.fit_transform(X['RadialHighwayAccessIndex'])
            
            # Concatenate standardized features with targets
            data_frame = pd.concat([X, y], axis=1)
            
            # Log success and return the data frame
            logging.info("Standardization completed successfully")
            
            return data_frame
        except Exception as e:
            # Log and raise exception if an error occurs during standardization
            logging.error("Error in Standardization: " + str(e))
            raise e

class DivideData(DataCleaningStrategy):
    def handle_data(self, df: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        """
        Divide the input DataFrame into training and testing sets.

        Args:
            df (pd.DataFrame): The input DataFrame.

        Returns:
            Union[pd.DataFrame, pd.Series]: Training and testing sets for features and labels.
        """
        try:
            # Split the DataFrame into features and labels
            X = df.drop('MedianHomeValue', axis=1)
            y = df['MedianHomeValue']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            logging.info("Data division completed successfully")
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error("Error in Data division: " + str(e))
            raise e

