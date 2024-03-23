import pandas as pd
import logging
# Definir une classe IngestData

# Configuration du niveau de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
logging.basicConfig(level=logging.INFO)

class IngestData:
    """
    Class to ingest a data from a given path.

    Parameters
    ----------
    path : str
        Path to the data file.

    Methods
    -------
    get_data()
        Get the data from the given path.
    """

    def __init__(self, path: str) -> None:
        """
        Initialize the IngestData instance.

        Parameters
        ----------
        path : str
            Path to the data file.
        """
        self.path = path


    def get_data(self) -> pd.DataFrame:
        """
        Get the data from the given path.

        Returns
        -------
        pd.DataFrame
            The data read from the path.
        """
        try:
            data = pd.read_csv(self.path)
            logging.info(f"Data read from path: {self.path} completed successfully.")
            return data
        except Exception as e:
            logging.error(f"Error reading data from path: {self.path}. Error: {e}")
            
        
