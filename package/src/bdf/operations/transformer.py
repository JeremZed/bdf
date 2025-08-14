import pandas as pd
import numpy as np

class Transformer:
    """
    A class for transforming data in a DataFrame.
    """

    def __init__(self, df):
        self.df = df

    def convert_dtypes(self, dtype_dict):
        """
        Converts the data types of the dataset's columns.

        Args:
            dtype_dict (dict): A dictionary where the keys are the column names and the values are the target types.

        Returns:
            pd.DataFrame: The DataFrame after type conversion.
        """
        return self.df.astype(dtype_dict)

    def normalize(self, columns=None):
        """
        Normalizes the specified columns of the dataset to a [0, 1] scale using the Min-Max method.

        Args:
            columns (list, optional): List of columns to normalize (default is to normalize all numerical columns).

        Returns:
            pd.DataFrame: The DataFrame after normalization.
        """
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()

        self.df[columns] = self.df[columns].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
        return self.df

    def standardize(self, columns=None):
        """
        Standardizes the specified columns of the dataset to a mean of 0 and a standard deviation of 1.

        Args:
            columns (list, optional): List of columns to standardize (default is to standardize all numerical columns).

        Returns:
            pd.DataFrame: The DataFrame after standardization.
        """
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()

        self.df[columns] = self.df[columns].apply(lambda x: (x - x.mean()) / x.std())
        return self.df
