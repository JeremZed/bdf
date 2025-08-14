import pandas as pd
import numpy as np

class Cleaner:
    """
    A class for cleaning data in a DataFrame.
    """

    def __init__(self, df):
        self.df = df

    def drop_missing_values(self, **kwargs):
        """
        Removes rows containing missing values from the dataset.

        Args:
            **kwargs: Additional arguments for the pandas `dropna` function.

        Returns:
            pd.DataFrame: The DataFrame after removing rows.
        """
        return self.df.dropna(**kwargs)

    def fill_missing(self, strategy=0, columns=None):
        """
        Fills missing values in the dataset.

        Args:
            strategy (int, str, optional): The filling strategy ('mean' for the mean, 'median' for the median,
                                      an integer or a float for a specific value).
            columns (list, optional): List of columns to fill (default is all columns).

        Returns:
            pd.DataFrame: The DataFrame after filling missing values.
        """
        if columns is None:
            columns = self.df.columns

        if strategy == 'mean':
            self.df[columns] = self.df[columns].fillna(self.df[columns].mean())
        elif strategy == 'median':
            self.df[columns] = self.df[columns].fillna(self.df[columns].median())
        elif isinstance(strategy, (int, float)):
            self.df[columns] = self.df[columns].fillna(strategy)

        return self.df

    def duplicated_values(self, filter=None, show=False):
        """
        Detects and counts duplicated values in the dataset.

        Args:
            filter (str, optional): Name of the column to filter for duplicates (default is to search all columns).
            show (bool, optional): If True, displays the duplicated occurrences (default is False).

        Returns:
            pd.DataFrame or int: The number of duplicates in the dataset or a DataFrame with the filtered duplicates.
        """
        if filter is not None:
            if not isinstance(filter, list):
                raise ValueError("The filter must be a list of features.")
            items = self.df.loc[self.df[filter].duplicated(keep=False), :]
        else:
            items = self.df.loc[self.df.duplicated(keep=False), :]

        if show:
            return items
        else:
            return len(items)

    def drop_duplicated_values(self, **kwargs):
        """
        Removes rows containing duplicated values from the dataset.

        Args:
            **kwargs: Additional arguments for the pandas `drop_duplicates` function.

        Returns:
            pd.DataFrame: The DataFrame after removing duplicates.
        """
        return self.df.drop_duplicates(**kwargs)
