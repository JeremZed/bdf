import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from bdf.operations.visualization import Viz

class Explorer:
    """
    A class for exploring data in a DataFrame.
    """

    def __init__(self, df):
        self.df = df

    def info(self, **kwargs):
        """
        Returns information about the dataset.

        Args:
            **kwargs: Additional arguments to pass to the pandas `info` function.

        Returns:
            None: Displays the dataset's information.
        """
        return self.df.info(**kwargs)

    def describe(self, **kwargs):
        """
        Returns a statistical summary of the columns, numerical by default.

        Args:
            **kwargs: Additional arguments to pass to the pandas `describe` function.

        Returns:
            pd.DataFrame: Statistical summary of the dataset.
        """
        return self.df.describe(**kwargs)

    def correlations(self, show_heatmap=False, figsize=(8,8)):
        """
        Calculates and displays the correlation matrix between the numerical columns of the dataset.

        Args:
            show_heatmap (bool, optional): If True, displays a heatmap of the correlation matrix (default is False).
            figsize (tuple, optional): Figure size for the heatmap (default is (8, 8)).

        Returns:
            pd.DataFrame: Correlation matrix of the numerical columns.
        """
        c = self.df.corr()

        if show_heatmap:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(1,1,1)
            sns.heatmap(c, cbar=True , annot=True, cmap="coolwarm", fmt="0.2f", ax=ax)
            plt.title("Correlation Heatmap")
            plt.tight_layout()
            plt.show()

        return c

    def top_values(self, n=10, filter=None, show_graph=False, nb_cols=3, w_graph=5, h_graph=5,figsize=None, show_y=False):
        """
        Returns the n most frequent values for each column or a specific column.

        Args:
            n (int, optional): Number of values to return (default is 10).
            filter (str, optional): Name of the column to filter for top values (default is to take all columns).

        Returns:
            pd.Series or pd.DataFrame: The top n values for each column or for the filtered column.
        """

        if filter is None:
            filter = self.df.columns

        data = {}

        for column in filter:

            counts = self.df[column].value_counts().head(n)

            data[column] = {
                "value": counts.index.tolist(),
                "count": counts.values.tolist(),
            }

        # Double header
        header = pd.MultiIndex.from_product([filter, ['value', 'count']])

        rows = []
        for i in range(n):
            row = []
            for column in filter:
                values = data[column]["value"]
                counts = data[column]["count"]

                # If we exceed the size of value_counts
                row.extend([values[i] if i < len(values) else None,
                            counts[i] if i < len(counts) else None])
            rows.append(row)

        result = pd.DataFrame(rows, columns=header)

        if show_graph:
            Viz.plot_top_values(result, nb_cols=nb_cols, w_graph=w_graph, h_graph=h_graph, figsize=figsize, show_y=show_y)

        return result

    def missing_values(self, show_heatmap=False,figsize=(8,8)):
        """
        Visualizes the proportion of missing values in the dataset.

        Args:
            show_heatmap (bool, optional): If True, displays a heatmap of missing values.
            figsize (tuple, optional): Figure size for the heatmap (default is (8, 8)).

        Returns:
            pd.DataFrame: Proportion of missing values for each column.
        """
        a = self.df.isna().sum() / self.df.shape[0]
        df = pd.DataFrame(a, columns=['ratio'])
        df['count'] = self.df.isna().sum()

        df.loc['BDF_total_of_values'] = [ df['count'].sum(), round((df['count'].sum() * 100) / (self.df.shape[0] * self.df.shape[1]), 2)  ]

        if show_heatmap == True:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(1,1,1)
            sns.heatmap(self.df.isna(), cbar=True , annot=False, cmap="coolwarm", fmt="0.2f")
            plt.title("Heatmap of missing values")
            plt.tight_layout()
            plt.show()

        return df.sort_values('ratio', ascending=False)

    def value_counts(self, column):
        """
        Returns the number of occurrences of each unique value in a given column.

        Args:
            column (str): The name of the column for which to get the unique values.

        Returns:
            pd.Series: Number of occurrences of each unique value in the column.
        """
        return self.df[column].value_counts()

    def dtypes(self, mode=None):
        """
        Returns the data types of the dataset's columns.

        Args:
            mode (str, optional): If 'count', returns the count of data types (default is to return the exact types of the columns).

        Returns:
            pd.Series or pd.DataFrame: The types of the columns or a count of the types.
        """
        if mode == "count":
            return self.df.dtypes.value_counts()
        else:
            return self.df.dtypes
