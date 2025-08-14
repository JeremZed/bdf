from bdf.utils import Tools
from bdf.operations.visualization import Viz
from bdf.operations.outliers import Outlier
from bdf.operations.cleaner import Cleaner
from bdf.operations.transformer import Transformer
from bdf.operations.explorer import Explorer

import pandas as pd
import numpy as np
import os

import seaborn as sns
import matplotlib.pyplot as plt

class Dataset:
    """
    Represents a dataset and provides tools for data manipulation, cleaning, and analysis.

    This class allows loading data from various formats (CSV, JSON, Excel, etc.), performing
    various cleaning operations (handling missing values, duplicates), transformations
    (normalization, standardization, type conversion), as well as statistical calculations
    and visualization.

    Attributes:
        df (pd.DataFrame): The DataFrame containing the dataset's data.
        options (dict): Configuration options for managing the dataset, such as verbosity level and dataset name.
        verbose (int): Detail level for logs (default is 0).
        name (str): Name of the dataset.
        cleaner (Cleaner): A Cleaner object for cleaning the data.
        transformer (Transformer): A Transformer object for transforming the data.
        explorer (Explorer): An Explorer object for exploring the data.
    """

    def __init__(self, data, options=None, **kwargs):
        """
        Initializes the dataset and loads the data.

        Args:
            data (str | pd.DataFrame | np.ndarray | list | dict):
                Data to load. Can be a file path or in-memory data.
            options (dict, optional):
                Additional options:
                - 'verbose' (int): Detail level for logs (default is 0).
                - 'name' (str): Name of the dataset (default is randomly generated).
            kwargs: Additional arguments for loading functions.
        """

        if not isinstance(data, (str, pd.DataFrame, np.ndarray, list, dict)):
            raise ValueError(f"Invalid data type: {type(data)}")

        self.options = options or {}

        self.verbose = self.options.get('verbose', 0)
        self.name = self.options.get('name', Tools.random_id())

        # Reset attributes and load data
        self.reset().load_data(data, **kwargs)

        self.cleaner = Cleaner(self.df)
        self.transformer = Transformer(self.df)
        self.explorer = Explorer(self.df)


    def reset(self):
        """
        Resets the dataset's attributes.

        Returns:
            Dataset: The current instance of the dataset after reset.
        """
        Tools.log("Resetting dataset...", self.verbose)

        self.df = None

        Tools.log("Reset complete.", self.verbose)

        return self

    def _load_from_file(self, filepath, **kwargs):
        """
        Loads data from a file.

        Args:
            filepath (str): Path to the file.
            kwargs: Additional arguments for pandas functions.

        Returns:
            pd.DataFrame: DataFrame loaded from the file.

        Raises:
            ValueError: If the file format is not supported.
        """
        extension = filepath.split('.')[-1]
        loaders = {
            "csv": pd.read_csv,
            "json": pd.read_json,
            "xlsx": pd.read_excel,
        }
        if extension not in loaders:
            raise ValueError(f"Unsupported extension: {extension}")

        try:
            return loaders[extension](filepath, **kwargs)
        except Exception as e:
            raise ValueError(f"Error loading file {filepath}: {e}")


    def load_data(self, data, **kwargs):
        """
        Loads data and transforms it into a DataFrame.

        Args:
            data (str | pd.DataFrame | np.ndarray | list | dict):
                Data to load.
            kwargs: Additional arguments for loading functions.

        Raises:
            ValueError: If the data type is not supported.
            FileNotFoundError: If the file path is invalid.

        Returns:
            Dataset: The current instance of the dataset after loading data.
        """

        Tools.log("Loading data...", self.verbose)

        if isinstance(data, str):
            if not os.path.exists(data):
                raise FileNotFoundError(f"File not found: {data}")
            self.df = self._load_from_file(data, **kwargs)

        elif isinstance(data, pd.DataFrame):
            self.df = data

        elif isinstance(data, (list, np.ndarray)):
            self.df = pd.DataFrame(data)

        elif isinstance(data, dict):
            self.df = pd.DataFrame.from_dict(data)

        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

        Tools.log("Data loaded successfully.", self.verbose)
        return self

    def head(self, n=5):
        """
        Displays the first n rows of the dataset.

        Args:
            n (int, optional): Number of rows to display (default is 5).

        Returns:
            pd.DataFrame: The first n rows of the dataset.
        """
        return self.df.head(n)

    def tail(self, n=5):
        """
        Displays the last n rows of the dataset.

        Args:
            n (int, optional): Number of rows to display (default is 5).

        Returns:
            pd.DataFrame: The last n rows of the dataset.
        """
        return self.df.tail(n)

    def info(self, **kwargs):
        """
        Returns information about the dataset.

        Args:
            **kwargs: Additional arguments to pass to the pandas `info` function.

        Returns:
            None: Displays the dataset's information.
        """
        return self.explorer.info(**kwargs)

    def describe(self, **kwargs):
        """
        Returns a statistical summary of the columns, numerical by default.

        Args:
            **kwargs: Additional arguments to pass to the pandas `describe` function.

        Returns:
            pd.DataFrame: Statistical summary of the dataset.
        """
        return self.explorer.describe(**kwargs)

    def shape(self):
        """
        Returns the shape of the dataset.

        Returns:
            tuple: The shape of the dataset as a tuple (number of rows, number of columns).
        """
        return self.df.shape

    def missing_values(self, show_heatmap=False,figsize=(8,8)):
        """
        Visualizes the proportion of missing values in the dataset.

        Args:
            show_heatmap (bool, optional): If True, displays a heatmap of missing values.
            figsize (tuple, optional): Figure size for the heatmap (default is (8, 8)).

        Returns:
            pd.DataFrame: Proportion of missing values for each column.
        """
        return self.explorer.missing_values(show_heatmap, figsize)

    def drop_missing_values(self, **kwargs):
        """
        Removes rows containing missing values from the dataset.

        Args:
            **kwargs: Additional arguments for the pandas `dropna` function.

        Returns:
            Dataset: The current instance of the dataset after removing rows.
        """
        self.df = self.cleaner.drop_missing_values(**kwargs)
        return self

    def fill_missing(self, strategy=0, columns=None):
        """
        Fills missing values in the dataset.

        Args:
            strategy (int, str, optional): The filling strategy ('mean' for the mean, 'median' for the median,
                                      an integer or a float for a specific value).
            columns (list, optional): List of columns to fill (default is all columns).

        Returns:
            Dataset: The current instance of the dataset after filling missing values.
        """
        self.df = self.cleaner.fill_missing(strategy, columns)
        return self

    def duplicated_values(self, filter=None, show=False):
        """
        Detects and counts duplicated values in the dataset.

        Args:
            filter (str, optional): Name of the column to filter for duplicates (default is to search all columns).
            show (bool, optional): If True, displays the duplicated occurrences (default is False).

        Returns:
            pd.DataFrame or int: The number of duplicates in the dataset or a DataFrame with the filtered duplicates.
        """
        return self.cleaner.duplicated_values(filter, show)

    def drop_duplicated_values(self, **kwargs):
        """
        Removes rows containing duplicated values from the dataset.

        Args:
            **kwargs: Additional arguments for the pandas `drop_duplicates` function.

        Returns:
            Dataset: The current instance of the dataset after removing duplicates.
        """
        self.df = self.cleaner.drop_duplicated_values(**kwargs)
        return self

    def dtypes(self, mode=None):
        """
        Returns the data types of the dataset's columns.

        Args:
            mode (str, optional): If 'count', returns the count of data types (default is to return the exact types of the columns).

        Returns:
            pd.Series or pd.DataFrame: The types of the columns or a count of the types.
        """
        return self.explorer.dtypes(mode)

    def convert_dtypes(self, dtype_dict):
        """
        Converts the data types of the dataset's columns.

        Args:
            dtype_dict (dict): A dictionary where the keys are the column names and the values are the target types.

        Returns:
            Dataset: The current instance of the dataset after type conversion.
        """
        self.df = self.transformer.convert_dtypes(dtype_dict)
        return self

    def normalize(self, columns=None):
        """
        Normalizes the specified columns of the dataset to a [0, 1] scale using the Min-Max method.

        Args:
            columns (list, optional): List of columns to normalize (default is to normalize all numerical columns).

        Returns:
            Dataset: The current instance of the dataset after normalization.
        """
        self.df = self.transformer.normalize(columns)
        return self

    def standardize(self, columns=None):
        """
        Standardizes the specified columns of the dataset to a mean of 0 and a standard deviation of 1.

        Args:
            columns (list, optional): List of columns to standardize (default is to standardize all numerical columns).

        Returns:
            Dataset: The current instance of the dataset after standardization.
        """
        self.df = self.transformer.standardize(columns)
        return self

    def value_counts(self, column):
        """
        Returns the number of occurrences of each unique value in a given column.

        Args:
            column (str): The name of the column for which to get the unique values.

        Returns:
            pd.Series: Number of occurrences of each unique value in the column.
        """
        return self.explorer.value_counts(column)

    def correlations(self, show_heatmap=False, figsize=(8,8)):
        """
        Calculates and displays the correlation matrix between the numerical columns of the dataset.

        Args:
            show_heatmap (bool, optional): If True, displays a heatmap of the correlation matrix (default is False).
            figsize (tuple, optional): Figure size for the heatmap (default is (8, 8)).

        Returns:
            pd.DataFrame: Correlation matrix of the numerical columns.
        """
        return self.explorer.correlations(show_heatmap, figsize)

    def top_values(self, n=10, filter=None, show_graph=False, nb_cols=3, w_graph=5, h_graph=5,figsize=None, show_y=False):
        """
        Returns the n most frequent values for each column or a specific column.

        Args:
            n (int, optional): Number of values to return (default is 10).
            filter (str, optional): Name of the column to filter for top values (default is to take all columns).

        Returns:
            pd.Series or pd.DataFrame: The top n values for each column or for the filtered column.
        """
        return self.explorer.top_values(n, filter, show_graph, nb_cols, w_graph, h_graph, figsize, show_y)

    def filter_rows(self, condition):
        """
        Filters the dataset's rows according to a specific condition.

        Args:
            condition (str or callable): The condition to apply to filter the rows.

        Returns:
            pd.DataFrame: A DataFrame containing only the rows that meet the condition.
        """
        try:
            return self.df.query(condition)
        except Exception as e:
            raise ValueError(f"Invalid condition: {condition}. Error: {str(e)}")

    def add_column(self, column_name, values):
        """
        Adds a new column to the dataset.

        Args:
            column_name (str): The name of the new column.
            values (list or pd.Series): The values of the column to add.

        Returns:
            Dataset: The current instance of the dataset after adding the new column.
        """
        self.df[column_name] = values
        return self

    def merge(self, other, on, how='inner'):
        """
        TODO: Review the function's arguments to be flexible with the pandas function and test it.

        Merges the dataset with another Dataset or DataFrame.

        Args:
            other (Dataset or pd.DataFrame): The other dataset or DataFrame to merge with.
            on (str or list): The name of the column(s) to merge on.
            how (str, optional): The merge method (default is 'inner').

        Returns:
            Dataset: The current instance of the dataset after merging.
        """
        if isinstance(other, Dataset):
            self.df = pd.merge(self.df, other.df, on=on, how=how)
        else:
            self.df = pd.merge(self.df, other, on=on, how=how)

        return self

    def sample(self, n=5):
        """
        Returns a random sample of n rows from the dataset.

        Args:
            n (int, optional): Number of rows to sample (default is 5).

        Returns:
            pd.DataFrame: A random sample of n rows from the dataset.
        """
        return self.df.sample(n)

    def to_csv(self, filepath):
        """
        Saves the dataset to a CSV file.

        Args:
            filepath (str): The path of the file where to save the dataset.

        Returns:
            None: Saves the dataset to a CSV file.
        """
        self.df.to_csv(filepath, index=False)

    def to_excel(self, filepath):
        """
        Saves the dataset to an Excel file.

        Args:
            filepath (str): The path of the file where to save the dataset.

        Returns:
            None: Saves the dataset to an Excel file.
        """
        self.df.to_excel(filepath, index=False)

    def to_json(self, filepath):
        """
        Saves the dataset to a JSON file.

        Args:
            filepath (str): The path of the file where to save the dataset.

        Returns:
            None: Saves the dataset to a JSON file.
        """
        self.df.to_json(filepath, orient="records")

    def _get_columns_by_type(self, columns=None, type_cols=[np.number]):
        """
        Returns the columns of the dataset to be processed, based on the specified parameters.

        If the `columns` parameter is `None`, the function selects all columns of the types specified in `type_cols`.

        Args:
            columns (list, optional): List of specific columns to process. If `None`, all columns of type `type_cols` are selected.
            type_cols (list, optional): List of column types to select. By default, selects numerical columns (type `np.number`).

        Returns:
            list: List of names of the columns to process, either those specified in `columns` or all columns matching `type_cols`.
        """
        if columns is None:
            return self.df.select_dtypes(include=type_cols).columns.tolist()

        return columns

    def has_only_features_numeric(self):
        """
            Checks if the dataset contains only numerical features.
        """

        return Tools.is_all_numeric(self.df)

    def outliers(self, columns=None, method=Outlier.METHOD_IQR, show_graph=False, nb_cols=2, w_graph=5, h_graph=5, show_y=False, figsize=None, **kwargs):
        """
            Identifies and returns the outliers of a DataFrame according to the specified method.

            This function calculates the outliers for the numerical columns of a DataFrame,
            or for a specified subset of columns. By default, the method used is the IQR (Interquartile Range).
            The function can also display graphs of the outliers as boxplots.

            Args:
                columns (list[str], optional): List of columns to analyze. If `None`, all numerical columns
                    of the DataFrame are used. Default: `None`.
                method (str, optional): Method for calculating outliers. Currently, only the `"IQR"` method is supported.
                    Default: `"IQR"`.
                show_graph (bool, optional): If `True`, displays graphs of the outliers as boxplots.
                    Default: `False`.
                nb_cols (int, optional): Number of boxplots displayed per row in the graph. Default: 2.
                w_graph (int, optional): Width of each boxplot in inches. Default: 5.
                h_graph (int, optional): Height of each boxplot in inches. Default: 5.
                show_y (bool, optional): If `True`, displays the y-axis graduations on the boxplots. Default: `False`.
                figsize (tuple, optional): Size of the figure `(width, height)` in inches. If specified, overrides
                    `w_graph` and `h_graph`. Default: `None`.
                **kwargs: Additional arguments passed to the chosen method (e.g., `threshold` for IQR).

            Raises:
                ValueError: If the specified method is not supported.
                ValueError: If the DataFrame contains non-numerical columns.
                ValueError: If the outliers cannot be calculated for any reason.

            Returns:
                pd.DataFrame: A DataFrame containing:
                    - `count`: The number of outliers for each column.
                    - `ratio`: The percentage of outliers relative to the total number of rows in the DataFrame.
                    - An additional row "BDF_total_of_values" with the total outliers and the overall ratio.

            Display:
                If `show_graph` is enabled, boxplots with the outliers highlighted in red
                are displayed.

            Example:
                >>> df = pd.DataFrame({
                ...     "feature1": [10, 12, 13, 500, 11],
                ...     "feature2": [15, 14, 500, 15, 13],
                ... })
                >>> obj = YourClass(df)
                >>> outliers = obj.outliers(columns=["feature1", "feature2"], method="IQR", show_graph=True)
                >>> print(outliers)
                            count  ratio
                feature1       1  20.0
                feature2       1  20.0
                BDF_total_of_values  2  20.0

            Notes:
                - Other methods like Z-score, Isolation Forest, or One-Class SVM can be added in the future.
                - The "BDF_total_of_values" row represents the combined totals of the outliers and their overall ratio.
            """

        if method not in [Outlier.METHOD_IQR, Outlier.METHOD_ZSCORE]:
            raise ValueError(f"The calculation method is not supported: {method}")

        if columns is None:
            columns = self.df.select_dtypes(include=np.number).columns.tolist()

        outliers = None
        if method == Outlier.METHOD_IQR:
            outliers = Outlier.iqr(self.df, columns, **kwargs)

        if method == Outlier.METHOD_ZSCORE:
            outliers, z_scores = Outlier.zscore(self.df, columns, **kwargs)

        # TODO method: Tukey's method
        # TODO method: Isolation Forest
        # TODO method: One-Class SVM
        # TODO method: Quartile and Decile Clipping Method
        # TODO method: DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
        # TODO method: K-means Clustering
        # TODO method: Regression-based methods (e.g., robust regression)

        if outliers is None:
            raise ValueError("Could not calculate outliers.")

        if show_graph:
            Viz.plot_outliers(self.df, outliers, columns, method=method, nb_cols=nb_cols, w_graph=w_graph, h_graph=h_graph, figsize=figsize, show_y=show_y)

            if method == Outlier.METHOD_ZSCORE:
                Viz.plot_zscore(z_scores, columns, nb_cols=nb_cols, w_graph=w_graph, h_graph=h_graph, figsize=figsize)

        result = pd.DataFrame(outliers.sum(), columns=['count'])
        result['ratio'] = result['count'].apply(lambda x : round((x * 100) / len(self.df), 2) )

        result.loc['BDF_total_of_values'] = [ result['count'].sum(), round((result['count'].sum() * 100) / (self.df.shape[0] * self.df.shape[1]), 2)  ]

        return result