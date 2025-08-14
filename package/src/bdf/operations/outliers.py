import pandas as pd
from scipy.stats import zscore
import matplotlib.pyplot as plt

from bdf.utils import Tools

class Outlier:

    """
    A class for detecting outliers in a DataFrame using the Interquartile Range (IQR) method.

    Methods:
        - iqr(df, features, threshold=1.5, q1=0.25, q3=0.75): Identifies outliers for specified columns in a DataFrame based on the IQR.
    """

    METHOD_IQR = "iqr"
    METHOD_ZSCORE = "zscore"

    @staticmethod
    def iqr(df, features, threshold=1.5, q1=0.25, q3=0.75):
        """
        Identifies outliers in a DataFrame using the IQR (Interquartile Range) method.

        Args:
            df (pd.DataFrame): The DataFrame containing the data to be analyzed.
            features (list): List of columns in the DataFrame for which to identify outliers.
            threshold (float, optional): Multiplier for the IQR to define the outlier bounds. Defaults to 1.5.
            q1 (float, optional): Lower quantile (Q1) to use for calculating the IQR. Defaults to 0.25 (25th percentile).
            q3 (float, optional): Upper quantile (Q3) to use for calculating the IQR. Defaults to 0.75 (75th percentile).

        Returns:
            pd.DataFrame: A binary DataFrame where each column corresponds to a feature, and each row indicates
                          whether the element is an outlier (`True`) or not (`False`).


        Example:
            >>> import pandas as pd
            >>> from outlier_detection import Outlier
            >>> data = pd.DataFrame({
                    "feature1": [1, 2, 3, 100],
                    "feature2": [5, 6, 7, 8]
                })
            >>> outliers = Outlier.iqr(data, features=["feature1", "feature2"])
            >>> print(outliers)
               feature1  feature2
            0     False     False
            1     False     False
            2     False     False
            3      True     False

        Notes:
            - The columns specified in `features` must contain numerical data.
            - The IQR method is sensitive to skewed distributions. For such cases, another method (e.g., Z-score) might be more appropriate.
        """

        if len(df) == 0:
            raise ValueError("The dataset must be populated.")

        if Tools.is_all_numeric(df[features]) == False:
            raise ValueError(f"The columns are not all of a numerical type.")

        outliers = pd.DataFrame(index=df.index)

        for feature in features:

            Q1 = df[feature].quantile(q1)
            Q3 = df[feature].quantile(q3)
            IQR = Q3 - Q1

            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR

            outliers[feature] = (df[feature] < lower_bound) | (df[feature] > upper_bound)

        return outliers

    @staticmethod
    def zscore(df, features, threshold=3):
        """
            Identifies outliers in a DataFrame using the Z-score method.

            This function calculates the Z-scores for each specified feature and identifies outliers based on
            a given threshold (default 3). An outlier is defined as a value whose Z-score is greater
            than the threshold or less than the opposite of the threshold.

            Args:
                df (pd.DataFrame): The DataFrame containing the data on which outliers will be identified.
                features (list[str]): List of column names to be analyzed for outliers.
                threshold (float, optional): Z-score threshold beyond which a value is considered an outlier.
                                            Default: 3.

            Returns:
                tuple:
                    - pd.DataFrame: A DataFrame indicating for each value if it is an outlier (True/False).
                    - pd.DataFrame: A DataFrame containing the calculated Z-scores for each specified feature.

            Raises:
                ValueError:
                    - If the `df` DataFrame is empty.
                    - If the columns specified in `features` are not all of a numerical type.

            Example:
                >>> import pandas as pd
                >>> import numpy as np
                >>> df = pd.DataFrame({
                >>>     'feature1': [10, 12, 13, 500, 11],
                >>>     'feature2': [15, 14, 500, 15, 13],
                >>> })
                >>> outliers, z_scores = zscore(df, features=['feature1', 'feature2'], threshold=3)

            Notes:
                - An outlier is a value whose Z-score is greater than the threshold or less than the opposite of the threshold.
                - This method assumes that the data follows a normal distribution.
                - The Z-scores are calculated via the `zscore` method from SciPy, which returns a measure of the distance in terms of standard deviations of the value from the mean.

        """

        if len(df) == 0:
            raise ValueError("The dataset must be populated.")

        if Tools.is_all_numeric(df[features]) == False:
            raise ValueError(f"The columns are not all of a numerical type.")

        outliers = pd.DataFrame(index=df.index)

        # We calculate the zscore for each feature
        z_scores = df.apply(zscore)

        for feature in features:
            outliers[feature] = (z_scores[feature] < -threshold) | (z_scores[feature] > threshold)

        return outliers, z_scores
