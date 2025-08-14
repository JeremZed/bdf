import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import math

from bdf.utils import Tools

class Viz:

    @staticmethod
    def plot_top_values(result_df, nb_cols=3, w_graph=5, h_graph=5, figsize=None, show_y=False):
        """
        Generates a bar chart for each feature of a DataFrame with a multi-index,
        representing the values and their counts.

        This method creates a figure with several subplots that show the
        most frequent (top) values for each feature in the DataFrame. Each
        subplot contains a bar chart with the values on the x-axis and the counts
        on the y-axis.

        Args:
            result_df (pd.DataFrame): A DataFrame with a multi-index where each column represents
                                    a feature with two levels: 'value' and 'count'.
                                    Example structure:
                                    - ('feature_name', 'value'): feature values
                                    - ('feature_name', 'count'): counts associated with the values.
            nb_cols (int, optional): Number of subplot columns in the figure. Defaults to 3.
            w_graph (int, optional): Width of an individual chart in the figure, in inches. Defaults to 5.
            h_graph (int, optional): Height of an individual chart in the figure, in inches. Defaults to 5.
            show_y (bool, optional): If True, displays the y-axis graduations. Defaults to False.

        Raises:
            ValueError: If the DataFrame does not contain the necessary columns or if the data is poorly formatted.

        Returns:
            None: The function generates a chart via matplotlib and displays it, but does not return anything.

        Notes:
            - The values on the X-axis are the different unique values for each feature.
            - The Y-axis represents the counts of the occurrences of these values.
            - If `show_y` is True, the Y-axis will display graduations adapted to the maximum of the counts.
            - The function will automatically hide unused subplots if the number of features is less than
            the number of available subplots.
            - The method uses `matplotlib.pyplot` to generate and display the charts.

        Example:
            # Example of use with a correctly formatted `result_df` DataFrame.
            Viz.plot_top_values(result_df, nb_cols=2, w_graph=6, h_graph=4, show_y=True)
        """

        if len(result_df) == 0:
            raise ValueError("The dataset must be populated.")

        num_features = len(result_df.columns.levels[0])
        num_rows = math.ceil((num_features / nb_cols))

        if figsize is None:
            figsize = (w_graph * nb_cols, h_graph * num_rows)

        fig, axes = plt.subplots(num_rows, nb_cols, figsize=figsize, sharey=False)

        if num_rows == 1:
            axes = np.array([axes])

        axes = axes.flatten()

        for i, feature in enumerate(result_df.columns.levels[0]):

            values = result_df[(feature, 'value')].astype(str).dropna()
            counts = result_df[(feature, 'count')].dropna()

            bars = axes[i].bar(values, counts, color='skyblue', edgecolor='black')

            # Value above each bar
            for bar, count in zip(bars, counts):
                axes[i].text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height(),
                            str(count),
                            ha='center', va='bottom', fontsize=10, color='black')


            axes[i].set_title(f"Top Values for {feature}", fontsize=14)
            axes[i].set_xlabel("Values", fontsize=12)
            axes[i].set_ylabel("Counts", fontsize=12)
            axes[i].tick_params(axis='x', rotation=90)

            # Additional graduations on the y-axis and presence of at least 5 graduations
            if show_y:
                y_max = counts.max()
                step = max(1, y_max // 5)
                axes[i].set_yticks(np.arange(0, y_max + (step * 2), step))
            else:
                axes[i].set_yticks([])

        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        plt.show()

    def plot_outliers(df, outliers, columns, method, nb_cols=2, w_graph=5, h_graph=5, figsize=None, show_y=False):
        """
            Displays boxplots to visualize the values and outliers of a DataFrame.

            This function generates boxplots for each specified feature and marks the outliers
            identified by the IQR method in red. The graphs are displayed on several rows
            and columns depending on the number of specified columns.

            Args:
                df (pd.DataFrame): The DataFrame containing the data to be visualized.
                outliers (pd.DataFrame): DataFrame of the same dimension as `df`, with booleans indicating
                    the outliers for each column (typical result of the IQR method).
                columns (list[str]): List of column names to be visualized.
                method (str): Outlier detection method.
                nb_cols (int, optional): Number of boxplots displayed per row. Defaults to 2.
                w_graph (int, optional): Width of each boxplot. Defaults to 5.
                h_graph (int, optional): Height of each boxplot. Defaults to 5.
                figsize (tuple, optional): Size of the figure (width, height). If specified, overrides
                    `w_graph` and `h_graph`. Defaults to None.
                show_y (bool, optional): If True, displays the y-axis ticks for the boxplots.
                    Defaults to False.

            Raises:
                ValueError: If the `df` DataFrame is empty.

            Example:
                >>> import pandas as pd
                >>> import numpy as np
                >>> import seaborn as sns
                >>> from matplotlib import pyplot as plt
                >>> df = pd.DataFrame({
                ...     'feature1': [10, 12, 13, 500, 11],
                ...     'feature2': [15, 14, 500, 15, 13],
                ... })
                >>> outliers = pd.DataFrame({
                ...     'feature1': [False, False, False, True, False],
                ...     'feature2': [False, False, True, False, False],
                ... }, index=df.index)
                >>> plot_outliers_iqr(df, outliers, columns=['feature1', 'feature2'])

            Notes:
                - The outliers are displayed in red.
                - If the total number of columns to be visualized is odd, the unused subplots
                will be hidden.
            """

        if len(df) == 0:
            raise ValueError("The dataset must be populated.")

        if Tools.all_features_present(columns, df.columns) == False:
            raise ValueError("Not all features are available.")

        num_features = len(columns)
        num_rows = math.ceil((num_features / nb_cols))

        if figsize is None:
            figsize = (w_graph * nb_cols, h_graph * num_rows)

        fig, axes = plt.subplots(num_rows, nb_cols, figsize=figsize, sharey=False)

        fig.suptitle(f'Representation of Outliers with the method: {method}', fontsize=16, y=1.0)

        if num_rows == 1:
            axes = np.array([axes])

        axes = axes.flatten()

        for i, c in enumerate(columns):

            sns.boxplot(x=df[c], color='lightblue', label='Box', ax=axes[i], showfliers=False)
            sns.scatterplot(x=df[c][outliers[c]], y=[0] * outliers[c].sum(), color='red', label='Outliers', s=100, ax=axes[i])

            # Title and labels
            axes[i].set_title(f"Outliers for {c}", fontsize=12)
            axes[i].set_xlabel(c, fontsize=10)
            axes[i].set_ylabel('Values', fontsize=10)

            if show_y:
                axes[i].set_yticks(np.arange(min(df[c]), max(df[c]) + 1, 1))

        # Hide unused subplots if the number of features is odd
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")


        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_zscore(z_scores, columns, nb_cols=2, w_graph=5, h_graph=5, figsize=None):
        """
            Displays histograms of Z-scores for each specified feature.

            This function generates a chart per feature in which the histogram of Z-scores is displayed, thus allowing to visualize the distribution of Z-scores for each column. The charts are organized in several rows and columns, depending on the number of specified columns.

            Args:
                z_scores (pd.DataFrame): DataFrame containing the calculated Z-scores for each feature.
                columns (list[str]): List of column names to be visualized in the chart.
                nb_cols (int, optional): Number of charts (boxplots) per row. Defaults to 2.
                w_graph (int, optional): Width of each chart. Defaults to 5.
                h_graph (int, optional): Height of each chart. Defaults to 5.
                figsize (tuple, optional): Size of the figure (width, height). If specified, overrides `w_graph` and `h_graph`. Defaults to None.

            Raises:
                ValueError: If the `z_scores` DataFrame is empty or if the specified columns are not present in the DataFrame.

            Example:
                >>> import pandas as pd
                >>> import numpy as np
                >>> z_scores = pd.DataFrame({
                >>>     'feature1': [0.5, -0.8, 1.2, -0.3, 2.0],
                >>>     'feature2': [1.5, -1.3, 0.8, 0.2, -0.7]
                >>> })
                >>> plot_zscore(z_scores, columns=['feature1', 'feature2'])

            Notes:
                - The Z-scores are displayed in a histogram for each feature.
                - If the number of charts is odd, the unused subplots will be hidden.
                - Each histogram represents the distribution of Z-scores for the corresponding feature.

        """

        if len(z_scores) == 0:
            raise ValueError("The dataset must be populated.")

        if Tools.all_features_present(columns, z_scores.columns) == False:
            raise ValueError("Not all features are available.")

        num_features = len(columns)
        num_rows = math.ceil((num_features / nb_cols))

        if figsize is None:
            figsize = (w_graph * nb_cols, h_graph * num_rows)

        fig, axes = plt.subplots(num_rows, nb_cols, figsize=figsize, sharey=False)
        fig.suptitle(f'Representation of the distribution of zscores', fontsize=16, y=1.0)

        if num_rows == 1:
            axes = np.array([axes])

        axes = axes.flatten()

        for i, c in enumerate(columns):

            axes[i].hist(z_scores[c], bins=15, alpha=0.5, color='g')

            axes[i].set_title(f"zscore for {c}", fontsize=12)
            axes[i].set_xlabel('Z-score', fontsize=10)
            axes[i].set_ylabel('Frequency', fontsize=10)

        # Hide unused subplots if the number of features is odd
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        plt.show()
