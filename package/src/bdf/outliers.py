import pandas as pd
from scipy.stats import zscore
import matplotlib.pyplot as plt

from bdf.tools import Tools

class Outlier:

    """
    Classe pour la détection des valeurs aberrantes (outliers) dans un DataFrame en utilisant la méthode de l'intervalle interquartile (IQR).

    Méthodes:
        - iqr(df, features, threshold=1.5, q1=0.25, q3=0.75): Identifie les outliers pour les colonnes spécifiées dans un DataFrame en fonction de l'IQR.
    """

    METHOD_IQR = "iqr"
    METHOD_ZSCORE = "zscore"

    @staticmethod
    def iqr(df, features, threshold=1.5, q1=0.25, q3=0.75):
        """
        Identifie les outliers dans un DataFrame en utilisant la méthode IQR (Interquartile Range).

        Args:
            df (pd.DataFrame): Le DataFrame contenant les données à analyser.
            features (list): Liste des colonnes du DataFrame pour lesquelles identifier les outliers.
            threshold (float, optional): Multiplicateur pour l'IQR pour définir les bornes des outliers. Par défaut, 1.5.
            q1 (float, optional): Quantile inférieur (Q1) à utiliser pour le calcul de l'IQR. Par défaut, 0.25 (25e percentile).
            q3 (float, optional): Quantile supérieur (Q3) à utiliser pour le calcul de l'IQR. Par défaut, 0.75 (75e percentile).

        Returns:
            pd.DataFrame: Un DataFrame binaire où chaque colonne correspond à une feature, et chaque ligne indique
                          si l'élément est un outlier (`True`) ou non (`False`).


        Exemple:
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
            - Les colonnes spécifiées dans `features` doivent contenir des données numériques.
            - La méthode IQR est sensible aux distributions asymétriques. Pour ces cas, une autre méthode (e.g., Z-score) pourrait être plus appropriée.
        """

        if len(df) == 0:
            raise ValueError("Le dataset doit être alimenté.")

        if Tools.is_all_numeric(df[features]) == False:
            raise ValueError(f"Les colonnes de sont pas toutes de type numérique.")

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
            Identifie les outliers dans un DataFrame en utilisant la méthode des Z-scores.

            Cette fonction calcule les Z-scores pour chaque feature spécifiée et identifie les outliers en fonction
            d'un seuil donné (par défaut 3). Un outlier est défini comme une valeur dont le Z-score est supérieur
            au seuil ou inférieur à l'opposé du seuil.

            Args:
                df (pd.DataFrame): Le DataFrame contenant les données sur lesquelles les outliers seront identifiés.
                features (list[str]): Liste des noms des colonnes à analyser pour les outliers.
                threshold (float, optional): Seuil de Z-score au-delà duquel une valeur est considérée comme un outlier.
                                            Par défaut : 3.

            Returns:
                tuple:
                    - pd.DataFrame: Un DataFrame indiquant pour chaque valeur si elle est un outlier (True/False).
                    - pd.DataFrame: Un DataFrame contenant les Z-scores calculés pour chaque feature spécifiée.

            Raises:
                ValueError:
                    - Si le DataFrame `df` est vide.
                    - Si les colonnes spécifiées dans `features` ne sont pas toutes de type numérique.

            Exemple:
                >>> import pandas as pd
                >>> import numpy as np
                >>> df = pd.DataFrame({
                >>>     'feature1': [10, 12, 13, 500, 11],
                >>>     'feature2': [15, 14, 500, 15, 13],
                >>> })
                >>> outliers, z_scores = zscore(df, features=['feature1', 'feature2'], threshold=3)

            Notes:
                - Un outlier est une valeur dont le Z-score est supérieur au seuil ou inférieur à l'opposé du seuil.
                - Cette méthode suppose que les données suivent une distribution normale.
                - Les Z-scores sont calculés via la méthode `zscore` de SciPy, qui renvoie une mesure de la distance en termes d'écarts-types de la valeur par rapport à la moyenne.

        """

        if len(df) == 0:
            raise ValueError("Le dataset doit être alimenté.")

        if Tools.is_all_numeric(df[features]) == False:
            raise ValueError(f"Les colonnes de sont pas toutes de type numérique.")

        outliers = pd.DataFrame(index=df.index)

        # On calcule le zscore pour chaque feature
        z_scores = df.apply(zscore)

        for feature in features:
            outliers[feature] = (z_scores[feature] < -threshold) | (z_scores[feature] > threshold)

        return outliers, z_scores
