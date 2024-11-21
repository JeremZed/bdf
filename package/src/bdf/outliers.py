import pandas as pd

class Outlier:

    """
    Classe pour la détection des valeurs aberrantes (outliers) dans un DataFrame en utilisant la méthode de l'intervalle interquartile (IQR).

    Méthodes:
        - iqr(df, features, threshold=1.5, q1=0.25, q3=0.75): Identifie les outliers pour les colonnes spécifiées dans un DataFrame en fonction de l'IQR.
    """

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


        outliers = pd.DataFrame(index=df.index)

        for feature in features:
            # Calcul des quartiles Q1 et Q3 et de l'IQR
            Q1 = df[feature].quantile(q1)
            Q3 = df[feature].quantile(q3)
            IQR = Q3 - Q1

            # Définir les bornes pour les outliers
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR

            # Identifier les outliers
            outliers[feature] = (df[feature] < lower_bound) | (df[feature] > upper_bound)

        return outliers