import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import math

class Viz:

    @staticmethod
    def plot_top_values(result_df, nb_cols=3, w_graph=5, h_graph=5, figsize=None, show_y=False):
        """
        Génère un graphique en barres pour chaque feature d'un DataFrame avec un multi-index,
        représentant les valeurs et leurs comptages.

        Cette méthode crée une figure avec plusieurs sous-graphiques (subplots) qui montrent les
        valeurs les plus fréquentes (top values) pour chaque feature dans le DataFrame. Chaque
        sous-graphe contient un graphique en barres avec les valeurs en abscisse et les comptages
        en ordonnée.

        Args:
            result_df (pd.DataFrame): Un DataFrame avec un multi-index dont chaque colonne représente
                                    un feature avec deux niveaux : 'value' et 'count'.
                                    Exemple de structure :
                                    - ('feature_name', 'value') : valeurs des features
                                    - ('feature_name', 'count') : comptages associés aux valeurs.
            nb_cols (int, optional): Nombre de colonnes de sous-graphiques dans la figure. Par défaut, 3.
            w_graph (int, optional): Largeur d'un graphique individuel dans la figure, en pouces. Par défaut, 5.
            h_graph (int, optional): Hauteur d'un graphique individuel dans la figure, en pouces. Par défaut, 5.
            show_y (bool, optional): Si True, affiche les graduations sur l'axe Y. Par défaut, False.

        Raises:
            ValueError: Si le DataFrame ne contient pas les colonnes nécessaires ou si les données sont mal formatées.

        Returns:
            None: La fonction génère un graphique via matplotlib et l'affiche, mais ne retourne rien.

        Notes:
            - Les valeurs sur l'axe des X sont les différentes valeurs uniques pour chaque feature.
            - L'axe des Y représente les comptages des occurrences de ces valeurs.
            - Si `show_y` est True, l'axe Y affichera des graduations adaptées au maximum des comptages.
            - La fonction masquera automatiquement les sous-graphiques inutilisés si le nombre de features est inférieur à
            celui des sous-graphiques disponibles.
            - La méthode utilise `matplotlib.pyplot` pour générer et afficher les graphiques.

        Example:
            # Exemple d'utilisation avec un DataFrame `result_df` correctement formaté.
            Viz.plot_top_values(result_df, nb_cols=2, w_graph=6, h_graph=4, show_y=True)
        """

        if len(result_df) == 0:
            raise ValueError("Le dataset doit être alimenté.")

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

            # Valeur au-dessus de chaque barre
            for bar, count in zip(bars, counts):
                axes[i].text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height(),
                            str(count),
                            ha='center', va='bottom', fontsize=10, color='black')


            axes[i].set_title(f"Top Values for {feature}", fontsize=14)
            axes[i].set_xlabel("Values", fontsize=12)
            axes[i].set_ylabel("Counts", fontsize=12)
            axes[i].tick_params(axis='x', rotation=90)

            # graduations supplémentaires sur l'axe y et présence d'au moins 5 graduations
            if show_y:
                y_max = counts.max()
                step = max(1, y_max // 5)
                axes[i].set_yticks(np.arange(0, y_max + (step * 2), step))
            else:
                axes[i].set_yticks([])

        # On masque les sous-graphiques inutilisés
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        plt.show()

    def plot_outliers(df, outliers, columns, method, nb_cols=2, w_graph=5, h_graph=5, figsize=None, show_y=False):
        """
            Affiche des boxplots pour visualiser les valeurs et les outliers d'un DataFrame.

            Cette fonction génère des boxplots pour chaque feature spécifiée et marque les outliers
            identifiés par la méthode IQR en rouge. Les graphes sont affichés sur plusieurs lignes
            et colonnes en fonction du nombre de colonnes spécifiées.

            Args:
                df (pd.DataFrame): Le DataFrame contenant les données à visualiser.
                outliers (pd.DataFrame): DataFrame de même dimension que `df`, avec des booléens indiquant
                    les outliers pour chaque colonne (résultat typique de la méthode IQR).
                columns (list[str]): Liste des noms des colonnes à visualiser.
                method (str): Méthode de detection des outliers.
                nb_cols (int, optional): Nombre de boxplots affichés par ligne. Par défaut : 2.
                w_graph (int, optional): Largeur de chaque boxplot. Par défaut : 5.
                h_graph (int, optional): Hauteur de chaque boxplot. Par défaut : 5.
                figsize (tuple, optional): Taille de la figure (largeur, hauteur). Si spécifiée, remplace
                    `w_graph` et `h_graph`. Par défaut : None.
                show_y (bool, optional): Si True, affiche les ticks de l'axe y pour les boxplots.
                    Par défaut : False.

            Raises:
                ValueError: Si le DataFrame `df` est vide.

            Exemple:
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
                - Les outliers sont affichés en rouge.
                - Si le nombre total de colonnes à visualiser est impair, les sous-graphiques inutilisés
                seront masqués.
            """

        if len(df) == 0:
            raise ValueError("Le dataset doit être alimenté.")

        num_features = len(columns)
        num_rows = math.ceil((num_features / nb_cols))

        if figsize is None:
            figsize = (w_graph * nb_cols, h_graph * num_rows)

        fig, axes = plt.subplots(num_rows, nb_cols, figsize=figsize, sharey=False)

        fig.suptitle(f'Représentation des Outliers avec la méthode : {method}', fontsize=16, y=1.0)

        if num_rows == 1:
            axes = np.array([axes])

        axes = axes.flatten()

        for i, c in enumerate(columns):

            sns.boxplot(x=df[c], color='lightblue', label='Box', ax=axes[i], showfliers=False)
            sns.scatterplot(x=df[c][outliers[c]], y=[0] * outliers[c].sum(), color='red', label='Outliers', s=100, ax=axes[i])

            # Titre et labels
            axes[i].set_title(f"Outliers pour {c}", fontsize=12)
            axes[i].set_xlabel(c, fontsize=10)
            axes[i].set_ylabel('Valeurs', fontsize=10)

            if show_y:
                axes[i].set_yticks(np.arange(min(df[c]), max(df[c]) + 1, 1))

        # Masquer les sous-graphes inutilisés si le nombre de features est impair
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")


        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_zscore(z_scores, columns, nb_cols=2, w_graph=5, h_graph=5, figsize=None):
        """
            Affiche les histogrammes des Z-scores pour chaque feature spécifiée.

            Cette fonction génère un graphique par feature dans lequel l'histogramme des Z-scores est affiché, permettant ainsi de visualiser la distribution des Z-scores pour chaque colonne. Les graphiques sont organisés en plusieurs lignes et colonnes, en fonction du nombre de colonnes spécifié.

            Args:
                z_scores (pd.DataFrame): DataFrame contenant les Z-scores calculés pour chaque feature.
                columns (list[str]): Liste des noms des colonnes à visualiser dans le graphique.
                nb_cols (int, optional): Nombre de graphiques (boxplots) par ligne. Par défaut : 2.
                w_graph (int, optional): Largeur de chaque graphique. Par défaut : 5.
                h_graph (int, optional): Hauteur de chaque graphique. Par défaut : 5.
                figsize (tuple, optional): Taille de la figure (largeur, hauteur). Si spécifiée, remplace `w_graph` et `h_graph`. Par défaut : None.

            Raises:
                ValueError: Si le DataFrame `z_scores` est vide ou si les colonnes spécifiées ne sont pas présentes dans le DataFrame.

            Exemple:
                >>> import pandas as pd
                >>> import numpy as np
                >>> z_scores = pd.DataFrame({
                >>>     'feature1': [0.5, -0.8, 1.2, -0.3, 2.0],
                >>>     'feature2': [1.5, -1.3, 0.8, 0.2, -0.7]
                >>> })
                >>> plot_zscore(z_scores, columns=['feature1', 'feature2'])

            Notes:
                - Les Z-scores sont affichés dans un histogramme pour chaque feature.
                - Si le nombre de graphiques est impair, les sous-graphiques inutilisés seront masqués.
                - Chaque histogramme représente la distribution des Z-scores pour la feature correspondante.

        """

        num_features = len(columns)
        num_rows = math.ceil((num_features / nb_cols))

        if figsize is None:
            figsize = (w_graph * nb_cols, h_graph * num_rows)

        fig, axes = plt.subplots(num_rows, nb_cols, figsize=figsize, sharey=False)
        fig.suptitle(f'Représentation de la distribution des zscores', fontsize=16, y=1.0)

        if num_rows == 1:
            axes = np.array([axes])

        axes = axes.flatten()

        for i, c in enumerate(columns):

            axes[i].hist(z_scores[c], bins=15, alpha=0.5, color='g')

            axes[i].set_title(f"zscore pour {c}", fontsize=12)
            axes[i].set_xlabel('Z-score', fontsize=10)
            axes[i].set_ylabel('Fréquence', fontsize=10)

        # Masquer les sous-graphes inutilisés si le nombre de features est impair
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        plt.show()
