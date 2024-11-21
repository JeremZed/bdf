import matplotlib.pyplot as plt
import numpy as np

class Viz:

    @staticmethod
    def plot_top_values(result_df, nb_cols=3, w_graph=5, h_graph=5, show_y=False):
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
        num_rows = round((num_features / nb_cols))

        fig, axes = plt.subplots(num_rows, nb_cols, figsize=(w_graph * nb_cols, h_graph * num_rows), sharey=False)
        axes = axes.flatten()

        # Gestion du cas où on visualise uniquement une feature
        if num_features == 1:
            axes = [axes]

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