import matplotlib.pyplot as plt
import numpy as np

class Viz:

    @staticmethod
    def plot_top_values(result_df, nb_cols=3, w_graph=5, h_graph=5, show_y=False):
        """
        Génère un graphique en barres représentant les résultats des top_values().

        Args:
            result_df (pd.DataFrame): Un DataFrame avec un double en-tête ('value', 'count')
                                    généré par la fonction group_value_counts.
        """

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