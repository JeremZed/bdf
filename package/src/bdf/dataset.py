from bdf.tools import Tools

class Dataset:
    """
        Gestion du dataset
    """

    def init(self, data, options={}):
        self.options = options

        if 'verbose' not in self.options:
            self.options['verbose'] = 0

        # raz des attributs et chargement des données
        self.reset().load_data(data)

    def reset(self):
        """ remise à zéro des attributs """
        Tools.log("reseting ... ", self.verbose)

        self.df = None

        Tools.log("reset done.", self.verbose)

        return self

    def load_data(self, data, **kwarg):
        """
        permet de charger les données dynamiquement
        gestion du type de la variable pour creer
        au final une instance de DataFrame
        """

        Tools.log("loading data...", self.verbose)

        # si type string alors
            # check de l'existence du fichier
                # si fichier existe, check de l'extension
                # pour chargement via pandas
                # csv, json

        # si type dataframe alors simple set
            # Tools.log("dataframe affecté avec succès. ")

        # si type numpy array alors création d'un df
            # Tools.log("dataframe créé avec succès à partir d'un numpy array")

        # si type list alors création d'un df

        # si type dict alors création d'un df

        # si aucun cas alors exception
            # raise Exception("ce type de données n'est pas pris en charge. ")

        return self