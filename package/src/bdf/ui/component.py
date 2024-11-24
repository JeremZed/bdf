class Component():
    def __init__(self, **kwargs):

        self.style = kwargs.get('style', '')
        self.dom = None

    def get_style(self):
        """ Permet de retourner la chaine de caractère de l'attribut style en html """
        return '' if self.style is None else f'style="{self.style}"'