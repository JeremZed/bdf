
from bdf.ui.component import Component

class Text(Component):
    """ Classe représentant le composant Text UI """

    def __init__(self, content, **kwargs):

        super().__init__(**kwargs)

        self.content = content


    def build(self, refresh=False):
        """
            Permet de retourner le DOM
        """

        if self.dom is None or refresh == True:
            self.dom = f'<p {self.get_style()} >{self.content}</p>'


