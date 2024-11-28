class Component():
    def __init__(self, **kwargs):

        self.attribute_style = kwargs.get('style', None)
        self.attribute_class = kwargs.get('class', None)
        self.content = kwargs.get('content', None)
        self.children = kwargs.get('children', None)
        self.tag = 'div'
        self.dom = None

    def build(self):
        """
            Permet de retourner le DOM
        """
        self.dom = f'''<{self.tag} {self.get_class()} {self.get_style()} >'''

        if self.children is not None:
            for child in self.children:
                child.build()
                self.dom = self.dom + child.render()
        else:
            if self.content is not None:
                self.dom = self.dom + f'{self.content}'

        self.dom = self.dom + f'''</{self.tag}>'''

    def get_style(self):
        """ Permet de retourner la chaine de caractère de l'attribut style en html """
        return '' if self.attribute_style is None else f'style="{self.attribute_style}"'

    def get_class(self):
        """ Permet de retourner la chaine de caractères de l'attribut class en html """
        return '' if self.attribute_class is None else f'class="{self.attribute_class}"'

    def get_dom(self, refresh=False):
        """ Permet d """
        if self.dom is None or refresh:
            self.build()

        return self.dom

    def render(self, refresh=False):
        """ Permet de retourner tout le dom préalablement construit """
        return self.get_dom(refresh)