from flask import Blueprint

from bdf.ui.components.text import TextBlock, TextInline
from bdf.ui.components.container import Container, Html, Head, Body

home = Blueprint('home', __name__)

@home.route('/')
def home_page():

    attributes =  {
        'class' : 'container'
    }
    elements = [
        Head(),
        Body(children=[
            TextBlock(content='Ici le contenu...', style='color:red;'),
            TextInline(content='Ici le contenu 2...', style='color:green;')
        ], **attributes)
    ]

    div = Html(children=elements)

    return div.render()