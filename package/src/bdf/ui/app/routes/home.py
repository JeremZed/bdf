from flask import Blueprint

home = Blueprint('home', __name__)

@home.route('/')
def home_page():
    return "Bienvenue sur la page d'accueil"