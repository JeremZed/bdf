from flask import Flask
from bdf.ui.app.routes.home import home
from bdf.ui.app.models import db
import os

class Application:
    """
        Class représentant l'application data Viz'
    """

    def __init__(self, name):
        self.name = name

        self.app = Flask(self.name)
        self.configure_app()
        self.init_db()
        self.handler_routes()

    def configure_app(self):

        basedir = os.path.abspath(os.path.dirname(__file__))
        basedir_instance = os.path.join(basedir, 'instance')

        self.app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir_instance, 'database.db')
        self.app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    def init_db(self):

        db.init_app(self.app)

        # Créer la base de données si elle n'existe pas
        if not os.path.exists(self.app.config['SQLALCHEMY_DATABASE_URI'].replace('sqlite:///', '')):
            with self.app.app_context():
                db.create_all()


    def handler_routes(self):

        self.app.register_blueprint(home)

    def run(self, debug=True):
        self.app.run(debug=debug)
