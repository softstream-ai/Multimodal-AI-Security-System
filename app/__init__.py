from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
import os

# Initialize SQLAlchemy
db = SQLAlchemy()

def create_app():
    app = Flask(__name__, 
                template_folder=os.path.join(os.path.dirname(__file__), '..', 'templates'),
                static_folder= os.path.join(os.path.dirname(__file__), '..', 'static'))

    # Configuration from the config.py file
    app.config.from_object('app.config.Config')

    # Initialize the database and migration
    db.init_app(app)
    migrate = Migrate(app, db)

    # Import blueprints/routes
    from app.routes import main as main_blueprint
    app.register_blueprint(main_blueprint)

    return app
