from flask import Flask
import os
from app.routes import main

UPLOAD_FOLDER = "app/static/uploads"

def create_app():
    app = Flask(__name__, template_folder="app/templates", static_folder="app/static")
    app.secret_key = "your_secret_key"  # Required for session management
    app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

    app.register_blueprint(main)
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure the upload folder exists
    return app

if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)