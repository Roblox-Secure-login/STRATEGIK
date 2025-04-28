from flask import Flask
from database import db
import models
from app import app

# Initialize the Flask app with the database
db.init_app(app)

# Create database tables if they don't exist
with app.app_context():
    db.create_all()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
