import os
from flask_sqlalchemy import SQLAlchemy

# Create a shared database instance to be used throughout the application
db = SQLAlchemy()