from app import app, db

# Create database tables if they don't exist
with app.app_context():
    import models  # Import models after app is created to avoid circular imports
    db.create_all()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
