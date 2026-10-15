from backend.app import app

# Vercel expects the Flask app object to be available at the project root.
# This file re-exports the real application instance from backend/app.py.
# The local app still runs from backend/app.py.

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
