# WSGI Entry Point
# ---------------
# This file serves as the WSGI (Web Server Gateway Interface) entry point
# for the Flask application. It is used by production WSGI servers like
# Gunicorn or uWSGI to start and manage the application.

import os
from app import app

if __name__ == "__main__":
    # This block only executes when the script is run directly
    # (not when imported as a module).
    # Typically used during development with the Flask development server.
    port = int(os.environ.get('PORT', '5000'))
    app.run(host='0.0.0.0', port=port)
