"""
E-Commerce Product Retrieval System
Flask API for managing products and performing similarity search.
"""

from flask import Flask

from services.manager_service import load_config, HOST, PORT
from utils.validation import init_validation_config
from routes.product_routes import product_bp
from routes.search_routes import search_bp
from routes.system_routes import system_bp


def create_app() -> Flask:
    """Create and configure the Flask application."""
    # Load configuration
    load_config()

    # Import after config is loaded to get updated values
    from services.manager_service import HOST, PORT, MAX_TOP_K, DEFAULT_TOP_K

    # Initialize validation config
    init_validation_config(MAX_TOP_K, DEFAULT_TOP_K)

    # Create Flask app
    app = Flask(__name__)

    # Register blueprints
    app.register_blueprint(product_bp)
    app.register_blueprint(search_bp)
    app.register_blueprint(system_bp)

    return app


app = create_app()


if __name__ == "__main__":
    from services.manager_service import HOST, PORT

    app.run(debug=True, host=HOST, port=PORT, use_reloader=False)
