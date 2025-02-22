# Database Management Script
# ------------------------
# This script serves as the entry point for database management commands.
# It provides a command-line interface for database operations such as:
# - Database initialization
# - Schema migrations
# - Data seeding
# - Database cleanup and maintenance

from src.manage_db import cli

if __name__ == '__main__':
    # Execute the command-line interface when run directly.
    # The cli() function contains the command definitions and handlers
    # for various database management tasks.
    cli()
