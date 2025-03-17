# Testing Database Migrations

This document explains how to test the database migration system both locally and in CI/CD environments.

## Local Testing

To test the database migrations locally, follow these steps:

1. **Set up a test database**

   You can create a test database using Docker:

   ```bash
   # Start a PostgreSQL container
   docker run --name test-postgres -e POSTGRES_PASSWORD=testing123 -e POSTGRES_USER=db -e POSTGRES_DB=frontend -p 5432:5432 -d postgres:16
   ```

2. **Wait for the container to be ready**

   ```bash
   sleep 5
   ```

3. **Set environment variables**

   ```bash
   # Export the database URL
   export DATABASE_URL=postgresql://db:testing123@localhost:5432/frontend
   ```

4. **Initialize the database schema**

   ```bash
   # Connect to the database and create tables
   psql $DATABASE_URL -c "
     CREATE TABLE IF NOT EXISTS users (
         id SERIAL PRIMARY KEY,
         username VARCHAR(255) UNIQUE NOT NULL,
         email VARCHAR(255) NOT NULL,
         password_hash TEXT NOT NULL,
         news_categories TEXT,
         created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
         updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
     );
     
     CREATE TABLE IF NOT EXISTS api_usage (
         id SERIAL PRIMARY KEY,
         api_name VARCHAR(50) NOT NULL,
         endpoint VARCHAR(255) NOT NULL,
         timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
         details JSONB
     );
   "
   ```

5. **Run the migration tests**

   ```bash
   # Run the test suite
   python -m tests.test_migrations
   
   # Test the migration script directly
   python scripts/apply_migrations.py
   ```

## CI/CD Testing

The CI/CD workflow has been updated to include tests for the database migration system:

1. The `db-migration-tests` job in GitHub Actions sets up a PostgreSQL container
2. It creates a basic database schema with tables required for the tests
3. It runs the `tests.test_migrations` test suite
4. It tests the migration script directly by running `scripts/apply_migrations.py`

### Tests Performed

The migration test suite verifies:

1. **Migration Table Creation**: Tests that the migrations tracking table is created correctly
2. **Migration Tracking**: Tests that migrations are properly tracked and not reapplied
3. **Admin User Creation**: Tests that the admin user is created correctly with the proper password hash
4. **Column Addition**: Tests that required columns are added to the users table if missing

## Adding New Migration Tests

When adding new migrations, follow these steps:

1. Add a new test method to `tests/test_migrations.py`
2. Update the migration function in `scripts/apply_migrations.py`
3. Add the new function to the `main()` function in `scripts/apply_migrations.py`
4. Include the test in your PR to ensure it passes CI

Example test method:

```python
def test_05_new_feature_migration(self):
    """Test a new feature migration."""
    # Ensure the migrations table exists
    ensure_migrations_table()
    
    # Reset the migration tracking
    with self.conn.cursor() as cur:
        cur.execute("DELETE FROM migrations WHERE migration_name = 'add_new_feature';")
        self.conn.commit()
    
    # Run the migration
    add_new_feature()
    
    # Check that the migration was applied
    self.assertTrue(has_migration_been_applied('add_new_feature'))
    
    # Verify the changes made by the migration
    # For example, check if a new table or column was created
```

## Troubleshooting Migration Tests

If the migration tests fail in CI, you can:

1. Check the CI logs for detailed error information
2. Run the tests locally to debug the issue
3. Verify that the migration function is idempotent
4. Check for correct error handling in the migration function

### Common Issues:

- **Migration not marked as applied**: Migration completed but failed to add an entry to the migrations table
- **Missing dependencies**: Migration relies on tables or columns that don't exist yet
- **SQL syntax errors**: Issues with SQL statements in the migration function
- **Connection issues**: Problems connecting to the test database 