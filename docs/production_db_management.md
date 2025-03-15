# Production Database Management

This document provides guidelines for managing the PostgreSQL database in production environments.

## Key Differences Between Development and Production

In production:
- The database volume is **persistent** between container restarts
- Initialization scripts only run on the first startup (when the volume is created)
- Admin user credentials should be changed from the default after deployment

## Database Initialization

The database is initialized through several mechanisms:

1. **PostgreSQL Init Scripts**: Files in `init-scripts/` are executed alphabetically only when the database is first created.
   - These scripts create the initial schema and default admin user.
   - They are designed to be idempotent (can be run multiple times without side effects).

2. **Migrations System**: The `scripts/apply_migrations.py` tracks database changes in a `migrations` table.
   - The migration script runs during deployment.
   - It only applies changes that haven't been applied before.
   - It ensures required database elements exist without overwriting data.

## Admin User Management

The default admin user (`admin` / `admin123`) is created during initial database setup. For security:

1. **Change the admin password** immediately after the first deployment:
   ```bash
   docker compose -f docker-compose.prod.yml exec frontend flask change-admin-password
   ```

2. **Environment variables** in `.env.prod` allow setting a custom admin email:
   - `ADMIN_EMAIL`: Default email for the admin user (only used on first initialization)
   - `ADMIN_PASSWORD`: Default password (only used on first initialization)

## Database Backups

Regular database backups are recommended:

```bash
# Create a backup
docker compose -f docker-compose.prod.yml exec db pg_dump -U frontend frontend > backup_$(date +%Y%m%d).sql

# Restore from backup (after stopping the containers)
cat backup_20250315.sql | docker compose -f docker-compose.prod.yml exec -T db psql -U frontend frontend
```

## Adding New Migrations

When making schema changes:

1. Create a function in `scripts/apply_migrations.py` for the change
2. Add a call to the function in the `main()` function
3. Make sure to check if the migration has already been applied
4. Use idempotent SQL operations (CREATE IF NOT EXISTS, etc.)

Example migration function:

```python
def add_new_feature_column():
    """Add a new column for feature X."""
    if has_migration_been_applied('add_new_feature_column'):
        print("New feature column migration already applied, skipping")
        return
    
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # Check if column exists
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.columns 
                    WHERE table_name = 'users' AND column_name = 'new_feature_enabled'
                );
            """)
            
            if not cur.fetchone()['exists']:
                cur.execute("ALTER TABLE users ADD COLUMN new_feature_enabled BOOLEAN DEFAULT FALSE;")
                conn.commit()
                print("Added new_feature_enabled column to users table")
        
        # Mark migration as applied
        mark_migration_applied('add_new_feature_column')
    except Exception as e:
        conn.rollback()
        print(f"Error adding new feature column: {e}")
    finally:
        conn.close()
```

## Troubleshooting

### Database Connection Issues

If the application can't connect to the database:

1. Verify PostgreSQL is running:
   ```bash
   docker compose -f docker-compose.prod.yml ps
   ```

2. Check database logs:
   ```bash
   docker compose -f docker-compose.prod.yml logs db
   ```

3. Ensure the database credentials in `.env` match those in the container:
   ```bash
   docker compose -f docker-compose.prod.yml exec db psql -U postgres -c "SELECT usename FROM pg_user;"
   ```

### Schema Issues

If the application reports missing tables or columns:

1. Run the migrations script manually:
   ```bash
   docker compose -f docker-compose.prod.yml exec frontend python /app/scripts/apply_migrations.py
   ```

2. Check the migration status in the database:
   ```bash
   docker compose -f docker-compose.prod.yml exec db psql -U frontend frontend -c "SELECT * FROM migrations ORDER BY applied_at DESC;"
   ``` 