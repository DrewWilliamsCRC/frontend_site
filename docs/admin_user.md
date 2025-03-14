# Admin User Management

This document describes how admin users are created and managed in the application.

## Default Admin User

A default admin user is automatically created during the initialization of the application:

- Username: `admin`
- Default Password: `admin123`
- Email: `admin@localhost`

**IMPORTANT**: For security reasons, you should change the default admin password immediately after the first login.

## How Admin User Creation Works

The admin user is created in three different ways to ensure it always exists:

1. **Database Initialization Script**: The `05-create-default-admin.sql` script creates the admin user during database initialization if it doesn't exist.

2. **Docker Entrypoint Script**: The `docker-entrypoint.sh` script includes code to create the admin user during container startup if it doesn't exist.

3. **Deployment Script**: The `scripts/deploy_prod.sh` script runs a Flask CLI command to ensure the admin user exists after deployment.

## Changing Admin Password

You can change the admin password in these ways:

### 1. Using the Web Interface

1. Log in as admin
2. Navigate to user settings
3. Change your password

### 2. Using the CLI Command (Recommended for Production)

This is the safest method as the password won't be stored in command history:

```bash
# For development environment
docker compose -f docker-compose.dev.yml exec frontend flask change-admin-password

# For production environment
docker compose -f docker-compose.prod.yml exec frontend flask change-admin-password
```

You'll be prompted to enter and confirm your new password.

### 3. Using the Flask Shell

Connect to the running container and use the Flask shell:

```bash
# For development environment
docker compose -f docker-compose.dev.yml exec frontend flask shell

# For production environment
docker compose -f docker-compose.prod.yml exec frontend flask shell
```

Then in the Python shell:

```python
from werkzeug.security import generate_password_hash
import psycopg2
from app import get_db_connection

# Set new password
new_password = "your-secure-password"
password_hash = generate_password_hash(new_password)

# Update in database
conn = get_db_connection()
with conn.cursor() as cur:
    cur.execute(
        "UPDATE users SET password_hash = %s WHERE username = 'admin'",
        (password_hash,)
    )
    conn.commit()
print("Admin password updated successfully")
conn.close()
```

## Adding Additional Admin Users

You can create additional admin users through the user management interface or using a similar process as described above for password changing.

## Security Considerations

- The default admin user is created with a simple password that should be changed immediately
- In production, consider using environment variables to set the initial admin password
- Implement account lockout policies to prevent brute force attacks
- Regularly audit admin accounts and their activities