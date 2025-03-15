-- Create default admin user if it doesn't exist
-- Using PostgreSQL's idempotent INSERT ON CONFLICT to avoid affecting existing users
INSERT INTO users (
    username,
    email,
    password_hash,
    news_categories,
    city_name
)
VALUES (
    'admin',
    'admin@localhost',
    -- This hash is generated for 'admin123' using Werkzeug 3.1.3 (exact version in the container)
    'scrypt:32768:8:1$BymU2FAmwmRqVLMp$3257820b2e6dfcfe5c71a02a05e0f52a4eb51797990de4a6b4a64a661bac21aad0d6c0991e79cc8571c41486a5d77e142484321ece91ec449a21c9ace0f59fdb',
    'general,technology,business',
    'San Francisco'
)
-- Do nothing if the admin user already exists (fully idempotent)
ON CONFLICT (username) DO NOTHING; 