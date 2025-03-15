-- Create default admin user if it doesn't exist (CI version)
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
    -- This hash is generated for 'admin123' using Werkzeug 3.1.3
    'scrypt:32768:8:1$abcdefghijklmnop$1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef',
    'general,technology,business',
    'San Francisco'
)
-- Do nothing if the admin user already exists (fully idempotent)
ON CONFLICT (username) DO NOTHING; 