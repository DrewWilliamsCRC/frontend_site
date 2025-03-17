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
    -- This hash is generated for 'admin123' using bcrypt
    '$2a$06$h4kpIbb60PIxm5jfYknt/ug5wLLplLYWv95HljZJVnJtTbwvzkrKi',
    'general,technology,business',
    'San Francisco'
)
-- Update password_hash if it's NULL
ON CONFLICT (username) DO UPDATE 
SET password_hash = EXCLUDED.password_hash
WHERE users.password_hash IS NULL; 