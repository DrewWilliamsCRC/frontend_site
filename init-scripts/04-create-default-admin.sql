-- Create default admin user if it doesn't exist
-- Using PostgreSQL's idempotent INSERT ON CONFLICT to avoid affecting existing users
INSERT INTO users (
    username,
    email,
    password_hash,
    news_categories,
    role
) VALUES (
    'admin',
    (SELECT var_value FROM init_vars WHERE var_name = 'admin_email'),
    (SELECT var_value FROM init_vars WHERE var_name = 'admin_password'),
    '{"categories": ["general", "technology", "business"]}'::jsonb,
    'admin'
)
WHERE NOT EXISTS (
    SELECT 1 FROM users WHERE username = 'admin'
); 