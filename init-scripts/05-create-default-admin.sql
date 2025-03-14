-- Create default admin user if it doesn't exist
DO
$do$
BEGIN
    -- Check if admin user exists
    IF NOT EXISTS (
        SELECT FROM users
        WHERE username = 'admin'
    ) THEN
        -- Insert admin user with hashed password
        -- Default password is 'admin123' - should be changed immediately after first login
        -- This hash is generated using Werkzeug's generate_password_hash() function
        INSERT INTO users (
            username,
            email,
            password_hash,
            news_categories
        ) VALUES (
            'admin',
            'admin@localhost',
            'pbkdf2:sha256:600000$vf9Ul3qpDdGQDSxs$f3cd8235cc3b562c523e08b885c66dd6c7c34c0ccd527b4a6e8cc9e10d39f9e9',
            'general,technology,business'
        );
        
        RAISE NOTICE 'Default admin user created';
    END IF;
END
$do$; 