-- Insert sample data for testing purposes

-- Check if admin user exists, if not create it
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM users WHERE username = 'admin') THEN
        INSERT INTO users (username, password, email, role, city_name, button_width, button_height, news_categories)
        VALUES (
            'admin', 
            'pbkdf2:sha256:150000$NLRXwvVR$7b9ccd62a9a529102f75c669a224921309afd198b7c7cf7251e3a8047da344a8', 
            'admin@example.com', 
            'admin',
            'New York',
            200,
            200,
            'general,business,technology'
        );
    END IF;
    
    -- Check if Drew user exists, if not create it
    IF NOT EXISTS (SELECT 1 FROM users WHERE username = 'Drew') THEN
        INSERT INTO users (username, password, email, role, city_name, button_width, button_height, news_categories)
        VALUES (
            'Drew', 
            'pbkdf2:sha256:150000$NLRXwvVR$7b9ccd62a9a529102f75c669a224921309afd198b7c7cf7251e3a8047da344a8', 
            'drew@example.com', 
            'admin',
            'Chicago',
            220,
            220,
            'general,business,technology,sports'
        );
    END IF;
    
    -- Add settings records for backward compatibility
    INSERT INTO settings (username, city_name, button_width, button_height, news_categories)
    VALUES 
        ('admin', 'New York', 200, 200, 'general,business,technology')
    ON CONFLICT (username) DO NOTHING;
    
    INSERT INTO settings (username, city_name, button_width, button_height, news_categories)
    VALUES 
        ('Drew', 'Chicago', 220, 220, 'general,business,technology,sports')
    ON CONFLICT (username) DO NOTHING;
END $$; 