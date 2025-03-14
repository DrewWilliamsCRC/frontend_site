-- Add missing columns to the users table
ALTER TABLE users 
ADD COLUMN IF NOT EXISTS city_name TEXT,
ADD COLUMN IF NOT EXISTS button_width INTEGER DEFAULT 200,
ADD COLUMN IF NOT EXISTS button_height INTEGER DEFAULT 200;

-- Log the change
INSERT INTO audit_log (table_name, action, new_data)
VALUES ('users', 'ALTER', '{"columns_added": ["city_name", "button_width", "button_height"]}'::jsonb);

-- Update users that match our admin user
UPDATE users
SET city_name = 'San Francisco'
WHERE username = 'admin' AND city_name IS NULL; 