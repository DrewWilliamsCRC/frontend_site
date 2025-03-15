-- This file is not needed anymore as we integrated the missing columns in the main schema

-- NOTE: This file is kept as a reference for future migrations, but the columns
-- are already included in the users table creation in 01-init-schema.sql

/*
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
*/

-- Add missing columns in an idempotent way
-- This script adds columns that might be missing in an existing database

-- Function to check if a column exists and add it if it doesn't
DO $$
BEGIN
    -- Add city_name to users if it doesn't exist
    IF NOT EXISTS (
        SELECT FROM information_schema.columns 
        WHERE table_name = 'users' AND column_name = 'city_name'
    ) THEN
        ALTER TABLE users ADD COLUMN city_name VARCHAR(255);
        RAISE NOTICE 'Added city_name column to users table';
    END IF;

    -- Add button_width to users if it doesn't exist
    IF NOT EXISTS (
        SELECT FROM information_schema.columns 
        WHERE table_name = 'users' AND column_name = 'button_width'
    ) THEN
        ALTER TABLE users ADD COLUMN button_width INTEGER DEFAULT 200;
        RAISE NOTICE 'Added button_width column to users table';
    END IF;
    
    -- Add button_height to users if it doesn't exist
    IF NOT EXISTS (
        SELECT FROM information_schema.columns 
        WHERE table_name = 'users' AND column_name = 'button_height'
    ) THEN
        ALTER TABLE users ADD COLUMN button_height INTEGER DEFAULT 200;
        RAISE NOTICE 'Added button_height column to users table';
    END IF;
END $$; 