-- Create users table
DROP TABLE IF EXISTS users CASCADE;
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(255) NOT NULL UNIQUE,
    email VARCHAR(255) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(50) DEFAULT 'user',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    theme VARCHAR(50) DEFAULT 'dark',
    layout VARCHAR(50) DEFAULT 'default',
    button_width INTEGER DEFAULT 220,
    button_height INTEGER DEFAULT 60,
    city_name VARCHAR(255) DEFAULT 'New York',
    news_categories JSONB DEFAULT '{"categories": ["business", "technology"]}'::jsonb,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create index on username for faster lookups
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);

-- Create index on email for faster lookups 