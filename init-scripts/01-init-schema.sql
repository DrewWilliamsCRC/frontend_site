-- Enable necessary extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Set default transaction isolation level
ALTER DATABASE frontend SET default_transaction_isolation TO 'read committed';

-- Set statement timeout (5 minutes)
ALTER DATABASE frontend SET statement_timeout = '300s';

-- Grant necessary permissions
GRANT CONNECT ON DATABASE frontend TO frontend;
GRANT USAGE ON SCHEMA public TO frontend;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO readonly;
GRANT ALL ON ALL TABLES IN SCHEMA public TO frontend;
GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO frontend;

-- Create or update functions
CREATE OR REPLACE FUNCTION audit_trigger_func() RETURNS trigger AS $$
BEGIN
    IF TG_OP = 'DELETE' THEN
        INSERT INTO audit_log (table_name, action, old_data)
        VALUES (TG_TABLE_NAME::text, 'DELETE', row_to_json(OLD));
        RETURN OLD;
    ELSIF TG_OP = 'UPDATE' THEN
        INSERT INTO audit_log (table_name, action, old_data, new_data)
        VALUES (TG_TABLE_NAME::text, 'UPDATE', row_to_json(OLD), row_to_json(NEW));
        RETURN NEW;
    ELSIF TG_OP = 'INSERT' THEN
        INSERT INTO audit_log (table_name, action, new_data)
        VALUES (TG_TABLE_NAME::text, 'INSERT', row_to_json(NEW));
        RETURN NEW;
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create tables if they don't exist
CREATE TABLE IF NOT EXISTS audit_log (
    id SERIAL PRIMARY KEY,
    table_name text NOT NULL,
    action text NOT NULL,
    old_data jsonb,
    new_data jsonb,
    changed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Handle users table migration safely
DO $$
BEGIN
    -- Check if users table exists
    IF NOT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'users') THEN
        -- Create new users table with UUID
        CREATE TABLE users (
            id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            username VARCHAR(255) NOT NULL UNIQUE,
            email VARCHAR(255) NOT NULL UNIQUE,
            password_hash VARCHAR(255) NOT NULL,
            role VARCHAR(50) DEFAULT 'user',
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            theme VARCHAR(50) DEFAULT 'dark',
            layout VARCHAR(50) DEFAULT 'default',
            button_width INTEGER DEFAULT 220,
            button_height INTEGER DEFAULT 60,
            city_name VARCHAR(255) DEFAULT 'New York',
            news_categories JSONB DEFAULT '{"categories": ["business", "technology"]}'::jsonb
        );
    ELSE
        -- Add any missing columns first
        ALTER TABLE users ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP;
        ALTER TABLE users ADD COLUMN IF NOT EXISTS theme VARCHAR(50) DEFAULT 'dark';
        ALTER TABLE users ADD COLUMN IF NOT EXISTS layout VARCHAR(50) DEFAULT 'default';
        ALTER TABLE users ADD COLUMN IF NOT EXISTS button_width INTEGER DEFAULT 220;
        ALTER TABLE users ADD COLUMN IF NOT EXISTS button_height INTEGER DEFAULT 60;
        ALTER TABLE users ADD COLUMN IF NOT EXISTS city_name VARCHAR(255) DEFAULT 'New York';
        ALTER TABLE users ADD COLUMN IF NOT EXISTS news_categories JSONB DEFAULT '{"categories": ["business", "technology"]}'::jsonb;

        -- Check if id column is UUID
        IF NOT EXISTS (
            SELECT FROM information_schema.columns 
            WHERE table_name = 'users' 
            AND column_name = 'id' 
            AND data_type = 'uuid'
        ) THEN
            -- First, drop foreign key constraints
            ALTER TABLE IF EXISTS api_usage DROP CONSTRAINT IF EXISTS api_usage_user_id_fkey;
            ALTER TABLE IF EXISTS custom_services DROP CONSTRAINT IF EXISTS custom_services_user_id_fkey;
            
            -- Add UUID column if it doesn't exist
            ALTER TABLE users ADD COLUMN IF NOT EXISTS new_id UUID DEFAULT uuid_generate_v4();
            
            -- Update existing rows with new UUIDs
            UPDATE users SET new_id = uuid_generate_v4() WHERE new_id IS NULL;
            
            -- Make new_id NOT NULL
            ALTER TABLE users ALTER COLUMN new_id SET NOT NULL;
            
            -- Drop old primary key if it exists
            ALTER TABLE users DROP CONSTRAINT IF EXISTS users_pkey;
            
            -- Add new primary key
            ALTER TABLE users ADD PRIMARY KEY (new_id);

            -- Convert api_usage user_id to UUID
            ALTER TABLE api_usage ADD COLUMN IF NOT EXISTS new_user_id UUID;
            UPDATE api_usage au 
            SET new_user_id = u.new_id 
            FROM users u 
            WHERE au.user_id = u.id;
            ALTER TABLE api_usage DROP COLUMN user_id;
            ALTER TABLE api_usage RENAME COLUMN new_user_id TO user_id;
            ALTER TABLE api_usage ALTER COLUMN user_id SET NOT NULL;

            -- Convert custom_services user_id to UUID
            ALTER TABLE custom_services ADD COLUMN IF NOT EXISTS new_user_id UUID;
            UPDATE custom_services cs 
            SET new_user_id = u.new_id 
            FROM users u 
            WHERE cs.user_id = u.id;
            ALTER TABLE custom_services DROP COLUMN user_id;
            ALTER TABLE custom_services RENAME COLUMN new_user_id TO user_id;
            ALTER TABLE custom_services ALTER COLUMN user_id SET NOT NULL;
            
            -- Rename columns in users table
            ALTER TABLE users RENAME COLUMN id TO old_id;
            ALTER TABLE users RENAME COLUMN new_id TO id;
            
            -- Drop old_id column
            ALTER TABLE users DROP COLUMN old_id;
            
            -- Recreate foreign key constraints
            ALTER TABLE api_usage ADD CONSTRAINT api_usage_user_id_fkey 
                FOREIGN KEY (user_id) REFERENCES users(id);
            ALTER TABLE custom_services ADD CONSTRAINT custom_services_user_id_fkey 
                FOREIGN KEY (user_id) REFERENCES users(id);
        END IF;
        
        -- Handle password to password_hash migration
        IF EXISTS (
            SELECT FROM information_schema.columns 
            WHERE table_name = 'users' 
            AND column_name = 'password'
        ) THEN
            -- Add password_hash column if it doesn't exist
            ALTER TABLE users ADD COLUMN IF NOT EXISTS password_hash VARCHAR(255);
            
            -- Copy data from password to password_hash
            UPDATE users SET password_hash = password WHERE password_hash IS NULL;
            
            -- Make password_hash NOT NULL
            ALTER TABLE users ALTER COLUMN password_hash SET NOT NULL;
            
            -- Drop old password column
            ALTER TABLE users DROP COLUMN password;
        ELSE
            -- Add password_hash column if it doesn't exist
            ALTER TABLE users ADD COLUMN IF NOT EXISTS password_hash VARCHAR(255) NOT NULL;
        END IF;
    END IF;
END $$;

-- Create other tables if they don't exist
CREATE TABLE IF NOT EXISTS api_usage (
    id SERIAL PRIMARY KEY,
    api_name VARCHAR(50) NOT NULL,
    endpoint VARCHAR(100) NOT NULL,
    request_params JSONB,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    user_id UUID REFERENCES users(id),
    response_status INTEGER,
    response_time DOUBLE PRECISION
);

CREATE TABLE IF NOT EXISTS alert_rules (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    rule_type VARCHAR(50) NOT NULL,
    rule_params JSONB NOT NULL,
    enabled BOOLEAN NOT NULL DEFAULT TRUE,
    last_triggered TIMESTAMP,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Add foreign key constraint to alert_rules after table creation
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints 
        WHERE constraint_name = 'alert_rules_user_id_fkey'
    ) THEN
        ALTER TABLE alert_rules ADD CONSTRAINT alert_rules_user_id_fkey 
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE;
    END IF;
END $$;

CREATE TABLE IF NOT EXISTS alert_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    alert_rule_id UUID NOT NULL REFERENCES alert_rules(id) ON DELETE CASCADE,
    triggered_at TIMESTAMP NOT NULL DEFAULT NOW(),
    data JSONB,
    notification_sent BOOLEAN NOT NULL DEFAULT FALSE
);

-- Create indexes if they don't exist
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_api_usage_timestamp ON api_usage(timestamp);
CREATE INDEX IF NOT EXISTS idx_api_usage_api_name ON api_usage(api_name);
CREATE INDEX IF NOT EXISTS idx_api_usage_user_id ON api_usage(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_log_timestamp ON audit_log(changed_at);
CREATE INDEX IF NOT EXISTS idx_alert_rules_user_id ON alert_rules(user_id);
CREATE INDEX IF NOT EXISTS idx_alert_history_rule_id ON alert_history(alert_rule_id);

-- Create or replace triggers
DROP TRIGGER IF EXISTS update_users_updated_at ON users;
CREATE TRIGGER update_users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS audit_users_trigger ON users;
CREATE TRIGGER audit_users_trigger
    AFTER INSERT OR UPDATE OR DELETE ON users
    FOR EACH ROW
    EXECUTE FUNCTION audit_trigger_func();

DROP TRIGGER IF EXISTS audit_api_usage_trigger ON api_usage;
CREATE TRIGGER audit_api_usage_trigger
    AFTER INSERT OR UPDATE OR DELETE ON api_usage
    FOR EACH ROW
    EXECUTE FUNCTION audit_trigger_func();

DROP TRIGGER IF EXISTS update_alert_rules_updated_at ON alert_rules;
CREATE TRIGGER update_alert_rules_updated_at
    BEFORE UPDATE ON alert_rules
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS audit_alert_rules_trigger ON alert_rules;
CREATE TRIGGER audit_alert_rules_trigger
    AFTER INSERT OR UPDATE OR DELETE ON alert_rules
    FOR EACH ROW
    EXECUTE FUNCTION audit_trigger_func();

DROP TRIGGER IF EXISTS audit_alert_history_trigger ON alert_history;
CREATE TRIGGER audit_alert_history_trigger
    AFTER INSERT OR UPDATE OR DELETE ON alert_history
    FOR EACH ROW
    EXECUTE FUNCTION audit_trigger_func();

-- Grant permissions
GRANT ALL ON users TO frontend;
GRANT SELECT ON users TO readonly;
GRANT ALL ON api_usage TO frontend;
GRANT SELECT ON api_usage TO readonly;
GRANT ALL ON alert_rules TO frontend;
GRANT SELECT ON alert_rules TO readonly;
GRANT ALL ON alert_history TO frontend;
GRANT SELECT ON alert_history TO readonly;

-- Create or replace maintenance function
CREATE OR REPLACE FUNCTION cleanup_old_records() RETURNS void AS $$
BEGIN
    -- Delete api_usage records older than 30 days
    DELETE FROM api_usage WHERE timestamp < NOW() - INTERVAL '30 days';
    -- Delete audit_log records older than 30 days
    DELETE FROM audit_log WHERE changed_at < NOW() - INTERVAL '30 days';
END;
$$ LANGUAGE plpgsql;

-- Set up autovacuum
ALTER TABLE api_usage SET (
    autovacuum_vacuum_scale_factor = 0.1,
    autovacuum_analyze_scale_factor = 0.05
);

ALTER TABLE audit_log SET (
    autovacuum_vacuum_scale_factor = 0.1,
    autovacuum_analyze_scale_factor = 0.05
);

-- Grant execute permission on functions
GRANT EXECUTE ON FUNCTION cleanup_old_records() TO frontend;
GRANT EXECUTE ON FUNCTION update_updated_at_column() TO frontend;
GRANT EXECUTE ON FUNCTION audit_trigger_func() TO frontend; 