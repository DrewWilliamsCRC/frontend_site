-- Enable necessary extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Set default transaction isolation level
ALTER DATABASE frontend SET default_transaction_isolation TO 'read committed';

-- Set statement timeout (5 minutes)
ALTER DATABASE frontend SET statement_timeout = '300s';

-- Create roles
CREATE ROLE readonly;
CREATE ROLE app_role WITH LOGIN PASSWORD 'app_password';

-- Grant necessary permissions
GRANT CONNECT ON DATABASE frontend TO app_role;
GRANT USAGE ON SCHEMA public TO app_role;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO readonly;
GRANT ALL ON ALL TABLES IN SCHEMA public TO app_role;
GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO app_role;

-- Create audit function
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

-- Create audit log table
CREATE TABLE IF NOT EXISTS audit_log (
    id SERIAL PRIMARY KEY,
    table_name text NOT NULL,
    action text NOT NULL,
    old_data jsonb,
    new_data jsonb,
    changed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create users table
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(255) NOT NULL UNIQUE,
    email VARCHAR(255) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    news_categories VARCHAR(255) DEFAULT 'general',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create api_usage table
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

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_api_usage_timestamp ON api_usage(timestamp);
CREATE INDEX IF NOT EXISTS idx_api_usage_api_name ON api_usage(api_name);
CREATE INDEX IF NOT EXISTS idx_api_usage_user_id ON api_usage(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_log_timestamp ON audit_log(changed_at);

-- Create updated_at trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Add triggers
CREATE TRIGGER update_users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER audit_users_trigger
    AFTER INSERT OR UPDATE OR DELETE ON users
    FOR EACH ROW
    EXECUTE FUNCTION audit_trigger_func();

CREATE TRIGGER audit_api_usage_trigger
    AFTER INSERT OR UPDATE OR DELETE ON api_usage
    FOR EACH ROW
    EXECUTE FUNCTION audit_trigger_func();

-- Create maintenance function
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
GRANT EXECUTE ON FUNCTION cleanup_old_records() TO app_role;
GRANT EXECUTE ON FUNCTION update_updated_at_column() TO app_role;
GRANT EXECUTE ON FUNCTION audit_trigger_func() TO app_role;
