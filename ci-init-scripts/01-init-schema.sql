-- Create users table
CREATE TABLE IF NOT EXISTS users (
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
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);

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

-- Create alert_rules table
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

-- Create alert_history table
CREATE TABLE IF NOT EXISTS alert_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    alert_rule_id UUID NOT NULL REFERENCES alert_rules(id) ON DELETE CASCADE,
    triggered_at TIMESTAMP NOT NULL DEFAULT NOW(),
    data JSONB,
    notification_sent BOOLEAN NOT NULL DEFAULT FALSE
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_api_usage_timestamp ON api_usage(timestamp);
CREATE INDEX IF NOT EXISTS idx_api_usage_api_name ON api_usage(api_name);
CREATE INDEX IF NOT EXISTS idx_api_usage_user_id ON api_usage(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_log_timestamp ON audit_log(changed_at);
CREATE INDEX IF NOT EXISTS idx_alert_rules_user_id ON alert_rules(user_id);
CREATE INDEX IF NOT EXISTS idx_alert_history_rule_id ON alert_history(alert_rule_id);

-- Add triggers
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
GRANT ALL ON users TO app_role;
GRANT SELECT ON users TO readonly;
GRANT ALL ON api_usage TO app_role;
GRANT SELECT ON api_usage TO readonly;
GRANT ALL ON alert_rules TO app_role;
GRANT SELECT ON alert_rules TO readonly;
GRANT ALL ON alert_history TO app_role;
GRANT SELECT ON alert_history TO readonly;

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