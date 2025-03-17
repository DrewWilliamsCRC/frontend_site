-- Drop triggers if they exist
DROP TRIGGER IF EXISTS update_alert_rules_updated_at ON alert_rules;
DROP TRIGGER IF EXISTS audit_alert_rules_trigger ON alert_rules;
DROP TRIGGER IF EXISTS audit_alert_history_trigger ON alert_history;

-- Create alert_rules table if it doesn't exist
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

-- Create index on user_id for faster queries
CREATE INDEX IF NOT EXISTS alert_rules_user_id_idx ON alert_rules(user_id);

-- Create alert_history table to store triggered alerts
CREATE TABLE IF NOT EXISTS alert_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    alert_rule_id UUID NOT NULL REFERENCES alert_rules(id) ON DELETE CASCADE,
    triggered_at TIMESTAMP NOT NULL DEFAULT NOW(),
    data JSONB,
    notification_sent BOOLEAN NOT NULL DEFAULT FALSE
);

-- Create index on alert_rule_id
CREATE INDEX IF NOT EXISTS alert_history_rule_id_idx ON alert_history(alert_rule_id);

-- Create trigger for updating the updated_at column
CREATE TRIGGER update_alert_rules_updated_at
    BEFORE UPDATE ON alert_rules
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Add audit triggers
CREATE TRIGGER audit_alert_rules_trigger
    AFTER INSERT OR UPDATE OR DELETE ON alert_rules
    FOR EACH ROW
    EXECUTE FUNCTION audit_trigger_func();

CREATE TRIGGER audit_alert_history_trigger
    AFTER INSERT OR UPDATE OR DELETE ON alert_history
    FOR EACH ROW
    EXECUTE FUNCTION audit_trigger_func();

-- Grant permissions
GRANT ALL ON alert_rules TO frontend;
GRANT SELECT ON alert_rules TO readonly;
GRANT ALL ON alert_history TO frontend;
GRANT SELECT ON alert_history TO readonly; 