-- Create alert_rules table
CREATE TABLE IF NOT EXISTS alert_rules (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
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
    id SERIAL PRIMARY KEY,
    alert_rule_id INTEGER NOT NULL REFERENCES alert_rules(id) ON DELETE CASCADE,
    message TEXT NOT NULL,
    context JSONB,
    triggered_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Create index on alert_rule_id for faster queries
CREATE INDEX IF NOT EXISTS alert_history_rule_id_idx ON alert_history(alert_rule_id);

-- Create notification_channels table
CREATE TABLE IF NOT EXISTS notification_channels (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    channel_type VARCHAR(50) NOT NULL,
    channel_config JSONB NOT NULL,
    enabled BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Create index on user_id for faster queries
CREATE INDEX IF NOT EXISTS notification_channels_user_id_idx ON notification_channels(user_id);

-- Create user_alert_preferences table to store user preferences for alerts
CREATE TABLE IF NOT EXISTS user_alert_preferences (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    notify_email BOOLEAN NOT NULL DEFAULT TRUE,
    notify_web BOOLEAN NOT NULL DEFAULT TRUE,
    notify_mobile BOOLEAN NOT NULL DEFAULT FALSE,
    cooldown_hours INTEGER NOT NULL DEFAULT 24,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    CONSTRAINT user_alert_preferences_unique_user UNIQUE (user_id)
);

-- Insert default alert preferences for existing users
INSERT INTO user_alert_preferences (user_id, notify_email, notify_web)
SELECT id, TRUE, TRUE FROM users
WHERE id NOT IN (SELECT user_id FROM user_alert_preferences)
ON CONFLICT (user_id) DO NOTHING; 