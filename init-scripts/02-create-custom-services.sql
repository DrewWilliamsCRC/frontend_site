-- Create custom services table
CREATE TABLE IF NOT EXISTS custom_services (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    url VARCHAR(255) NOT NULL,
    icon_url VARCHAR(255),
    category VARCHAR(100),
    section VARCHAR(100),
    display_order INTEGER,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create index for category
CREATE INDEX IF NOT EXISTS idx_custom_services_category ON custom_services(category);

-- Create index for user_id
CREATE INDEX IF NOT EXISTS idx_custom_services_user_id ON custom_services(user_id);

-- Create index for section and display_order
CREATE INDEX IF NOT EXISTS idx_custom_services_section_order ON custom_services(section, display_order);

-- Add audit trigger
CREATE TRIGGER audit_custom_services_trigger
    AFTER INSERT OR UPDATE OR DELETE ON custom_services
    FOR EACH ROW
    EXECUTE FUNCTION audit_trigger_func();

-- Add updated_at trigger
CREATE TRIGGER update_custom_services_updated_at
    BEFORE UPDATE ON custom_services
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Grant permissions
GRANT ALL ON custom_services TO frontend;
GRANT SELECT ON custom_services TO readonly; 