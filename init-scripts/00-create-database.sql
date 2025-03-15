-- Create necessary monitoring user
DO
$do$
BEGIN
   IF NOT EXISTS (
      SELECT FROM pg_catalog.pg_roles
      WHERE rolname = 'netdata') THEN
      CREATE ROLE netdata WITH LOGIN PASSWORD 'netdata';
   END IF;
END
$do$;

-- Create readonly role if it doesn't exist
DO
$do$
BEGIN
   IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'readonly') THEN
      CREATE ROLE readonly;
   END IF;
END
$do$;

-- Create database if it doesn't exist
-- Note: This database should be created by Docker using POSTGRES_DB environment variable
-- This script assumes the database already exists and is being run within that database context

-- Grant necessary permissions
GRANT CONNECT ON DATABASE frontend TO netdata;
GRANT USAGE ON SCHEMA public TO netdata;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO netdata;
GRANT SELECT ON ALL SEQUENCES IN SCHEMA public TO netdata;

-- Set default privileges for future tables
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO netdata;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON SEQUENCES TO netdata; 