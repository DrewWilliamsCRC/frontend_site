-- Create netdata user if it doesn't exist
DO
$do$
BEGIN
   IF NOT EXISTS (
      SELECT FROM pg_catalog.pg_roles
      WHERE  rolname = 'netdata') THEN
      CREATE ROLE netdata WITH LOGIN PASSWORD 'netdata';
   END IF;
END
$do$;

-- Create frontend database if it doesn't exist
SELECT 'CREATE DATABASE frontend'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'frontend')\gexec

-- Connect to frontend database to grant permissions
\c frontend

-- Grant necessary permissions
GRANT CONNECT ON DATABASE frontend TO netdata;
GRANT USAGE ON SCHEMA public TO netdata;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO netdata;
GRANT SELECT ON ALL SEQUENCES IN SCHEMA public TO netdata;

-- Set default privileges for future tables
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO netdata;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON SEQUENCES TO netdata; 