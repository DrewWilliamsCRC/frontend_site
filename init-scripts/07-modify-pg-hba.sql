-- Modify pg_hba.conf to allow trust authentication for netdata user
ALTER SYSTEM SET password_encryption = 'scram-sha-256';
ALTER SYSTEM SET password_encryption = 'md5';

-- NOTE: The manual modification of pg_hba.conf doesn't work in Docker
-- Left for reference but commented out
/*
-- Create a new pg_hba.conf entry for netdata user
CREATE OR REPLACE FUNCTION modify_pg_hba()
RETURNS void AS $$
BEGIN
    -- Add trust authentication for netdata user
    EXECUTE 'ALTER SYSTEM SET pg_hba.conf = $1' USING 
        'local   all             netdata                                  trust' || E'\n' ||
        'host    all             netdata         127.0.0.1/32           trust' || E'\n' ||
        'host    all             netdata         ::1/128                trust' || E'\n' ||
        current_setting('pg_hba.conf');
END;
$$ LANGUAGE plpgsql;

SELECT modify_pg_hba();
DROP FUNCTION modify_pg_hba();
*/ 