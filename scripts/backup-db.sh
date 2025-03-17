#!/bin/bash

# Exit on error
set -e

# Load environment variables
source ../.env

# Configuration
BACKUP_DIR="/var/backups/postgres"
BACKUP_RETENTION_DAYS=7
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="${BACKUP_DIR}/frontend_${DATE}.sql.gz"

# Ensure backup directory exists
mkdir -p ${BACKUP_DIR}

# Create backup
echo "Creating backup: ${BACKUP_FILE}"
PGPASSWORD="${POSTGRES_PASSWORD}" pg_dump \
    -h localhost \
    -p 5432 \
    -U "${POSTGRES_USER}" \
    -d frontend \
    --clean \
    --if-exists \
    --no-owner \
    --no-acl \
    --no-privileges \
    --no-tablespaces \
    --no-comments \
    --no-security-labels \
    --no-sync \
    --no-single-transaction \
    --no-locks \
    --no-tables \
    --no-data \
    --schema-only \
    | gzip > "${BACKUP_FILE}"

# Remove old backups
echo "Removing backups older than ${BACKUP_RETENTION_DAYS} days"
find ${BACKUP_DIR} -type f -name "frontend_*.sql.gz" -mtime +${BACKUP_RETENTION_DAYS} -delete

# Verify backup
echo "Verifying backup file..."
if gzip -t "${BACKUP_FILE}"; then
    echo "Backup completed and verified: ${BACKUP_FILE}"
else
    echo "Backup verification failed!"
    exit 1
fi

# Create checksum
md5sum "${BACKUP_FILE}" > "${BACKUP_FILE}.md5"

# List remaining backups
echo -e "\nCurrent backups:"
ls -lh ${BACKUP_DIR} 