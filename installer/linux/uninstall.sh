#!/bin/bash
#
# NeurDB Uninstaller for Native Linux Installation
#

set -e

# Must run as root
if [ "$(id -u)" -ne 0 ]; then
    echo "Error: This script must be run as root (use sudo)."
    exit 1
fi

echo "NeurDB Uninstaller"
echo "=================="

# 1. Stop and disable systemd services
echo "Stopping NeurDB services..."
systemctl stop neurdb-ai 2>/dev/null || true
systemctl stop neurdb-db 2>/dev/null || true
systemctl disable neurdb-ai 2>/dev/null || true
systemctl disable neurdb-db 2>/dev/null || true

# 2. Remove service unit files
echo "Removing systemd units..."
rm -f /etc/systemd/system/neurdb-db.service
rm -f /etc/systemd/system/neurdb-ai.service

# 3. Reload systemd
systemctl daemon-reload

# 4. Remove binary installation
echo "Removing /opt/neurdb/..."
rm -rf /opt/neurdb

# 5. Remove PATH integration
rm -f /etc/profile.d/neurdb.sh
rm -f /usr/local/bin/neurdb-psql

# 6. Optionally remove data
if [ -d "/var/lib/neurdb" ]; then
    echo ""
    echo "WARNING: /var/lib/neurdb contains your database data."
    read -rp "Delete database data? This CANNOT be undone. [y/N] " CONFIRM
    if [[ "$CONFIRM" =~ ^[Yy]$ ]]; then
        rm -rf /var/lib/neurdb
        echo "Database data removed."
    else
        echo "Database data preserved at /var/lib/neurdb"
    fi
fi

# Remove log directory
rm -rf /var/log/neurdb

# 7. Remove the neurdb system user
if id -u neurdb &>/dev/null; then
    echo "Removing neurdb system user..."
    userdel neurdb 2>/dev/null || true
fi

echo ""
echo "NeurDB has been uninstalled."
