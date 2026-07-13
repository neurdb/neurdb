#!/bin/bash
#
# NeurDB Native Linux Installer
# Called from install.sh --native
#
# This script installs NeurDB directly on the host without Docker.
# Requires Ubuntu 20.04+, Debian 11+, or RHEL/Rocky 8+.
#

set -e

# Defaults
VERSION="latest"
PORT=5432
INSTALL_DIR="/opt/neurdb"
DATA_DIR="/var/lib/neurdb/data"
LOG_DIR="/var/log/neurdb"
REGISTRY_URL="https://github.com/neurdb/neurdb/releases"

usage() {
    echo "NeurDB Native Linux Installer"
    echo ""
    echo "Usage: install-native.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --version VERSION   NeurDB version to install (default: latest)"
    echo "  --port PORT         PostgreSQL port (default: 5432)"
    echo "  -h, --help          Show this help message"
    exit 0
}

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --version)  VERSION="$2"; shift ;;
        --port)     PORT="$2"; shift ;;
        -h|--help)  usage ;;
        # Skip flags handled by the parent install.sh
        --native|--cpu|--gpu) ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Must run as root
if [ "$(id -u)" -ne 0 ]; then
    echo "Error: This script must be run as root (use sudo)."
    exit 1
fi

# 1. Check the OS
echo "Checking system requirements..."
ARCH=$(uname -m)
if [ "$ARCH" != "x86_64" ]; then
    echo "Error: Only x86_64 architecture is supported. Detected: $ARCH"
    exit 1
fi

if [ -f /etc/os-release ]; then
    . /etc/os-release
    case "$ID" in
        ubuntu)
            if [ "${VERSION_ID%%.*}" -lt 20 ]; then
                echo "Error: Ubuntu 20.04 or later is required. Detected: $VERSION_ID"
                exit 1
            fi
            PKG_MANAGER="apt"
            ;;
        debian)
            if [ "${VERSION_ID%%.*}" -lt 11 ]; then
                echo "Error: Debian 11 or later is required. Detected: $VERSION_ID"
                exit 1
            fi
            PKG_MANAGER="apt"
            ;;
        rhel|rocky|centos|almalinux)
            if [ "${VERSION_ID%%.*}" -lt 8 ]; then
                echo "Error: RHEL/Rocky 8 or later is required. Detected: $VERSION_ID"
                exit 1
            fi
            PKG_MANAGER="dnf"
            ;;
        *)
            echo "Warning: Unsupported distribution '$ID'. Proceeding anyway..."
            PKG_MANAGER="apt"
            ;;
    esac
else
    echo "Error: Cannot detect OS. /etc/os-release not found."
    exit 1
fi

# 2. Create system user
echo "Creating neurdb system user..."
if ! id -u neurdb &>/dev/null; then
    useradd --system --no-create-home --shell /usr/sbin/nologin neurdb
fi

# 3. Install system runtime dependencies
echo "Installing runtime dependencies..."
if [ "$PKG_MANAGER" = "apt" ]; then
    apt-get update
    apt-get install -y --no-install-recommends \
        librocksdb6.11 libwebsockets16 libcjson1 \
        libreadline8 libicu70 zlib1g libssl3 \
        libopencv-core4.5d \
        python3 python3-venv curl locales
    locale-gen en_US.UTF-8 || true
elif [ "$PKG_MANAGER" = "dnf" ]; then
    dnf install -y \
        rocksdb libwebsockets cjson \
        readline icu zlib openssl-libs \
        opencv-core \
        python3 curl glibc-langpack-en
fi

# 4. Download and verify the tarball
echo "Downloading NeurDB..."
if [ "$VERSION" = "latest" ]; then
    DOWNLOAD_URL="${REGISTRY_URL}/latest/download/neurdb-latest-linux-${ARCH}.tar.gz"
    SUMS_URL="${REGISTRY_URL}/latest/download/SHA256SUMS"
else
    DOWNLOAD_URL="${REGISTRY_URL}/download/v${VERSION}/neurdb-${VERSION}-linux-${ARCH}.tar.gz"
    SUMS_URL="${REGISTRY_URL}/download/v${VERSION}/SHA256SUMS"
fi

TMPDIR=$(mktemp -d)
curl -fSL "$DOWNLOAD_URL" -o "$TMPDIR/neurdb.tar.gz"
curl -fSL "$SUMS_URL" -o "$TMPDIR/SHA256SUMS"

echo "Verifying download integrity..."
cd "$TMPDIR"
sha256sum -c SHA256SUMS
cd /

# 5. Extract to /opt/neurdb
echo "Installing to ${INSTALL_DIR}..."
rm -rf "$INSTALL_DIR"
mkdir -p "$INSTALL_DIR"
tar -xzf "$TMPDIR/neurdb.tar.gz" -C "$INSTALL_DIR" --strip-components=1
rm -rf "$TMPDIR"

# 6. Create data and log directories
echo "Setting up directories..."
mkdir -p "$DATA_DIR"
mkdir -p "$LOG_DIR"
chown -R neurdb:neurdb "$DATA_DIR"
chown -R neurdb:neurdb "$LOG_DIR"

# 7. Initialise the database cluster
if [ ! -f "$DATA_DIR/PG_VERSION" ]; then
    echo "Initializing database cluster..."
    sudo -u neurdb "$INSTALL_DIR/bin/initdb" -D "$DATA_DIR" -U neurdb

    # Patch postgresql.conf to load nr_kernel extensions
    sed -i '/^#*shared_preload_libraries/d' "$DATA_DIR/postgresql.conf"
    echo "shared_preload_libraries = 'pg_hint_plan, nr_molqo, nr_ext, nram, pg_neurstore'" >> "$DATA_DIR/postgresql.conf"

    # Set the port
    if [ "$PORT" != "5432" ]; then
        sed -i "s/^#*port = .*/port = ${PORT}/" "$DATA_DIR/postgresql.conf"
    fi

    # Allow local connections
    echo "listen_addresses = 'localhost'" >> "$DATA_DIR/postgresql.conf"
fi

# 8. Install systemd service units
echo "Installing systemd services..."
if [ -f "$INSTALL_DIR/installer/neurdb-db.service" ]; then
    cp "$INSTALL_DIR/installer/neurdb-db.service" /etc/systemd/system/
    cp "$INSTALL_DIR/installer/neurdb-ai.service" /etc/systemd/system/
else
    # Use co-located service files
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    cp "$SCRIPT_DIR/neurdb-db.service" /etc/systemd/system/
    cp "$SCRIPT_DIR/neurdb-ai.service" /etc/systemd/system/
fi
systemctl daemon-reload

# 9. Enable and start services
echo "Starting NeurDB services..."
systemctl enable neurdb-db neurdb-ai
systemctl start neurdb-db

# Wait for DB to be ready
echo -n "Waiting for database to start "
MAX_WAIT=60
ELAPSED=0
until sudo -u neurdb "$INSTALL_DIR/bin/psql" -h localhost -p "$PORT" -U neurdb -c '\q' 2>/dev/null; do
    printf '.'
    sleep 1
    ELAPSED=$((ELAPSED + 1))
    if [ "$ELAPSED" -ge "$MAX_WAIT" ]; then
        echo ""
        echo "Warning: Database did not become ready within ${MAX_WAIT}s."
        echo "Check logs: journalctl -u neurdb-db"
        exit 1
    fi
    # Create neurdb database if needed
    sudo -u neurdb "$INSTALL_DIR/bin/createdb" -h localhost -p "$PORT" -U neurdb neurdb 2>/dev/null || true
done
echo " OK"

# Install nr_pipeline extension
sudo -u neurdb "$INSTALL_DIR/bin/psql" -h localhost -p "$PORT" -U neurdb -c 'CREATE EXTENSION IF NOT EXISTS nr_pipeline;'

# Start AI runtime
systemctl start neurdb-ai

# Wait for AI server
echo -n "Waiting for AI runtime server to start "
ELAPSED=0
until curl --output /dev/null --silent --head --fail http://127.0.0.1:8090/; do
    printf '.'
    sleep 1
    ELAPSED=$((ELAPSED + 1))
    if [ "$ELAPSED" -ge "$MAX_WAIT" ]; then
        echo ""
        echo "Warning: AI server did not start within ${MAX_WAIT}s."
        echo "Check logs: journalctl -u neurdb-ai"
        break
    fi
done
echo " OK"

# 10. Install PATH integration
echo 'export PATH="/opt/neurdb/bin:$PATH"' > /etc/profile.d/neurdb.sh
chmod 644 /etc/profile.d/neurdb.sh

# Install convenience wrapper
cat > /usr/local/bin/neurdb-psql << 'WRAPPER'
#!/bin/bash
exec /opt/neurdb/bin/psql -h localhost -U neurdb -d neurdb "$@"
WRAPPER
chmod +x /usr/local/bin/neurdb-psql

# 11. Print connection info
echo ""
echo "============================================"
echo "  NeurDB has been installed!"
echo "============================================"
echo ""
echo "  Connect with:"
echo "    neurdb-psql"
echo "    # or"
echo "    psql -h localhost -p ${PORT} -U neurdb -d neurdb"
echo ""
echo "  Service management:"
echo "    systemctl status neurdb-db neurdb-ai"
echo "    systemctl restart neurdb-db"
echo "    journalctl -u neurdb-db -f"
echo ""
echo "  Uninstall:"
echo "    /opt/neurdb/installer/uninstall.sh"
echo ""
