# Installation

## Quick Install (Docker)

NeurDB can be installed with a single command using Docker.

**Prerequisites:**

| Requirement | Version |
|---|---|
| Docker Engine | 20.10+ |
| OS | Linux (x86_64), Windows 10/11 with Docker Desktop |
| GPU drivers (optional) | NVIDIA 470+ with CUDA 11.8 |

**Linux:**
```bash
curl -fsSL https://github.com/neurdb/neurdb/releases/latest/download/install.sh | bash
```

**Windows (PowerShell as Administrator):**
```powershell
irm https://github.com/neurdb/neurdb/releases/latest/download/install.ps1 | iex
```

**Manual Docker install:**
```bash
# GPU (auto-detected if nvidia-smi is available)
bash installer/linux/install.sh --gpu

# CPU only
bash installer/linux/install.sh --cpu

# Custom port and persistent data
bash installer/linux/install.sh --port 15432 --data-dir /data/neurdb
```

**Python client library:**
```bash
pip install neurdb
```

## Build from Source (Development)

Our database is based on the PostgreSQL 16.3 with [doc](https://www.postgresql.org/docs/16/)

### Clone the latest code

```bash
git clone https://github.com/neurdb/neurdb.git
cd neurdb
# Give Docker container write permission
chmod -R 777 .
```

### Build with Docker

```bash
# Release build (builds both CPU and GPU variants, optimized, no debug tools)
bash build.sh --release

# Development build (with source mounting and debug port)
bash build.sh --gpu
bash build.sh --cpu
```

Wait until the following prompt shows:

```
Please use 'control + c' to exit the logging print
...
Press CTRL+C to quit
```

### Native Build on Linux

> [!WARNING]
> If you have previously compiled using Docker (`bash build.sh`), the source tree will contain leftover build artifacts with absolute symlinks pointing to Docker container paths. You **MUST** completely clean the repository before attempting a native build:
> `git clean -xfd` (Warning: this deletes all untracked files)

> [!NOTE]
> Tested on Ubuntu 22.04 (x86_64). Other Debian-based distributions should also work.

**1. Install system dependencies:**

```bash
sudo apt-get update
sudo apt-get install -y \
    python3-dev python3-pip python-is-python3 \
    build-essential gcc make cmake pkg-config \
    clang flex bison \
    libreadline-dev zlib1g-dev libicu-dev \
    libssl-dev libclang-dev llvm-dev \
    libcurl4-openssl-dev libwebsockets-dev libcjson-dev \
    librocksdb-dev libpqxx-dev libopencv-dev \
    curl git locales
```

Generate locale (needed by PostgreSQL):

```bash
sudo locale-gen en_US.UTF-8
sudo update-locale LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8
```

**2. Set environment variables:**

```bash
# Change to the root of the neurdb repo
cd neurdb

# Set environment variables
export NEURDBPATH="$(pwd)"
export NR_BUILD_PATH="$NEURDBPATH/build"
export NR_PSQL_PATH="$NR_BUILD_PATH/psql"
export NR_DBDATA_PATH="$NR_BUILD_PATH/data"
export MULTIARCH_LIBDIR="/usr/lib/$(dpkg-architecture -qDEB_HOST_MULTIARCH)"
export PKG_CONFIG_PATH="$MULTIARCH_LIBDIR/pkgconfig"
export LD_LIBRARY_PATH="$MULTIARCH_LIBDIR:$LD_LIBRARY_PATH"
export LIBCLANG_PATH="$(llvm-config --libdir)"
export NR_DBENGINE_PATH="$NEURDBPATH/dbengine"
export NR_KERNEL_PATH="$NR_DBENGINE_PATH/nr_kernel"
export NR_AIENGINE_PATH="$NEURDBPATH/aiengine"
export NR_API_PATH="$NEURDBPATH/api"
```

**3. Build the DB engine (PostgreSQL):**

```bash
mkdir -p $NR_BUILD_PATH/dbengine $NR_PSQL_PATH
cd $NR_BUILD_PATH/dbengine

# Configure and build (out-of-source build)
$NR_DBENGINE_PATH/configure --prefix=$NR_PSQL_PATH --enable-debug
make -j

# If buidl errors, clean previous build artifacts and try again
# rm -rf $NR_BUILD_PATH/dbengine/*

make install
```

**4. Build pg_hint_plan extension:**

```bash
mkdir -p $NR_BUILD_PATH/contrib && cd $NR_BUILD_PATH/contrib
if [ ! -d "pg_hint_plan" ]; then
  git clone https://github.com/ossc-db/pg_hint_plan.git
fi
cd pg_hint_plan && git checkout PG16
make PG_CONFIG=$NR_PSQL_PATH/bin/pg_config clean || true
make PG_CONFIG=$NR_PSQL_PATH/bin/pg_config
make PG_CONFIG=$NR_PSQL_PATH/bin/pg_config install
```

**5. Initialize and start the database:**

```bash
# Initialize data directory (skip if already exists)
if [ ! -d "$NR_DBDATA_PATH" ]; then
  mkdir -p $NR_DBDATA_PATH
  $NR_PSQL_PATH/bin/initdb -D $NR_DBDATA_PATH
else
  chmod 0750 $NR_DBDATA_PATH
fi

# Start the database server
$NR_PSQL_PATH/bin/pg_ctl -D $NR_DBDATA_PATH -l $NR_BUILD_PATH/logfile start

# If the server fails to start with "could not start server", port 5432 already in use, then stop the postgresql instance or change the port
# sudo systemctl stop postgresql

# Wait for the server to be ready, then create the neurdb database
until $NR_PSQL_PATH/bin/psql -h localhost -p 5432 -U $USER -c '\q' 2>/dev/null; do
  echo 'NeurDB is unavailable - sleeping'
  sleep 1
  $NR_PSQL_PATH/bin/createdb -h localhost -p 5432 neurdb 2>/dev/null || true
done
```

> [!TIP]
> If the server fails to start with `could not start server`, check the log with
> `cat $NR_BUILD_PATH/logfile`. Common causes:
> - **Port 5432 already in use**: stop the other PostgreSQL instance (`sudo systemctl stop postgresql`) or change the port in `$NR_DBDATA_PATH/postgresql.conf`.
> - **Permission denied on data directory**: run `chmod 0750 $NR_DBDATA_PATH`.

**6. Build NR kernel extensions:**

```bash
cd $NR_KERNEL_PATH
export PG_CONFIG=$NR_PSQL_PATH/bin/pg_config
make clean || true
make
make install
```

Register extensions and restart:

```bash
sed -i '/^#*shared_preload_libraries/d' $NR_DBDATA_PATH/postgresql.conf
echo "shared_preload_libraries = 'pg_hint_plan, nr_molqo, nr_ext, nram, pg_neurstore'" >> $NR_DBDATA_PATH/postgresql.conf
$NR_PSQL_PATH/bin/pg_ctl -D $NR_DBDATA_PATH -l $NR_BUILD_PATH/logfile restart

# Wait for the server to be ready after restart
until $NR_PSQL_PATH/bin/psql -h localhost -p 5432 -U $USER -c '\q' 2>/dev/null; do
  echo 'NeurDB is unavailable - sleeping'
  sleep 1
done
```

**7. Install AI engine dependencies and start the server:**

```bash
# CPU only
pip install -r $NR_AIENGINE_PATH/runtime/requirements.cpu.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Or GPU (CUDA 11.x)
pip install -r $NR_AIENGINE_PATH/runtime/requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu116

# Start AI engine server
cd $NR_AIENGINE_PATH/runtime
export NR_LOG_LEVEL=INFO
nohup python server.py &

# Wait for the AI engine to be ready
echo -n 'Waiting for AI engine to start '
until curl --output /dev/null --silent --head --fail http://127.0.0.1:8090/; do
  printf '.'
  sleep 1
done
echo ' OK'
```

**8. Install the NeurDB Python client API:**

```bash
mkdir -p $NR_BUILD_PATH/api/python
cp -r $NR_API_PATH/python/* $NR_BUILD_PATH/api/python/
cd $NR_BUILD_PATH/api/python
touch setup.cfg
pip install -e .
rm setup.cfg
```

**9. Create the pipeline extension:**

```bash
$NR_PSQL_PATH/bin/psql -h localhost -p 5432 -U $USER -d neurdb \
    -c 'CREATE EXTENSION IF NOT EXISTS nr_pipeline;'
```

### Native Build on Windows

NeurDB does **not** build directly on Windows. Use **WSL2** (Windows Subsystem for Linux) to get a full Linux environment, then follow the Linux steps above.

**1. Install WSL2** (PowerShell as Administrator):

```powershell
wsl --install -d Ubuntu-22.04
```

Restart your machine if prompted, then launch the **Ubuntu** terminal from the Start menu.

**2. (Optional) Enable GPU pass-through:**

If you have an NVIDIA GPU and want CUDA support inside WSL2, install the
[NVIDIA CUDA on WSL driver](https://developer.nvidia.com/cuda/wsl) on the **Windows** side.
Verify with:

```bash
nvidia-smi   # should show your GPU inside WSL2
```

**3. Follow the Linux build steps** listed above inside the WSL2 terminal.

> [!TIP]
> Your Windows filesystem is accessible under `/mnt/c/` inside WSL2, but for best build
> performance, clone the repo into the Linux filesystem (e.g., `~/neurdb`).

## Development

[DB engine dev](./doc/db_dev.md)

[AI engine dev](./doc/ai_dev.md)
