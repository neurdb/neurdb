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

You can natively compile and start the environment using the included `Makefile`:

```bash
# 1. Install prerequisites (Requires sudo)
make deps

# 2. Build and install the DB engine, extensions, API, and CPU AI engine
make install

# (Optional: Build the GPU AI engine instead)
# make install AI_ENGINE_MODE=gpu

# 3. Start the database background service and the AI server
make start
```

> [!TIP]
> If the server fails to start with `could not start server`, check the log with
> `cat build/logfile`. Common causes:
> - **Port 5432 already in use**: stop the other PostgreSQL instance (`sudo systemctl stop postgresql`) or change the port in `build/data/postgresql.conf`.
> - **Permission denied on data directory**: run `chmod 0750 build/data`.

When you are done, you can stop the services:
```bash
make stop
```

For advanced usage or troubleshooting, here are the other available commands:
* `make clean`: Removes the build outputs to force a recompile but keeps the data directory.
* `make distclean`: Deletes the entire `build/` directory including your data and configuration.

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
