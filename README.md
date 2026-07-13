![logo](./doc/logo.jpg)

[![NeurDB Website](https://img.shields.io/badge/Website-neurdb.com-blue)](https://neurdb.com)
[![Github](https://img.shields.io/badge/Github-100000.svg?logo=github&logoColor=white)](https://github.com/neurdb/neurdb)
![GitHub contributors](https://img.shields.io/github/contributors-anon/neurdb/neurdb)
![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![paper-24-1](https://img.shields.io/badge/DOI-10.1007/s11432--024--4125--9-B31B1B.svg)](http://scis.scichina.com/en/2024/200901.pdf)
[![paper-24-2](https://img.shields.io/badge/CIDR--25%20Paper-F7C15D)](https://vldb.org/cidrdb/papers/2025/p29-zhao.pdf)


NeurDB is an AI-powered autonomous data system.

![NeurDB demo](assets/demo.gif)

## Installation

### Quick Install (Docker)

NeurDB can be installed with a single command using Docker.

**Prerequisites:**

| Requirement | Version |
|---|---|
| Docker Engine | 20.10+ |
| OS | Linux (x86_64), Windows 10/11 with Docker Desktop |
| GPU drivers (optional) | NVIDIA 470+ with CUDA 11.8 |

**Using pre-built Docker images:**

*Linux:*
```bash
curl -fsSL https://github.com/neurdb/neurdb/releases/latest/download/install.sh | bash
```

*Windows (PowerShell as Administrator):*
```powershell
irm https://github.com/neurdb/neurdb/releases/latest/download/install.ps1 | iex
```

**Manually Build from Docker:**
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

### Building from Source

For native Linux builds, you can simply use the top-level `Makefile` to install prerequisites, build the engine, and start the services:

```bash
# 1. Install prerequisites (Requires sudo)
make deps

# 2. Build and install DB, AI engine, and Python client
make install

# 3. Start PostgreSQL and the AI Server
make start
```

For detailed instructions spanning Docker builds, custom ports, GPU support, and Windows development, please see **[INSTALL.md](./INSTALL.md#build-from-source-development)**.

## Usage

### Connecting to NeurDB

NeurDB is PostgreSQL-compatible, so you can connect using any PostgreSQL client. The default port is `5432`.

**Using `psql`:**
```bash
psql -h localhost -p 5432 -U neurdb -d neurdb
```

### In-Database AI with SQL

NeurDB extends SQL with a `PREDICT` statement for in-database AI inference:

```sql
-- Create a table and load data
CREATE TABLE frappe_test (
    click_rate INT, feature1 INT, feature2 INT,
    feature3 INT, feature4 INT, feature5 INT,
    feature6 INT, feature7 INT, feature8 INT,
    feature9 INT, feature10 INT
);

COPY frappe_test FROM '/path/to/data.csv' DELIMITER ',' CSV HEADER;

-- Configure training parameters
SET nr_task_batch_size TO 60;
SET nr_task_num_batches TO 100;

-- Train and predict in a single statement
PREDICT VALUE OF click_rate FROM frappe_test TRAIN ON *;
```

### Python Client

Install the NeurDB Python client and use it to manage models programmatically:

```bash
pip install neurdb
```

```python
from neurdb import NeurDB, ModelSerializer
import torch

# Connect to a running NeurDB instance
db = NeurDB(db_host="localhost", db_port="5432")

# Serialize and save a PyTorch model
model = torch.nn.Linear(10, 2)
pickled = ModelSerializer.serialize_model(model)
model_id = db.save_model(pickled)

# Load and restore the model
loaded = db.load_model(model_id)
restored_model = ModelSerializer.deserialize_model(loaded)

# Register the model for in-database inference
db.register_model(model_id, "my_table", ["feat1", "feat2"], ["target"])

# Clean up
db.close()
```

For more details, see the [Python client documentation](./api/python/README.md).

## Citation

Our vision paper can be found in:

```
@article{neurdb-scis-24,
  author = {Beng Chin Ooi and
            Shaofeng Cai and
            Gang Chen and
            Yanyan Shen and
            Kian-Lee Tan and
            Yuncheng Wu and
            Xiaokui Xiao and
            Naili Xing and
            Cong Yue and
            Lingze Zeng and
            Meihui Zhang and
            Zhanhao Zhao},
  title  =  {NeurDB: An AI-powered Autonomous Data System},
  journal=  {SCIENCE CHINA Information Sciences},
  year   =  {2024},
  pages  =  {-},
  url    =  {https://www.sciengine.com/SCIS/doi/10.1007/s11432-024-4125-9},
  doi    =  {https://doi.org/10.1007/s11432-024-4125-9}
}
```
