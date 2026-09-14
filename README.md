<div align="center">

![logo](./doc/logo.jpg)


[![NeurDB Website](https://img.shields.io/badge/Home-Page-blue)](https://neurdb.com) [![paper-24-1](https://img.shields.io/badge/SCIS-Position%20Paper-B31B1B.svg)](http://scis.scichina.com/en/2024/200901.pdf) [![paper-24-2](https://img.shields.io/badge/CIDR-PoC%20Version-F7C15D)](https://vldb.org/cidrdb/papers/2025/p29-zhao.pdf) [![GitHub release](https://img.shields.io/badge/release-v0.5-F7C15D)](https://github.com/neurdb/neurdb/releases) ![GitHub contributors](https://img.shields.io/github/contributors-anon/neurdb/neurdb) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

</div>

---

## Overview

**NeurDB** is a cutting-edge AI-powered autonomous database system that revolutionizes how organizations manage, query, and interact with their data. NeurDB seamlessly integrates artificial intelligence capabilities directly into the database layer, in all major components, enabling intelligent data processing and autonomous optimization.

### Why NeurDB?

Modern data systems struggle to keep up with the growing complexity of AI-driven applications. Traditional databases rely on manual tuning and external machine learning pipelines, resulting in fragmented workflows, high maintenance costs, and limited adaptability.

NeurDB redefines this paradigm by making AI a first-class citizen inside the database. It is not just a database with AI features — it is an AI-native data system that learns, adapts, and optimizes itself in real time.

Key advantages:

- **AI-Native Architecture**: AI and data processing are deeply fused within the database engine, enabling seamless model training, inference, and management.
- **Intelligent Analytics**: Built-in AI operators let users run predictive and generative analytics directly with SQL, without external pipelines.
- **Autonomous Operation**: Self-tuning, self-scaling, and self-healing mechanisms continuously optimize performance and resource usage.
- **Unified AI & Data Platform**: One system for both data management and AI lifecycle, ensuring stronger security, lower latency, and simplified workflows.



## Quick Start

### Prerequisites

- Docker and Docker Compose
<!-- - Git -->
- 8GB+ RAM recommended
- (Optional) NVIDIA GPU with CUDA support for GPU acceleration

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


## Demo

See NeurDB in action:

![NeurDB demo](assets/demo.gif)



## Architecture

NeurDB consists of three main components:

![NeurDB arch](assets/neurdb-v1-arch.jpg)

1.	**AI Layer (NeurIDA)**: This layer manages AI models and analytics tasks inside the database and supports the lifecycle of in-database machine learning models.
    - *Model Selection (VLDB 2024, ICML 2026)* – Automatically ranks and selects suitable models for database tasks.
    - *Model Slicing (VLDB 2025)* – Decomposes large models into smaller slices to enable efficient execution and deployment.
    - *Model Construction (VLDB 2026)*  – Supports the composition and integration of multiple models for AI analytics and transactional workloads.


2. **Database Engine Layer**: Built on top of an enhanced PostgreSQL engine, this layer integrates learned optimization, learned concurrency control, and runtime adaptive execution.
   - *NeurQO (SIGMOD 2027)* – A learned query optimizer that performs fast-adaptive query optimization through query-state abstraction and workload feedback.
   - *NeurEngine (VLDB 2027)* – Implements the planner, optimizer, and executor, and supports unified execution graph construction and CPU–GPU co-scheduling to execute both traditional database operators and AI operators.
   - *NeurCC (SIGMOD 2026)* – A learned concurrency control framework that models concurrency control as a learnable function and dynamically adapts to workload changes.

3.	**Adaptive Data Access Components**: NeurDB incorporates adaptive data access modules that improve indexing and caching performance under dynamic workloads.
	- *NeurIndex* – A general learned index framework for adaptive index design and optimization under dynamic workloads, with *Selix (SIGMOD 2026)* as one implementation that enables DRL-based index adaptation.
	- *NeurCache* – A workload-aware caching framework that jointly manages model weights, features, and data through a unified cache structure.

4. **Storage Layer**: A dual-format storage system supporting both key–value (RocksDB), heap storage, and model storage (*NeurStore*).
   - *NeurStore* – Provides efficient storage and management for deep learning models.

5. **Benchmarking and Evaluation**
NeurDB also includes benchmarking frameworks for evaluating AI-powered database systems and learned components.

   - *NeurBench (SIGMOD 2026)* – A unified benchmark for evaluating learned database components under data and workload drift.
   - *NL2SQLBench (VLDB 2026)* – A modular benchmark for evaluating LLM-enabled NL2SQL systems.

1. **Tools and Utilities**
   NeurDB provides additional system tools to support model interpretation and system analysis.

   - *CoShap (SIGMOD 2026)* – A scalable method for Shapley value approximation to support model interpretation and feature contribution analysis.

<!-- ### Development Setup

For contributors looking to develop NeurDB:

- [DBEngine Development Guide](./doc/db_dev.md)
- [AIEngine Development Guide](./doc/ai_dev.md) -->



## Publications

NeurDB is backed by rigorous academic research. Our work has been published in top-tier venues:

### Papers

1. **NeurDB: An AI-powered Autonomous Data System** [[PDF]](http://scis.scichina.com/en/2024/200901.pdf)
   *SCIENCE CHINA Information Sciences, 2024*
2. **NeurDB: On the Design and Implementation of an AI-powered Autonomous Database** [[PDF]](https://vldb.org/cidrdb/papers/2025/p29-zhao.pdf)
   *CIDR 2025*
3. **Database Native Model Selection: Harnessing Deep Neural Networks in Database Systems** [[PDF]](https://www.vldb.org/pvldb/vol17/p1020-xing.pdf)
   *VLDB 2024*
4. **Powering In-Database Dynamic Model Slicing for Structured Data Analytics** [[PDF]](https://www.vldb.org/pvldb/vol17/p4813-zeng.pdf)
   *VLDB, 2025*
5.  **NeurStore: Efficient In-database Deep Learning Model Management System**
   *SIGMOD 2026*
6. **Modeling Concurrency Control as a Learnable Function**
   *SIGMOD 2026*
7. **On Self-Designing Learned Indexes**
   *SIGMOD 2026*
8. **NL2SQLBench: A Modular Benchmarking Framework for LLM-Enabled NL2SQL Solutions**
   *VLDB 2026*
9.  **NeurBench: A Benchmark Suite for Learned Database Components with Drift Modeling**
   *SIGMOD 2026*
10. **CoShap: A Scalable Coalition Growth Approach to Shapley Value Approximation**
      *SIGMOD 2026*
11. **pTNAS: Progressive Neural Architecture Search for Tabular Data**
    ICML 2026
12. **NeurIDA: Dynamic Modeling for Effective In-Database Analytics**
    VLDB 2026
13. **NQO: Query Optimization as a Learnable Function**
    *SIGMOD 2027*
14. **Towards Effective Orchestration of AI x DB Workloads [Vision]**
    *VLDB 2027*


### Citation

If you use NeurDB in your research, please cite:

```bibtex
@article{neurdb-scis-24,
  author  = {Beng Chin Ooi and Shaofeng Cai and Gang Chen and
             Yanyan Shen and Kian-Lee Tan and Yuncheng Wu and
             Xiaokui Xiao and Naili Xing and Cong Yue and
             Lingze Zeng and Meihui Zhang and Zhanhao Zhao},
  title   = {NeurDB: An AI-powered Autonomous Data System},
  journal = {SCIENCE CHINA Information Sciences},
  year    = {2024},
  url     = {https://www.sciengine.com/SCIS/doi/10.1007/s11432-024-4125-9},
  doi     = {10.1007/s11432-024-4125-9}
}
```
</div>
