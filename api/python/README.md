# NeurDB Python Client

Python client library for [NeurDB](https://neurdb.com) — an AI-powered autonomous data system.

## Installation

### Install from Local Distribution

To install the package from a local checkout (non-editable):

```bash
pip install .
```

Note: changes to source files are not reflected until you reinstall:

```bash
pip uninstall neurdb
pip install .
```
### Install from Python Package Index (PyPI)

```bash
pip install neurdb
```

## Quickstart

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

## Prerequisites

A running NeurDB server is required. See the [main repository](https://github.com/neurdb/neurdb) for installation instructions.

## Development

```bash
# Install in editable mode
pip install -e ".[dev]"

# Run tests
pytest
```


## Links

- [NeurDB Website](https://neurdb.com)
- [GitHub Repository](https://github.com/neurdb/neurdb)
- [Bug Tracker](https://github.com/neurdb/neurdb/issues)
