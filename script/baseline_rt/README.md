# baseline_rt — minimal neurdbrt for baseline inference

This directory is a **copy** of only the neurdbrt modules needed when `conn.load_model(id).unpack()` runs: the pickle resolves the model class as `neurdbrt.model.armnet.model.ARMNetModel`, so we provide that module (and its dependencies) here **without** dataloader, cache, or app.

- **Purpose**: Run `script/baseline_inference.py` without pulling the full `aiengine/runtime/neurdbrt` (avoids circular imports: cache → data_dispatcher → app → hook/setup → dataloader).
- **Usage**: `baseline_inference.py` inserts `script/baseline_rt` at the front of `sys.path`, so `import neurdbrt.model.armnet.model` loads this copy.
- **Contents**: `neurdbrt/model/armnet/` — `model.py`, `entmax.py`, `layer.py`, and a minimal `__init__.py` that only exports `ARMNetModel` (no builder, no `register_model`).

When the runtime or server runs, they use the full `aiengine/runtime` neurdbrt; only the baseline script uses this stub.
