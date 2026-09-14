# trails

`trails` brings zero-cost neural architecture search (NAS) utilities to
[torch_frame](https://github.com/pyg-team/torch-frame) models focused on structured
tabular data. The module groups two key parts:

- reusable search spaces (`search_space/`) with NAS-friendly variants of the
  default MLP and ResNet architectures;
- zero-cost proxy utilities (`proxies/`) implementing the ExpressFlow score that
  can evaluate candidate models without training.

## Components

- `search_space.mlp.TrailsMLP`
  Torch Frame MLP that lets you control the hidden width of every layer via the
  `hidden_dims` list while keeping encoder and decoder logic intact. It exports a
  `forward_wo_embedding` helper so proxies can bypass the costly embedding step.

- `search_space.resnet.TrailsResNet`
  Residual MLP-style network that accepts per-block widths through `block_widths`
  and exposes the same `forward_wo_embedding` shortcut. It reuses
  `FCResidualBlock` from Torch Frame for stable training behaviour.

- `proxies.expressflow.express_flow_score`
  Implementation of the ExpressFlow zero-cost proxy. It linearises ReLU-backed
  modules, perturbs the inputs, backpropagates once, then aggregates activation–
  gradient products with configurable weighting (`traj_width`, `traj`, `width`).

## Quick start

```python
import torch
from torch_frame import stype

from neurdbrt.model.trails.search_space.mlp import TrailsMLP
from neurdbrt.model.trails.proxies.expressflow import express_flow_score

# Dummy metadata — in practice take these from your Torch Frame dataset.
col_stats = {"num_col": {}}
col_names_dict = {stype.numerical: ["num_col"], stype.categorical: []}

model = TrailsMLP(
    channels=64,
    out_channels=1,
    num_layers=3,
    col_stats=col_stats,
    col_names_dict=col_names_dict,
    hidden_dims=[64, 32],
)

# Batch-shaped tensor after encoding step; use real TensorFrame in production.
batch = torch.randn(8, 64)

score, latency = express_flow_score(
    arch=model,
    batch_data=batch,
    device="cpu",
    use_wo_embedding=True,
    epsilon=1e-5,
    weight_mode="traj_width",
)

print(f"ExpressFlow score={score:.3f} (computed in {latency:.4f}s)")
```

## Tips

- Always wrap ExpressFlow calls in `torch.no_grad()` or keep gradients disabled to
  avoid accidental parameter updates.
- The proxy linearises model parameters in-place, so call it on evaluation clones
  or let it run while the model is in `eval()` mode.
- Use the `blocks_choices` and `channel_choices` class attributes as convenient
  search space hints for discrete NAS algorithms.
