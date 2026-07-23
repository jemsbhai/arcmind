# ArcMind

**A causal, dual-rate sequence backbone for streaming sensor policies.**

[![PyPI version](https://img.shields.io/pypi/v/arcmind)](https://pypi.org/project/arcmind/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

> **Alpha research release.** The architecture and streaming API are tested.
> Existing experimental checkpoints predate the current causal-memory revision
> and are not paper results. APIs may change before 1.0.

## Overview

ArcMind is an experimental backbone for low-dimensional sensor streams under
partial observability. It combines:

- a selective state-space fast path that updates on every sensor frame;
- a lower-rate exact-attention path over a bounded ring buffer of compressed,
  strictly prior decision states;
- learned relative-age embeddings that make memory order observable; and
- a learned gate that fuses recurrent and recalled representations.

The implementation is sensor-to-action: raw floating-point channels are
projected directly into the model dimension and the output head predicts a
continuous action vector. ArcMind is not currently a language model,
vision-language-action model, or offline-RL algorithm.

```text
sensor frame -> tokenizer -> selective SSM ----------------------+
                              |                                  |
                              +-> periodic compressed memory     |
                                      |                          |
current decision state -> bounded exact recall over prior slots  |
                              |                                  |
                              +---------- learned fusion <-------+
                                             |
                                          action
```

For a fixed configuration, recurrent state and episodic memory remain bounded
during streaming inference. Performance and hardware suitability are empirical
questions covered by the
[research protocol](https://github.com/jemsbhai/arcmind/blob/master/docs/research_protocol.md);
this README intentionally does not claim state-of-the-art results.

## Installation

Requirements: Python 3.10 or newer and PyTorch 2.1 or newer.

```bash
pip install arcmind
```

The optional Minari-backed offline-control adapters are installed with:

```bash
pip install "arcmind[datasets]"
```

Benchmark authoring utilities are available through `arcmind[benchmarks]`.

For development:

```bash
git clone https://github.com/jemsbhai/arcmind.git
cd arcmind
pip install -e ".[dev]"
pytest
```

## Batch use

```python
import torch

from arcmind import ArcMindConfig, ArcMindModel

config = ArcMindConfig.robotics_small()
model = ArcMindModel(config)

sensor_data = torch.randn(2, 100, config.num_sensor_channels)
actions = model(sensor_data)

assert actions.shape == (2, 100, config.action_dim)
```

The batch path is causal and stateless across calls. Slow-path decisions occur at
`sensor_freq_hz / decision_freq_hz`, and each decision can recall only snapshots
written by earlier decisions.

## Streaming use

```python
model.eval()
model.init_streaming(batch_size=1)

with torch.no_grad():
    for sensor_frame in sensor_stream:
        action = model.step(sensor_frame)
        actuator.send(action)
```

Call `init_streaming()` at every episode boundary. It clears the SSM state,
episodic memory, held slow-path output, and step counter. Unit tests require the
streaming recurrence to match causal batch execution.

## Custom configuration

```python
config = ArcMindConfig(
    num_sensor_channels=9,
    d_model=96,
    num_ssm_layers=6,
    ssm_state_dim=12,
    num_attn_layers=1,
    num_attn_heads=3,
    attn_window_size=32,
    num_memory_slots=64,
    memory_compress_ratio=4,
    action_dim=4,
    sensor_freq_hz=200.0,
    decision_freq_hz=20.0,
)
model = ArcMindModel(config)
```

`attn_window_size` is the maximum number of recent decision snapshots visible
to exact recall. `num_memory_slots` controls retained episode history and may be
larger than that window.

The registered ablations are available through configuration flags:
`ablate_ssm`, `ablate_attention`, `ablate_memory`, `ablate_gating`, and
`ablate_temporal_encoding`.

## Presets

Counts are generated from the current source and checked by the test suite.
Hardware labels are intentionally omitted until matched-device measurements
exist.

| Preset | Parameters | SSM share | Exact-attention share |
|---|---:|---:|---:|
| `iot_tiny` | 246,356 | 73.2% | 20.6% |
| `robotics_small` | 1,692,070 | 84.2% | 12.2% |
| `robotics_medium` | 10,354,252 | 82.1% | 15.6% |

The fast path is a compact, pure-PyTorch input-dependent selective SSM. It is
inspired by selective state-space modeling but is not the published Mamba or
Mamba-2 block and does not use their optimized kernels.

## Research and reproducibility

The
[research protocol](https://github.com/jemsbhai/arcmind/blob/master/docs/research_protocol.md)
defines the falsifiable claim, required baselines, benchmark families,
ablations, statistics, efficiency measurements, and prohibited claims.
The companion
[benchmark audit](https://github.com/jemsbhai/arcmind/blob/master/docs/literature_and_baselines.md)
records the primary-source rationale, implementation status, and
like-for-like versus contextual comparison boundary.

The planned evidence has three tracks:

1. POBAX and POPGym for the memory mechanism;
2. low-dimensional RoboMimic for state-based robot imitation; and
3. UEA plus MONSTER for multivariate sensor classification diagnostics.

UCI HAR, Opportunity, MuJoCo, and AntMaze examples remain development smoke
tests. They are not sufficient evidence for the proposed contribution.

No existing checkpoint or metric should be cited as a result of the current
architecture. Paper results must be regenerated from committed configurations,
seed manifests, immutable dataset identifiers, and machine-readable raw output.

## Project status

- [x] Causal batch and recurrent execution
- [x] Selective SSM fast path
- [x] Bounded, chronological, time-aware exact recall
- [x] Learned memory compression and fast/slow gating
- [x] Configuration presets and component ablations
- [x] PyPI package, reproducible artifacts, and release-gated unit tests
- [x] Pre-registered research and evaluation protocol
- [x] Local delayed-recall benchmark, matched baselines, and aggregation
- [x] Commit-pinned JAX/POBAX environment and streaming inference parity
- [x] Shared discrete/continuous JAX PPO learner
- [x] Parameter-matched recurrent, convolutional, SSM, and attention controls
- [x] Registered ArcMind ablation adapters with effective parameter counts
- [x] S5RL reset-aware structured-SSM policy adapter
- [x] Privileged-observation reference adapters and evidence-link gates
- [ ] Registered multi-seed experiments
- [ ] Reproducible paper tables and figures
- [ ] Pretrained weights

## Citation

A paper citation will be added after the registered experiments. Until then:

```bibtex
@software{arcmind2026,
  title  = {ArcMind: A Dual-Rate Sequence Backbone for Streaming Sensor Policies},
  author = {Syed, Muntaser},
  year   = {2026},
  url    = {https://github.com/jemsbhai/arcmind}
}
```

## License

ArcMind is released under the MIT License. See [LICENSE](LICENSE).
