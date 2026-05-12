# ArcMind

**Dual-timescale hybrid SSM+Attention architecture for efficient robotics and IoT.**

[![PyPI version](https://img.shields.io/pypi/v/arcmind)](https://pypi.org/project/arcmind/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-44%20passing-brightgreen.svg)]()

> ⚠️ **Alpha release** — architecture is functional and tested, pretrained weights coming soon. API may change.

## What is ArcMind?

ArcMind is a neural architecture purpose-built for robotics and IoT edge deployment. It combines a fast State Space Model (SSM) path for continuous sensor stream processing with a slow exact attention path for episodic memory recall — all at 245K to 10.3M parameters.

### The problem

Current approaches to "LMs for robotics" fall into two camps:

1. **Massive VLA models** (RT-2, OpenVLA, Pi-Zero) — 7B–55B parameters, cloud-dependent, text-centric tokenization. Can't run on edge hardware.
2. **Shoehorned text SLMs** (quantized Phi, TinyLlama on Jetson) — general-purpose text models forced onto constrained devices. Not designed for sensor streams.

Neither is purpose-built for the sensor-stream → reasoning → action loop that robotics and IoT actually need.

### The approach

ArcMind introduces a **dual-timescale hybrid** that mirrors how robotic control actually works:

- **Fast path (SSM):** Processes every sensor frame at hardware rate (100–1000 Hz). O(n) time, O(1) decode memory, no KV cache. Produces smooth, physically plausible control signals.
- **Slow path (Attention):** Tiny exact attention (1–2 layers, 2–4 heads) runs at decision rate (1–10 Hz) over an episodic memory buffer for precise spatial/temporal recall.
- **Sensor-native tokenization:** Raw sensor frames projected directly into model dimension via learned linear layers. No vocabulary table — eliminates the ~40% parameter overhead of embedding tables in small text LMs.
- **Episodic memory:** Fixed-size ring buffer of compressed environment state snapshots. Enables landmark recall and obstacle memory without a growing KV cache.

## Installation

**Requirements:** Python ≥ 3.10, PyTorch ≥ 2.1

```bash
pip install arcmind
```

For development:

```bash
git clone https://github.com/jemsbhai/arcmind.git
cd arcmind
pip install -e ".[dev]"
pytest tests/ -v  # 44 tests, all passing
```

## Quick Start

```python
import torch
from arcmind import ArcMindConfig, ArcMindModel

# Create a model from a preset
config = ArcMindConfig.robotics_small()
model = ArcMindModel(config)

# Simulate a sensor stream (batch=1, 100 timesteps, 12 channels)
sensor_data = torch.randn(1, 100, config.num_sensor_channels)

# Forward pass
model.reset_memory(batch_size=1)
actions = model(sensor_data)
print(actions.shape)  # (1, 100, 6)

# Inspect parameter breakdown
for component, count in model.count_parameters().items():
    print(f"  {component}: {count:,}")
```

### Custom Configuration

```python
# Build a custom config for your specific robot/sensor setup
config = ArcMindConfig(
    num_sensor_channels=9,    # e.g., 3-axis accel + 3-axis gyro + 3-axis mag
    d_model=96,
    num_ssm_layers=6,
    ssm_state_dim=12,
    num_attn_layers=1,
    num_attn_heads=3,
    num_memory_slots=32,
    action_dim=4,             # e.g., 4 motor commands
    sensor_freq_hz=200.0,
    decision_freq_hz=20.0,
)
model = ArcMindModel(config)
```

## Model Presets

All parameter counts independently verified via test suite.

| Preset | Params | SSM | Attention | Tokenizer | Target Hardware |
|--------|--------|-----|-----------|-----------|-----------------|
| `iot_tiny` | 245K | 73.5% | 20.3% | 0.2% | Cortex-M7, ESP32-S3 |
| `robotics_small` | 1.7M | 84.7% | 11.7% | 0.1% | Jetson Orin Nano, RPi 5 |
| `robotics_medium` | 10.3M | 82.4% | 15.3% | 0.1% | Desktop GPU, Jetson AGX |

Key property: the sensor tokenizer is consistently <1% of total parameters, confirming the embedding-table-free design eliminates the parameter overhead that dominates sub-100M text LMs.

## Architecture

```
Sensor Stream → SensorTokenizer → SSMCore (fast, 100-1000 Hz)
                                      ↓ periodic snapshot
                                 EpisodicMemory (ring buffer)
                                      ↓ read
SSM output → SlowAttention (slow, 1-10 Hz) ← memory slots
                   ↓ gated fusion
              ActionHead → action output
```

**Design rationale:**

- SSM:Attention parameter ratio of ~5:1 to ~8:1 extends the Granite 4 (9:1) and Jamba (7:1) ratios, justified by sensor streams being temporally smoother than text.
- Only 2–4 attention heads for recall, based on retrieval-aware distillation research showing 2–3 heads suffice for recall in SSM hybrids.
- Episodic ring buffer adapted from Expansion Span's reserved attention context for distant retrieval, applied here to compressed environment state snapshots.

## Project Status

- [x] Core architecture (SSM + Attention + Memory + Gating)
- [x] Sensor-native tokenizer
- [x] Three validated presets (IoT, Robotics-S, Robotics-M)
- [x] Full test suite (44 tests)
- [x] PyPI package
- [ ] Training pipeline
- [ ] Benchmark evaluation (MuJoCo, sensor datasets)
- [ ] Pretrained weights on HuggingFace
- [ ] Research paper

## Citation

Paper forthcoming. For now:

```bibtex
@software{arcmind2026,
  title={ArcMind: Dual-Timescale Hybrid SSM+Attention for Efficient Robotics and IoT},
  author={Syed, Muntaser},
  year={2026},
  url={https://github.com/jemsbhai/arcmind},
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
