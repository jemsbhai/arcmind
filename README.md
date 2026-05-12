# ArcMind

**Dual-timescale hybrid SSM+Attention architecture for efficient robotics and IoT language models.**

ArcMind is a novel neural architecture that combines a fast State Space Model (SSM) path for continuous sensor stream processing with a slow exact attention path for episodic memory recall, purpose-built for edge-deployed robotics and IoT applications at 5–100M parameter scale.

## Key Ideas

- **Sensor-native tokenization** — raw sensor frames projected directly into model dimension via learned linear layers. No vocabulary table, no embedding overhead.
- **Fast SSM path** — Mamba-style selective state space layers process sensor streams at hardware rate (100–1000 Hz) with O(n) time and O(1) decode memory.
- **Slow attention path** — tiny exact attention (1–2 layers, 2–4 heads) runs at decision rate (1–10 Hz) over episodic memory for precise recall.
- **Episodic memory ring buffer** — fixed-size compressed snapshots of environment state, enabling spatial/temporal recall without a growing KV cache.

## Installation

```bash
pip install arcmind
```

For development:

```bash
git clone https://github.com/jemsbhai/arcmind.git
cd arcmind
pip install -e ".[dev]"
```

## Quick Start

```python
import torch
from arcmind import ArcMindConfig, ArcMindModel

# Create a small robotics model
config = ArcMindConfig.robotics_small()
model = ArcMindModel(config)

# Simulate a sensor stream (batch=1, 100 timesteps, 12 channels)
sensor_data = torch.randn(1, 100, config.num_sensor_channels)

# Run inference
model.reset_memory(batch_size=1)
actions = model(sensor_data)
print(actions.shape)  # (1, 100, 6)

# Check parameter count
print(model.count_parameters())
```

### Model Presets

| Preset | Target | Params | Hardware |
|--------|--------|--------|----------|
| `ArcMindConfig.iot_tiny()` | 5–15M | MCU/NPU (Cortex-M7, ESP32) |
| `ArcMindConfig.robotics_small()` | 30–50M | Jetson Orin Nano, RPi 5 |
| `ArcMindConfig.robotics_medium()` | 50–100M | Desktop GPU, Jetson AGX |

## Development

```powershell
# Run tests
pytest tests/ -v

# Run tests with coverage
pytest tests/ -v --cov=arcmind --cov-report=term

# Lint
ruff check arcmind/
```

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

## Citation

Paper forthcoming.

## License

MIT License. See [LICENSE](LICENSE) for details.
