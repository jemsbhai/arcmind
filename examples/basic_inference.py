"""
ArcMind basic inference example.

Demonstrates forward passes through all three model presets,
parameter breakdowns, episodic memory behavior, and the
dual-timescale architecture in action.

Usage:
    python examples/basic_inference.py
"""

import time

import torch

from arcmind import ArcMindConfig, ArcMindModel


def print_header(text: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {text}")
    print(f"{'=' * 60}")


def demo_preset(name: str, config: ArcMindConfig, seq_len: int = 100) -> None:
    """Run a full forward pass on a preset and report diagnostics."""
    print_header(f"Preset: {name}")

    model = ArcMindModel(config)
    model.eval()

    # Parameter breakdown
    counts = model.count_parameters()
    print("\n  Parameters:")
    for component, count in counts.items():
        if component == "total":
            print(f"    {'─' * 38}")
        pct = count / counts["total"] * 100 if component != "total" else 100.0
        print(f"    {component:20s}  {count:>10,}  ({pct:5.1f}%)")

    # Sensor input simulation
    batch_size = 1
    sensor_data = torch.randn(batch_size, seq_len, config.num_sensor_channels)
    print("\n  Input:")
    print(f"    Sensor channels:    {config.num_sensor_channels}")
    print(f"    Sequence length:    {seq_len} frames")
    print(f"    Sensor rate:        {config.sensor_freq_hz} Hz")
    print(f"    Decision rate:      {config.decision_freq_hz} Hz")
    print(
        f"    Decision stride:    {model.decision_stride} "
        f"(1 slow step per {model.decision_stride} fast steps)"
    )
    print(f"    Simulated duration: {seq_len / config.sensor_freq_hz:.2f} seconds")

    # Forward pass with timing. Batch calls use sequence-local memory and do
    # not mutate persistent streaming state.
    with torch.no_grad():
        start = time.perf_counter()
        actions = model(sensor_data)
        elapsed_ms = (time.perf_counter() - start) * 1000

    print("\n  Output:")
    print(f"    Action shape:       {tuple(actions.shape)}")
    print(f"    Action dim:         {config.action_dim}")
    print(f"    Inference time:     {elapsed_ms:.1f} ms (CPU, batch={batch_size})")
    print(f"    Per-frame:          {elapsed_ms / seq_len:.2f} ms/frame")

    # Persistent memory remains empty after stateless batch execution.
    occupancy = model.memory.get_occupancy()
    print("\n  Persistent Streaming Memory:")
    print(f"    Slots total:        {config.num_memory_slots}")
    print(f"    Slots written:      {occupancy}")
    print(f"    Buffer shape:       {tuple(model.memory.read().shape)}")


def demo_custom_config() -> None:
    """Show how to build a custom config for a specific sensor setup."""
    print_header("Custom Config: 9-axis IMU robot arm")

    config = ArcMindConfig(
        num_sensor_channels=9,      # 3 accel + 3 gyro + 3 magnetometer
        d_model=96,
        num_ssm_layers=6,
        ssm_state_dim=12,
        num_attn_layers=1,
        num_attn_heads=3,
        num_memory_slots=32,
        action_dim=7,               # 7-DOF robot arm
        sensor_freq_hz=200.0,
        decision_freq_hz=20.0,
    )

    model = ArcMindModel(config)
    total = model.count_parameters()["total"]
    print("\n  Config: 9ch input, d_model=96, 6 SSM layers, 1 attn layer")
    print(f"  Total parameters: {total:,}")

    # Simulate 2 seconds of sensor data at 200 Hz
    sensor_data = torch.randn(1, 400, config.num_sensor_channels)
    with torch.no_grad():
        actions = model(sensor_data)

    print(f"  Input:  {tuple(sensor_data.shape)} (2 seconds at 200 Hz)")
    print(f"  Output: {tuple(actions.shape)} (7-DOF joint commands)")
    print("  Batch memory: sequence-local and cleared when forward() returns")


def demo_episodic_memory() -> None:
    """Show the ring buffer wrap-around behavior."""
    print_header("Episodic Memory: Ring Buffer Behavior")

    config = ArcMindConfig.iot_tiny()
    model = ArcMindModel(config)
    model.reset_memory(batch_size=1)

    print(f"\n  Buffer size: {config.num_memory_slots} slots")
    print(f"  Writing {config.num_memory_slots + 5} snapshots (expect wrap-around)...\n")

    for i in range(config.num_memory_slots + 5):
        snapshot = torch.randn(1, config.d_model)
        model.memory.write(snapshot)
        if i < 5 or i >= config.num_memory_slots - 1:
            print(f"    Write #{i + 1:3d}  →  occupancy: {model.memory.get_occupancy()}")
        elif i == 5:
            print("    ...")

    print(
        f"\n  After {config.num_memory_slots + 5} writes, "
        f"occupancy caps at {model.memory.get_occupancy()}"
    )
    print("  Oldest entries have been overwritten (FIFO ring buffer)")


def main():
    print("\n" + "=" * 60)
    print("  ArcMind — Basic Inference Demo")
    print("=" * 60)

    # Run all three presets
    demo_preset("iot_tiny", ArcMindConfig.iot_tiny(), seq_len=50)
    demo_preset("robotics_small", ArcMindConfig.robotics_small(), seq_len=100)
    demo_preset("robotics_medium", ArcMindConfig.robotics_medium(), seq_len=200)

    # Custom config
    demo_custom_config()

    # Memory behavior
    demo_episodic_memory()

    print_header("Done")
    print("\n  All demos completed successfully.\n")


if __name__ == "__main__":
    main()
