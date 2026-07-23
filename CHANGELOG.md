# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-07-23

### Fixed
- Made exact recall strictly past-only instead of duplicating the current state
- Enforced the configured local attention and episodic-memory window
- Added relative-age encoding so episodic recall is sensitive to temporal order
- Made batch memory updates causal so actions cannot attend to future snapshots
- Preserved training gradients through the learned memory compressor
- Aligned batch and streaming slow-path cadence and ring-buffer semantics
- Excluded empty memory slots and restored chronological reads after buffer wrap
- Switched AntMaze rollouts to stateful streaming inference
- Modernized PyPI license metadata and explicit public package exports
- Made batch execution stateless across independent sequences and minibatches
- Preserved the model dtype in streaming recurrent state
- Restored batch and streaming parity for convolution width one
- Included all public dataset adapters in reproducible source distributions

### Added
- Registered research protocol covering claims, benchmarks, baselines, and statistics
- Temporal-order and attention-window regression tests
- Local delayed-recall benchmark with validation-only checkpoint selection
- Parameter-matched MLP, GRU, LSTM, and causal Transformer benchmark baselines
- Per-seed result schema with bootstrap and interquartile-mean aggregation
- Commit-pinned WSL2 JAX/POBAX environment and GPU smoke validation
- Pure-JAX streaming reference with numerical parity across memory wraparound
- Shared reset-aware JAX PPO learner with causal prior-action/reward inputs
- Shared categorical and diagonal-Gaussian policy distributions for discrete
  and continuous POBAX tasks
- Parameter-matched MLP, frame-stack, memory-trace, Elman, GRU, LSTM, TCN,
  LRU, S4D, S5RL, MS4/MS4N, Transformer, Transformer-XL, and GTrXL policy
  adapters
- Executable ArcMind ablations with instantiated and effective parameter counts
- Primary-source benchmark and state-of-the-art baseline audit
- Project scaffolding and package structure
- Sensor-native tokenizer (linear projection, no vocabulary table)
- Selective SSM core (fast path for continuous sensor streams)
- Tiny exact attention module (slow path for episodic recall)
- Episodic memory ring buffer with learned compression
- Dual-timescale ArcMind model assembling fast and slow paths
- Model configuration presets (IoT-tiny, robotics-small, robotics-medium)
- Test suite with unit tests for all modules
