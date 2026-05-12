# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Project scaffolding and package structure
- Sensor-native tokenizer (linear projection, no vocabulary table)
- Mamba-2 SSM core (fast path for continuous sensor streams)
- Tiny exact attention module (slow path for episodic recall)
- Episodic memory ring buffer with learned compression
- Dual-timescale ArcMind model assembling fast and slow paths
- Model configuration presets (IoT-tiny, robotics-small, robotics-medium)
- Test suite with unit tests for all modules
