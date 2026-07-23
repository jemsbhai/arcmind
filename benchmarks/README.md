# ArcMind Benchmarks

This directory contains the local-first evaluation harness for the ICLR
research program. The machine-readable registration is in `protocol.yaml`; the
broader scientific contract is in `docs/research_protocol.md`.

## Delayed sensor recall

The diagnostic task interleaves key-value write and query events with
sensor-rate nuisance frames. A query target is the value most recently written
to its key. Labels occur only at query boundaries, and each query retains its
lag from the relevant write.

This isolates four questions:

1. can a model beat a memoryless policy?
2. does the selective SSM retain the association?
3. does bounded exact recall improve the SSM?
4. does explicit relative age improve exact recall when keys are overwritten?

The task is mechanistic evidence, not a substitute for POBAX.

## Smoke run

```bash
python -m benchmarks.run_delayed_recall \
  --quick \
  --models memoryless_mlp gru lstm causal_transformer \
           arcmind_ssm_only arcmind_unordered arcmind
```

## Calibrated local pilot

```bash
python -m benchmarks.run_delayed_recall \
  --seeds 1103 \
  --epochs 20 \
  --train-examples 1024 \
  --validation-examples 256 \
  --test-examples 512 \
  --device cuda \
  --output-dir benchmark_results/pilot
```

## Registered local diagnostic

Omitting explicit sizes and seeds uses `protocol.yaml`:

```bash
python -m benchmarks.run_delayed_recall \
  --device cuda \
  --output-dir benchmark_results/registered
```

Run long jobs as one model and seed per process so interruption cannot discard
other cells. Every cell writes an independent JSON record.

## Aggregation

```bash
python -m benchmarks.aggregate benchmark_results/registered
```

Aggregation preserves raw record paths and reports mean, median, interquartile
mean, and a percentile-bootstrap 95% interval. A one-seed pilot is useful only
for debugging and calibration; its interval is not inferential evidence.

## Selection and test isolation

Training selects the checkpoint with minimum validation negative
log-likelihood. The selected checkpoint is then evaluated on the fixed test
split exactly once. Result files record the selected epoch and
`test_evaluations: 1`.

Train, validation, and test examples use fixed split seeds shared by every
model. Model initialization and data-loader order use the registered training
seed. Baselines are width-matched to the full ArcMind parameter count within
10%.

## POBAX

POBAX is JAX-native. Accuracy comparisons will use a single JAX PPO learner,
rollout collector, optimizer, and update schedule. The commit-pinned
environment and PyTorch/JAX streaming parity gate are documented in
`benchmarks/pobax/README.md`. The shared discrete/continuous PPO learner and
the parameter-matched MLP, frame-stack, memory-trace, RNN, GRU, LSTM, TCN,
LRU, S4D, S5RL, MS4/MS4N, Transformer, Transformer-XL, GTrXL, and ArcMind
policy cores are implemented. Remaining pre-registration work is the
privileged-observation upper-reference path, equal-budget tuning grid, frozen
seed manifest, and POBAX result aggregation. Separate PyTorch and JAX learners
will not be compared as if they were controlled baselines.
