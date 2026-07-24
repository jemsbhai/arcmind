# ArcMind benchmark and baseline audit

Status: living primary-source audit, updated 2026-07-23.

This document separates like-for-like backbone controls from methods that
change the learner, supervision, observation privileges, or task. It is not a
claim that ArcMind exceeds any listed method.

## Benchmark selection

### Primary: POBAX

The published
[POBAX paper](https://rlj.cs.umass.edu/2025/papers/RLJ_RLC_2025_153.pdf)
defines the primary RL suite. Its selection criterion is directly aligned with
the ArcMind hypothesis: environments should span distinct forms of partial
observability and exhibit a measurable gap between memoryless and
better-informed agents. The official source is
[pinned to commit
`a5e1d62d14e4efe783885b9d4f19cffa2a568eec`](https://github.com/taodav/pobax/tree/a5e1d62d14e4efe783885b9d4f19cffa2a568eec).

The registered subset and published reference protocol are:

| Source environment identifier | Partial-observability category | Published training steps | Tuning seeds | Final seeds |
|---|---|---:|---:|---:|
| `tmaze_10` | object uncertainty and tracking | 1,000,000 | 5 | 30 |
| `rocksample_11_11` | object uncertainty | 5,000,000 | 5 | 30 |
| `battleship_10` | spatial uncertainty and episode nonstationarity | 10,000,000 | 10 | 30 |
| `Walker-V-v0` | moment features | 50,000,000 | 5 | 30 |
| `HalfCheetah-V-v0` | moment features | 50,000,000 | 5 | 30 |
| `Navix-DMLab-Maze-01-v0` | spatial uncertainty and episode nonstationarity | 10,000,000 | 5 | 30 |

`HalfCheetah-P-v0` exists in the library but is not one of the masked MuJoCo
tasks validated for the published POBAX benchmark. Battleship uses an action
mask, but action masking is not a category of partial observability in the
paper. Simple Chain remains an infrastructure check because its single action
cannot discriminate policy quality.

The pinned
[masked-environment map](https://github.com/taodav/pobax/blob/a5e1d62d14e4efe783885b9d4f19cffa2a568eec/pobax/envs/__init__.py#L49-L86)
defines Walker-P as dimensions 0 through 7, Walker-V as dimensions 8 through
16, and Walker-F as all dimensions 0 through 16. Their observation widths are
therefore 8, 9, and 17. All three use the same Walker2d dynamics, a continuous
6-value action bounded to `[-1, 1]`, and a 1,000-step episode limit
([wrapper](https://github.com/taodav/pobax/blob/a5e1d62d14e4efe783885b9d4f19cffa2a568eec/pobax/envs/wrappers/gymnax.py#L242-L273)).
The published upper-reference pair is Walker-V with Walker-F
([recurrent task](https://github.com/taodav/pobax/blob/a5e1d62d14e4efe783885b9d4f19cffa2a568eec/scripts/hyperparams/masked_mujoco/walker_v_ppo.py#L15-L25),
[full-state reference](https://github.com/taodav/pobax/blob/a5e1d62d14e4efe783885b9d4f19cffa2a568eec/scripts/hyperparams/masked_mujoco/walker_v_ppo_perfect_memory_memoryless.py#L16-L29)).
Under the common causal input contract, adding previous action, reward, and
reset gives Walker-V 17 policy-input values and Walker-F 25. HalfCheetah-V
selects 8 observation values and is paired with the 17-value HalfCheetah-F
environment
([recurrent task](https://github.com/taodav/pobax/blob/a5e1d62d14e4efe783885b9d4f19cffa2a568eec/scripts/hyperparams/masked_mujoco/halfcheetah_v_ppo.py#L15-L25),
[full-state reference](https://github.com/taodav/pobax/blob/a5e1d62d14e4efe783885b9d4f19cffa2a568eec/scripts/hyperparams/masked_mujoco/halfcheetah_v_ppo_perfect_memory_memoryless.py#L16-L29)).

### Secondary and diagnostic suites

- [POPGym](https://arxiv.org/abs/2303.01859), with
  [official source at commit
  `410d5aa626dae8024f498354d8781a0d1870c399`](https://github.com/proroklab/popgym/tree/410d5aa626dae8024f498354d8781a0d1870c399),
  supplies controlled memory primitives and scalable memory-length sweeps.
  Its positional MLP control guards against crediting memory for position
  inference. POBAX itself does not supply this control.
- [Memory Gym](https://www.jmlr.org/papers/v26/24-0043.html) supplies finite
  and endless 2D memory tasks with official GRU and Transformer-XL references.
  Its [official source is commit
  `a94f2b60d1769ea44df3226561488768e1dff9f4`](https://github.com/MarcoMeter/endless-memory-gym/tree/a94f2b60d1769ea44df3226561488768e1dff9f4).
  It is the preferred first external stress suite.
- [Memory Maze](https://openreview.net/pdf?id=yHLvIlE9RGN) is a 3D
  embodied-memory stretch test after the low-dimensional policy-backbone claim
  is established. Its [official source is commit
  `4030901cef3b7d4e7f92e62099e8be378303dc0a`](https://github.com/jurgisp/memory-maze/tree/4030901cef3b7d4e7f92e62099e8be378303dc0a).
- [POPGym Arcade](https://arxiv.org/abs/2503.01450) supplies
  hardware-accelerated pixel tasks with paired fully and partially observable
  variants. Its observability-gap and memory-contamination analyses make it a
  useful later stress suite, but its visual input and counterfactual analysis
  change the question relative to the primary low-dimensional POBAX
  comparison. The reproducible package release is pinned by its PyPI
  0.0.7 attestation to
  [source commit
  `d061b611718ae55d095791b4ea7046b5266cafd4`](https://github.com/bolt-research/popgym-arcade/tree/d061b611718ae55d095791b4ea7046b5266cafd4).
- The 2026 [memory-rewriting benchmark](https://arxiv.org/abs/2601.15086)
  tests whether a memory can replace stale information rather than merely
  retain it. It is a useful stress test, but remains a preprint and should not
  displace POBAX as the primary suite.
- State-based [RoboMimic](https://proceedings.mlr.press/v164/mandlekar22a.html)
  evaluates the architecture as a behavior-cloning backbone without adding a
  vision-encoder confound.
- [MONSTER](https://arxiv.org/abs/2502.15122) and the official
  [UEA archive](https://timeseriesclassification.com/) evaluate multivariate
  sensor classification as a separate diagnostic track.

## Like-for-like policy-backbone controls

Every implemented control below receives the same causal input (current
observation, previous action, previous reward, and reset flag), actor-critic
heads, collection path, PPO loss, optimizer, update schedule, environment
seeds, and evaluation path. Width is chosen by exact scalar parameter count.

| Family | Required control | Implementation status | Reason |
|---|---|---|---|
| no learned memory | MLP, positional MLP, four-frame MLP | MLP and frame stack implemented; positional MLP required | Establishes observation, timestep-aware, and short-window floors. POPGym found the positional control important enough to recommend in future memory comparisons. |
| conventional recurrence | Elman RNN, GRU, LSTM | implemented | Strong recurrent RL is mandatory: careful recurrent model-free RL matched or exceeded specialized methods on 18/21 environments in the [ICML 2022 study](https://proceedings.mlr.press/v162/ni22a.html). |
| RL-specific fading memory | Fast and Forgetful Memory | required | [FFM, NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/e3bf2f0f10774c474de22a12cb060e2c-Abstract-Conference.html) is a direct drop-in RL memory baseline evaluated on POPGym. |
| learned matrix memory | Stable Hadamard Memory | required | [SHM, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/b6446566965fa38e183650728ab70318-Paper-Conference.pdf) is a direct learned write, calibration, and read baseline evaluated against GRU and FFM on POPGym. |
| finite convolution | causal dilated TCN | implemented | Separates finite receptive field from indefinite recurrence. |
| fixed fading memory | memory traces, official and shared lanes | shared adaptation implemented; official lane required | The [ICML 2025 method](https://proceedings.mlr.press/v267/eberhard25a.html) uses compact exponential moving averages. The official source is pinned at [`fcfdacc0b0a06dc181b49b9ef95893dbae7f2bcd`](https://github.com/onnoeberhard/memory-traces/tree/fcfdacc0b0a06dc181b49b9ef95893dbae7f2bcd). The existing core preserves the recurrence but traces the shared augmented policy input and uses a shared parameter-matched trunk. It must be labeled `memory_trace_shared`. A separate observation-only, separate-head source lane is required. |
| stable linear recurrence | LRU | implemented | [ICML 2024](https://proceedings.mlr.press/v235/lu24h.html) identifies the Deep Linear Recurrent Unit as a strong alternative to Transformers in POMDPs; its stable parameterization comes from the [LRU paper](https://proceedings.mlr.press/v202/orvieto23a.html). The official POMDP implementation is pinned at [`e74ac70d17fde88b0f005f2aba7ac780a5b1b143`](https://github.com/CTP314/TFPORL/tree/e74ac70d17fde88b0f005f2aba7ac780a5b1b143). |
| structured SSM | S4D | implemented | Tests stable input-invariant diagonal dynamics. S4D follows the [paper](https://arxiv.org/abs/2206.11893) and [official implementation](https://github.com/state-spaces/s4). |
| RL-specific structured SSM | S5RL | implemented | S5 is a multi-input/multi-output SSM with an [official JAX implementation](https://github.com/lindermanlab/S5); [S5RL](https://proceedings.neurips.cc/paper_files/paper/2023/hash/92d3d2a9801211ca3693ccb2faa1316f-Abstract-Conference.html) specifically adapts it to reset-aware RL. Its current official repository is pinned at [`12e5d42be3a6bde81cce4234f8be4e119e4318b6`](https://github.com/luchris429/popjaxrl/tree/12e5d42be3a6bde81cce4234f8be4e119e4318b6). |
| input-selective state-space model | Mamba1 | source-audited core implemented | [Mamba](https://openreview.net/forum?id=tEYskw1VY2) provides the modern input-selective state-space control. The implemented source contract is pinned at [`10b5d6358f27966f6a40e4bf0baa17a460688128`](https://github.com/state-spaces/mamba/tree/10b5d6358f27966f6a40e4bf0baa17a460688128). Mamba-2 and Mamba-3 remain contextual until they have matched online-PPO evidence. |
| constant-state gated linear attention | AGaLiTe | required | [AGaLiTe](https://openreview.net/forum?id=lh6vOAHuvo) is an accepted TMLR method for partially observable online RL with PureJaxRL PPO and a constant-state approximate gated linear-attention core. Its official JAX/Flax implementation is pinned at [`101acbecc121a258ad8f7e58e2f782f546674979`](https://github.com/subho406/agalite/tree/101acbecc121a258ad8f7e58e2f782f546674979). It is a mandatory modern executable baseline. |
| exact attention | full-window causal Transformer | implemented | Provides the high-cost exact-context reference. |
| recurrent attention | Transformer-XL, GTrXL | implemented | Tests segment recurrence and the GRU-gated, identity-map architecture introduced by [GTrXL](https://proceedings.mlr.press/v119/parisotto20a.html). |
| proposed hybrid | ArcMind plus five registered ablations | implemented | Isolates the fast path, exact memory, temporal ordering, fusion gate, and full hybrid. |

### Memory Traces source contract

The source audit uses the official
[ICML 2025 proceedings paper](https://proceedings.mlr.press/v267/eberhard25a.html)
and the MIT-licensed
[repository at commit
`fcfdacc0b0a06dc181b49b9ef95893dbae7f2bcd`](https://github.com/onnoeberhard/memory-traces/tree/fcfdacc0b0a06dc181b49b9ef95893dbae7f2bcd).
The executable trace and license bytes at that revision match the conference
snapshot `a565a2e3cf8710439e58e4051dffc452ef2896cb`.

For fixed decay \(\lambda_j\), the PPO implementation updates
\(z_t^{(j)}=\lambda_j z_{t-1}^{(j)}+(1-\lambda_j)y_t\), then concatenates
the trace states in trace-major order. The trace has no learned parameters,
gate, bias, activation, or normalization. The official T-Maze policy traces
observations only. It resets each trace at an episode boundary and immediately
incorporates the new initial observation. Its actor and critic are separate
two-layer, width-64 tanh networks with orthogonal initialization. This is the
contract for `memory_trace_official`.

The existing ArcMind core instead traces the complete shared causal input,
including previous action, previous reward, and reset metadata. It uses a
shared parameter-matched trunk and Xavier initialization. The recurrence is
correct, but the complete policy is an adaptation. Registered results must
call it `memory_trace_shared`, not an official reproduction.

The paper defines the T-Maze decay as \((k-1)/k\). Its public corridor-64
example rounds this value to `0.985`, while the exact value is `0.984375`.
The paper reports 1,024,000,000 training steps for its main T-Maze study,
whereas the public example uses 20,480,000 steps. No official POBAX decays or
parameter-matching rule exist. POBAX decay values therefore require a frozen
tuning registration and cannot be described as author-selected settings.

### AGaLiTe source contract

The source audit uses the
[TMLR OpenReview paper](https://openreview.net/forum?id=lh6vOAHuvo), its
[final author manuscript](https://arxiv.org/abs/2310.15719), and the
Apache-2.0
[repository at commit
`101acbecc121a258ad8f7e58e2f782f546674979`](https://github.com/subho406/agalite/tree/101acbecc121a258ad8f7e58e2f782f546674979).
The paper and executable use different finite-channel algorithms, so they
cannot share one unqualified implementation label.

The released recurrence stores exactly `R` cosine channels with frequencies
`linspace(-pi, pi, R)`. Its phase counter starts at one, so the first token
uses phase two, and an episode reset clears prior memory without resetting the
phase. The released readout divides by
`2 * R * dot(s, q) + 1e-5`. The surrounding block uses pre-attention
LayerNorm, the AGaLiTe attention update, two GTrXL-style GRU residual gates,
and a two-layer feedforward block.

The paper instead defines `r + 1` channels at frequencies
`2 * pi * i / r`. Its operational readout and one appendix derivation differ
by a factor of four. Published experiments report `r = 1`, while released
configurations use `R = 2`. Registered code must therefore use separate names:
`agalite_source_compat` for the pinned executable, `agalite_shared` for the
parameter-matched shared-PPO port, and an optional paper-equation audit
ablation. A close source port must retain Apache-2.0 attribution and identify
all modifications.

### Positional MLP source contract

POBAX contains no positional MLP. Its
[`backend="positional"` argument](https://github.com/taodav/pobax/blob/a5e1d62d14e4efe783885b9d4f19cffa2a568eec/pobax/envs/wrappers/gymnax.py#L242-L273)
selects a Brax physics backend and does not add position or time to the policy
input. The required baseline is instead a shared-head JAX adaptation of the
pinned POPGym positional MLP.

In the official POPGym source, `MLP` enables `embedding="sine"` and
`BasicMLP` disables it
([definitions](https://github.com/proroklab/popgym/blob/410d5aa626dae8024f498354d8781a0d1870c399/popgym/baselines/ray_models/ray_mlp.py#L10-L55)).
For episode limit \(L\), feature width \(d\), position \(t\), and pair index
\(i\), the fixed encoding is
\(\sin(t\exp(-\log(L)2i/d))\) and
\(\cos(t\exp(-\log(L)2i/d))\)
([implementation](https://github.com/proroklab/popgym/blob/410d5aa626dae8024f498354d8781a0d1870c399/popgym/baselines/models/embeddings.py#L7-L17)).
POPGym creates it at the hidden feature width and blends it with the feature
vector as \((1-\alpha)x+\alpha PE\), where the learned scalar \(\alpha\)
starts at 0.5 and is clipped to `[0, 1]`
([setup](https://github.com/proroklab/popgym/blob/410d5aa626dae8024f498354d8781a0d1870c399/popgym/baselines/ray_models/base_model.py#L121-L160),
[blend](https://github.com/proroklab/popgym/blob/410d5aa626dae8024f498354d8781a0d1870c399/popgym/baselines/ray_models/base_model.py#L240-L276)).

The JAX adaptation preserves this encoding and learned blend, with one
reset-aware `int32` episode counter per vector worker. It retains the shared
causal input, actor, critic, PPO implementation, and parameter-matching
procedure, and counts \(\alpha\) as a parameter. It is not a literal
reproduction of POPGym's RLlib preprocessing stack, and a scalar normalized
timestep is not an acceptable substitute. Its tests cover independent resets,
step versus scan parity, a pinned-formula fixture, distinct outputs for equal
observations at distinct positions, absence of observation history in state,
and inclusion of \(\alpha\) in parameter matching. POPGym's source
tracks elapsed episode position as recurrent state
([counter path](https://github.com/proroklab/popgym/blob/410d5aa626dae8024f498354d8781a0d1870c399/popgym/baselines/ray_models/base_model.py#L176-L179),
[advance path](https://github.com/proroklab/popgym/blob/410d5aa626dae8024f498354d8781a0d1870c399/popgym/baselines/ray_models/base_model.py#L277-L317)).

### FFM and SHM shared-learner compatibility

The [official FFM repository at commit
`b3f94d2a0f35ba05089faf19ab1df846057cf8b6`](https://github.com/proroklab/ffm/tree/b3f94d2a0f35ba05089faf19ab1df846057cf8b6)
contains a JAX implementation whose call accepts the input sequence, recurrent
state, and episode-done flags. FFM can therefore be adapted as a policy core in
the shared JAX PPO without changing the learner, loss, collector, or optimizer.
The implemented core tests parameters, initialization, asynchronous resets,
sequence outputs, recurrent state, and gradients.

The [official SHM `v1.1` source at commit
`40d73d44936e47a29e2c76a481d93c434b857ea1`](https://github.com/thaihungle/SHM/tree/40d73d44936e47a29e2c76a481d93c434b857ea1)
is PyTorch. The released memory module uses ordinary differentiable recurrence
and does not require an auxiliary RL loss. The JAX policy core exposes initial
memory, asynchronous reset masks, explicit random addresses, and a time-major
scan. Fixed-address tests match an independent translation of the released
equations. Distributional tests cover address sampling, and PPO stores and
replays each collection-time address during loss recomputation.

The SHM audit found two source-level incompatibilities that must remain
explicit. First, the paper and the official POMDP implementation sample
uniformly from all 128 address rows, but the standalone and POPGym `v1.1`
implementations call `uniform_(0, 1).long()`. Values in that interval truncate
to zero, so those paths always select address row zero. Second, the
[POMDP implementation clamps the recurrent matrix to `[-100, 100]` after each
write](https://github.com/thaihungle/SHM/blob/40d73d44936e47a29e2c76a481d93c434b857ea1/pomdp-baselines/torchkit/shm.py#L68-L75),
while the
[POPGym path leaves the recurrence unclamped](https://github.com/thaihungle/SHM/blob/40d73d44936e47a29e2c76a481d93c434b857ea1/popgym/baselines/ray_models/ray_shm.py#L83-L87).
The paper notes that cumulative products can occasionally overflow and names
gradient clipping as a remedy, but its method equations do not specify the
forward-state clamp.

The JAX port therefore has two named source modes. `paper_uniform` uses the
paper's uniform addressing and the official POMDP cell's recurrent-state
clamp. `v1_1_popgym_compat` preserves the released POPGym behavior, including
row-zero addressing and an unclamped recurrence. The main shared-learner
baseline should use `paper_uniform`. It is a source-audited scientific
baseline, not an exact reproduction of the published POPGym agent.

The implemented memory-trace adapter applies the fixed exponential averages to
the common causal policy input rather than the original paper's
environment-specific one-hot observation encoder. The implemented MS4/MS4N
replaces sequence-classification pooling with causal actor-critic heads. The
S5RL core preserves the authors' HiPPO-LegS DPLR initialization, ZOH
discretization, conjugate-symmetric recurrence, reset behavior, and full-GLU
residual block, but uses the common direct input projection and actor-critic
heads. These differences must be stated in the paper and checked against
author-code results where possible.

POBAX labels its attention agent Transformer-XL. The pinned repository enables
GRU-style gates in the transformer configuration by default, while GTrXL is a
specific architecture from the
[ICML 2020 paper](https://proceedings.mlr.press/v119/parisotto20a.html).
ArcMind must record the actual gating and normalization settings. The main text
should use Transformer-XL for a POBAX author configuration unless those
settings verify that it is exactly GTrXL.

## Upper-reference taxonomy

POBAX's `perfect_memory` flag does not have one consistent semantic meaning.
The
[pinned environment factory](https://github.com/taodav/pobax/blob/a5e1d62d14e4efe783885b9d4f19cffa2a568eec/pobax/envs/__init__.py#L171-L319)
maps it to a full observation in some tasks, a persistent cue or accumulated
history in others, and a broken wrapper in Simple Chain. Each paper artifact
must record the reference class instead of grouping all such runs under
"perfect memory."

| Primary task | Upper invocation | Runtime observation | Reference class | Eligibility |
|---|---|---:|---|---|
| `Walker-V-v0` | `Walker-F-v0` | 17 floats | full Markov observation | public upper reference |
| `HalfCheetah-V-v0` | `HalfCheetah-F-v0` | 17 floats | full Markov observation | public upper reference |
| `rocksample_11_11` | same identifier, `perfect_memory=True` | 33 floats | full Markov observation with true rock morality | public upper reference |
| `Navix-DMLab-Maze-01-v0` | `Navix-DMLab-Maze-F-01-v0` | `(21, 41, 6)`, or 5,166 flattened values | full Markov map, position, and direction | public upper reference after matched adapter |
| `tmaze_10` | same identifier, `perfect_memory=True` | 4 floats | persistent-cue upper reference | public upper reference |
| `battleship_10` | same identifier, `perfect_memory=True` | raw `(10, 10)` integer history | perfect-recall history | requires corrected shared-learner adapter |
| `simple_chain` | same identifier, `perfect_memory=True` | reset fails | intended full state, broken upstream | excluded |

RockSample's wrapper exposes one-hot position and true rock morality
([source](https://github.com/taodav/pobax/blob/a5e1d62d14e4efe783885b9d4f19cffa2a568eec/pobax/envs/jax/rocksample.py#L93-L125)).
The Navix full task exposes a position-encoded full map, agent position, and
direction, with the same 2,000-step maze specification as its partial pair
([representation](https://github.com/taodav/pobax/blob/a5e1d62d14e4efe783885b9d4f19cffa2a568eec/pobax/envs/jax/navix_mazes.py#L295-L355),
[registrations](https://github.com/taodav/pobax/blob/a5e1d62d14e4efe783885b9d4f19cffa2a568eec/pobax/envs/jax/navix_mazes.py#L413-L464)).
T-Maze retains the goal cue at every step but does not expose the complete
latent state
([source](https://github.com/taodav/pobax/blob/a5e1d62d14e4efe783885b9d4f19cffa2a568eec/pobax/envs/jax/tmaze.py#L12-L65)).
Battleship retains every hit and miss but does not reveal the hidden ship board
([state and wrapper](https://github.com/taodav/pobax/blob/a5e1d62d14e4efe783885b9d4f19cffa2a568eec/pobax/envs/jax/battleship.py#L15-L99)).
Simple Chain's intended full wrapper calls nonexistent instance state, so it
cannot be used at the pinned commit
([source](https://github.com/taodav/pobax/blob/a5e1d62d14e4efe783885b9d4f19cffa2a568eec/pobax/envs/jax/simple_chain.py#L77-L89)).

These methods receive information unavailable to the primary deployed policy.
They are diagnostic upper references, not like-for-like baselines. They are
also not guaranteed empirical ceilings: a trained policy can underperform
despite receiving more information because the input representation and
finite optimization problem have changed. Keep them out of the
parameter-matched backbone table and report them in a separately labeled
reference table.

The implemented shared-runner aliases are
`tmaze_10-perfect-memory`, `rocksample_11_11-fully-observable`, and
`Navix-DMLab-Maze-01-fully-observable`, together with
`battleship_10-perfect-recall`. They hash the exact source invocation, restrict
the model to the matched memoryless policy, and inherit the corresponding
primary interaction budget. Both primary and upper RockSample observations
have 33 values at runtime. The upper reference changes the meaning of the
rock features to true morality rather than increasing input width. The Navix
alias invokes the separately registered full source task.

The Battleship adapter corrects two interface hazards. Its public
perfect-recall wrapper returns an array rather than the ordinary observation
and action-mask dictionary, so the adapter preserves the `(10, 10)` history
and reconstructs a row-major legal mask from `hits_misses == 0`. The
unregistered state wrapper is not a shortcut:
it declares `(rows, cols, 2)`, returns a `(2, rows, cols)` stack, and also
drops the action mask
([source](https://github.com/taodav/pobax/blob/a5e1d62d14e4efe783885b9d4f19cffa2a568eec/pobax/envs/jax/battleship.py#L67-L205)).
A future true privileged-state variant would still need an explicitly
corrected local wrapper exposing hit history and the hidden board. It would be
classified separately from POBAX's public perfect-recall reference.

The Navix partial task produces a `(3, 3, 2)` runtime observation, or 18
values, while the full reference has 5,166 values
([partial field-of-view wrapper](https://github.com/taodav/pobax/blob/a5e1d62d14e4efe783885b9d4f19cffa2a568eec/pobax/envs/wrappers/nx.py#L59-L132),
[full representation](https://github.com/taodav/pobax/blob/a5e1d62d14e4efe783885b9d4f19cffa2a568eec/pobax/envs/jax/navix_mazes.py#L350-L464)).
The shared input also contains the previous action and two boundary flags, so
the primary and full policy widths are 23 and 5,171. The full observation is
flattened into a memoryless MLP with hidden width six. Its target is the
primary partial-task ArcMind parameter budget, not a larger budget induced by
the privileged input width, and every learned parameter is counted.

## Algorithm-level and contextual comparisons

These methods cannot be placed in the main parameter-matched backbone table
without changing the experimental question:

- [Lambda discrepancy](https://proceedings.neurips.cc/paper_files/paper/2024/hash/73073ccb3bc559fd001e66b9079d6d5e-Abstract-Conference.html)
  adds a second critic and an auxiliary discrepancy objective. Report the
  POBAX author result in a separate algorithm-level table.
- [Memory Traces](https://proceedings.mlr.press/v267/eberhard25a.html) can be
  used as a fixed policy representation in the shared learner, while its
  author-code result remains a separate compatibility reproduction.
- [Memoroids](https://proceedings.neurips.cc/paper_files/paper/2024/hash/19f7f755908372efb25826d61959cdf9-Abstract-Conference.html)
  changes batching and the training formulation for associative recurrences.
  It is a training-method comparison, not a pure backbone.
- [Recall to Imagine](https://recall2imagine.github.io/) adds a world model
  and imagined rollouts.
- [Memo](https://proceedings.neurips.cc/paper_files/paper/2025/hash/96889893231d651898b0de42fdbee3a6-Abstract-Conference.html)
  targets learned memory management for embodied agents at a different system
  scale. Its official source is
  [pinned at `6ec28e0b35ebc01749753371629e8842caaae816`](https://github.com/gunshi/memo/tree/6ec28e0b35ebc01749753371629e8842caaae816).
- [RATE](https://openreview.net/forum?id=kByN4v0M3e) evaluates recurrent
  attention in offline RL over stored trajectories, not the shared online PPO
  regime.
- [ELMUR](https://arxiv.org/abs/2510.07151) is an offline,
  Decision-Transformer-style ICLR 2026 poster, not an online PPO backbone.
- [GPO](https://arxiv.org/abs/2505.15418) uses privileged full-state guidance
  during training. It is an algorithm-level contextual comparison.
- [Mamba-2](https://proceedings.mlr.press/v235/dao24a.html) and
  [Mamba-3](https://openreview.net/forum?id=HwCvaJOiCj) are current sequence
  models, but neither is an established matched online-PPO baseline.
- [LinOSS](https://openreview.net/forum?id=GRMfXcAAFh) is an ICLR 2025 oral
  supervised sequence model. Its RL evidence comes through a different sensor
  dropout study, so it is an optional linear-memory control.
- The [Kalman state-space layer](https://openreview.net/forum?id=rfPns0WJyg)
  is a TMLR 2025 probabilistic recurrent layer evaluated with off-policy
  actor-critic methods. No official source was located. It becomes mandatory
  only if ArcMind makes sensor-noise or uncertainty claims.
- [MS4 and MS4N](https://arxiv.org/abs/2605.27406) are supervised
  multivariate time-series models. Their current ArcMind cores are causal
  policy adaptations, not established RL baselines.
- [When Sensors Fail](https://arxiv.org/abs/2603.04648), an ICLR 2026 CAO
  workshop poster, studies persistent sensor dropout rather than POBAX. Its
  Transformer, GTrXL, GRU, LRU, and LinOSS comparison reinforces the need for
  exact-attention and stable-recurrence controls, but its published returns are
  not directly comparable to the primary task suite.
- Privileged-state critics, expert distillation, and asymmetric methods use
  information unavailable to the deployed policy. They are upper references
  or training-regime comparisons, not pure memory-backbone baselines.

The 2026 preprint
[Why Linear Recurrent Memory Works in Partially Observable RL](https://arxiv.org/abs/2605.31261)
provides additional motivation for testing stable linear filters, but it is not
peer-reviewed evidence and does not replace the LRU, S5RL, GRU, or LSTM
comparisons.

## Internal research-folder review

The four internal reports on subquadratic attention are language-model
analyses, not sensor-control studies. They are retained as hypothesis-generating
material only. Three ideas transfer plausibly to ArcMind and therefore receive
direct controls: fixed-state recurrence trades exact retrieval for compression,
a small exact path may repair recall failures, and regular bounded access
patterns are more likely to yield practical speedups than irregular dynamic
sparsity.

The reports do **not** justify importing language-model context thresholds,
FlashAttention or KV-cache speedups, million-token claims, mobile deployment
claims, or absolute statements about Mamba/Transformer recall into this paper.
They also do not establish that the hybrid principle transfers to POMDP
control. That transfer is the falsifiable question tested by the SSM-only,
attention-only, full-hybrid, and memory-ordering ablations. Primary papers and
author code, rather than the internal reports, determine every baseline
equation and empirical comparison.

## Registration implications

There is no defensible single "state of the art" number across these suites:
methods differ in observations, learner, privileged information, interaction
budget, and evaluation protocol. The paper must therefore lead with paired
within-protocol comparisons and uncertainty, then place published author-code
numbers in explicitly non-comparable context tables.

Before registered POBAX runs:

1. implement and parity-test the positional MLP, FFM, SHM, and privileged
   observation adapters;
2. reproduce a small author-code FFM or SHM reference cell and an S5RL or
   memory-trace reference cell;
3. classify every artifact as smoke, pilot, preliminary, or registered final;
4. treat quick and 131,072-step cells as infrastructure evidence only;
5. use the published interaction budget for every method in each registered
   task cell;
6. target 30 paired final seeds, matching the published POBAX result count,
   and label any 10-seed table preliminary;
7. freeze one small tuning grid with the same trial count per architecture;
8. give memory traces a budget capable of reproducing their slow-learning
   behavior, or label the comparison budget-limited;
9. run fully observable or privileged references where POBAX supplies them;
10. freeze seed manifests and raw-result schemas before evaluating final
    seeds;
11. retain throughput, latency, state size, and peak-memory measurements; and
12. reserve the USD 10 cloud allowance for an independent environment or
    author-code reproduction check, using the local RTX 4090 Laptop GPU for
    development and registered runs.
