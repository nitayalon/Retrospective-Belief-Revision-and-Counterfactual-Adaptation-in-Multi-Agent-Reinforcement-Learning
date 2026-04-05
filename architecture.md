# HIM+HER: Codebase Specification
## Hindsight Inconsistency Mitigation + Hindsight Experience Replay for Multi-Agent RL

> **Implementation note:** This codebase uses **JAX** for all compute (networks, likelihood,
> Bellman updates, HER relabeling) and **NumPy** for all stateful operations outside the
> JIT boundary (replay buffer storage, HIM relabeling, environment stepping). This boundary
> is enforced throughout. See Section 5 for the design rationale.

---

## 1. Project Overview

This document specifies the full architecture, dependencies, directory structure, and
implementation requirements for the HIM+HER codebase. The system implements a two-phase
hindsight-based framework for ego agents in multi-agent environments: (1) **HIM** —
retrospective revision of the ego's model of other agents when observed behavior is
inconsistent with predictions, and (2) **HER** — goal relabeling under the updated model
for off-policy policy improvement.

The codebase is designed to support four benchmark environments (Predator-Prey, Cooperative
Navigation, Intersection Negotiation, Hide-and-Seek), four experimental baselines, and clean
ablations isolating HIM from HER.

---

## 2. Repository Structure

```
him_her/
├── README.md
├── setup.py
├── requirements.txt
├── configs/                        # YAML experiment configs
│   ├── predator_prey.yaml
│   ├── cooperative_nav.yaml
│   ├── intersection.yaml
│   ├── hide_and_seek.yaml
│   └── defaults.yaml
│
├── him_her/                        # Main package
│   ├── __init__.py
│   │
│   ├── agents/                     # Ego agent implementations
│   │   ├── __init__.py
│   │   ├── base_agent.py           # Abstract ego agent
│   │   ├── him_her_agent.py        # Full HIM+HER agent
│   │   ├── him_only_agent.py       # HIM ablation (no HER)
│   │   ├── her_only_agent.py       # HER ablation (no HIM)
│   │   └── baseline_agent.py       # Vanilla MARL, static model, Bayesian baselines
│   │
│   ├── models/                     # Other-agent model representations
│   │   ├── __init__.py
│   │   ├── base_model.py           # Abstract model class + dataclasses
│   │   ├── reward_model.py         # Latent reward function model
│   │   ├── policy_model.py         # Explicit policy model of other
│   │   ├── type_model.py           # Discrete type model (for intersection task)
│   │   └── model_set.py            # Hypothesis set M, prior p(m), likelihood computation
│   │
│   ├── him/                        # HIM module
│   │   ├── __init__.py
│   │   ├── inconsistency.py        # JAX likelihood computation, threshold detection
│   │   ├── model_revision.py       # MAP model selection (JAX); trajectory relabeling (NumPy)
│   │   └── belief_updater.py       # Belief distribution over M
│   │
│   ├── her/                        # HER module
│   │   ├── __init__.py
│   │   ├── goal_sampler.py         # Goal relabeling strategies (future, episode, random)
│   │   ├── reward_relabeler.py     # Recomputes r_e(s, a, g_tilde, m_tilde)
│   │   └── her_buffer.py           # HER-aware replay buffer extension
│   │
│   ├── replay/                     # Replay buffer
│   │   ├── __init__.py
│   │   └── replay_buffer.py        # NumPy-backed buffer; stores (s, a_e, a_o, s', g, m)
│   │
│   ├── networks/                   # Flax neural network components
│   │   ├── __init__.py
│   │   ├── actor.py                # Flax Linen: goal- and model-conditioned policy
│   │   ├── critic.py               # Flax Linen: Q(s, a, g, m) value network
│   │   └── encoder.py              # Optional: encode model m as embedding
│   │
│   ├── training/                   # Training loop
│   │   ├── __init__.py
│   │   ├── train_state.py          # NEW: HIMHERTrainState pytree + create_train_state()
│   │   ├── trainer.py              # Hybrid training loop (JAX update step + NumPy episode loop)
│   │   └── evaluator.py            # Evaluation and metric logging
│   │
│   ├── envs/                       # Environment wrappers
│   │   ├── __init__.py
│   │   ├── base_env.py             # Abstract multi-agent environment wrapper
│   │   ├── predator_prey.py        # MAgent/PettingZoo pursuit wrapper
│   │   ├── cooperative_nav.py      # MPE simple_spread wrapper
│   │   ├── intersection.py         # highway-env intersection-v0 wrapper
│   │   └── hide_and_seek.py        # Hide-and-seek wrapper
│   │
│   ├── other_agents/               # Other-agent (non-ego) implementations
│   │   ├── __init__.py
│   │   ├── scripted_other.py       # Deterministic scripted agents (for controlled experiments)
│   │   └── type_agents.py          # Typed agents: aggressive, cautious, reciprocal, etc.
│   │
│   └── utils/
│       ├── __init__.py
│       ├── config.py               # Config loading and validation
│       ├── logging.py              # WandB / TensorBoard integration
│       ├── seeding.py              # Reproducibility: JAX PRNGKey + numpy seeds
│       └── math_utils.py           # Log-sum-exp, KL divergence, etc. (JAX)
│
├── scripts/
│   ├── train.py                    # Entry point
│   ├── evaluate.py                 # Run a trained checkpoint, log metrics
│   └── ablation.py                 # Run full ablation sweep across agent types
│
├── tests/
│   ├── test_him.py
│   ├── test_her.py
│   ├── test_replay_buffer.py
│   ├── test_envs.py
│   ├── test_model_set.py
│   └── test_train_state.py
│
└── notebooks/
    ├── visualize_belief_updates.ipynb
    ├── likelihood_curves.ipynb
    └── policy_comparison.ipynb
```

---

## 3. Dependencies

### 3.1 Core (JAX stack — replaces PyTorch)

| Package | Version | Purpose |
|---|---|---|
| `python` | ≥ 3.10 | Language |
| `jax` | ≥ 0.4.25 | Autodiff, JIT, vmap |
| `jaxlib` | ≥ 0.4.25 | XLA backend (use `jaxlib[cuda12]` for GPU) |
| `flax` | ≥ 0.8.0 | Neural network library (Linen API) |
| `optax` | ≥ 0.2.0 | Optimizers (Adam, gradient clipping) |
| `chex` | ≥ 0.1.86 | Shape/type assertions, pytree utilities |
| `orbax-checkpoint` | ≥ 0.5.0 | JAX-native checkpointing of pytrees |
| `numpy` | ≥ 1.24 | Replay buffer, env stepping (outside JAX boundary) |
| `scipy` | ≥ 1.11 | `logsumexp` for belief normalisation |
| `gymnasium` | ≥ 0.29 | Base RL environment API |
| `pettingzoo` | ≥ 1.24 | Multi-agent environments (pursuit, MPE) |
| `hydra-core` | ≥ 1.3 | Config management |
| `omegaconf` | ≥ 2.3 | Config composition |

> **Removed from original spec:** `torch`, `stable-baselines3`.
> All SAC/DDPG logic is implemented directly using Flax + Optax.

### 3.2 Environments

| Package | Version | Purpose |
|---|---|---|
| `pettingzoo[mpe]` | ≥ 1.24 | MPE cooperative navigation |
| `pettingzoo[magent]` | ≥ 1.24 | Predator-prey (pursuit) |
| `highway-env` | ≥ 1.8 | Intersection negotiation |
| `pygame` | ≥ 2.5 | Environment rendering |

### 3.3 Logging and Experiment Tracking

| Package | Version | Purpose |
|---|---|---|
| `wandb` | ≥ 0.16 | Experiment tracking, sweep management |
| `tensorboard` | ≥ 2.14 | Optional local logging |
| `matplotlib` | ≥ 3.8 | Plotting |
| `seaborn` | ≥ 0.13 | Publication-quality figures |

### 3.4 Notes on Environment Setup

**Hide-and-Seek:** OpenAI's original `multi-agent-emergence` codebase requires MuJoCo 2.x
and is not pip-installable. Options:
- Use the `mujoco` Python bindings (≥ 3.0, free) with a re-implementation of the
  hide-and-seek physics environment, or
- Use a lightweight substitute: `pettingzoo.mpe.simple_tag_v3` with a custom hide-and-seek
  scenario (simpler but sufficient for the HIM ablation).

**MAgent/Pursuit:** Install via `pip install pettingzoo[magent]`. Requires `magent2` which
has Linux/macOS wheels; Windows users should use WSL2 or Docker.

**JAX GPU install:** By default `pip install jax` installs the CPU-only build. For GPU:
```bash
pip install -U "jax[cuda12]"
```
Verify with `python -c "import jax; print(jax.devices())"`.

---

## 4. Core Data Structures

All dataclasses use `np.ndarray` (not `jnp.ndarray`) because they live in the NumPy replay
buffer. Conversion to JAX arrays happens once per gradient step via `to_jax()` (see §6.6).

### 4.1 Transition

The fundamental unit stored in the replay buffer:

```python
@dataclass
class Transition:
    state: np.ndarray           # s_t
    ego_action: np.ndarray      # a^e_t
    other_action: np.ndarray    # a^o_t
    next_state: np.ndarray      # s_{t+1}
    goal: np.ndarray            # g (original)
    model_id: int               # index into ModelSet — the assumed model m at time t
    reward: float               # r_e(s_t, a^e_t, g, m) — can be recomputed
    done: bool
```

### 4.2 Model

An element of the hypothesis set $\mathcal{M}$:

```python
@dataclass
class AgentModel:
    model_id: int
    name: str                       # e.g., "obedient", "food_motivated"
    reward_weights: np.ndarray      # Parameterizes r_o^m if using reward model
    policy_params: np.ndarray       # Parameters of pi_o^m — passed to JAX likelihood fn
    prior: float                    # p(m) — prior probability, sums to 1 over M
```

> **Note:** `policy` is no longer a `Callable` field. Instead `policy_params` is a plain
> array that gets passed into a pure JAX function for likelihood computation. This keeps
> `AgentModel` a plain pytree-compatible dataclass with no embedded Python callables.

### 4.3 Episode

```python
@dataclass
class Episode:
    transitions: List[Transition]
    total_reward: float
    model_used: int                 # model_id throughout this episode
    him_triggered: bool             # was HIM applied at end of episode?
    revised_model: Optional[int]    # model_id after HIM, if triggered
```

### 4.4 ModelSet

```python
@dataclass
class ModelSet:
    models: List[AgentModel]
    stacked_policy_params: np.ndarray   # shape (|M|, param_dim) — for vmap over models
    log_priors: np.ndarray              # shape (|M|,) — log p(m) for each model
```

`stacked_policy_params` is pre-computed at init so `all_model_log_likelihoods` (§6.1) can
`vmap` over the model dimension without Python-level iteration.

---

## 5. JAX/NumPy Boundary

This is the single most important architectural constraint in the codebase.

```
┌─────────────────────────────────────────────────────┐
│                   JAX SIDE (jit-compiled)            │
│                                                      │
│  • Likelihood computation (him/inconsistency.py)     │
│  • MAP model selection (him/model_revision.py)       │
│  • Bellman critic update (training/trainer.py)       │
│  • Actor update (training/trainer.py)                │
│  • HER reward recomputation (her/reward_relabeler.py)│
│  • Soft target update                                │
│  • All neural network forward passes                 │
└───────────────────┬─────────────────────────────────┘
                    │  jnp.array(batch) — one conversion per grad step
                    │  int(map_id)      — one conversion per HIM trigger
┌───────────────────▼─────────────────────────────────┐
│                  NUMPY SIDE (plain Python)           │
│                                                      │
│  • env.step(), env.reset()                           │
│  • Replay buffer storage and sampling                │
│  • HIM relabeling (buffer.relabel_episode)           │
│  • Episode collection loop                           │
│  • Checkpoint saving/loading (via orbax)             │
└─────────────────────────────────────────────────────┘
```

**Rules — enforce these in code review:**
1. Never call `jax.jit` on any function that touches `env.step()`.
2. Never store JAX arrays in the replay buffer — convert to NumPy immediately after each step.
3. Never represent the replay buffer as a JAX pytree.
4. Convert a sampled batch to JAX arrays **once** at the start of each gradient step.
5. `current_model_id` must be encoded as an embedding **before** entering `update_step` to
   avoid JAX retracing the function on every HIM trigger (see §15, item 5).

---

## 6. Module Specifications

### 6.1 `him/inconsistency.py`

All functions here are pure JAX — JIT-compiled and vmapped over the model dimension.
The key advantage over PyTorch: likelihoods under all $|\mathcal{M}|$ models are computed
in a single `vmap` call rather than a Python loop.

```python
import jax
import jax.numpy as jnp

def single_step_log_prob(
    policy_params: jnp.ndarray,
    state: jnp.ndarray,
    action: jnp.ndarray
) -> jnp.ndarray:
    """log pi_o^m(a | s) for one timestep. Pure function."""
    logits = model_forward(policy_params, state)    # task-specific, injected
    return jax.nn.log_softmax(logits)[action]


@jax.jit
def trajectory_log_likelihood(
    policy_params: jnp.ndarray,
    states: jnp.ndarray,            # shape (T, obs_dim)
    actions: jnp.ndarray            # shape (T,)
) -> jnp.ndarray:
    """log L(m | tau) = sum_t log pi_o^m(a^o_t | s_t). vmap over timesteps."""
    log_probs = jax.vmap(single_step_log_prob, in_axes=(None, 0, 0))(
        policy_params, states, actions
    )
    return jnp.sum(log_probs)


@jax.jit
def all_model_log_likelihoods(
    stacked_policy_params: jnp.ndarray,     # shape (|M|, param_dim)
    states: jnp.ndarray,
    actions: jnp.ndarray
) -> jnp.ndarray:
    """
    Compute log L(m | tau) for ALL models simultaneously via vmap.
    Returns shape (|M|,).
    """
    return jax.vmap(trajectory_log_likelihood, in_axes=(0, None, None))(
        stacked_policy_params, states, actions
    )


@jax.jit
def is_inconsistent_ratio(
    current_model_id: int,
    all_log_likelihoods: jnp.ndarray,   # shape (|M|,)
    log_priors: jnp.ndarray,            # shape (|M|,)
    ratio_delta: float
) -> jnp.ndarray:
    """
    Trigger HIM if the best alternative model's log-posterior exceeds
    the current model's by more than ratio_delta.
    Preferred over absolute threshold — avoids per-task tuning of epsilon.
    """
    log_posteriors = all_log_likelihoods + log_priors
    best_alternative = jnp.max(
        jnp.where(
            jnp.arange(len(log_priors)) == current_model_id,
            -jnp.inf,
            log_posteriors
        )
    )
    return best_alternative - log_posteriors[current_model_id] > ratio_delta
```

---

### 6.2 `him/model_revision.py`

MAP selection runs in JAX; buffer relabeling is a NumPy mutation outside the JIT boundary.

```python
@jax.jit
def select_map_model(
    stacked_policy_params: jnp.ndarray,
    log_priors: jnp.ndarray,
    states: jnp.ndarray,
    actions: jnp.ndarray
) -> jnp.ndarray:
    """
    m_tilde = argmax_{m in M} [log p(m) + log L(m | tau)]
    Returns scalar index into ModelSet.
    """
    log_likelihoods = all_model_log_likelihoods(stacked_policy_params, states, actions)
    return jnp.argmax(log_likelihoods + log_priors)


def relabel_trajectory_in_buffer(
    buffer: ReplayBuffer,
    episode_start_idx: int,
    episode_length: int,
    new_model_id: int,      # int(jax_scalar) — convert before this call
) -> None:
    """
    NumPy in-place mutation. Called OUTSIDE the JAX jit boundary.
    Rewards are NOT recomputed here — that happens during HER gradient updates.
    """
    buffer.relabel_episode(episode_start_idx, episode_length, new_model_id)
```

---

### 6.3 `him/belief_updater.py`

Used by the Bayesian baseline and the soft-HIM variant. Belief is maintained in NumPy log-space;
JAX is called only for the likelihood computation.

```python
class BeliefUpdater:
    def __init__(self, model_set: ModelSet):
        self.log_belief = model_set.log_priors.copy()   # numpy array

    def update(
        self,
        stacked_policy_params: jnp.ndarray,
        states: jnp.ndarray,
        actions: jnp.ndarray,
    ) -> np.ndarray:
        """
        Bayesian update: log b'(m) = log b(m) + log L(m | tau), then normalise.
        Returns normalised posterior as numpy array.
        """
        log_likelihoods = np.array(
            all_model_log_likelihoods(stacked_policy_params, states, actions)
        )
        log_unnorm = self.log_belief + log_likelihoods
        self.log_belief = log_unnorm - scipy.special.logsumexp(log_unnorm)
        return np.exp(self.log_belief)

    def map_model_id(self) -> int:
        return int(np.argmax(self.log_belief))

    def sample_model_id(self, rng: np.random.Generator) -> int:
        """Thompson sampling — sample from posterior."""
        return int(rng.choice(len(self.log_belief), p=np.exp(self.log_belief)))
```

---

### 6.4 `her/goal_sampler.py`

Pure NumPy — operates on the episode before JAX conversion.

```python
class GoalSampler:
    STRATEGIES = ["future", "episode", "random", "final"]

    def sample_goals(
        self,
        episode: Episode,
        transition_idx: int,
        strategy: str = "future",
        k: int = 4,
        rng: np.random.Generator = None,
    ) -> List[np.ndarray]:
        """
        Returns k relabeled goals as achieved-state observations (numpy arrays).
        "future":  sample from states at t' > t within the same episode.
        "episode": sample from any state in the episode.
        "final":   use the final state of the episode.
        "random":  sample from the full replay buffer.
        """
```

---

### 6.5 `her/reward_relabeler.py`

The reward function must be a pure JAX callable (injected per environment) so it can be
called inside the JIT-compiled `update_step`.

```python
def relabel_reward(
    state: jnp.ndarray,
    ego_action: jnp.ndarray,
    new_goal: jnp.ndarray,
    new_model_reward_weights: jnp.ndarray,
    reward_fn: Callable,        # r_e(s, a, g, reward_weights) — pure JAX, task-specific
) -> jnp.ndarray:
    """
    Recomputes r_e(s_t, a^e_t, g_tilde, m_tilde).
    Called inside the JIT-compiled update_step — all inputs must be JAX arrays.
    """
    return reward_fn(state, ego_action, new_goal, new_model_reward_weights)
```

---

### 6.6 `replay/replay_buffer.py`

Stays entirely in NumPy. Pre-allocated arrays (no Python list growth). Explicit `to_jax()`
conversion utility is the only JAX touchpoint.

```python
class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, action_dim: int, goal_dim: int):
        # Pre-allocate — never use Python lists for storage
        self.states       = np.zeros((capacity, obs_dim),    dtype=np.float32)
        self.ego_actions  = np.zeros((capacity, action_dim), dtype=np.float32)
        self.other_actions= np.zeros((capacity, action_dim), dtype=np.float32)
        self.next_states  = np.zeros((capacity, obs_dim),    dtype=np.float32)
        self.goals        = np.zeros((capacity, goal_dim),   dtype=np.float32)
        self.model_ids    = np.zeros(capacity,               dtype=np.int32)
        self.rewards      = np.zeros(capacity,               dtype=np.float32)
        self.dones        = np.zeros(capacity,               dtype=bool)
        self._episode_boundaries: List[Tuple[int, int]] = []  # (start_idx, length)
        self._ptr = 0
        self._size = 0
        self._capacity = capacity

    def add(self, transition: Transition) -> None: ...

    def sample(self, batch_size: int) -> dict:
        """Returns a dict of numpy arrays — NOT JAX arrays."""
        ...

    def relabel_episode(
        self, episode_start_idx: int, episode_length: int, new_model_id: int
    ) -> None:
        """In-place NumPy update. O(T). Called outside JAX boundary."""
        end = episode_start_idx + episode_length
        self.model_ids[episode_start_idx:end] = new_model_id

    def to_jax(self, batch: dict) -> dict:
        """
        Convert a sampled numpy batch to JAX arrays.
        Called ONCE per gradient step at the start of update_step.
        """
        return jax.tree_util.tree_map(jnp.array, batch)
```

---

### 6.7 `networks/actor.py` and `networks/critic.py`

Flax Linen modules. Parameters are **not** stored inside the module — Flax modules are
stateless. Parameters are initialised separately and passed explicitly everywhere.

```python
import flax.linen as nn
import jax.numpy as jnp
from typing import Sequence

class Actor(nn.Module):
    """
    pi_e(a | s, g, m_embed)
    Input: [s || g || encode(m)]
    Output: (mean, log_std) of Gaussian action distribution
    """
    hidden_sizes: Sequence[int]
    action_dim: int

    @nn.compact
    def __call__(self, obs, goal, model_embed):
        x = jnp.concatenate([obs, goal, model_embed], axis=-1)
        for h in self.hidden_sizes:
            x = nn.relu(nn.Dense(h)(x))
        mean    = nn.Dense(self.action_dim)(x)
        log_std = nn.Dense(self.action_dim)(x)
        return mean, jnp.clip(log_std, -5, 2)


class Critic(nn.Module):
    """
    Q(s, a, g, m_embed) — double-Q to reduce overestimation bias.
    """
    hidden_sizes: Sequence[int]

    @nn.compact
    def __call__(self, obs, action, goal, model_embed):
        x = jnp.concatenate([obs, action, goal, model_embed], axis=-1)
        q1, q2 = x, x
        for h in self.hidden_sizes:
            q1 = nn.relu(nn.Dense(h)(q1))
            q2 = nn.relu(nn.Dense(h)(q2))
        return nn.Dense(1)(q1), nn.Dense(1)(q2)
```

**Model encoding** (`networks/encoder.py`): Configurable via `model_encoding` in YAML.
- `onehot`: `jax.nn.one_hot(model_id, num_classes)` — simple, works for small $|\mathcal{M}|$.
- `reward_weights`: concatenate `model.reward_weights` directly — generalises to unseen types.

The encoding must happen **before** `update_step` is called so that `current_model_id`
(a Python int) never enters the JIT trace as a dynamic value (see §15, item 5).

---

### 6.8 `training/train_state.py` — NEW (no PyTorch equivalent)

All mutable training state lives in a single immutable pytree. Updating any field returns a
new `HIMHERTrainState` via `.replace(...)` — standard Flax pattern.

```python
from flax.training.train_state import TrainState
from flax import struct
import optax

@struct.dataclass
class HIMHERTrainState:
    actor_state:          TrainState       # params + optimizer state
    critic_state:         TrainState       # params + optimizer state
    target_critic_params: dict             # soft-updated target network
    log_belief:           jnp.ndarray      # shape (|M|,) — log posterior over M
    current_model_id:     int              # index of MAP model (Python int, not traced)
    step:                 int              # global gradient step counter


def create_train_state(
    rng: jax.random.PRNGKey,
    config,
    obs_dim: int,
    goal_dim: int,
    action_dim: int,
    model_embed_dim: int,
    log_priors: np.ndarray,
) -> HIMHERTrainState:
    actor  = Actor(config.hidden_sizes, action_dim)
    critic = Critic(config.hidden_sizes)

    rng, actor_key, critic_key = jax.random.split(rng, 3)

    dummy_obs    = jnp.zeros((1, obs_dim))
    dummy_goal   = jnp.zeros((1, goal_dim))
    dummy_embed  = jnp.zeros((1, model_embed_dim))
    dummy_action = jnp.zeros((1, action_dim))

    actor_params  = actor.init(actor_key,   dummy_obs, dummy_goal, dummy_embed)
    critic_params = critic.init(critic_key, dummy_obs, dummy_action, dummy_goal, dummy_embed)

    return HIMHERTrainState(
        actor_state=TrainState.create(
            apply_fn=actor.apply, params=actor_params, tx=optax.adam(config.lr_actor)
        ),
        critic_state=TrainState.create(
            apply_fn=critic.apply, params=critic_params, tx=optax.adam(config.lr_critic)
        ),
        target_critic_params=critic_params,
        log_belief=jnp.array(log_priors),
        current_model_id=0,
        step=0,
    )
```

---

### 6.9 `training/trainer.py`

Split into a **pure JIT-compiled update function** and a **stateful Python episode loop**.
Never mix these two halves.

```python
# ── Pure, JIT-compiled update step ────────────────────────────────────────────

@jax.jit
def update_step(
    train_state: HIMHERTrainState,
    batch: dict,                    # JAX arrays — already converted via to_jax()
    model_embed: jnp.ndarray,       # pre-computed BEFORE entering jit
    rng: jax.random.PRNGKey,
    config,
) -> Tuple[HIMHERTrainState, dict]:
    """Fully functional — no side effects. Safe to JIT."""

    def critic_loss_fn(critic_params):
        rng_action, _ = jax.random.split(rng)
        next_mean, next_log_std = train_state.actor_state.apply_fn(
            train_state.actor_state.params,
            batch["next_states"], batch["goals"], model_embed,
        )
        next_actions, next_log_probs = sample_tanh_gaussian(next_mean, next_log_std, rng_action)
        q1_t, q2_t = train_state.critic_state.apply_fn(
            {"params": train_state.target_critic_params},
            batch["next_states"], next_actions, batch["goals"], model_embed,
        )
        q_target = batch["rewards"] + config.gamma * (
            jnp.minimum(q1_t, q2_t) - config.alpha * next_log_probs
        ) * (1.0 - batch["dones"])
        q1, q2 = train_state.critic_state.apply_fn(
            {"params": critic_params},
            batch["states"], batch["actions"], batch["goals"], model_embed,
        )
        return jnp.mean((q1 - q_target)**2 + (q2 - q_target)**2)

    critic_loss, critic_grads = jax.value_and_grad(critic_loss_fn)(
        train_state.critic_state.params
    )
    new_critic_state = train_state.critic_state.apply_gradients(grads=critic_grads)
    new_target_params = jax.tree_util.tree_map(
        lambda t, s: config.tau * s + (1.0 - config.tau) * t,
        train_state.target_critic_params,
        new_critic_state.params,
    )
    # Actor update follows the same pattern (omitted for brevity)

    return train_state.replace(
        critic_state=new_critic_state,
        target_critic_params=new_target_params,
        step=train_state.step + 1,
    ), {"critic_loss": critic_loss}


# ── Stateful episode loop — plain Python, NOT jit-compiled ────────────────────

class HIMHERTrainer:
    def train(self, total_episodes: int) -> None:
        rng = jax.random.PRNGKey(self.config.training.seed)
        numpy_rng = np.random.default_rng(self.config.training.seed)
        train_state = create_train_state(rng, ...)
        buffer = ReplayBuffer(...)

        for ep in range(total_episodes):

            # Episode collection (NumPy / env side)
            trajectory, episode_start_idx = self.collect_episode(train_state)
            buffer.add_episode(trajectory)

            # HIM: JAX likelihood, then NumPy relabeling
            states  = jnp.array(np.stack([t.state        for t in trajectory.transitions]))
            actions = jnp.array(np.stack([t.other_action for t in trajectory.transitions]))
            all_log_liks = all_model_log_likelihoods(
                jnp.array(self.model_set.stacked_policy_params), states, actions
            )
            triggered = bool(is_inconsistent_ratio(
                train_state.current_model_id, all_log_liks,
                jnp.array(self.model_set.log_priors), self.config.him.ratio_delta,
            ))
            if triggered:
                new_model_id = int(jnp.argmax(
                    all_log_liks + jnp.array(self.model_set.log_priors)
                ))
                buffer.relabel_episode(             # NumPy mutation
                    episode_start_idx, len(trajectory.transitions), new_model_id
                )
                train_state = train_state.replace(current_model_id=new_model_id)

            # Gradient steps (JAX)
            for _ in range(self.config.training.gradient_steps):
                numpy_batch = buffer.sample(self.config.training.batch_size)
                her_batch   = self.apply_her(numpy_batch, trajectory, numpy_rng)
                jax_batch   = buffer.to_jax(her_batch)          # convert once
                # Encode model BEFORE entering jit
                model_embed = encode_model(train_state.current_model_id, self.config)
                rng, step_key = jax.random.split(rng)
                train_state, metrics = update_step(
                    train_state, jax_batch, model_embed, step_key, self.config
                )

            self.log(ep, train_state, metrics, triggered)
```

---

## 7. Environment Wrappers

Environment stepping is always outside the JAX boundary — never JIT-compile anything that
calls `env.step()`.

```python
class BaseMultiAgentEnv(ABC):
    @abstractmethod
    def reset(self) -> Tuple[np.ndarray, Dict]:
        """Returns (obs, info) as numpy arrays."""

    @abstractmethod
    def step(self, ego_action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Returns (next_obs, reward, terminated, truncated, info).
        info must contain:
            - "other_action": np.ndarray
            - "achieved_goal": np.ndarray
            - "desired_goal": np.ndarray
        All arrays are numpy, not JAX.
        """

    @abstractmethod
    def compute_reward(
        self, state: np.ndarray, ego_action: np.ndarray,
        goal: np.ndarray, model: AgentModel,
    ) -> float:
        """
        NumPy version of r_e(s, a, g, m). Used for debugging and unit tests.
        Also expose compute_reward_jax (pure JAX) for use inside update_step.
        """

    @abstractmethod
    def get_other_action_log_probability(
        self, policy_params: np.ndarray, state: np.ndarray, action: np.ndarray,
    ) -> float:
        """
        NumPy version of log pi_o^m(a^o | s). For debugging and unit tests only.
        The JAX version lives in him/inconsistency.py.
        """
```

### 7.1 Task-Specific Notes

**Predator-Prey (`predator_prey.py`):**
- Wrap `pettingzoo.magent2.adversarial_pursuit_v4`.
- Other agent cycles between `evasive` and `territorial` at a switch point sampled uniformly
  in $[T/3, 2T/3]$ per run (unknown to ego).
- `get_other_action_log_probability`: Boltzmann distribution over scripted policy preferences.

**Cooperative Navigation (`cooperative_nav.py`):**
- Wrap `pettingzoo.mpe.simple_spread_v3`.
- One other agent has a hidden additive reward bonus toward landmark 0.
- `compute_reward_jax`: binary sparse (all landmarks covered) or shaped (sum of min distances).

**Intersection (`intersection.py`):**
- Wrap `highway_env.envs.IntersectionEnv`.
- Three model types: `aggressive` (always advances), `cautious` (always yields), `reciprocal`
  (mirrors ego's last action). The reciprocal type creates a non-stationary effective model.
- Collision → `reward = -1`, successful crossing → `reward = +1`.

**Hide-and-Seek (`hide_and_seek.py`):**
- Use `pettingzoo.mpe.simple_tag_v3` as a lightweight substitute.
- Three hider strategies: `corner_preference`, `obstacle_user`, `random_walk`.
- **HER is a no-op in this task** — implement a `no_her=True` flag the trainer checks before
  applying goal relabeling.

---

## 8. Baselines

All baselines share the same environment wrappers, replay buffer, and Flax networks. They
differ only in which parts of the training loop are active.

| Baseline | Class | HIM | HER | Notes |
|---|---|---|---|---|
| No opponent model | `VanillaAgent` | ✗ | ✗ | Standard SAC; other treated as env dynamics |
| Static model | `StaticModelAgent` | ✗ | ✓ | Fixed model at init; no revision |
| Bayesian (no HER) | `BayesianAgent` | ✓ (soft) | ✗ | `BeliefUpdater` posterior; no relabeling |
| Full HIM+HER | `HIMHERAgent` | ✓ (hard MAP) | ✓ | Full Algorithm 1 |

Ablations:
- `HIMOnlyAgent`: HIM revision but standard ER (no goal relabeling)
- `HEROnlyAgent`: HER goal relabeling with a fixed static model (no revision)

---

## 9. Configuration System

```yaml
# configs/predator_prey.yaml
defaults:
  - defaults
  - _self_

env:
  name: predator_prey
  n_agents: 2
  switch_episode: null            # null = sampled randomly per run
  max_episode_length: 50

model_set:
  models:
    - name: evasive
      prior: 0.5
    - name: territorial
      prior: 0.5

him:
  ratio_mode: true                # recommended: use likelihood ratio, not absolute threshold
  ratio_delta: 2.0                # log-posterior gap required to trigger HIM

her:
  strategy: future
  k: 4                            # relabeled goals per transition

training:
  total_episodes: 5000
  batch_size: 256
  gradient_steps: 50
  replay_capacity: 100000
  lr_actor: 3e-4
  lr_critic: 3e-4
  gamma: 0.99
  tau: 0.005                      # soft target update coefficient
  alpha: 0.2                      # SAC entropy coefficient
  seed: 42

agent:
  type: him_her                   # him_her | him_only | her_only | static | vanilla | bayesian
  hidden_sizes: [256, 256]
  model_encoding: onehot          # onehot | reward_weights

logging:
  wandb: true
  project: him_her
  log_interval: 10                # episodes
  checkpoint_interval: 500        # episodes
```

---

## 10. Metrics and Logging

### Per-episode
- `episode_reward`
- `him_triggered`
- `model_id_before_him`, `model_id_after_him`
- `log_likelihood` — $\log \mathcal{L}(m | \tau)$ at end of episode
- `model_posterior` — full posterior over $\mathcal{M}$
- `detection_lag` — episodes since true switch until HIM triggers (Predator-Prey, Hide-and-Seek only)
- Task-specific: `capture_rate`, `coverage_rate`, `collision_rate`, `search_time`

### Per gradient step
- `critic_loss`, `actor_loss`
- `her_fraction` — proportion of batch that was HER-relabeled
- `mean_relabeled_reward`

---

## 11. Testing Requirements

```bash
pytest tests/ -v
```

- `test_him.py`: likelihood computation on synthetic trajectories with known probabilities;
  verify `vmap` over models gives the same result as a Python loop; ratio detection at boundary
  conditions; MAP selection given a 3-model set with controlled likelihoods.
- `test_her.py`: goal relabeling produces valid transitions; `relabel_reward` is deterministic
  given (state, action, goal, model); $k$ relabeled transitions per real transition.
- `test_replay_buffer.py`: episode relabeling updates only the target episode's `model_id`;
  capacity rollover; sampling is uniform; `to_jax()` produces correct dtypes and shapes.
- `test_envs.py`: each wrapper returns correct observation shape, reward type, and `info`
  dict keys; `compute_reward` callable with relabeled (goal, model) pairs.
- `test_model_set.py`: prior sums to 1; posterior update correct against hand-computed Bayesian
  update; MAP selection correct; `stacked_policy_params` shape is `(|M|, param_dim)`.
- `test_train_state.py`: `create_train_state` produces correct shapes; `update_step` is
  JIT-compilable (call twice — second call must not retrace); `.replace(...)` returns a valid
  pytree; checkpointing round-trip via orbax preserves all params.

---

## 12. Reproducibility

- All JAX randomness flows through an explicit PRNG key split from
  `jax.random.PRNGKey(config.training.seed)`. **Never use `np.random.seed()` as a substitute
  for JAX randomness** — JAX has no global random state and will silently produce correlated
  samples if a key is reused without splitting.
- NumPy seed (buffer sampling, goal sampling): `np.random.default_rng(seed)` — pass this
  generator explicitly, never use the global state.
- Environment seeds: `env.reset(seed=seed)`.
- Log full Hydra config and a git hash to WandB for every run.
- Checkpointing via `orbax-checkpoint`:

```python
import orbax.checkpoint as ocp

checkpointer = ocp.PyTreeCheckpointer()
checkpointer.save(f"runs/{run_id}/ep{ep}", train_state)                         # save
train_state = checkpointer.restore(f"runs/{run_id}/ep{ep}", item=train_state)   # restore
```

Resume from checkpoint with `--resume`.

---

## 13. Installation

```bash
git clone https://github.com/yourlab/him_her.git
cd him_her

conda create -n him_her python=3.10
conda activate him_her

# JAX CPU
pip install -U jax jaxlib

# JAX GPU (CUDA 12)
pip install -U "jax[cuda12]"

pip install -r requirements.txt
pip install -e .

# Verify JAX
python -c "import jax; print(jax.devices())"

# Verify env
python -c "from him_her.envs import PredatorPreyEnv; e = PredatorPreyEnv(); print(e.reset())"
```

**requirements.txt** (pinned):

```
jax>=0.4.25
jaxlib>=0.4.25
flax>=0.8.0
optax>=0.2.0
chex>=0.1.86
orbax-checkpoint>=0.5.0
numpy>=1.24.0
scipy>=1.11.0
gymnasium>=0.29.0
pettingzoo[mpe,magent]>=1.24.0
highway-env>=1.8.0
hydra-core>=1.3.0
omegaconf>=2.3.0
wandb>=0.16.0
matplotlib>=3.8.0
seaborn>=0.13.0
pygame>=2.5.0
pytest>=7.4.0
```

---

## 14. Running Experiments

**Single run:**
```bash
python scripts/train.py --config-name predator_prey agent.type=him_her training.seed=42
```

**Ablation sweep:**
```bash
python scripts/ablation.py \
  --tasks predator_prey cooperative_nav intersection hide_and_seek \
  --agents him_her him_only her_only bayesian static vanilla \
  --seeds 42 43 44 45 46 \
  --wandb_project him_her_ablations
```

**Evaluate a checkpoint:**
```bash
python scripts/evaluate.py \
  --checkpoint runs/predator_prey/him_her/seed42/ep5000 \
  --n_episodes 100 \
  --render
```

---

## 15. Open Design Questions (for Discussion Section)

1. **Threshold adaptation:** `ratio_mode=True` is the default and avoids per-task tuning of
   an absolute $\epsilon$. Whether `ratio_delta` itself should be adapted (e.g., via a running
   mean of log-posterior gaps) is left as an empirical question.

2. **Soft vs. hard HIM:** The default is hard MAP selection. The `BayesianAgent` baseline
   tests soft HIM (posterior-weighted mixture policy). Comparing these is one of the core ablations.

3. **Model parameterization:** `onehot` is the default for small discrete $|\mathcal{M}|$.
   `reward_weights` encoding generalises to unseen model types but requires a learned inference
   network — left for future work.

4. **Depth of recursion:** The framework assumes the other agent has a fixed (though latent)
   model. Extending to the case where the other is also learning and modeling the ego requires
   a nested I-POMDP treatment — left for future work.

5. **JAX retracing risk:** `current_model_id` is a Python `int` in `HIMHERTrainState`. If
   passed as a raw int into `update_step`, JAX will retrace the function on every HIM trigger
   (i.e., every time the model changes). The fix — enforced in §6.9 — is to convert
   `current_model_id` to a model embedding **before** calling `update_step`, so the traced
   function always receives a fixed-shape JAX array regardless of which model is active.