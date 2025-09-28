# SPOinPyTorch API Reference:

The documentation for public classes and functions in SPOinPyTorch.

## SPOAgent:

Main agent implementing the Simple Policy Optimization algorithm.

```python
SPOAgent(
    state_dim: int,
    action_dim: int,
    config: dict,
    is_discrete: bool = True,
    action_low: Optional[List[float]] = None,
    action_high: Optional[List[float]] = None,
    device: str = 'cpu'
)
```

**Parameters:**
- state_dim (int): Observation dimension.
- action_dim (int): Action dimension.
- config (dict): Hyperparameters (see Config.get_dict()).
- is_discrete (bool): Use categorical (True) or normal (False) policy.
- action_low (List[float] | None): Per dimension min bounds for continuous actions.
- action_high (List[float] | None): Per dimension max bounds for continuous actions.
- device (str): 'cpu' or 'cuda'.

### Methods:
- get_action_and_value(state: torch.Tensor, action: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
  - Inputs: state shape (B, state_dim), dtype float32 on the same device; optional action
  - Returns:
    - action: (B,) for discrete; (B, action_dim) for continuous
    - log_prob: (B,)
    - entropy: (B,)
    - value: (B,)
  - Notes: For continuous, actions are clamped to [action_low, action_high] if provided. log_prob/entropy are summed across action dims.

- get_value(state: torch.Tensor) -> torch.Tensor
  - Inputs: state (B, state_dim)
  - Returns: value (B,)

- compute_gae(rewards: torch.Tensor, dones: torch.Tensor, values: torch.Tensor, next_value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
  - Shapes: rewards/dones/values (T, N), next_value (N,) or (1,)
  - Returns: advantages (T, N), returns (T, N) where returns = advantages + values
  - Uses gamma and gae_lambda from Config

- update(states, actions, old_log_probs, advantages, returns) -> Dict[str, float]
  - All tensors batched and device-aligned
  - Returns keys: policy_loss, value_loss, entropy_loss, total_loss
  - Optimization: Adam; gradient clipping if max_grad_norm is set
  - Policy objective (SPO): r * A - (|A|/(2 * epsilon)) * (r-1)^2, where r = exp(new_log_prob - old_log_prob)

## Configurations:

Container for hyperparameters with convenient update/get methods.

```python
class Config:
    env_name: str = "LunarLanderContinuous-v3"
    seed: int
    total_timesteps: int
    steps_per_batch: int
    update_epochs: int
    num_minibatches: int
    learning_rate: float
    gamma: float
    gae_lambda: float
    epsilon: float
    entropy_coeff: float
    value_loss_coeff: float
    max_grad_norm: float
    actor_hidden_dims: List[int]
    critic_hidden_dims: List[int]
    normalize_advantages: bool
```

### Methods:
- update(new_config: Dict[str, Any]) -> None
- get_dict() -> Dict[str, Any]

**Attributes:**
- env_name (str): Gymnasium environment ID; default "LunarLanderContinuous-v3" for examples.

- seed (int): Random seed for reproducibility.

- total_timesteps (int): Total environment steps to collect; like 1e6.

- steps_per_batch (int): On policy rollout length per update; like 2048 for continuous control.

- update_epochs (int): Number of passes over collected data per update; like 10.

- num_minibatches (int): Minibatches per epoch; like 32. Effective batch size = steps_per_batch; minibatch size = steps_per_batch / num_minibatches.

- learning_rate (float): Adam LR; typical range 1e-4 to 1e-3; default around 3e-4.

- gamma (float): Discount factor in [0.9, 0.999]; default 0.99.

- gae_lambda (float): GAE parameter in [0.9, 1.0]; default 0.95.

- epsilon (float): SPO penalty scale; typical 0.1 to 0.3; smaller values enforce tighter trust region.

- entropy_coeff (float): Weight for policy entropy bonus; typical 0.0 to 0.02.

- value_loss_coeff (float): Weight for value loss term; typical 0.5 to 1.0.

- max_grad_norm (float): Global gradient clip norm; typical 0.5 to 1.0; set 0 or None to disable.

- actor_hidden_dims (List[int]): MLP widths for policy network; like [64,64] or [256,256].

- critic_hidden_dims (List[int]): MLP widths for value network; may mirror actor dims.

- normalize_advantages (bool): If True, standardize advantages before updates (recommended).


## Actor:

The Policy network.

```python
Actor(state_dim: int, action_dim: int, hidden_dims: List[int] = [64, 64], is_discrete: bool = True)
```

- Discrete: returns logits (B, action_dim)
- Continuous: returns (mean, log_std) where
  - mean: (B, action_dim)
  - log_std: broadcastable to (B, action_dim)

## Critic:

Value function network.

```python
Critic(state_dim: int, hidden_dims: List[int] = [64, 64])
```

- Forward outputs value (B,)

## Utility Functions:

- layer_init(layer: nn.Module, std: float = sqrt(2), bias_const: float = 0.0) -> nn.Module
  - Orthogonal weight init, constant bias

## Examples:

See the Usage Guide below for complete usage patterns and training loops.

An example code block:

```python
import gymnasium as gym, torch
from SPOinPyTorch import SPOAgent, Config

env = gym.make("LunarLanderContinuous-v3")

agent = SPOAgent(8, 2, Config().get_dict(), is_discrete=False, action_low=[-1,-1], action_high=[1,1])
s = torch.randn(1, 8)
action, logp, ent, v = agent.get_action_and_value(s)
```


## Training Loop Details:

Complete training generally follows an on policy rollout → advantage/return computation → multi epoch update over minibatches.

- Rollout buffer shapes (time major):
  - rewards, dones, values: (T, N) where T is steps per rollout and N is number of parallel envs
  - next_value: (N,) at rollout end
- Advantage estimation: Generalized Advantage Estimation (GAE)
  - advantages, returns = compute_gae(rewards, dones, values, next_value)
  - Optionally normalize advantages to zero mean and unit std
- Policy update (SPO objective):
  - r = exp(new_log_prob - old_log_prob)
  - L_policy = r*A - (|A|/(2*epsilon))*(r-1)^2
- Value loss: MSE between predicted value and return.
- Entropy bonus: encourages exploration; weight through entropy_coeff
- Multi epoch updates: iterate update_epochs times over shuffled minibatches (num_minibatches).

Pseudo code:

```python
buffer = collect_rollout(env, agent, steps_per_batch)  #returns states, actions, rewards, dones, values, log_probs
next_value = agent.get_value(last_obs)
advs, rets = agent.compute_gae(buffer.rewards, buffer.dones, buffer.values, next_value)

for epoch in range(config.update_epochs):
    for batch in iterate_minibatches(buffer.states, buffer.actions, buffer.log_probs, advs, rets, config.num_minibatches):
        loss_info = agent.update(*batch)
```

#### References:
- Paper: https://arxiv.org/abs/2401.16025
- README: [../README.md](../README.md)

## Usage Guide:

This section contains the installation, setup and usage guides.

### Installation:

From PyPI:

```bash
pip install SPOinPyTorch
```

From source:

```bash
git clone https://github.com/RedZapdos123/SPOinPyTorch.git
cd SPOinPyTorch
pip install -e .
```

### Basic Usage:

```python
import torch
import gymnasium as gym
from SPOinPyTorch import SPOAgent, Config

#Create environment.
env = gym.make("LunarLanderContinuous-v3")

#Initialize configuration and agent.
config = Config()
agent = SPOAgent(
    state_dim=8,
    action_dim=2,
    config=config.get_dict(),
    is_discrete=False,
    action_low=[-1, -1],
    action_high=[1, 1],
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

#Simple interaction loop (illustrative).
state, _ = env.reset()
for step in range(1000):
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    action, log_prob, entropy, value = agent.get_action_and_value(state_tensor)
    next_state, reward, terminated, truncated, _ = env.step(action.cpu().numpy()[0])
    state = next_state if not (terminated or truncated) else env.reset()[0]
```

### Training on LunarLander (sketch):

Minimal on policy rollout and update pattern (refer to examples for batching, logging, and multiple epochs/minibatches):

```python
import torch
import gymnasium as gym
from SPOinPyTorch import SPOAgent, Config

env = gym.make("LunarLanderContinuous-v3")
config = Config()
agent = SPOAgent(8, 2, config.get_dict(), is_discrete=False, action_low=[-1, -1], action_high=[1, 1])

obs, _ = env.reset()
states, actions, rewards, dones, values, log_probs = [], [], [], [], [], []
for t in range(config.steps_per_batch):
    s = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    a, lp, ent, v = agent.get_action_and_value(s)
    obs_next, r, term, trunc, _ = env.step(a.cpu().numpy()[0])

    states.append(s)
    actions.append(a)
    rewards.append(torch.tensor([r], dtype=torch.float32))
    dones.append(torch.tensor([term or trunc], dtype=torch.float32))
    values.append(v.detach())
    log_probs.append(lp.detach())

    obs = obs_next if not (term or trunc) else env.reset()[0]

states = torch.cat(states)
actions = torch.stack(actions)
rewards = torch.cat(rewards)
dones = torch.cat(dones)
values = torch.cat(values)
with torch.no_grad():
    next_value = agent.get_value(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
advantages, returns = agent.compute_gae(rewards.unsqueeze(1), dones.unsqueeze(1), values.unsqueeze(1), next_value)

loss_info = agent.update(states, actions, torch.stack(log_probs), advantages.squeeze(1), returns.squeeze(1))
print(loss_info)
```

### Running Provided Examples:

The `examples/` directory contains complete implementations for LunarLander, including training, evaluation, hyperparameter optimization, and visualization.

```bash
cd examples/lunar_lander
python train.py
python evaluate.py
python hyperparameter_optimization.py
python visualization.py
```

### Custom Configuration:

```python
from SPOinPyTorch import Config

config = Config()
config.learning_rate = 3e-4
config.actor_hidden_dims = [256, 256, 256]
config.critic_hidden_dims = [256, 256, 256]
config.total_timesteps = 1_000_000

agent = SPOAgent(
    state_dim=8, action_dim=2, config=config.get_dict(),
    is_discrete=False, device='cuda'
)
```

### Evaluation and Visualization:

- See `examples/lunar_lander/evaluate.py` for deterministic evaluation and score logging.
- See `examples/lunar_lander/visualization.py` for plotting learning curves and diagnostics.


## Best Practices:

- Advantage normalization: Enable through `Config.normalize_advantages` for stable updates.
- Epsilon (ε) schedule: Consider decaying ε slightly over training to reduce step sizes as the policy converges.
- Entropy coefficient: Start at 0.0–0.01 for continuous control; tune upward if policy collapses prematurely.
- Gradient clipping:Set `max_grad_norm` to 0.5–1.0 to mitigate rare spikes.
- Value function: Ensure critic capacity (hidden dims) is sufficient; underfitting values hurts advantage estimation.
- Observation scaling: Normalize inputs per environment recommendation; avoid extremely large/small magnitudes.
- Seeding: Set `torch.manual_seed` and environment seeds for reproducibility.

## Troubleshooting:

- NaNs in loss or values:
  - Reduce learning rate; verify rewards are finite; check observation normalization.
  - Clamp log_std for continuous policies if the environment is very sensitive.
- Poor learning progress:
  - Increase `steps_per_batch`; try more `update_epochs`; verify correct device placement.
  - Tune ε and `entropy_coeff`; ensure `old_log_probs` are computed from the rollout policy.
- Unstable policy updates:
  - Use smaller ε; enable gradient clipping; normalize advantages.
- Action bounds violations (continuous):
  - Provide `action_low`/`action_high` matching the environment; actions are clamped to these.

## Version Compatibility:

- Python: 3.11+
- PyTorch: 2.0+
- Gymnasium: 0.29+

GPU acceleration is supported when `device='cuda'` and a compatible CUDA build of PyTorch is installed.

## Environment Support:

- Designed for Gymnasium environments.
- Discrete and continuous action spaces supported:
  - Discrete: Categorical policy over action_dim.
  - Continuous: Diagonal Normal policy parameterized by mean and log_std per action dimension.

## References:

- Simple Policy Optimization (arXiv:2401.16025): https://arxiv.org/abs/2401.16025
- Project README: [README.md](../README.md)
