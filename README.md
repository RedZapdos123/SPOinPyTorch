# SPOinPyTorch:

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A PyTorch implementation of Simple Policy Optimization (SPO) for reinforcement learning in continuous and discrete control. SPO is an alternative to Proximal Policy Optimization (PPO) featuring a smooth, unconstrained policy objective with a quadratic penalty around the probability ratio 1, offering improved stability and strong theoretical properties.

Paper: Simple Policy Optimization (arXiv:2401.16025):
- arXiv: https://arxiv.org/abs/2401.16025.
- DOI: https://doi.org/10.48550/arXiv.2401.16025.

## Overview:

- Clean, modular implementation of the SPO algorithm.
- Simple, familiar API and configuration utilities.
- Optimized defaults for continuous control (like LunarLanderContinuous-v3).
- GPU acceleration (CUDA) supported.
- Examples for training, evaluation, HPO, and visualization.

## Installation:

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

## Requirements:

- Python: 3.11+.
- PyTorch: 2.0+ (install per your CUDA setup from https://pytorch.org/).
- Gymnasium: 0.29+ (for environments like LunarLanderContinuous-v3).
- OS: Linux, macOS, or Windows', and CUDA optional for GPU acceleration.

**Install core dependencies:**

```bash
pip install torch gymnasium numpy
```

Advice: Prefer the official PyTorch install command for your platform to get the correct CUDA build.

## Documentation:

- API reference (includes usage): [docs/API.md](docs/API.md).

## SPO vs PPO (from the Research Paper):

- Objective: PPO uses min(r * A, clip(r, 1-ε, 1+ε) * A)', and SPO uses a smooth penalty objective: r * A - (|A|/(2ε)) * (r-1)^2.
- Smoothness: SPO avoids clipping discontinuities, providing stable gradients for first order optimizers.
- Trust region behavior: The quadratic penalty keeps the probability ratio r near 1 ± ε, improving constraint adherence without second order methods.
- Empirical performance: The paper reports SPO outperforming PPO in several settings, particularly with large, complete networks.



## Examples:

See the examples directory for complete scripts:
- examples/lunar_lander/train.py.
- examples/lunar_lander/evaluate.py.
- examples/lunar_lander/hyperparameter_optimization.py.
- examples/lunar_lander/visualization.py.

## Research:

SPO is an unconstrained first order policy optimization method that replaces PPO's clipping with a smooth quadratic penalty centered at probability ratio 1. This yields stable gradients and implicitly constrains the ratio without second order methods.

Objective:

```
L_SPO = r * A - (|A| / (2 * ε)) * (r - 1)^2
```

Advantages over PPO (per the paper):
- Smooth, unconstrained objective amenable to standard first order optimizers.
- Stronger theoretical properties for constraining the probability ratio within a trust region neighborhood.
- Improved stability due to quadratic penalty and absence of clipping discontinuities.
- Competitive or superior empirical performance to PPO, particularly for large, complete networks.



## Best Practices:

- Normalize advantages: Set `Config.normalize_advantages = True` for improved stability.
- Tuning epsilon (ε): Start with ε in [0.1, 0.3]', and larger ε reduces the penalty and allows larger updates.
- Learning rate: 3e-4 is a good default', and try 1e-4 to 1e-3 depending on reward scale and network size.
- Rollout length: `steps_per_batch` controls bias variance trade off', and longer rollouts improve value targets but may increase correlation.
- Minibatching: Use multiple epochs over minibatches for better sample efficiency', and shuffle data each epoch.
- Gradient clipping: Set `max_grad_norm` to prevent rare spikes in gradients.
- Seeding: Fix `Config.seed` and set `torch.backends.cudnn.deterministic=True` if strict reproducibility is required.
- Device: Move tensors and models to the same device', and prefer float32', and avoid implicit CPU to GPU and reverse transfers in the loop.

## Troubleshooting:

- Action out of bounds (continuous): Ensure `action_low` and `action_high` reflect your environment’s bounds', and actions are clamped accordingly.
- NaN losses: Reduce learning rate', and enable gradient clipping', and verify rewards are finite and observations are normalized.
- Poor performance vs PPO: Tune ε and entropy coefficient', and ensure advantages are computed with correct `gamma` and `gae_lambda`.
- Log-prob mismatch: Make sure `old_log_probs` passed to `update` were computed by the same policy that generated the actions.
- CUDA errors: Confirm torch and CUDA versions match your driver', and run `torch.cuda.is_available()` and reinstall per pytorch.org instructions.

## License:

This project is licensed under the MIT License. See LICENSE for details.

## Citation:

If you use this library or the SPO algorithm in your work, please cite the paper and this repository.

```bibtex
@article{xie2024spo,
  title={Simple Policy Optimization},
  author={Xie, Zhengpeng and Zhang, Qiang and Yang, Fan and Hutter, Marco and Xu, Renjing},
  journal={arXiv preprint arXiv:2401.16025},
  year={2024},
  doi={10.48550/arXiv.2401.16025}
}

@software{spoinpytorch2025,
  title={SPOinPyTorch: Simple Policy Optimization in PyTorch},
  author={SPOinPyTorch Contributors},
  year={2025},
  url={https://github.com/RedZapdos123/SPOinPyTorch}
}
```