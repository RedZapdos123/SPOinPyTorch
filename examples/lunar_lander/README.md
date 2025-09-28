# LunarLander SPO Examples

This directory contains example scripts demonstrating how to use the SPOinPyTorch library to train and evaluate Simple Policy Optimization (SPO) agents on the LunarLanderContinuous-v3 environment.

## Prerequisites

Make sure you have installed the SPOinPyTorch package and its dependencies:

```bash
pip install SPOinPyTorch[examples]
```

Or if running from source:

```bash
pip install -e .[examples]
```

## Scripts

### 1. `train.py` - Training Script

Train a new SPO agent from scratch:

```bash
python train.py
```

**Features:**
- Trains on LunarLanderContinuous-v3 environment
- Uses optimized hyperparameters from research
- Saves checkpoints during training
- Provides real-time training progress
- Early stopping when target performance is reached

**Outputs:**
- `checkpoints/best_model.pth` - Best performing model
- `checkpoints/final_model.pth` - Final model after training
- `logs/training_log.txt` - Detailed training log
- `logs/training_history.json` - Training metrics in JSON format

### 2. `evaluate.py` - Evaluation Script

Evaluate a trained SPO agent:

```bash
python evaluate.py
```

**Features:**
- Loads trained model checkpoints
- Evaluates performance over 100 episodes
- Creates detailed evaluation plots
- Optionally renders episodes for visual inspection

**Outputs:**
- `evaluation_results.json` - Evaluation metrics
- `evaluation_plots.png` - Performance visualization
- `evaluation_analysis.png` - Detailed analysis plots

### 3. `hyperparameter_optimization.py` - Hyperparameter Tuning

Optimize hyperparameters using Optuna:

```bash
python hyperparameter_optimization.py
```

**Features:**
- Uses Optuna for efficient hyperparameter search
- Supports pruning for faster optimization
- Saves best hyperparameters for later use
- Reduced training time for faster iteration

**Outputs:**
- `best_hyperparameters.json` - Optimized hyperparameters
- `optuna_checkpoints/` - Model checkpoints from trials

### 4. `visualization.py` - Training Visualization

Visualize training progress and results:

```bash
python visualization.py
```

**Features:**
- Loads training logs or history files
- Creates comprehensive training progress plots
- Analyzes reward progression and learning curves
- Supports both saved plots and interactive display

**Outputs:**
- `spo_training_progress.png` - Training progress visualization
- `spo_reward_analysis.png` - Detailed reward analysis

## Usage Examples

### Basic Training and Evaluation

```bash
# Train a new agent
python train.py

# Evaluate the trained agent
python evaluate.py

# Visualize the training progress
python visualization.py
```

### Hyperparameter Optimization Workflow

```bash
# Optimize hyperparameters (this may take several hours)
python hyperparameter_optimization.py

# Use the optimized hyperparameters by updating train.py
# or by loading them in your custom training script

# Train with optimized hyperparameters
python train.py

# Evaluate the optimized agent
python evaluate.py
```

### Custom Configuration

You can modify the training configuration by editing the `Config` class parameters in the scripts or by creating your own configuration:

```python
from SPOinPyTorch import Config

# Create custom configuration
config = Config()
config.learning_rate = 0.001
config.total_timesteps = 1_000_000
config.actor_hidden_dims = [512, 512, 512]

# Use in training
agent = SPOAgent(
    state_dim=8,
    action_dim=2,
    config=config.get_dict(),
    is_discrete=False,
    action_low=[-1, -1],
    action_high=[1, 1],
    device='cuda'
)
```

## Expected Results

With the default configuration, you should expect:

- **Training Time**: ~2-4 hours on modern GPU
- **Final Performance**: 220+ average reward
- **Success Rate**: 80%+ episodes with reward ≥ 200
- **Convergence**: Around 600K-800K timesteps

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce `num_envs` in training scripts
2. **Slow training**: Ensure CUDA is available and being used
3. **Poor performance**: Try hyperparameter optimization
4. **Import errors**: Make sure SPOinPyTorch is properly installed

### Performance Tips

1. Use GPU acceleration when available
2. Adjust `num_envs` based on your hardware
3. Use hyperparameter optimization for best results
4. Monitor training progress with visualization tools

## File Structure

```
lunar_lander/
├── README.md                           # This file
├── train.py                           # Training script
├── evaluate.py                        # Evaluation script
├── hyperparameter_optimization.py     # Hyperparameter tuning
├── visualization.py                   # Training visualization
├── checkpoints/                       # Model checkpoints (created during training)
├── logs/                             # Training logs (created during training)
└── optuna_checkpoints/               # Optimization checkpoints (created during tuning)
```

## Next Steps

After running these examples, you can:

1. Adapt the code for other continuous control environments
2. Experiment with different network architectures
3. Implement custom reward functions or environment modifications
4. Compare SPO performance with other RL algorithms
5. Use the trained models in your own applications
