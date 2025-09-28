"""
SPOinPyTorch: Simple Policy Optimization in PyTorch

A PyTorch implementation of the Simple Policy Optimization (SPO) algorithm
for reinforcement learning in continuous control environments.

Based on the research paper: https://arxiv.org/abs/2401.16025
"""

from .agent import SPOAgent
from .models import Actor, Critic
from .config import Config

__version__ = "0.1.0"
__author__ = "SPOinPyTorch Contributors"
__email__ = ""
__description__ = "Simple Policy Optimization (SPO) algorithm implementation in PyTorch"

__all__ = [
    "SPOAgent",
    "Actor", 
    "Critic",
    "Config",
]
