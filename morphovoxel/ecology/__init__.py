"""Modular finite-resource ecology."""
from .environment import EcologyWorld
from .simulator import ecology_step

__all__ = ["EcologyWorld", "ecology_step"]

