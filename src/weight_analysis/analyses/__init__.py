"""Checkpoint-level descriptive analyses with explicit inputs."""

from .drift import common, describe_checkpoint, scope_rows
from .factorial import factorial, validate_factorial_design

__all__ = ["common", "describe_checkpoint", "scope_rows", "factorial", "validate_factorial_design"]
