"""Base model class, specifying the interface.."""

import abc
from typing import Optional

from acme.agents.jax.mcts import types

import dm_env


class Model(dm_env.Environment, abc.ABC):
  """Base (abstract) class for models used for planning via MCTS."""

  @abc.abstractmethod
  def load_checkpoint(self):
    """Loads a saved model state, if it exists."""

  @abc.abstractmethod
  def save_checkpoint(self):
    """Saves the model state so that we can reset it after a rollout."""

  @abc.abstractmethod
  def update(
      self,
      timestep: dm_env.TimeStep,
      action: types.Action,
      next_timestep: dm_env.TimeStep,
  ) -> dm_env.TimeStep:
    """Updates the model given an observation, action, reward, and discount."""

  @abc.abstractmethod
  def reset(self, initial_state: Optional[types.Observation] = None):
    """Resets the model, optionally to an initial state."""

  @property
  @abc.abstractmethod
  def needs_reset(self) -> bool:
    """Returns whether or not the model needs to be reset."""