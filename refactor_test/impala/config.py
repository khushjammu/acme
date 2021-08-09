from acme.adders import reverb as adders_reverb
import dataclasses
import numpy as np
from acme import types
from typing import Union

@dataclasses.dataclass
class IMPALAConfig:
  """Configuration options for IMPALA."""
  seed: int = 0

  # Loss options
  batch_size: int = 16 # 256
  min_replay_size: int = 10  # Minimum replay size.
  sequence_length: int = 20
  sequence_period: int = 20
  discount: float = 0.99
  entropy_cost: float = 0.01
  baseline_cost: float = 0.5
  max_abs_reward: float = np.inf
  max_gradient_norm: float = np.inf

  # Optimizer options
  learning_rate: float = 5e4
  terminal_learning_rate: float = 5e-8  # Final learning rate scheduler applies
  schedule_steps: int = 5 # Number of steps between starting LR and terminal LR
  adam_momentum_decay: float = 0.0
  adam_variance_decay: float = 0.99

  # Replay options
  max_queue_size: Union[int, types.Batches] = types.Batches(10)
  samples_per_insert: float = 32  # Ratio of learning samples to insert.

  def __post_init__(self):
    if isinstance(self.max_queue_size, types.Batches):
      self.max_queue_size *= self.batch_size

  # How many gradient updates to perform per learner step.
  num_sgd_steps_per_step: int = 1

  # Checkpointing configuration
  # Interval (in steps) for learner checkpointing
  checkpoint_interval: int = 10_000
  base_checkpoint_dir: str = "checkpoints/"

  # Logging configuration
  base_log_dir: str = "logs/" # Base log directory for all experimental runs.
  # Interval for logging global high-score and global return distribution
  universal_stats_interval: int = 100
