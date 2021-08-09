from acme.adders import reverb as adders_reverb
import dataclasses
import numpy as np

@dataclasses.dataclass
class IMPALAConfig:
  """Configuration options for IMPALA."""
  seed: int = 0

  # Loss options
  batch_size: int = 16
  sequence_length: int = 20
  sequence_period: int = 20
  discount: float = 0.99
  entropy_cost: float = 0.01
  baseline_cost: float = 0.5
  max_abs_reward: float = np.inf
  max_gradient_norm: float = np.inf

  # Optimizer options
  learning_rate: float = 1e-4
  adam_momentum_decay: float = 0.0
  adam_variance_decay: float = 0.99

  # Replay options
  max_queue_size: Union[int, types.Batches] = types.Batches(10)

  def __post_init__(self):
    if isinstance(self.max_queue_size, types.Batches):
      self.max_queue_size *= self.batch_size
