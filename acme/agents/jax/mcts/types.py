# python3
# Copyright 2018 DeepMind Technologies Limited. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Type aliases and assumptions that are specific to the MCTS agent."""
from typing import Callable

from acme.jax import networks
import haiku as hk
import jax.numpy as jnp

# Assumption: actions are scalar and discrete (integral).
Action = int

# Assumption: observations are array-like.
Observation = jnp.ndarray

# Assumption: rewards and discounts are scalar.
Reward = float
Discount = float

# Notation: policy logits/probabilities are simply a vector of floats.
Probs = jnp.ndarray

# Notation: the value function is scalar-valued.
Value = float


ModelFn = Callable[[Observation, Action], networks.Params]

PolicyValueInitFn = Callable[[networks.PRNGKey, Observation, hk.LSTMState],
                             networks.Params]
PolicyValueFn = Callable[[networks.Params, Observation, hk.LSTMState],
                         networks.LSTMOutputs]

