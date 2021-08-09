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

"""SgdLearner takes steps of SGD on a LossFn."""

import functools
from typing import Dict, Iterator, List, NamedTuple, Optional, Tuple

import acme
from acme.adders import reverb as adders
from acme.agents import agent
from acme.jax import networks as networks_lib
from acme.jax import utils
from acme.utils import async_utils
from acme.utils import counting
from acme.utils import loggers
import jax
import jax.numpy as jnp
import optax
import reverb
import rlax
import typing_extensions
from copy import deepcopy


class ReverbUpdate(NamedTuple):
  """Tuple for updating reverb priority information."""
  keys: jnp.ndarray
  priorities: jnp.ndarray

class LossExtra(NamedTuple):
  """Extra information that is returned along with loss value."""
  metrics: Dict[str, jnp.DeviceArray]
  reverb_update: Optional[ReverbUpdate] = None

class LossFn(typing_extensions.Protocol):
  """A LossFn calculates a loss on a single batch of data."""

  def __call__(self,
               network: networks_lib.FeedForwardNetwork,
               params: networks_lib.Params,
               target_params: networks_lib.Params,
               batch: reverb.ReplaySample,
               key: networks_lib.PRNGKey) -> Tuple[jnp.DeviceArray, LossExtra]:
    """Calculates a loss on a single batch of data."""

class TrainingState(NamedTuple):
  """Holds the agent's training state."""
  params: networks_lib.Params
  opt_state: optax.OptState
  steps: int
  rng_key: networks_lib.PRNGKey

class MCTSLoss(LossFn):
  """Clipped double q learning with prioritization on TD error."""
  discount: float = 0.99
  importance_sampling_exponent: float = 0.2
  max_abs_reward: float = 1.
  huber_loss_parameter: float = 1.

  # loss function to be vectorised
  def stonks(self, v_tm1, r_t, discount_t, v_t, labels, logits):
    value_loss = rlax.td_learning(
      v_tm1=v_tm1,
      r_t=r_t,
      discount_t=discount_t,
      v_t=v_t,
    )

    policy_loss = rlax.categorical_cross_entropy(
      labels=labels,
      logits=logits
    )

    return value_loss + policy_loss

  def __call__(
      self,
      network: networks_lib.FeedForwardNetwork,
      params: networks_lib.Params,
      batch: reverb.ReplaySample,
      key: networks_lib.PRNGKey,
  ) -> Tuple[jnp.DeviceArray, LossExtra]:
    """Calculate a loss on a single batch of data."""
    del key
    transitions: types.Transition = batch.data
    print(transitions)
    print(batch.data)
    print(batch.info) # need to figure out how to get the pi
    keys, probs, *_ = batch.info

    logits, value = network.apply(params, transitions.observation)
    _, target_value = network.apply(params, transitions.next_observation)
    target_value = jax.lax.stop_gradient(target_value)

    scaled_discount = (transitions.discount * self.discount).astype(jnp.float32)
    clipped_reward = jnp.clip(transitions.reward, -self.max_abs_reward,
                   self.max_abs_reward).astype(jnp.float32)

    batch_loss = jax.vmap(self.stonks)

    loss = batch_loss(
      v_tm1=value,
      r_t=clipped_reward,
      discount_t=scaled_discount,
      v_t=target_value,
      labels=transitions.extras["pi"],
      logits=logits
      )
    print("loss:", loss)

    reverb_update = ReverbUpdate(
        keys=keys, priorities=jnp.abs(td_error).astype(jnp.float64))
    extra = LossExtra(metrics={}, reverb_update=reverb_update)

    return loss, extra
    # # Forward pass.
    # q_tm1 = network.apply(params, transitions.observation)
    # q_t_value = network.apply(target_params, transitions.next_observation)
    # q_t_selector = network.apply(params, transitions.next_observation)

    # # Cast and clip rewards.
    # d_t = (transitions.discount * self.discount).astype(jnp.float32)
    # r_t = jnp.clip(transitions.reward, -self.max_abs_reward,
    #                self.max_abs_reward).astype(jnp.float32)

    # # Compute double Q-learning n-step TD-error.
    # batch_error = jax.vmap(rlax.double_q_learning)
    # td_error = batch_error(q_tm1, transitions.action, r_t, d_t, q_t_value,
    #                        q_t_selector)
    # batch_loss = rlax.huber_loss(td_error, self.huber_loss_parameter)

    # # Importance weighting.
    # importance_weights = (1. / probs).astype(jnp.float32)
    # importance_weights **= self.importance_sampling_exponent
    # importance_weights /= jnp.max(importance_weights)

    # # Reweight.
    # loss = jnp.mean(importance_weights * batch_loss)  # []
    # reverb_update = ReverbUpdate(
    #     keys=keys, priorities=jnp.abs(td_error).astype(jnp.float64))
    # extra = LossExtra(metrics={}, reverb_update=reverb_update)
    # return loss, extra

# TODO: migrate this to multicore (commented version below)
class MCTSLearner(acme.Learner):
  """An Acme learner based around SGD on batches.
  This learner currently supports optional prioritized replay and assumes a
  TrainingState as described above.
  """

  def __init__(self,
               network: networks_lib.FeedForwardNetwork,
               loss_fn: LossFn,
               optimizer: optax.GradientTransformation,
               data_iterator: Iterator[reverb.ReplaySample],
               random_key: networks_lib.PRNGKey,
               replay_client: Optional[reverb.Client] = None,
               counter: Optional[counting.Counter] = None,
               logger: Optional[loggers.Logger] = None,
               num_sgd_steps_per_step: int = 1):
    """Initialize the SGD learner."""
    self.network = network

    # Internalize the loss_fn with network.
    self._loss = jax.jit(functools.partial(loss_fn, self.network))

    # SGD performs the loss, optimizer update and periodic target net update.
    def sgd_step(state: TrainingState,
                 batch: reverb.ReplaySample) -> Tuple[TrainingState, LossExtra]:
      print("learner: sgd_step started")
      next_rng_key, rng_key = jax.random.split(state.rng_key)
      # Implements one SGD step of the loss and updates training state
      (loss, extra), grads = jax.value_and_grad(self._loss, has_aux=True)(
          state.params, batch, rng_key)
      extra.metrics.update({'total_loss': loss})

      # Apply the optimizer updates
      updates, new_opt_state = optimizer.update(grads, state.opt_state)
      new_params = optax.apply_updates(state.params, updates)

      # Periodically update target networks.
      steps = state.steps + 1
      new_training_state = TrainingState(
          new_params, new_opt_state, steps, next_rng_key)
      return new_training_state, extra

    def postprocess_aux(extra: LossExtra) -> LossExtra:
      reverb_update = jax.tree_map(lambda a: jnp.reshape(a, (-1, *a.shape[2:])),
                                   extra.reverb_update)
      return extra._replace(
          metrics=jax.tree_map(jnp.mean, extra.metrics),
          reverb_update=reverb_update)

    sgd_step = utils.process_multiple_batches(sgd_step, num_sgd_steps_per_step,
                                              postprocess_aux)
    self._sgd_step = jax.jit(sgd_step)

    # Internalise agent components
    # self._data_iterator = utils.prefetch(data_iterator)
    self._data_iterator = data_iterator
    self._counter = counter or counting.Counter()
    self._logger = logger or loggers.TerminalLogger('learner', time_delta=1.)

    # Initialize the network parameters
    key_params, key_target, key_state = jax.random.split(random_key, 3)
    initial_params = self.network.init(key_params)
    self._state = TrainingState(
        params=initial_params,
        opt_state=optimizer.init(initial_params),
        steps=0,
        rng_key=key_state,
    )

    # Update replay priorities
    def update_priorities(reverb_update: Optional[ReverbUpdate]) -> None:
      if reverb_update is None or replay_client is None:
        return
      else:
        replay_client.mutate_priorities(
            table=adders.DEFAULT_PRIORITY_TABLE,
            updates=dict(zip(reverb_update.keys, reverb_update.priorities)))
    self._replay_client = replay_client
    self._async_priority_updater = async_utils.AsyncExecutor(update_priorities)

  def step(self):
    """Takes one SGD step on the learner."""
    print("learner: stepping")
    batch = next(self._data_iterator)
    print("learner: fetched batch")
    self._state, extra = self._sgd_step(self._state, batch)

    if self._replay_client:
      reverb_update = extra.reverb_update._replace(keys=batch.info.key)
      self._async_priority_updater.put(reverb_update)

    # Update our counts and record it.
    result = self._counter.increment(steps=1)
    result.update(extra.metrics)
    self._logger.write(result)

  def get_variables(self, names: List[str]) -> List[networks_lib.Params]:
    return [self._state.params]

  def save(self) -> TrainingState:
    return self._state

  def restore(self, state: TrainingState):
    self._state = state
