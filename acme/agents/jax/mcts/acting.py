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

"""MCTS (AlphaZero) JAX actor."""

from typing import Callable, Generic, Optional, Tuple, TypeVar, Union

from acme import adders
from acme import core
from acme import types
from acme.agents.jax import actor_core
from acme.jax import networks as network_lib
from acme.jax import utils
from acme.jax import variable_utils
import dm_env
import jax
import numpy as np
import search

# Useful type aliases.
RecurrentState = TypeVar('RecurrentState')

# Signatures for functions that sample from parameterised stochastic policies.
FeedForwardPolicy = Callable[
    [network_lib.Params, network_lib.PRNGKey, network_lib.Observation],
    Union[network_lib.Action, Tuple[network_lib.Action, types.NestedArray]]]
RecurrentPolicy = Callable[[
    network_lib.Params, network_lib.PRNGKey, network_lib
    .Observation, RecurrentState
], Tuple[Union[network_lib.Action, Tuple[network_lib.Action,
                                         types.NestedArray]], RecurrentState]]


class MCTSActor(core.Actor):
  """A simple feed-forward actor implemented in JAX.

  An actor based on a policy which takes observations and outputs actions. It
  also adds experiences to replay and updates the actor weights from the policy
  on the learner.
  """

  _prev_timestep: dm_env.TimeStep

  def __init__(
      self,
      policy,
      random_key: network_lib.PRNGKey,
      variable_client: variable_utils.VariableClient,
      num_actions: int,
      num_simulations: int,
      discount: int,
      model = None, # todo: sort out environment model
      adder: Optional[adders.Adder] = None,
      has_extras: bool = False,
      backend: Optional[str] = 'cpu',
  ):
    """Initializes a feed forward actor.

    Args:
      policy: A value-policy network.
      random_key: Random key.
      variable_client: The variable client to get policy parameters from.
      adder: An adder to add experiences to.
      has_extras: Flag indicating whether the policy returns extra
        information (e.g. q-values) in addition to an action.
      backend: Which backend to use for running the policy.
    """
    self._num_actions = num_actions
    self._actions = list(range(self._num_actions))
    self._num_simulations = num_simulations
    self._discount = discount
    self._random_key = random_key
    self._has_extras = has_extras
    self._extras: types.NestedArray = ()

    # Adding batch dimension inside jit is much more efficient than outside.
    def batched_policy(
        params: network_lib.Params, key: network_lib.PRNGKey,
        observation: network_lib.Observation
    ) -> Tuple[Union[network_lib.Action, Tuple[
        network_lib.Action, types.NestedArray]], network_lib.PRNGKey]:
      # TODO(b/161332815): Make JAX Actor work with batched or unbatched inputs.
      key, key2 = jax.random.split(key)
      observation = utils.add_batch_dim(observation)
      # output = policy(params, key2, observation)
      logits, value = policy(params, key2, observation)
      return (
        utils.squeeze_batch_dim(logits), 
        utils.squeeze_batch_dim(value)
        ), key

    # this policy is the state-value model
    self._policy = jax.jit(batched_policy, backend=backend)

    def forward(observation):
      (logits, value), self._random_key = self._policy(
        self._client.params, 
        self._random_key, 
        observation)
      return logits, value
    self._forward = forward

    self._adder = adder
    self._client = variable_client

    # todo: pass these in somewhere
    self._model = model
    self._probs = np.ones(
        shape=(self._num_actions,), dtype=np.float32) / self._num_actions


  def select_action(self,
                    observation: network_lib.Observation) -> types.NestedArray:
    if self._model.needs_reset:
      self._model.reset(observation)

    root = search.mcts(
        observation,
        model=self._model,
        search_policy=search.puct,
        evaluation=self._forward,
        num_simulations=self._num_simulations,
        num_actions=self._num_actions,
        discount=self._discount,
    )

    probs = search.visit_count_policy(root)
    action = np.int32(np.random.choice(self._actions, p=probs))

    # Save the policy probs so that we can add them to replay in `observe()`.
    self._probs = probs.astype(np.float32)

    # result, self._random_key = self._policy(self._client.params,
    #                                         self._random_key, observation)


    # if self._has_extras:
    #   action, self._extras = result
    # else:
    #   action = result
    # return utils.to_numpy(action)
    return action

  def observe_first(self, timestep: dm_env.TimeStep):
    self._prev_timestep = timestep
    if self._adder:
      self._adder.add_first(timestep)

  def observe(self, action: network_lib.Action, next_timestep: dm_env.TimeStep):
    print("self._prev_timestep", self._prev_timestep)
    print("next_timestep", next_timestep)
    self._model.update(self._prev_timestep, action, next_timestep)
    self._prev_timestep = next_timestep

    if self._adder:
      self._adder.add(action, next_timestep, extras={'pi': self._probs})
      # self._adder.add(action, next_timestep, extras=self._extras)

  def update(self, wait: bool = False):
    self._client.update(wait)
