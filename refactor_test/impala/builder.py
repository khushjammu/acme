"""
Actor
- just run continuously, populating the self-play buffer until learner sends a terminate signal

Learner
- continuously sample from the replay buffer until has done sufficient learning steps
- at regular intervals, perform an evaluation

Cache # ?
- fetch params from the learner every x seconds

SharedStorage
- store the terminate signal

CustomConfig
- holds all the config-related stuff (e.g. print intervals, learning rate)
"""

from acme import specs
from acme.jax import utils
from acme.jax import networks as networks_lib
from acme.agents import replay
from acme.agents.jax import actors
from acme.agents.jax.impala import learning

from acme.adders import reverb as adders
from typing import Generic, List, Optional, Sequence, TypeVar

import jax.numpy as jnp
import optax
import haiku as hk
import numpy as np

from acme.utils import counting
from acme.utils import loggers

import functools
import gym
from acme import wrappers

from variable_utils import RayVariableClient


class IMPALARAMNetwork(hk.RNNCore):
  """A simple recurrent network for RAM states."""

  def __init__(self, num_actions: int):
    super().__init__(name='impala_ram_network')
    self._torso = hk.Sequential([
        lambda x: jnp.reshape(x, [np.prod(x.shape)]),
        hk.nets.MLP([50, 50]),
    ])
    self._core = hk.LSTM(20)
    self._head = networks_lib.PolicyValueHead(num_actions)

  def __call__(self, inputs, state):
    embeddings = self._torso(inputs)
    embeddings, new_state = self._core(embeddings, state)
    logits, value = self._head(embeddings)
    return (logits, value), new_state

  def initial_state(self, batch_size: int):
    return self._core.initial_state(batch_size)


class Builder():
  def __init__(self, config):
    self.config = config
    self.spec = specs.make_environment_spec(self.environment_factory())

  def environment_factory(self, evaluation: bool = False, level: str = 'BreakoutNoFrameskip-v4'):
    """Creates environment."""
    # todo: add configurable ram-states
    env = gym.make(level, full_action_space=True, obs_type="ram")
    max_episode_len = 108_000 if evaluation else 50_000

    return wrappers.wrap_all(env, [
        wrappers.GymAtariRAMAdapter,
        # wrappers.GymAtariAdapter,
        functools.partial(
            wrappers.AtariRAMWrapper,
            # wrappers.AtariWrapper,
            to_float=True,
            max_episode_len=max_episode_len,
            # zero_discount_on_life_loss=True,
        ),
        wrappers.SinglePrecisionWrapper,
    ])

  def network_factory(self):
    """Creates networks."""

    def forward_fn(x, s):
      model = IMPALARAMNetwork(self.spec.actions.num_values)
      return model(x, s)

    def initial_state_fn(batch_size: Optional[int] = None):
      model = IMPALARAMNetwork(self.spec.actions.num_values)
      return model.initial_state(batch_size)

    def unroll_fn(inputs, state):
      model = IMPALARAMNetwork(self.spec.actions.num_values)
      return hk.static_unroll(model, inputs, state)

    forward_fn_transformed = hk.without_apply_rng(hk.transform(
      forward_fn,
      apply_rng=True))
    unroll_fn_transformed = hk.without_apply_rng(hk.transform(
      unroll_fn,
      apply_rng=True))
    initial_state_fn_transformed = hk.without_apply_rng(hk.transform(
      initial_state_fn,
      apply_rng=True))

    return forward_fn_transformed, unroll_fn_transformed, initial_state_fn_transformed

  def make_actor(
      self,
      forward_fn,
      initial_state_init_fn, initial_state_fn,
      random_key,
      adder=None,
      variable_source=None,
      temp_client_key=None
    ):
    """Creates an actor."""
    assert variable_source is not None, "make_actor doesn't support None for `variable_source` right now"

    variable_client = RayVariableClient(
        client=variable_source,
        key='',
        # variables={'policy': policy_network.variables},
        update_period=1,
        temp_client_key=temp_client_key
    )

    variable_client.update_and_wait()

    acting.IMPALAActor(
      forward_fn=jax.jit(forward_fn, backend='cpu'),
      initial_state_init_fn=initial_state_init_fn,
      initial_state_fn=initial_state_fn,
      rng=hk.PRNGSequence(random_key),
      adder=adder,
      variable_client=variable_client,
    )

    return actor

  def make_adder(self, reverb_client):
    """Creates a reverb adder."""
    return adders.SequenceAdder(
      client=reverb_client,
      period=self.config.sequence_period,
      sequence_length=self.config.sequence_length,
    )
    #return adders.NStepTransitionAdder(reverb_client, self.config.n_step, self.config.discount)

  def make_learner(
      self,
      unroll_init_fn, unroll_fn,
      initial_state_init_fn, initial_state_fn,
      optimizer,
      data_iterator,
      reverb_client,
      random_key,
      logger=None,
      checkpoint=None
    ):
    # TODO: add a sexy logger here
    # TODO: remove checkpoint=None ?

    learner = learning.IMPALALearner(
      obs_spec=self.spec.observations,
      unroll_init_fn=unroll_init_fn,
      unroll_fn=unroll_fn,
      initial_state_init_fn=initial_state_init_fn,
      initial_state_fn=initial_state_fn,
      iterator=data_iterator,
      random_key=random_key,
      # counter=counter,
      logger=logger,
      optimizer=optimizer,
      discount=self.config.discount,
      entropy_cost=self.config.entropy_cost,
      baseline_cost=self.config.baseline_cost,
      max_abs_reward=self.config.max_abs_reward,
    )

    return learner

  def make_optimizer(self):
    schedule = optax.linear_schedule(
      init_value=self.config.learning_rate,
      end_value=self.config.terminal_learning_rate,
      transition_steps=self.config.schedule_steps
    )

    optimizer = optax.chain(
      optax.clip_by_global_norm(self.config.max_gradient_norm),
      optax.adam(schedule),
    )
    return optimizer
