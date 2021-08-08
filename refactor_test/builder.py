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
from acme.agents.jax import actors
from acme.agents.jax.dqn import learning

from acme.adders import reverb as adders
from typing import Generic, List, Optional, Sequence, TypeVar

import optax
import haiku as hk

from acme.utils import counting
from acme.utils import loggers

import functools
import gym
from acme import wrappers

from variable_utils import RayVariableClient


class Builder():
  def __init__(self, config):
    self.config = config

  def environment_factory(self, evaluation: bool = False, level: str = 'BreakoutNoFrameskip-v4'):
    """Creates environment."""
    # todo: add configurable ram-states
    env = gym.make(level, full_action_space=True, obs_type="ram")
    max_episode_len = 108_000 if evaluation else 50_000

    return wrappers.wrap_all(env, [
        wrappers.GymAtariRAMAdapter,
        functools.partial(
            wrappers.AtariRAMWrapper,
            to_float=True,
            max_episode_len=max_episode_len,
            # zero_discount_on_life_loss=True,
        ),
        wrappers.SinglePrecisionWrapper,
    ])

    # env = gym.make(level, full_action_space=True)
    # max_episode_len = 108_000 if evaluation else 50_000

    # return wrappers.wrap_all(env, [
    #     wrappers.GymAtariAdapter,
    #     functools.partial(
    #         wrappers.AtariWrapper,
    #         to_float=True,
    #         max_episode_len=max_episode_len,
    #         zero_discount_on_life_loss=True,
    #     ),
    #     wrappers.SinglePrecisionWrapper,
    # ])
  

  def network_factory(self):
    """Creates network."""
    spec = specs.make_environment_spec(self.environment_factory())

    def network(x):
      model = hk.Sequential([
          # networks_lib.AtariTorso(),
          hk.Flatten(),
          # hk.nets.MLP([50, 50, spec.actions.num_values])
          hk.nets.MLP([512, 1024, 2048, spec.actions.num_values])
      ])
      return model(x)

    # Make network purely functional
    network_hk = hk.without_apply_rng(hk.transform(network, apply_rng=True))
    dummy_obs = utils.add_batch_dim(utils.zeros_like(spec.observations))

    network = networks_lib.FeedForwardNetwork(
      init=lambda rng: network_hk.init(rng, dummy_obs),
      apply=network_hk.apply)

    return network

  def make_actor(self, policy_network, random_key, adder = None, variable_source = None, temp_client_key=None):
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

    actor = actors.FeedForwardActor(
      policy=policy_network,
      random_key=random_key,
      variable_client=variable_client, # need to write a custom wrapper around learner so it calls .remote
      adder=adder)
    return actor

  def make_adder(self, reverb_client):
    """Creates a reverb adder."""
    return adders.NStepTransitionAdder(reverb_client, self.config.n_step, self.config.discount)

  def make_learner(self, network, optimizer, data_iterator, reverb_client, random_key, logger=None, checkpoint=None):
    # TODO: add a sexy logger here
    learner = learning.DQNLearner(
      network=network,
      random_key=random_key,
      optimizer=optimizer,
      discount=self.config.discount,
      importance_sampling_exponent=self.config.importance_sampling_exponent,
      target_update_period=self.config.target_update_period,
      iterator=data_iterator,
      replay_client=reverb_client,
      logger=logger,
    )
    return learner

  def make_optimizer(self):
    optimizer = optax.chain(
      optax.clip_by_global_norm(self.config.max_gradient_norm),
      optax.adam(self.config.learning_rate),
    )
    return optimizer