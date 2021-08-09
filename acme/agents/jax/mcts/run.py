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

"""Tests for DQN agent."""

from absl.testing import absltest
import acme
from acme import specs
from acme import wrappers
from acme.agents import agent
from acme.agents import replay
from acme.agents.jax import dqn
from acme.agents.jax.mcts.models import simulator
from acme.jax import networks as networks_lib
from acme.jax import utils
from acme.testing import fakes
import haiku as hk
import numpy as np
import optax
import jax
import jax.numpy as jnp
import bsuite

from acme.jax.variable_utils import VariableClient

from learning import MCTSLoss, MCTSLearner
from acting import MCTSActor

import config

config = config.MCTSConfig()


environment = bsuite.load_from_id("catch/0")
environment = wrappers.SinglePrecisionWrapper(raw_environment)
spec = specs.make_environment_spec(environment)

# Create a fake environment to test with.
# environment = fakes.DiscreteEnvironment(
#     num_actions=5,
#     num_observations=10,
#     obs_shape=(10, 5),
#     obs_dtype=np.float32,
#     episode_length=10)
# spec = specs.make_environment_spec(environment)

model = simulator.Simulator(environment)

def network(x):
  model = hk.Sequential([
      hk.Flatten(), 
      hk.nets.MLP([256, 1024]),
      networks_lib.PolicyValueHead(spec.actions.num_values)
  ])
  return model(x)

# Make network purely functional
network_hk = hk.without_apply_rng(hk.transform(network, apply_rng=True))
dummy_obs = utils.add_batch_dim(utils.zeros_like(spec.observations))


network = networks_lib.FeedForwardNetwork(
    init=lambda rng: network_hk.init(rng, dummy_obs),
    apply=network_hk.apply)

key_learner, key_actor = jax.random.split(jax.random.PRNGKey(config.seed))



loss_fn = MCTSLoss()
optimizer = optax.adam(5e-4)

extra_spec = {
    'pi':
        specs.Array(
            shape=(spec.actions.num_values,), dtype=np.float32)
}

print("FUCK YOU KHUSH BATCH SIZE:", config.batch_size)

reverb_replay = replay.make_reverb_prioritized_nstep_replay(
    environment_spec=spec,
    extra_spec=extra_spec,
    n_step=config.n_step,
    batch_size=config.batch_size,
    max_replay_size=config.max_replay_size,
    min_replay_size=1,
    # min_replay_size=config.min_replay_size,
    priority_exponent=config.priority_exponent,
    discount=config.discount,
)

learner = MCTSLearner(
  network=network,
  loss_fn=loss_fn,
  optimizer=optimizer,
  data_iterator=reverb_replay.data_iterator,
  random_key=key_learner,
  # replay_client: Optional[reverb.Client] = None,
  # counter: Optional[counting.Counter] = None,
  # logger: Optional[loggers.Logger] = None,
  )

def policy(params: networks_lib.Params,key,
           observation: jnp.ndarray) -> jnp.ndarray:
  return network.apply(params, observation)

# Construct the agent.

actor = MCTSActor(
    policy=policy,
    random_key=key_actor,
    variable_client=VariableClient(learner, ''),
    num_actions=spec.actions.num_values,
    num_simulations=10,
    discount=1.,
    model=model, # todo: sort out environment model
    adder=reverb_replay.adder
    # adder: Optional[adders.Adder] = None,
)

class MCTS(agent.Agent):
  """A single-process MCTS agent."""

  def __init__(self, actor, learner):
    # Now create the agent components: actor & learner.
    actor = actor
    learner = learner
    # The parent class combines these together into one 'agent'.
    super().__init__(
        actor=actor,
        learner=learner,
        min_observations=20,
        observations_per_step=1,
    )

myagent = MCTS(actor, learner)


# agent = dqn.DQN(
#     environment_spec=spec,
#     network=network,
#     batch_size=10,
#     samples_per_insert=2,
#     min_replay_size=10)

# Try running the environment loop. We have no assertions here because all
# we care about is that the agent runs without raising any errors.
loop = acme.EnvironmentLoop(environment, myagent)
loop.run(num_episodes=20)
