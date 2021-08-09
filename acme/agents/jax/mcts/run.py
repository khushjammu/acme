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
from acme.agents.jax import dqn
from acme.jax import networks as networks_lib
from acme.jax import utils
from acme.testing import fakes
import haiku as hk
import numpy as np


# Create a fake environment to test with.
environment = fakes.DiscreteEnvironment(
    num_actions=5,
    num_observations=10,
    obs_shape=(10, 5),
    obs_dtype=np.float32,
    episode_length=10)
spec = specs.make_environment_spec(environment)

model = simulator.Simulator(environment)

def network(x):
  model = hk.Sequential([
      hk.Flatten(), 
      hk.nets.MLP([256, 1024, 2048]),
      networks.PolicyValueHead(spec.actions.num_values)
  ])
  return model(x)

# Make network purely functional
network_hk = hk.without_apply_rng(hk.transform(network, apply_rng=True))
dummy_obs = utils.add_batch_dim(utils.zeros_like(spec.observations))


network = networks_lib.FeedForwardNetwork(
    init=lambda rng: network_hk.init(rng, dummy_obs),
    apply=network_hk.apply)

key_learner, key_actor = jax.random.split(jax.random.PRNGKey(config.seed))



loss_fn = mcts.MCTSLoss()
optimizer = optax.adam(5e-4)

reverb_replay = replay.make_reverb_prioritized_nstep_replay(
    environment_spec=specs.make_environment_spec(builder.environment_factory()),
    n_step=config.n_step,
    batch_size=config.batch_size,
    max_replay_size=config.max_replay_size,
    min_replay_size=config.min_replay_size,
    priority_exponent=config.priority_exponent,
    discount=config.discount,
)

learner = mcts.MCTSLearner(
  network=network,
  loss_fn=loss_fn,
  optimizer=optimizer,
  data_iterator=reverb_replay.data_iterator,
  random_key=key_learner,
  # replay_client: Optional[reverb.Client] = None,
  # counter: Optional[counting.Counter] = None,
  # logger: Optional[loggers.Logger] = None,
  )



# Construct the agent.

actor = mcts.MCTSActor(
    policy=network,
    random_key=key_actor,
    variable_client=VariableClient(learner, ''),
    model = model, # todo: sort out environment model
    # adder: Optional[adders.Adder] = None,
)

# agent = dqn.DQN(
#     environment_spec=spec,
#     network=network,
#     batch_size=10,
#     samples_per_insert=2,
#     min_replay_size=10)

# Try running the environment loop. We have no assertions here because all
# we care about is that the agent runs without raising any errors.
loop = acme.EnvironmentLoop(environment, actor)
loop.run(num_episodes=20)
