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

import time, datetime

import acme
from acme import specs
from acme import datasets
from acme.jax import utils
from acme.jax import variable_utils
from acme.jax import networks as networks_lib
from acme.agents import replay
from acme.agents.jax import actors
from acme.agents.jax.dqn import learning
# from acme.agents.jax.dqn import config as dqn_config
# from acme.testing import fakes

from acme.adders import reverb as adders
from typing import Generic, List, Optional, Sequence, TypeVar
from acme import types

import tensorflow as tf
import ray
import jax
import jax.numpy as jnp
import rlax
import optax
import reverb
import numpy as np
import haiku as hk

import operator
import tree

from acme.utils import counting
from acme.utils import loggers

import functools
import gym
from acme import wrappers
import uuid
import pickle
import argparse

from variable_utils import RayVariableClient
from environment_loop import CustomEnvironmentLoop
from config import DQNConfig

# jax.config.update('jax_platform_name', "cpu")
parser = argparse.ArgumentParser(description='Run some stonks.')

parser.add_argument('--total_learning_steps', type=float, default=2e8, help='Number of training steps to run.')
parser.add_argument('--num_actors', type=int, default=5, help='Number of actors to run.')
parser.add_argument('--max_result_cache_size', type=int, default=1000, help='Max size of SharedStorage result cache.')
parser.add_argument("--force_cpu", help="Force all workers to use CPU.", action="store_true")
parser.add_argument("--enable_checkpointing", help="Learner will checkpoint at preconfigured intervals.", action="store_true")
parser.add_argument("--initial_checkpoint", help="Learner will load from initial checkpoint before training.", action="store_true")
parser.add_argument("--initial_checkpoint_path", type=str, default="initial_checkpoint", help="Initial checkpoint for learner. `initial_checkpoint` must be True.")


# flags.DEFINE_integer('total_learning_steps', 2e8, 'Number of training steps to run.')
# flags.DEFINE_integer('num_actors', 5, 'Number of actors to run.')
# flags.DEFINE_bool('force_cpu', False, 'Force all workers to use CPU.')

# flags.DEFINE_bool('enable_checkpointing', False, 'Learner will checkpoint at preconfigured intervals.')
# flags.DEFINE_bool('initial_checkpoint', False, 'Learner will load from initial checkpoint before training.')
# flags.DEFINE_string('initial_checkpoint_path', "initial_checkpoint", 'Initial checkpoint for learner. `initial_checkpoint` must be True.')

# FLAGS = flags.FLAGS

config = DQNConfig(
  learning_rate=5e-4,
  # learning_rate=625e-7,
  # samples_per_insert=0.5
)

# def environment_factory(evaluation: bool = False, level: str = 'BreakoutNoFrameskip-v4'):
#   """Creates environment."""
#   env = gym.make(level, full_action_space=True, obs_type="ram")
#   max_episode_len = 108_000 if evaluation else 50_000

#   return wrappers.wrap_all(env, [
#       wrappers.GymAtariRAMAdapter,
#       functools.partial(
#           wrappers.AtariRAMWrapper,
#           to_float=True,
#           max_episode_len=max_episode_len,
#           # zero_discount_on_life_loss=True,
#       ),
#       wrappers.SinglePrecisionWrapper,
#   ])

def environment_factory(evaluation: bool = False, level: str = 'BreakoutNoFrameskip-v4'):
  """Creates environment."""
  env = gym.make(level, full_action_space=True)
  max_episode_len = 108_000 if evaluation else 50_000

  return wrappers.wrap_all(env, [
      wrappers.GymAtariAdapter,
      functools.partial(
          wrappers.AtariWrapper,
          to_float=True,
          max_episode_len=max_episode_len,
          zero_discount_on_life_loss=True,
      ),
      wrappers.SinglePrecisionWrapper,
  ])

spec = specs.make_environment_spec(environment_factory())

def network_factory():
  """Creates network."""
  def network(x):
    model = hk.Sequential([
        networks_lib.AtariTorso(),
        hk.Flatten(),
        hk.nets.MLP([50, 50, spec.actions.num_values])
        # hk.nets.MLP([512, 1024, 2048, spec.actions.num_values])
    ])
    return model(x)

  # Make network purely functional
  network_hk = hk.without_apply_rng(hk.transform(network, apply_rng=True))
  dummy_obs = utils.add_batch_dim(utils.zeros_like(spec.observations))

  network = networks_lib.FeedForwardNetwork(
    init=lambda rng: network_hk.init(rng, dummy_obs),
    apply=network_hk.apply)

  return network

def make_actor(policy_network, random_key, adder = None, variable_source = None, temp_client_key=None):
  """Creates an actor."""
  assert variable_source is not None, "make_actor doesn't support None for `variable_source` right now"

  variable_client = RayVariableClient(
      client=variable_source,
      key='',
      # variables={'policy': policy_network.variables},
      update_period=100,
      temp_client_key=temp_client_key
  )

  variable_client.update_and_wait()

  actor = actors.FeedForwardActor(
    policy=policy_network,
    random_key=random_key,
    variable_client=variable_client, # need to write a custom wrapper around learner so it calls .remote
    adder=adder)
  return actor

def make_adder(reverb_client):
  """Creates a reverb adder."""
  return adders.NStepTransitionAdder(reverb_client, config.n_step, config.discount)

def make_learner(network, optimizer, data_iterator, reverb_client, random_key, logger=None, checkpoint=None):
  # TODO: add a sexy logger here
  learner = learning.DQNLearner(
    network=network,
    random_key=random_key,
    optimizer=optimizer,
    discount=config.discount,
    importance_sampling_exponent=config.importance_sampling_exponent,
    target_update_period=config.target_update_period,
    iterator=data_iterator,
    replay_client=reverb_client,
    logger=logger
  )
  return learner

def make_optimizer():
  optimizer = optax.chain(
    optax.clip_by_global_norm(config.max_gradient_norm),
    optax.adam(config.learning_rate),
  )
  return optimizer


class ActorLogger():
  def __init__(self, interval=1, disable_printing=False):
    self.data = []
    self.counter = 0
    self.interval = interval
    self.disable_printing = disable_printing
    if self.disable_printing: print("actor logger printing temporarily disabled")

  def write(self, s):
    self.data.append(s)
    if self.counter % self.interval == 0:
      if not self.disable_printing: print(s)
      self.counter += 1

@ray.remote
class SharedStorage():
    """
    Class which run in a dedicated thread to store the network weights and some information.
    """
    def __init__(self, max_result_cache_size=10000):
      self.max_result_cache_size = max_result_cache_size
      self.current_checkpoint = {
        "steps": 0,
        "results": [],
        "highscore": 0
      }
      self.writer = tf.summary.create_file_writer("/home/aryavohra/tf_summaries/stonks_histogram")


    def get_info(self, keys):
      if isinstance(keys, str):
        return self.current_checkpoint[keys]
      elif isinstance(keys, list):
        return {key: self.current_checkpoint[key] for key in keys}
      else:
        raise TypeError


    def add_result(self, result):
      self.current_checkpoint["results"].append(result)

      if result["episode_return"] > self.current_checkpoint["highscore"]:
        self.current_checkpoint["highscore"] = result["episode_return"]
        with self.writer.as_default():
          tf.summary.scalar(
            "actors/highscore",
            self.current_checkpoint["highscore"],
            step=self.current_checkpoint["steps"]
          )

      if len(self.current_checkpoint["results"]) > self.max_result_cache_size:
        return_cache = [r["episode_return"].item() for r in self.current_checkpoint["results"]]
        with self.writer.as_default():
          tf.summary.histogram(
            "actors/return_histogram",
            return_cache,
            step=self.current_checkpoint["steps"]
          )
        
        # clearing the result cache
        self.current_checkpoint["results"] = []

      self.current_checkpoint["steps"] += 1

    def set_info(self, keys, values=None):
      if isinstance(keys, str) and values is not None:
        self.current_checkpoint[keys] = values
      elif isinstance(keys, dict):
        self.current_checkpoint.update(keys)
      else:
        raise TypeError



@ray.remote(num_cpus=1)
class ActorRay():
  """Glorified wrapper for environment loop."""
  
  def __init__(self, reverb_address, variable_source, shared_storage, id=None, verbose=False):
    self._verbose = verbose
    self._id = str(id) or uuid.uuid1()

    self._shared_storage = shared_storage

    self._client = reverb.Client(reverb_address)

    print("A - flag 0.5")

    network = network_factory()
    def policy(params: networks_lib.Params, key: jnp.ndarray,
               observation: jnp.ndarray) -> jnp.ndarray:
      action_values = network.apply(params, observation) # how will this work when they're on different devices?
      return rlax.epsilon_greedy(config.epsilon).sample(key, action_values)


    # print("A - flag 1")
    # todo: make this proper splitting and everything
    random_key=jax.random.PRNGKey(1701)

    self._actor = make_actor(
      policy, 
      random_key,
      adder=make_adder(self._client),
      variable_source=variable_source,
      temp_client_key=self._id
    )

    print("A - flag 2")
    self._environment = environment_factory()
    self._counter = counting.Counter() # prefix='actor'
    self._logger = ActorLogger(
      # interval=10, # log every 10 steps
      # disable_printing=(type(id) == int and (id % 4 == 0)) # only get every 4th actor to print shit
    ) # TODO: use config for `interval` arg

    self._env_loop = CustomEnvironmentLoop(
      self._environment, 
      self._actor, 
      counter=self._counter,
      logger=self._logger,
      should_update=True
      )

    print("A - flag 3")


    # TODO: migrate all print statements to the logger
    # or should i? logger is for the environment loop
    if self._verbose: print(f"Actor {self._id}: instantiated on {jnp.ones(3).device_buffer.device()}.")
  
  def ready(self):
    return True

  def run(self):
    if self._verbose: print(f"Actor {self._id}: beginning training.")

    steps=0

    while not ray.get(self._shared_storage.get_info.remote("terminate")):
      result = self._env_loop.run_episode()
      result.update({
        "id": self._id
        })
      self._logger.write(result)

      self._shared_storage.add_result.remote(result)

      steps += result['episode_length']

    if self._verbose: print(f"Actor {self._id}: terminated at {steps} steps.") 
    # todo: get it to print some info here?

@ray.remote # max_concurrency=1 + N(cacher nodes)
class LearnerRay():
  def __init__(self, reverb_address, shared_storage, enable_checkpointing=False, verbose=False):
    self._verbose = verbose
    self._enable_checkpointing = enable_checkpointing
    self._shared_storage = shared_storage
    self._client = reverb.Client(reverb_address)

    print("L - flag 0.5")

    data_iterator = datasets.make_reverb_dataset(
      table="priority_table",
      server_address=reverb_address,
      batch_size=config.batch_size,
      prefetch_size=4,
    ).as_numpy_iterator()

    print("L - flag 1")
    # todo: sort out the key

    # disabled the logger because it's not toooo useful
    # self._logger = ActorLogger()
    random_key = jax.random.PRNGKey(1701)
    self._learner = make_learner(
      network_factory(), 
      make_optimizer(), 
      data_iterator, 
      self._client,
      random_key,
      # logger=self._logger
    )

    print("L - flag 2")
    print("devices:", jax.devices())
    if self._verbose: print(f"Learner: instantiated on {jnp.ones(3).device_buffer.device()}.")

  @staticmethod
  def _calculate_num_learner_steps(num_observations: int, min_observations: int, observations_per_step: float) -> int:
    """Calculates the number of learner steps to do at step=num_observations."""
    n = num_observations - min_observations
    if observations_per_step > 1:
      # One batch every 1/obs_per_step observations, otherwise zero.
      return int(n % int(observations_per_step) == 0)
    else:
      # Always return 1/obs_per_step batches every observation.
      return int(1 / observations_per_step)

  def get_variables(self, names: Sequence[str]) -> List[types.NestedArray]:
    """This has to be called by a wrapper which uses the .remote postfix."""
    return self._learner.get_variables(names)

  def save_checkpoint(self, path):
    weights_to_save = self._learner.get_variables("")

    # path = "/home/aryavohra/temp/acme/refactor_test/checkpoint"
    # path = "checkpoint"

    # todo: checkpoint_directory
    with open(path, 'wb') as f:
      pickle.dump(weights_to_save, f)

    if self._verbose: print("Learner: checkpoint saved successfully.")
    return True # todo: can we remove this?

  def load_checkpoint(self, path):
    with open(path, 'rb') as f:
      weights = pickle.load(f)

    self._learner.restore_from_single_weights(weights)

    if self._verbose: print("Learner: checkpoint restored successfully.")

    # once we've loaded the weights, wtf do we do with them

  def run(self, total_learning_steps: int = 2e8):
    if self._verbose: print("Learner: starting training.")

    while self._client.server_info()["priority_table"].current_size < max(config.batch_size, config.min_replay_size):
      time.sleep(0.1)

    observations_per_step = config.batch_size / config.samples_per_insert
    steps_completed = 0

    # TODO: migrate to the learner internal counter instance
    while steps_completed < total_learning_steps:
      steps = self._calculate_num_learner_steps(
        num_observations=self._client.server_info()["priority_table"].current_size,
        min_observations=max(config.batch_size, config.min_replay_size),
        observations_per_step=observations_per_step
        )

      for _ in range(steps):
        self._learner.step()
        steps_completed += 1

        if self._enable_checkpointing and (steps_completed % config.checkpoint_interval == 0):
          self.save_checkpoint(f"checkpoint-{steps_completed}.pickle")

        # todo: add evaluation
        # perhaps make a coordinator which runs learner for x steps, then calls an eval actor?
        # if steps_completed % config.eval_interval == 0:
        #   pass

    if self._verbose: print(f"Learner complete at {steps_completed}. Terminating actors.")
    self._shared_storage.set_info.remote({
      "terminate": True
    })

if __name__ == '__main__':
  ray.init(address="auto")

  args = parser.parse_args()

  if args.force_cpu: jax.config.update('jax_platform_name', "cpu")

  storage = SharedStorage.remote()
  storage.set_info.remote({
    "terminate": False
  })

  reverb_replay = replay.make_reverb_prioritized_nstep_replay(
      environment_spec=spec,
      n_step=config.n_step,
      batch_size=config.batch_size,
      max_replay_size=config.max_replay_size,
      min_replay_size=config.min_replay_size,
      priority_exponent=config.priority_exponent,
      discount=config.discount,
  )

  learner = LearnerRay.options(max_concurrency=2).remote(
    "localhost:8000",
    storage,
    enable_checkpointing=args.enable_checkpointing,
    verbose=True
  )

  # important to force the learner onto TPU
  ray.get(learner.get_variables.remote(""))

  # load the initial checkpoint if relevant
  if args.initial_checkpoint:
    ray.get(learner.load_checkpoint.remote(args.initial_checkpoint_path))

  actors = [ActorRay.remote(
    "localhost:8000", 
    learner, 
    storage,
    verbose=True,
    id=i
  ) for i in range(args.num_actors)] # 50

  [a.run.remote() for a in actors]

  # actor.run.remote()
  # learner.run.remote(total_learning_steps=200)
  learner.run.remote(total_learning_steps=args.total_learning_steps)


  while not ray.get(storage.get_info.remote("terminate")):
    time.sleep(1)






