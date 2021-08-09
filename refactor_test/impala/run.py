from acme import specs
from acme import types
from acme import datasets
from acme.utils import counting
from acme.jax import networks as networks_lib
from acme.agents import replay
from typing import Generic, List, Optional, Sequence, TypeVar

import ray
import jax
import jax.numpy as jnp
import rlax
import reverb
import tensorflow as tf
import numpy as np

import os
import time
import uuid
import pickle
import argparse
import datetime

from variable_utils import RayVariableClient
from environment_loop import CustomEnvironmentLoop
from config import IMPALAConfig
from builder import Builder
from loggers import ActorLogger, LearnerTensorboardLogger

parser = argparse.ArgumentParser(description='Run some stonks.')

parser.add_argument('--total_learning_steps', type=float, default=2e8, help='Number of training steps to run.')
parser.add_argument('--num_actors', type=int, default=5, help='Number of actors to run.')
parser.add_argument('--max_result_cache_size', type=int, default=1000, help='Max size of SharedStorage result cache.')
parser.add_argument("--force_cpu", help="Force all workers to use CPU.", action="store_true")
parser.add_argument("--enable_checkpointing", help="Learner will checkpoint at preconfigured intervals.", action="store_true")
parser.add_argument("--initial_checkpoint", help="Learner will load from initial checkpoint before training.", action="store_true")
parser.add_argument("--initial_checkpoint_path", type=str, default="initial_checkpoint", help="Initial checkpoint for learner. `initial_checkpoint` must be True.")
parser.add_argument("--enable_tensorboard", help="Learner and actor will write key statistics to tensorboard.", action="store_true")


config = IMPALAConfig(
  learning_rate=2e-4,
)

builder = Builder(config)

@ray.remote
class SharedStorage():
    """
    Class which run in a dedicated thread to store the network weights and some information.
    """
    def __init__(self, max_result_cache_size=1000, log_dir=None):
      self.max_result_cache_size = max_result_cache_size
      self.current_checkpoint = {
        "steps": 0,
        "results": [],
        "highscore": 0,
        "printed_warning": False # used to print a warning ONCE if self.writer is None and we try to add_result
      }
      if log_dir:
        self.writer = tf.summary.create_file_writer(log_dir) # "/home/aryavohra/tf_summaries/stonks_histogram"
      else:
        self.writer = None

    def get_info(self, keys):
      if isinstance(keys, str):
        return self.current_checkpoint[keys]
      elif isinstance(keys, list):
        return {key: self.current_checkpoint[key] for key in keys}
      else:
        raise TypeError

    def add_result(self, result):
      self.current_checkpoint["results"].append(result)
      
      if self.writer:
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
            tf.summary.scalar(
              f"actors/past_{self.max_result_cache_size}_avg_return",
              sum(return_cache)/len(return_cache),
              step=self.current_checkpoint["steps"]
            )

            tf.summary.histogram(
              "actors/return_histogram",
              return_cache,
              step=self.current_checkpoint["steps"]
            )

          # clearing the result cache
          self.current_checkpoint["results"] = []
      else:
        if not self.current_checkpoint["printed_warning"]:
          print("WARNING: shared storage `add_result` called but tensorboard not enabled. Silencing future warnings...")
          self.current_checkpoint["printed_warning"] = True
          
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
  
  def __init__(self, reverb_address, variable_source, shared_storage, log_dir=None, id=None, verbose=False):
    self._verbose = verbose
    self._id = str(id) or uuid.uuid1()

    self._shared_storage = shared_storage

    self._client = reverb.Client(reverb_address)

    print("A - flag 0.5")

    forward_fn_transformed, \
    unroll_fn_transformed, \
    initial_state_fn_transformed = builder.network_factory()

    # print("A - flag 1")
    # todo: make this proper splitting and everything
    random_key=jax.random.PRNGKey(1701)

    self._actor = builder.make_actor(
      forward_fn_transformed.apply,
      initial_state_fn_transformed.init,
      initial_state_fn_transformed.apply,
      random_key,
      adder=builder.make_adder(self._client),
      variable_source=variable_source,
      temp_client_key=self._id
    )

    print("A - flag 2")
    self._environment = builder.environment_factory()
    self._counter = counting.Counter() # prefix='actor'
    self._logger = ActorLogger() # TODO: use config for `interval` arg

    self._env_loop = CustomEnvironmentLoop(
      self._environment, 
      self._actor, 
      counter=self._counter,
      logger=self._logger,
      should_update=True
    )

    if log_dir:
      self._tensorboard_writer = tf.summary.create_file_writer(f"{log_dir}/actor-{self._id}")
    else:
      self._tensorboard_writer = None

    print("A - flag 3")


    if self._verbose: print(f"Actor {self._id}: instantiated on {jnp.ones(3).device_buffer.device()}.")
  
  def ready(self):
    return True

  def log_to_tensorboard(self, result):
    """Logs statistics to `self._tensorboard_logger`."""

    with self._tensorboard_writer.as_default():
      tf.summary.scalar("episode_return", result["episode_return"], step=result["episodes"])
      tf.summary.scalar("episode_length", result["episode_length"], step=result["episodes"])
      tf.summary.scalar("steps_per_second", result["steps_per_second"], step=result["episodes"])
      tf.summary.scalar("total_steps", result["steps"], step=result["episodes"])
    
  def run(self):
    if self._verbose: print(f"Actor {self._id}: beginning training.")

    steps=0

    while not ray.get(self._shared_storage.get_info.remote("terminate")):
      result = self._env_loop.run_episode()
      result.update({
        "id": self._id
        })

      if self._tensorboard_writer:
        self.log_to_tensorboard(result)

      self._logger.write(result)

      self._shared_storage.add_result.remote(result) # we add to shared storage too for calculating return distribution etc.
      steps += result['episode_length']

    if self._verbose: print(f"Actor {self._id}: terminated at {steps} steps.") 

@ray.remote(resources={"tpu": 1})
class LearnerRay():
  def __init__(self, reverb_address, shared_storage, random_key, log_dir=None, enable_checkpointing=False, verbose=False):
    self._verbose = verbose
    self._enable_checkpointing = enable_checkpointing
    self._shared_storage = shared_storage
    self._client = reverb.Client(reverb_address)

    print("devices:", jax.devices())

    print("L - flag 0.5")


    # disabled the logger because it's not toooo useful
    # self._logger = ActorLogger()

    if log_dir:
      self._tensorboard_writer = tf.summary.create_file_writer(f"{log_dir}/learner")
      self._tensorboard_logger = LearnerTensorboardLogger(self._tensorboard_writer)
    else:
      self._tensorboard_writer = None
      self._tensorboard_logger = None

    # tensorboard_logging_func = log_to_tensorboard if log_dir else 

    # learner {'steps': 17, 'total_loss': DeviceArray(0.01486394, dtype=float32)}

    forward_fn_transformed, \
    unroll_fn_transformed, \
    initial_state_fn_transformed = builder.network_factory()

    optimizer = builder.make_optimizer()

    print("L - flag 1")

    data_iterator = datasets.make_reverb_dataset(
      table="priority_table",
      server_address=reverb_address,
      batch_size=config.batch_size,
      prefetch_size=4,
    ).as_numpy_iterator()

    print("L - flag 2")

    self._learner = builder.make_learner(
      unroll_fn_transformed.init,
      unroll_fn_transformed.apply,
      initial_state_fn_transformed.init,
      initial_state_fn_transformed.apply,
      optimizer, 
      data_iterator, 
      self._client,
      random_key,
      logger=self._tensorboard_logger
    )

    print("L - flag 3")
    
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

  def save_checkpoint(self, path: str):
    """Saves entire learner state to a checkpoint file."""

    state_to_save = self._learner.save()

    # create directory if doesn't exist
    if not os.path.exists(config.base_checkpoint_dir):
        os.makedirs(config.base_checkpoint_dir)

    with open(config.base_checkpoint_dir + path, 'wb') as f:
      pickle.dump(state_to_save, f)

    if self._verbose: print("Learner: checkpoint saved successfully.")
    return True # todo: can we remove this?

  def load_checkpoint(self, path):
    with open(path, 'rb') as f:
      state = pickle.load(f)

    self._learner.restore(state)
    self._shared_storage.set_info.remote({
      "steps": state.steps
    })

    if self._verbose: print("Learner: checkpoint restored successfully.")

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

    if self._verbose: print(f"Learner complete at {steps_completed}. Terminating actors.")
    self._shared_storage.set_info.remote({
      "terminate": True
    })

if __name__ == '__main__':
  ray.init()
  # ray.init(address="auto")

  args = parser.parse_args()

  if args.force_cpu: jax.config.update('jax_platform_name', "cpu")
    
  LOG_DIR = config.base_log_dir + str(datetime.datetime.now()) + "/" if args.enable_tensorboard else None

  storage = SharedStorage.remote(
    max_result_cache_size=config.universal_stats_interval,
    log_dir=LOG_DIR)

  storage.set_info.remote({
    "terminate": False
  })

  # todo: sort out the key
  random_key = jax.random.PRNGKey(1701)

  forward_fn_transformed, \
  unroll_fn_transformed, \
  initial_state_fn_transformed = builder.network_factory()

  extra_spec = {
    'core_state': initial_state_fn_transformed.apply(initial_state_fn_transformed.init(random_key)),
    'logits': np.ones(shape=(builder.spec.actions.num_values,), dtype=np.float32)
  }

  r_queue = replay.make_reverb_online_queue(
    environment_spec=builder.spec,
    extra_spec=extra_spec,
    max_queue_size=builder.config.max_queue_size,
    sequence_length=builder.config.sequence_length,
    sequence_period=builder.config.sequence_period,
    batch_size=builder.config.batch_size,
  )

  print("devices:", jax.devices())

  learner = LearnerRay.options(max_concurrency=2).remote(
    "localhost:8000",
    storage,
    random_key,
    log_dir=LOG_DIR, 
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
    log_dir=LOG_DIR,
    verbose=True,
    id=i
  ) for i in range(args.num_actors)] # 50

  [a.run.remote() for a in actors]

  learner.run.remote(total_learning_steps=args.total_learning_steps)

  while not ray.get(storage.get_info.remote("terminate")):
    time.sleep(1)
