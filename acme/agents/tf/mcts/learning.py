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

"""A MCTS "AlphaZero-style" learner."""

from typing import List, Optional

import acme
from acme.tf import utils as tf2_utils
from acme.utils import counting
from acme.utils import loggers
import numpy as np
import sonnet as snt
import tensorflow as tf


class AZLearner(acme.Learner):
  """AlphaZero-style learning."""

  def __init__(
      self,
      network: snt.Module,
      optimizer: snt.Optimizer,
      dataset: tf.data.Dataset,
      discount: float,
      logger: Optional[loggers.Logger] = None,
      counter: Optional[counting.Counter] = None,
  ):

    # Logger and counter for tracking statistics / writing out to terminal.
    self._counter = counting.Counter(counter, 'learner')
    self._logger = logger or loggers.TerminalLogger('learner', time_delta=30.)

    # Internalize components.
    # TODO(b/155086959): Fix type stubs and remove.
    self._iterator = iter(dataset)  # pytype: disable=wrong-arg-types
    self._optimizer = optimizer
    self._network = network
    self._variables = network.trainable_variables
    self._discount = np.float32(discount)

  # @tf.function
  def _step(self) -> tf.Tensor:
    """Do a step of SGD on the loss."""

    inputs = next(self._iterator)
    # print("inputs:", inputs)
    o_t, _, r_t, d_t, o_tp1, extras = inputs.data
    pi_t = extras['pi']

    with tf.GradientTape() as tape:
      # Forward the network on the two states in the transition.
      logits, value = self._network(o_t)
      _, target_value = self._network(o_tp1)
      target_value = tf.stop_gradient(target_value)

      
      r_t = np.asarray([0., 0., 1., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=np.float32)

      scaled_discount = np.asarray(
      [0.99, 0.99, 0., 0.99, 0.99, 0.99, 0., 0., 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99], dtype=np.float32)

      logits = np.asarray(
      [[ 0.07668373,  0.00641519,  0.06191878],
       [ 0.18665414, -0.14294943,  0.01194256],
       [ 0.18619043, -0.18134044,  0.17319885],
       [ 0.03082304,  0.04289203,  0.12380577],
       [-0.0079467,   0.05012767,  0.04710817],
       [ 0.02678695, -0.07670858,  0.02872688],
       [ 0.18619043, -0.18134044,  0.17319885],
       [ 0.18619043, -0.18134044,  0.17319885],
       [-0.06816303,  0.01560557,  0.0005657],
       [-0.0079467,   0.05012767,  0.04710817],
       [-0.06816303,  0.01560557,  0.0005657],
       [-0.00499794, -0.0093834,   0.10516889],
       [-0.00499794, -0.0093834,   0.10516889],
       [ 0.07668373,  0.00641519,  0.06191878],
       [ 0.03082304,  0.04289203,  0.12380577],
       [ 0.07668373,  0.00641519,  0.06191878]], dtype=np.float32)

      value = np.asarray(
      [-0.01639128, -0.10810072, -0.17826547, -0.01005482, -0.03273902, -0.07094201,
       -0.17826547, -0.17826547, -0.06661676, -0.03273902, -0.06661676, -0.06111332,
       -0.06111332, -0.01639128, -0.01005482, -0.01639128], dtype=np.float32)

      target_value = np.asarray(
      [-0.10810072, -0.06111332, -0.00666812, -0.24884203, -0.06147643, -0.06661676,
       -0.00666812, -0.00666812,  0.02674196, -0.06147643,  0.02674196, -0.07094201,
       -0.07094201, -0.10810072, -0.24884203, -0.10810072], dtype=np.float32)

      pi_t = np.asarray(
      [[0.34, 0.32, 0.34],
       [0.44, 0.24, 0.32],
       [0.02, 0.02, 0.96],
       [0.34, 0.34, 0.32],
       [0.18, 0.64, 0.18],
       [0.36, 0.3,  0.34],
       [0.02, 0.02, 0.96],
       [0.02, 0.02, 0.96],
       [0.32, 0.36, 0.32],
       [0.18, 0.64, 0.18],
       [0.32, 0.36, 0.32],
       [0.32, 0.32, 0.36],
       [0.32, 0.32, 0.36],
       [0.34, 0.32, 0.34],
       [0.34, 0.34, 0.32],
       [0.34, 0.32, 0.34]], dtype=np.float32)


      # print("r_t:", r_t.numpy())
      # print("scaled_discount:", (self._discount * d_t).numpy())
      # print("logits:", logits.numpy())
      # print("value:", value.numpy())
      # print("target_value:", target_value.numpy())
      # print("pi_t:", pi_t.numpy())

      # Value loss is simply on-policy TD learning.
      value_loss = tf.square(r_t + scaled_discount * target_value - value)
      # value_loss = tf.square(r_t + self._discount * d_t * target_value - value)

      # Policy loss distills MCTS policy into the policy network.
      policy_loss = tf.nn.softmax_cross_entropy_with_logits(
          logits=logits, labels=pi_t)

      # Compute gradients.
      loss = tf.reduce_mean(value_loss + policy_loss)
      print("value_loss:", tf.reduce_mean(value_loss))
      print("policy_loss:", tf.reduce_mean(policy_loss))
      print("loss:", loss)
      import sys; sys.exit(-1)
      gradients = tape.gradient(loss, self._network.trainable_variables)

    self._optimizer.apply(gradients, self._network.trainable_variables)

    return loss

  def step(self):
    """Does a step of SGD and logs the results."""
    loss = self._step()
    self._logger.write({'loss': loss})

  def get_variables(self, names: List[str]) -> List[List[np.ndarray]]:
    """Exposes the variables for actors to update from."""
    return tf2_utils.to_numpy(self._variables)
