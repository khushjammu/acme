"""MCTS actor implementation."""

from typing import Optional

from acme import adders
from acme import core
from acme.agents.jax.mcts import types
from acme.jax import variable_utils
import dm_env
import haiku as hk
import jax
import jax.numpy as jnp


class MCTSActor(core.Actor):
  def __init__(
      self,
      environment_spec: specs.EnvironmentSpec,
      model_fn: types.ModelFn,
      policy_value_fn: types.PolicyValueFn,
      num_simulations: int,
      rng_key: networks_lib.PRNGKey,
      variable_client: Optional[variable_utils.VariableClient] = None,
      adder: Optional[adders.Adder] = None,
  ):

  # Store these for later use.
  self._adder = adder
  self._variable_client = variable_client
  self._model_fn = model_fn
  self._policy_value_fn = policy_value_fn
  self._rng_key = rng_key

  # Internalize hyperparameters.
  self._num_actions = environment_spec.actions.num_values
  self._num_simulations = num_simulations
  self._actions = list(range(self._num_actions))
  self._discount = discount # we got rid of this before, right?

  # Make sure not to use a random policy after checkpoint restoration by
  # assigning variables before running the environment loop.
  if self._variable_client is not None:
    self._variable_client.update_and_wait()

  # We need to save the policy so as to add it to replay on the next step.
  self._probs = np.ones(shape=(self._num_actions,), dtype=np.float32) / self._num_actions


  def _forward(self, observation: types.Observation) -> Tuple[types.Probs, types.Value]:
    """Performs a forward pass of the policy-value network."""
    logits, value = self._network(observation)
    probs = jax.nn.softmax(logits)

    return probs, value


  def select_action(self, observation: types.Observation) -> types.Action:
  	"""Builds MCTS plan to select optimal action for a given observation."""
    if self._state is None:
      self._state = self._initial_state

    # Compute a fresh MCTS plan.
    root = search.mcts(
    		next(self._rng),
        observation,
        model=self._model_fn,
        search_policy=search.puct,
        evaluation=self._forward,
        num_simulations=self._num_simulations,
        num_actions=self._num_actions,
        discount=self._discount,
    )

    # The agent's policy is softmax w.r.t. the *visit counts* as in AlphaZero.
		probs = search.visit_count_policy(root)
		action = jax.random.choice(next(self._rng), self._actions, p=probs)

		# Save the policy probs so that we can add them to replay in `observe()`.
		self._probs = probs.astype(jnp.float32)

		return action


		def update(self, wait: bool = False):
		  """Fetches the latest variables from the variable source, if needed."""
		  if self._variable_client:
		    self._variable_client.update(wait)


		def observe_first(self, timestep: dm_env.TimeStep):
		  self._prev_timestep = timestep
		  if self._adder:
		    self._adder.add_first(timestep)


		def observe(self, action: types.Action, next_timestep: dm_env.TimeStep):
		  """Updates the agent's internal model and adds the transition to replay."""
		  self._model.update(self._prev_timestep, action, next_timestep) # this doesn't really conform to jax principles...
		  self._prev_timestep = next_timestep

		  if self._adder:
		    self._adder.add(action, next_timestep, extras={'pi': self._probs})


  # @property
  # def _params(self) -> Optional[hk.Params]:
  #   if self._variable_client is None:
  #     # If self._variable_client is None then we assume self._forward  does not
  #     # use the parameters it is passed and just return None.
  #     return None
  #   return self._variable_client.params
