"""A Monte Carlo Tree Search implementation."""

from typing import Callable, Dict

from acme.agents.tf.mcts import models
from acme.agents.tf.mcts import types

import dataclasses
import jax.numpy as jnp


@dataclasses.dataclass
class Node:
  """A MCTS node."""

  reward: float = 0.
  visit_count: int = 0
  terminal: bool = False
  prior: float = 1.
  total_value: float = 0.
  children: Dict[types.Action, 'Node'] = dataclasses.field(default_factory=dict)

  def expand(self, prior: jnp.ndarray):
    """Expands this node, adding child nodes."""
    assert prior.ndim == 1  # Prior should be a flat vector.
    for a, p in enumerate(prior):
      self.children[a] = Node(prior=p)

  @property
  def value(self) -> types.Value:  # Q(s, a)
    """Returns the value from this node."""
    if self.visit_count:
      return self.total_value / self.visit_count
    return 0.

  @property
  def children_visits(self) -> jnp.ndarray:
    """Return array of visit counts of visited children."""
    return jnp.array([c.visit_count for c in self.children.values()])

  @property
  def children_values(self) -> jnp.ndarray:
    """Return array of values of visited children."""
    return jnp.array([c.value for c in self.children.values()])


SearchPolicy = Callable[[Node], types.Action]


def mcts(
    rng_key: networks.PRNGKey,
    observation: types.Observation,
    model: models.Model,
    search_policy: SearchPolicy,
    evaluation: types.EvaluationFn,
    num_simulations: int,
    num_actions: int,
    discount: float = 1.,
    dirichlet_alpha: float = 1,
    exploration_fraction: float = 0.,
) -> Node:
  """Does Monte Carlo tree search (MCTS), AlphaZero style."""

  # Evaluate the prior policy for this state.
  prior, value = evaluation(observation)
  assert prior.shape == (num_actions,)

  # Add exploration noise to the prior.
  noise = jnp.random.dirichlet(rng_key, alpha=[dirichlet_alpha] * num_actions)
  prior = prior * (1 - exploration_fraction) + noise * exploration_fraction

  # Create a fresh tree search.
  root = Node()
  root.expand(prior)

  # Save the model state so that we can reset it for each simulation.
  model.save_checkpoint()
  for _ in range(num_simulations):
    # Start a new simulation from the top.
    trajectory = [root]
    node = root

    # Generate a trajectory.
    timestep = None
    while node.children:
      # Select an action according to the search policy.
      action = search_policy(node)

      # Point the node at the corresponding child.
      node = node.children[action]

      # Step the simulator and add this timestep to the node.
      timestep = model.step(action)
      node.reward = timestep.reward or 0.
      node.terminal = timestep.last()
      trajectory.append(node)

    if timestep is None:
      raise ValueError('Generated an empty rollout; this should not happen.')

    # Calculate the bootstrap for leaf nodes.
    if node.terminal:
      # If terminal, there is no bootstrap value.
      value = 0.
    else:
      # Otherwise, bootstrap from this node with our value function.
      prior, value = evaluation(timestep.observation)

      # We also want to expand this node for next time.
      node.expand(prior)

    # Load the saved model state.
    model.load_checkpoint()

    # Monte Carlo back-up with bootstrap from value function.
    ret = value
    while trajectory:
      # Pop off the latest node in the trajectory.
      node = trajectory.pop()

      # Accumulate the discounted return
      ret *= discount
      ret += node.reward

      # Update the node.
      node.total_value += ret
      node.visit_count += 1

  return root


def bfs(node: Node) -> types.Action:
  """Breadth-first search policy."""
  visit_counts = jnp.array([c.visit_count for c in node.children.values()])
  return argmax(-visit_counts)


def puct(node: Node, ucb_scaling: float = 1.) -> types.Action:
  """PUCT search policy, i.e. UCT with 'prior' policy."""
  # Action values Q(s,a).
  value_scores = jnp.array([child.value for child in node.children.values()])
  check_numerics(value_scores)

  # Policy prior P(s,a).
  priors = jnp.array([child.prior for child in node.children.values()])
  check_numerics(priors)

  # Visit ratios.
  visit_ratios = jnp.array([
      jnp.sqrt(node.visit_count) / (child.visit_count + 1)
      for child in node.children.values()
  ])
  check_numerics(visit_ratios)

  # Combine.
  puct_scores = value_scores + ucb_scaling * priors * visit_ratios
  return argmax(puct_scores)


def visit_count_policy(root: Node, temperature: float = 1.) -> types.Probs:
  """Probability weighted by visit^{1/temp} of children nodes."""
  visits = root.children_visits
  if jnp.sum(visits) == 0:  # uniform policy for zero visits
    visits += 1
  rescaled_visits = visits**(1 / temperature)
  probs = rescaled_visits / jnp.sum(rescaled_visits)
  check_numerics(probs)

  return probs


def argmax(values: jnp.ndarray) -> types.Action:
  """Argmax with random tie-breaking."""
  check_numerics(values)
  max_value = jnp.max(values)
  return jnp.int32(jnp.random.choice(jnp.flatnonzero(values == max_value)))


def check_numerics(values: jnp.ndarray):
  """Raises a ValueError if any of the ijnp.ts are NaN or Inf."""
  if not jnp.isfinite(values).all():
    raise ValueError('check_numerics failed. Ijnp.ts: {}. '.format(values))