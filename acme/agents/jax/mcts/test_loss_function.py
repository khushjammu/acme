"""
Overall conclusion is that the loss function works fine. There is
some disrepency, but it's most likely due to numerical imprecision
(on the order of ~1e-5).
"""


import rlax
import numpy as np
import jax
import jax.numpy as jnp

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

# (1.3484426
loss_actual = 1.3484426
print("definitions are ok")
# loss function to be vectorised
def stonks(v_tm1, r_t, discount_t, v_t, labels, logits):
	value_loss = rlax.td_learning(
	  v_tm1=v_tm1,
	  r_t=r_t,
	  discount_t=discount_t,
	  v_t=v_t,
	)
	value_loss = jnp.square(value_loss)

	policy_loss = rlax.categorical_cross_entropy(
	  labels=labels,
	  logits=logits
	)

	return value_loss + policy_loss

batch_loss_fn = jax.vmap(stonks)

# print("value", value)
# print("r_t", r_t)
# print("scaled_discount", scaled_discount)
# print("target_value", target_value)
# print("pi_t", pi_t)
# print("logits", logits)

batch_loss = batch_loss_fn(
  v_tm1=value,
  r_t=r_t,
  discount_t=scaled_discount,
  v_t=target_value,
  labels=pi_t,
  logits=logits
  )

def custom(r, d, tv, v, pt, lg):
	value_loss = jnp.square(r + d * tv - v)
	policy_loss = rlax.categorical_cross_entropy(
		labels=pt,
		logits=lg
		)

	return value_loss+policy_loss, value_loss, policy_loss

loss = jnp.mean(batch_loss)
custom_loss, val, pol = jax.vmap(custom)(r_t, scaled_discount, target_value, value, pi_t, logits)
print("custom loss:", jnp.mean(custom_loss), jnp.mean(val), jnp.mean(pol))
print("calculated loss:", loss)
print("actual loss:", loss_actual)
print("disrepency:", loss_actual-loss)