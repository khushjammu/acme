r_t = np.asarray([0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1.], dtype=np.float32)

scaled_discount = np.asarray(
[0.99 0.99 0.99 0.99 0.99 0.99 0.99 0.99 0.99 0.99 0.99 0.99 0.99 0.99
0.99 0.  ], dtype=np.float32)

logits = np.asarray(
[[-0.05347703  0.08023488 -0.1022791 ]
[ 0.00907817  0.22380453 -0.00403775]
[ 0.11095882  0.04761696 -0.13537525]
[ 0.06463103  0.1569905  -0.19233918]
[ 0.03386218  0.21535848 -0.01245719]
[ 0.03386218  0.21535848 -0.01245719]
[-0.05347703  0.08023488 -0.1022791 ]
[ 0.00907817  0.22380453 -0.00403775]
[-0.03797547  0.13095018 -0.12524539]
[ 0.00907817  0.22380453 -0.00403775]
[ 0.11095882  0.04761696 -0.13537525]
[-0.05347703  0.08023488 -0.1022791 ]
[ 0.00964257  0.10704721 -0.04755305]
[ 0.11095882  0.04761696 -0.13537525]
[ 0.0227496   0.18665592 -0.01798243]
[ 0.0237124   0.06998112 -0.09491985]], dtype=np.float32)

value = np.asarray(
[ 0.07586078 -0.06986783 -0.02092129  0.07167659  0.09633503  0.09633503
0.07586078 -0.06986783  0.18152489 -0.06986783 -0.02092129  0.07586078
0.09137869 -0.02092129  0.17242938  0.04854046], dtype=np.float32)

target_value = np.asarray(
[ 0.18152489  0.07586078  0.17242938 -0.0875344   0.07167659  0.07167659
0.18152489  0.07586078  0.04854046  0.07586078  0.17242938  0.18152489
0.02781944  0.17242938  0.09633503  0.03026186], dtype=np.float32)

pi_t = np.asarray(
[[0.06 0.86 0.08]
[0.18 0.58 0.24]
[0.3  0.34 0.36]
[0.28 0.48 0.24]
[0.24 0.46 0.3 ]
[0.24 0.46 0.3 ]
[0.06 0.86 0.08]
[0.18 0.58 0.24]
[0.04 0.92 0.04]
[0.18 0.58 0.24]
[0.3  0.34 0.36]
[0.06 0.86 0.08]
[0.28 0.36 0.36]
[0.3  0.34 0.36]
[0.28 0.38 0.34]
[0.02 0.96 0.02]], dtype=np.float32)

loss = 1.1339607

# loss function to be vectorised
def stonks(self, v_tm1, r_t, discount_t, v_t, labels, logits):
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

batch_loss_fn = jax.vmap(self.stonks)

batch_loss = batch_loss_fn(
  v_tm1=value,
  r_t=r_t,
  discount_t=scaled_discount,
  v_t=target_value,
  labels=pi_t,
  logits=logits
  )

loss = jnp.mean(batch_loss)
print("calculated loss:", loss)
print("actual loss:", loss)