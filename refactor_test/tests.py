# TODO: add all the various tests we use here

### TEST FOR CHECKPOINTING

# if __name__ == '__main__':
#   ray.init(address="auto")

#   storage = SharedStorage.remote()
#   storage.set_info.remote({
#     "terminate": False
#   })

#   reverb_replay = replay.make_reverb_prioritized_nstep_replay(
#       environment_spec=spec,
#       n_step=config.n_step,
#       batch_size=config.batch_size,
#       max_replay_size=config.max_replay_size,
#       min_replay_size=config.min_replay_size,
#       priority_exponent=config.priority_exponent,
#       discount=config.discount,
#   )

#   learner = LearnerRay.options(max_concurrency=2).remote(
#     "localhost:8000",
#     storage,
#     verbose=True
#   )

#   # important to force the learner onto TPU
#   ray.get(learner.get_variables.remote(""))

#   # ray.get(learner.save_checkpoint.remote())
#   old_params = ray.get(learner.get_variables.remote(""))
#   ray.get(learner.load_checkpoint.remote("/home/aryavohra/acme/refactor_test/checkpoint"))
#   new_params = ray.get(learner.get_variables.remote(""))

#   assert old_params != new_params, "no checkpoint update took place"

#   if old_params != new_params:
#     print("yipee")

#   while not ray.get(storage.get_info.remote("terminate")):
#     time.sleep(1)