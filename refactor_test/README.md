# Distributed Acme

## Features
- **Ray-enabled multi-actor training** — by populating the replay buffer faster, we accelerate training 🏎

- **Tensorboard integration for experiment tracking and statistics** — we track per-actor performance stats, learner stats, as well as a few neat aggregated metrics (e.g per-1000 episode return histograms) 📈

![Screenshot 2021-08-09 at 7 26 28 PM](https://user-images.githubusercontent.com/8716483/128699358-4122a65d-a504-47f4-be18-22874986e1f6.png)

- **RAM states** — framestacking-enabled RAM states are integrated! (This achieved-wide max return on Breakout: !) 👾

- **New DQN config** — we've added the hyperparameters we used to achieve the current highscore 🔢

- **Checkpointing** — we pickle params at a configurable interval so that we can play around with visualisations etc.

## How-To

We've pre-loaded a ~430 mean return checkpoint on node-4 and node-5 for everyone to play with.

### To enter the testing environment
1. SSH into node-4 or node-5
2. `sudo su ubuntu`
3. `cd acme/distributed/dqn`

### To continue training from a checkpoint
1. `python run.py --enable_checkpointing --enable_tensorboard --initial_checkpoint --initial_checkpoint /home/ubuntu/checkpoint-810000.pickle`

*Note: to tunnel the Tensorboard port, use the following GCP CLI command:*

`gcloud alpha compute tpus tpu-vm ssh node-5 --zone=us-central1-f --project="pearl2-320514" -- -L {desired_local_port}:localhost:6006`



