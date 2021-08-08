import tensorflow as tf

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

class LearnerTensorboardLogger():
  def __init__(self, tensorboard_writer):
    self._tensorboard_writer = tensorboard_writer

  def write(self, result):
    with self._tensorboard_writer.as_default():
        tf.summary.scalar("total_loss", result["total_loss"], step=result["steps"])