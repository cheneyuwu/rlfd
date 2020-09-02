import tensorflow as tf

tfk = tf.keras
tfl = tfk.layers


class Actor(tfk.Model):

  def __init__(self, dimo, dimu, max_u, layer_sizes, name="pi"):
    super().__init__(name=name)
    self._dimo = dimo
    self._dimu = dimu
    self._max_u = max_u

    self._mlp_layers = []
    for size in layer_sizes:
      layer = tfl.Dense(units=size,
                        activation="relu",
                        kernel_initializer="glorot_normal")
      self._mlp_layers.append(layer)
    self._output_layer = tfl.Dense(units=self._dimu[0],
                                   activation="tanh",
                                   kernel_initializer="glorot_normal")
    # Create weights
    self([tf.zeros([0, *self._dimo])])

  @tf.function
  def call(self, inputs):
    res = inputs[0]
    for l in self._mlp_layers:
      res = l(res)
    res = self._output_layer(res)
    return res * self._max_u


class Critic(tfk.Model):

  def __init__(self, dimo, dimu, max_u, layer_sizes, name="q"):
    super().__init__(name=name)

    self._dimo = dimo
    self._dimu = dimu
    self._max_u = max_u

    self._mlp_layers = []
    for size in layer_sizes:
      layer = tfl.Dense(units=size,
                        activation="relu",
                        kernel_initializer="glorot_normal")
      self._mlp_layers.append(layer)
    self._output_layer = tfl.Dense(units=1, kernel_initializer="glorot_normal")
    # Create weights
    self([tf.zeros([0, *self._dimo]), tf.zeros([0, *self._dimu])])

  @tf.function
  def call(self, inputs):
    o, u = inputs
    res = tf.concat([o, u / self._max_u], axis=-1)
    for l in self._mlp_layers:
      res = l(res)
    res = self._output_layer(res)
    return res
