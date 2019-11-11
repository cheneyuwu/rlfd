"""Adopted from Eric Jang's normalizing flow tutorial: https://blog.evjang.com/2018/01/nf1.html
"""

import tensorflow as tf  # pylint: ignore-module
import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

# Simple Normalizing Flow
# =============================================================================
class LeakyReLU(tfb.Bijector):
    def __init__(self, alpha=0.5, validate_args=False, name="leaky_relu"):
        super(LeakyReLU, self).__init__(
            forward_min_event_ndims=1, inverse_min_event_ndims=1, validate_args=validate_args, name=name
        )
        self.alpha = alpha

    def _forward(self, x):
        return tf.where(tf.greater_equal(x, 0), x, self.alpha * x)

    def _inverse(self, y):
        return tf.where(tf.greater_equal(y, 0), y, 1.0 / self.alpha * y)

    def _inverse_log_det_jacobian(self, y):
        event_dims = (
            1
        )  # self._event_dims_tensor(y)  ## Note: this wil break for objects of size other than N x dim(vector)

        I = tf.ones_like(y)
        J_inv = tf.where(tf.greater_equal(y, 0), I, 1.0 / self.alpha * I)
        # abs is actually redundant here, since this det Jacobian is > 0
        log_abs_det_J_inv = tf.log(tf.abs(J_inv))
        return tf.reduce_sum(log_abs_det_J_inv, axis=event_dims)


class ToyNF:
    def __init__(self, base_dist, num_layers=6, d=2, r=2):
        self.bijectors = []

        for i in range(num_layers):
            with tf.variable_scope("bijector_%d" % i):
                V = tf.get_variable("V", [d, r], dtype=tf.float32)  # factor loading
                shift = tf.get_variable("shift", [d], dtype=tf.float32)  # affine shift
                L = tf.get_variable("L", [d * (d + 1) / 2], dtype=tf.float32)  # lower triangular

                self.bijectors.append(
                    tfb.Affine(scale_tril=tfd.fill_triangular(L), scale_perturb_factor=V, shift=shift)
                )

                alpha = tf.abs(tf.get_variable("alpha", [], dtype=tf.float32)) + 0.01
                self.bijectors.append(LeakyReLU(alpha=alpha))

        # Last layer is affine. Note that tfb.Chain takes a list of bijectors in the *reverse* order
        # that they are applied..
        mlp_bijector = tfb.Chain(list(reversed(self.bijectors[:-1])), name="2d_mlp_bijector")

        self.dist = tfd.TransformedDistribution(distribution=base_dist, bijector=mlp_bijector)

    def __call__(self, input):
        return self.dist.log_prob(input)


# Modern Normalizing Flows
# =============================================================================
class MAF:
    def __init__(self, base_dist, dim, num_bijectors=6, layer_sizes=[512, 512]):
        # build layers
        self.bijectors = []
        for _ in range(num_bijectors):
            self.bijectors.append(
                tfb.MaskedAutoregressiveFlow(
                    shift_and_log_scale_fn=tfb.masked_autoregressive_default_template(
                        hidden_layers=layer_sizes,
                        kernel_initializer=tf.initializers.glorot_normal(),
                        bias_initializer=None,
                    )
                )
            )
            self.bijectors.append(tfb.Permute(permutation=list(range(0, dim))[::-1]))
        # discard the last Permute layer.
        flow_bijector = tfb.Chain(list(reversed(self.bijectors[:-1])))
        self.dist = tfd.TransformedDistribution(distribution=base_dist, bijector=flow_bijector)
        # output
        self.log_prob = self.dist.log_prob
        self.prob = self.dist.prob


class RealNVP:
    def __init__(self, base_dist, dim, num_masked, num_bijectors=6, layer_sizes=[512, 512]):
        # build layers
        self.bijectors = []
        for _ in range(num_bijectors):
            self.bijectors.append(
                tfb.RealNVP(
                    num_masked=num_masked,
                    shift_and_log_scale_fn=tfb.real_nvp_default_template(
                        hidden_layers=layer_sizes,
                        kernel_initializer=tf.initializers.glorot_normal(),
                        bias_initializer=None,
                    ),
                )
            )
            self.bijectors.append(tfb.Permute(permutation=list(range(0, dim))[::-1]))
        # discard the last Permute layer.
        flow_bijector = tfb.Chain(list(reversed(self.bijectors[:-1])))
        self.dist = tfd.TransformedDistribution(distribution=base_dist, bijector=flow_bijector)
        # output
        self.log_prob = self.dist.log_prob
        self.prob = self.dist.prob
