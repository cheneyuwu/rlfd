import os
import pickle
osp = os.path

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfk = tf.keras

from rlfd import logger, memory, normalizer, policies
from rlfd.agents import agent, cql, sac_networks


class CQLDP(cql.CQL):
  """This agent adds on top of normal CQL a double penalty to the Q value
  regularizer.
  """

  def _cql_criticq_loss_graph(self, o, g, o_2, g_2, u, r, n, done, step):
    # Normalize observations
    norm_o = self._o_stats.normalize(o)
    norm_g = self._g_stats.normalize(g)
    norm_o_2 = self._o_stats.normalize(o_2)
    norm_g_2 = self._g_stats.normalize(g_2)

    pi_2, logprob_pi_2 = self._actor([norm_o_2, norm_g_2])

    # Immediate reward
    target_q = r
    # Shaping reward
    if self.online_data_strategy == "Shaping":
      potential_curr = self.shaping.potential(o=o, g=g, u=u)
      potential_next = self.shaping.potential(o=o_2, g=g_2, u=pi_2)
      target_q += (1.0 - done) * tf.pow(self.gamma,
                                        n) * potential_next - potential_curr
    # Q value from next state
    target_next_q1 = self._criticq1_target([norm_o_2, norm_g_2, pi_2])
    target_next_q2 = self._criticq2_target([norm_o_2, norm_g_2, pi_2])
    target_next_min_q = tf.minimum(target_next_q1, target_next_q2)
    target_q += ((1.0 - done) * tf.pow(self.gamma, n) *
                 (target_next_min_q - self.alpha * logprob_pi_2))
    target_q = tf.stop_gradient(target_q)

    td_loss_q1 = self._huber_loss(target_q, self._criticq1([norm_o, norm_g, u]))
    td_loss_q2 = self._huber_loss(target_q, self._criticq2([norm_o, norm_g, u]))
    td_loss = td_loss_q1 + td_loss_q2
    # Being Conservative (Eqn.4)
    # second term
    max_term_q1 = self._criticq1([norm_o, norm_g, u])
    max_term_q2 = self._criticq2([norm_o, norm_g, u])
    # first term (uniform)
    num_samples = 10
    tiled_norm_o = tf.tile(tf.expand_dims(norm_o, axis=1),
                           [1, num_samples] + [1] * len(self.dimo))
    tiled_norm_g = tf.tile(tf.expand_dims(norm_g, axis=1),
                           [1, num_samples] + [1] * len(self.dimg))
    uni_u_dist = tfd.Uniform(low=-self.max_u * tf.ones(self.dimu),
                             high=self.max_u * tf.ones(self.dimu))
    uni_u = uni_u_dist.sample((tf.shape(u)[0], num_samples))
    logprob_uni_u = tf.reduce_sum(uni_u_dist.log_prob(uni_u),
                                  axis=list(range(2, 2 + len(self.dimu))),
                                  keepdims=True)
    uni_q1 = self._criticq1([tiled_norm_o, tiled_norm_g, uni_u])
    uni_q2 = self._criticq2([tiled_norm_o, tiled_norm_g, uni_u])
    # apply double side penalty
    uni_q1 = tf.abs(uni_q1 + 1000)
    uni_q2 = tf.abs(uni_q2 + 1000)
    # apply double side penalty
    uni_q1_logprob_uni_u = uni_q1 - logprob_uni_u
    uni_q2_logprob_uni_u = uni_q2 - logprob_uni_u
    # first term (policy)
    pi, logprob_pi = self._actor([tiled_norm_o, tiled_norm_g])
    pi_q1 = self._criticq1([tiled_norm_o, tiled_norm_g, pi])
    pi_q2 = self._criticq2([tiled_norm_o, tiled_norm_g, pi])
    # apply double side penalty
    pi_q1 = tf.abs(pi_q1 + 1000)
    pi_q2 = tf.abs(pi_q2 + 1000)
    # apply double side penalty
    pi_q1_logprob_pi = pi_q1 - logprob_pi
    pi_q2_logprob_pi = pi_q2 - logprob_pi
    # Note: log(2N) not included in this case since it is constant.
    log_sum_exp_q1 = tf.math.reduce_logsumexp(tf.concat(
        (uni_q1_logprob_uni_u, pi_q1_logprob_pi), axis=1),
                                              axis=1)
    log_sum_exp_q2 = tf.math.reduce_logsumexp(tf.concat(
        (uni_q2_logprob_uni_u, pi_q2_logprob_pi), axis=1),
                                              axis=1)
    cql_loss_q1 = (tf.exp(self.cql_log_alpha) *
                   (log_sum_exp_q1 - max_term_q1 - self.cql_tau))
    cql_loss_q2 = (tf.exp(self.cql_log_alpha) *
                   (log_sum_exp_q2 - max_term_q2 - self.cql_tau))
    cql_loss = cql_loss_q1 + cql_loss_q2

    criticq_loss = tf.reduce_mean(td_loss) + tf.reduce_mean(cql_loss)
    tf.summary.scalar(name='criticq_loss vs {}'.format(step.name),
                      data=criticq_loss,
                      step=step)
    return criticq_loss