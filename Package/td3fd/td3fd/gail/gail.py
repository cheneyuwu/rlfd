import pickle

import numpy as np
import tensorflow as tf

from td3fd import logger
from td3fd.gail.model import Discriminator, Generator, ValueNet
from td3fd.memory import RingReplayBuffer, UniformReplayBuffer, iterbatches
from td3fd.normalizer import Normalizer
from td3fd.util.tf_util import GetFlat, SetFromFlat, flatgrad, intprod


class GAIL(object):
    def __init__(
        self,
        num_epochs,
        scope,
        buffer_size,
        num_demo,
        policy_step,
        disc_step,
        gen_layer_sizes,
        disc_layer_sizes,
        max_kl,
        gen_ent_coeff,
        disc_ent_coeff,
        norm_eps,
        norm_clip,
        lam,
        cg_damping,
        cg_iters,
        vf_iters,
        vf_batch_size,
        max_u,
        input_dims,
        eps_length,
        fix_T,
        gamma,
        info,
    ):
        """
        Implementation of DDPG that is used in combination with Hindsight Experience Replay (HER). Added functionality
        to use demonstrations for training to Overcome exploration problem.

        Args:
        """
        # Store initial args passed into the function
        self.init_args = locals()

        # Parameters
        self.scope = scope
        self.buffer_size = buffer_size
        self.num_demo = num_demo
        self.policy_step = policy_step
        self.disc_step = disc_step
        self.gen_layer_sizes = gen_layer_sizes
        self.disc_layer_sizes = disc_layer_sizes
        self.max_kl = max_kl
        self.gen_ent_coeff = gen_ent_coeff
        self.disc_ent_coeff = disc_ent_coeff
        self.norm_eps = norm_eps
        self.norm_clip = norm_clip
        self.lam = lam
        self.cg_damping = cg_damping
        self.cg_iters = cg_iters
        self.vf_iters = vf_iters
        self.vf_batch_size = vf_batch_size
        self.max_u = max_u
        self.input_dims = input_dims
        self.eps_length = eps_length
        self.fix_T = fix_T
        self.gamma = gamma
        self.info = info
        self.num_epochs = num_epochs

        # Prepare parameters
        self.dimo = self.input_dims["o"]
        self.dimg = self.input_dims["g"]
        self.dimu = self.input_dims["u"]

        # Get a tf session
        self.sess = tf.get_default_session()
        assert self.sess != None, "must have a default session before creating DDPG"
        with tf.variable_scope(scope):
            self.scope = tf.get_variable_scope()
            self._create_memory()
            self._create_network()
        self.saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope.name))

    def get_actions(self, o, g, compute_q=False):
        assert compute_q == False
        vals = self.generator_output_tf
        feed = {}
        feed[self.policy_inputs_tf["o"]] = o.reshape(-1, self.dimo)
        if self.dimg != 0:
            feed[self.policy_inputs_tf["g"]] = g.reshape(-1, self.dimg)
        ret = self.sess.run(vals, feed_dict=feed)
        return ret

    def get_rewards(self, o, g, u):
        feed = {}
        feed[self.disc_inputs_tf["po"]] = o.reshape(-1, self.dimo)
        if self.dimg != 0:
            feed[self.disc_inputs_tf["pg"]] = g.reshape(-1, self.dimg)
        feed[self.disc_inputs_tf["pu"]] = u.reshape(-1, self.dimu)
        ret = self.sess.run(self.disc_reward_tf, feed_dict=feed)
        return ret

    def get_values(self, o, g):
        feed = {}
        feed[self.policy_inputs_tf["o"]] = o.reshape(-1, self.dimo)
        if self.dimg != 0:
            feed[self.policy_inputs_tf["g"]] = g.reshape(-1, self.dimg)
        ret = self.sess.run(self.val_output_tf, feed_dict=feed)
        return ret

    def init_demo_buffer(self, demo_file):
        """Initialize the demonstration buffer.
        """
        # load the demonstration data from data file
        self.demo_buffer.load_from_file(data_file=demo_file, num_demo=self.num_demo)

    def store_episode(self, episode_batch):
        """
        fixed_T
            array of batch_size x (T or T+1) x dim_key ('o' and 'ag' is of size T+1, others are of size T)
        else
            array of (batch_size x T) x dim_key
        """
        self.policy_buffer.store_episode(episode_batch)

    def save_policy(self, path):
        """Pickles the current policy for later inspection.
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)

    def save_weights(self, path):
        self.saver.save(self.sess, path)

    def load_weights(self, path):
        self.saver.restore(self.sess, path)

    def clear_buffer(self):
        self.policy_buffer.clear_buffer()

    def _add_vtarg_and_adv(self, rollout):
        rollout["adv"] = gaelam = np.empty((rollout["done"].shape[0], 1), dtype=np.float32)
        rew = rollout["pr"]
        lastgaelam = 0
        for t in reversed(range(rollout["done"].shape[0])):
            if rollout["done"][t]:
                lastgaelam = 0
            delta = rew[t] + self.gamma * rollout["pv_2"][t] - rollout["pv"][t]
            gaelam[t] = lastgaelam = delta + self.gamma * self.lam * lastgaelam
        rollout["tdlamret"] = rollout["adv"] + rollout["pv"]

    def train_policy(self):

        rollout = self.policy_buffer.sample()  # get all
        self._add_vtarg_and_adv(rollout)

        # Inputs
        atarg = rollout["adv"]
        atarg = (atarg - atarg.mean()) / atarg.std()  # standardized advantage function estimate
        pv_before = rollout["pv"]  # predicted value function before udpate

        # Update normalizer
        self.policy_o_stats.update(rollout["o"])
        if self.dimg != 0:
            self.policy_g_stats.update(rollout["g"])

        # Update generator using TRPO
        # assign generator params to generator_old
        self.sess.run(self.update_generator_op)
        # compute loss
        losses = [
            self.gen_loss_tf,
            self.gen_mean_kl_tf,
            self.gen_ent_loss_tf,
            self.gen_surrgain_loss_tf,
            self.gen_mean_ent_tf,
        ]
        loss_names = ["gen_loss", "gen_mean_kl", "gen_ent_loss", "gen_surrgain_loss", "gen_mean_ent"]
        vals = losses + [self.gen_grad_tf]
        feed = {}
        feed[self.policy_inputs_tf["o"]] = rollout["o"]
        if self.dimg != 0:
            feed[self.policy_inputs_tf["g"]] = rollout["g"]
        feed[self.policy_inputs_tf["u"]] = rollout["u"]
        feed[self.policy_inputs_tf["atarg"]] = atarg
        *loss_before, grad = self.sess.run(vals, feed_dict=feed)

        def fisher_vector_product(p):
            vals = self.gen_fvp_tf
            feed[self.policy_inputs_tf["o"]] = rollout["o"][::5]
            if self.dimg != 0:
                feed[self.policy_inputs_tf["g"]] = rollout["g"][::5]
            feed[self.policy_inputs_tf["u"]] = rollout["u"][::5]
            feed[self.policy_inputs_tf["atarg"]] = atarg[::5]
            feed[self.flat_tangent] = p
            ret = self.sess.run(vals, feed_dict=feed)
            ret += self.cg_damping * p
            return ret

        if np.allclose(grad, 0):
            logger.info("Got zero gradient. not updating")
        else:
            step_dir = cg(fisher_vector_product, grad, cg_iters=self.cg_iters, verbose=False)
            assert np.isfinite(step_dir).all()
            shs = 0.5 * step_dir.dot(fisher_vector_product(step_dir))
            lm = np.sqrt(shs / self.max_kl)
            full_step = step_dir / lm
            expected_improve = grad.dot(full_step)
            surr_before = loss_before[0]
            step_size = 1.0
            th_before = self.get_gen_var_flat()
            for _ in range(10):
                th_new = th_before + full_step * step_size
                self.set_gen_var_flat(th_new)
                meanlosses = surr, kl, *_ = self.sess.run(losses, feed_dict=feed)
                improve = surr - surr_before
                # logger.info("Expected: %.3f Actual: %.3f" % (expected_improve, improve))
                if (np.isfinite(meanlosses).all()) and (kl < self.max_kl * 1.5) and (improve > 0):
                    logger.info("Stepsize OK!")
                    break
                step_size *= 0.5
            else:
                logger.info("Couldn't compute a good step")
                self.set_gen_var_flat(th_before)

        # Update Value Function
        for _ in range(self.vf_iters):
            vals = [self.val_loss_tf, self.val_update_op]
            if self.dimg != 0:
                # TODO: should we add a normalizer here?
                for (mb_o, mb_g, mb_val) in iterbatches(
                    (rollout["o"], rollout["g"], rollout["tdlamret"]),
                    include_final_partial_batch=False,
                    batch_size=self.vf_batch_size,
                ):
                    feed = {
                        self.policy_inputs_tf["o"]: rollout["o"],
                        self.policy_inputs_tf["g"]: rollout["g"],
                        self.policy_inputs_tf["ret"]: rollout["tdlamret"],
                    }
                    loss, _ = self.sess.run(vals, feed_dict=feed)
            else:
                for (mb_o, mb_val) in iterbatches(
                    (rollout["o"], rollout["tdlamret"]),
                    include_final_partial_batch=False,
                    batch_size=self.vf_batch_size,
                ):
                    feed = {
                        self.policy_inputs_tf["o"]: rollout["o"],
                        self.policy_inputs_tf["ret"]: rollout["tdlamret"],
                    }
                    loss, _ = self.sess.run(vals, feed_dict=feed)

    def train_disc(self):
        rollout = self.policy_buffer.sample()  # get all
        num_transitions = rollout["u"].shape[0]
        batch_size = num_transitions // self.disc_step

        disc_losses = []
        vals = [self.disc_loss_tf, self.disc_update_op]
        if self.dimg != 0:
            for (policy_o, policy_g, policy_u) in iterbatches(
                (rollout["o"], rollout["g"], rollout["u"]), include_final_partial_batch=False, batch_size=batch_size
            ):
                demo_batch = self.demo_buffer.sample(num_transitions)
                self.disc_o_stats.update(np.concatenate((policy_o, demo_batch["o"]), axis=0))
                self.disc_g_stats.update(np.concatenate((policy_g, demo_batch["g"]), axis=0))
                feed = {
                    self.disc_inputs_tf["po"]: policy_o,
                    self.disc_inputs_tf["pg"]: policy_g,
                    self.disc_inputs_tf["pu"]: policy_u,
                    self.disc_inputs_tf["do"]: demo_batch["o"],
                    self.disc_inputs_tf["dg"]: demo_batch["g"],
                    self.disc_inputs_tf["du"]: demo_batch["u"],
                }
                loss, _ = self.sess.run(vals, feed_dict=feed)
                disc_losses.append(loss)
        else:
            for (policy_o, policy_u) in iterbatches(
                (rollout["o"], rollout["u"]), include_final_partial_batch=False, batch_size=batch_size
            ):
                demo_batch = self.demo_buffer.sample(num_transitions)
                self.disc_o_stats.update(np.concatenate((policy_o, demo_batch["o"]), axis=0))
                feed = {
                    self.disc_inputs_tf["po"]: policy_o,
                    self.disc_inputs_tf["pu"]: policy_u,
                    self.disc_inputs_tf["do"]: demo_batch["o"],
                    self.disc_inputs_tf["du"]: demo_batch["u"],
                }
                loss, _ = self.sess.run(vals, feed_dict=feed)
                disc_losses.append(loss)
        return np.mean(disc_losses)

    def logs(self, prefix=""):
        logs = []
        logs.append((prefix + "policy_stats_o/mean", np.mean(self.sess.run([self.policy_o_stats.mean_tf]))))
        logs.append((prefix + "policy_stats_o/std", np.mean(self.sess.run([self.policy_o_stats.std_tf]))))
        if self.dimg != 0:
            logs.append((prefix + "policy_stats_g/mean", np.mean(self.sess.run([self.policy_g_stats.mean_tf]))))
            logs.append((prefix + "policy_stats_g/std", np.mean(self.sess.run([self.policy_g_stats.std_tf]))))
        logs.append((prefix + "disc_stats_o/mean", np.mean(self.sess.run([self.disc_o_stats.mean_tf]))))
        logs.append((prefix + "disc_stats_o/std", np.mean(self.sess.run([self.disc_o_stats.std_tf]))))
        if self.dimg != 0:
            logs.append((prefix + "disc_stats_g/mean", np.mean(self.sess.run([self.disc_g_stats.mean_tf]))))
            logs.append((prefix + "disc_stats_g/std", np.mean(self.sess.run([self.disc_g_stats.std_tf]))))
        return logs

    def _create_memory(self):
        # buffer shape
        buffer_shapes = {}
        if self.fix_T:
            # demonstration buffer
            buffer_shapes["o"] = (self.eps_length + 1, self.dimo)
            buffer_shapes["u"] = (self.eps_length, self.dimu)
            buffer_shapes["r"] = (self.eps_length, 1)
            if self.dimg != 0:  # for multigoal environment - or states that do not change over episodes.
                buffer_shapes["ag"] = (self.eps_length + 1, self.dimg)
                buffer_shapes["g"] = (self.eps_length, self.dimg)
            for key, val in self.input_dims.items():
                if key.startswith("info"):
                    buffer_shapes[key] = (self.eps_length, *(tuple([val]) if val > 0 else tuple()))
            self.demo_buffer = UniformReplayBuffer(buffer_shapes, self.buffer_size, self.eps_length)
            # policy buffer
            policy_buffer_shapes = buffer_shapes.copy()
            policy_buffer_shapes["pr"] = (self.eps_length, 1)
            policy_buffer_shapes["pv"] = (self.eps_length + 1, 1)
            # TODO: later we should move this above, but now it can only be here since the demo_data.npz does not have done signal
            policy_buffer_shapes["done"] = (self.eps_length, 1)
            self.policy_buffer = UniformReplayBuffer(policy_buffer_shapes, self.buffer_size, self.eps_length)
        else:
            # demonstration buffer
            buffer_shapes["o"] = (self.dimo,)
            buffer_shapes["o_2"] = (self.dimo,)
            buffer_shapes["u"] = (self.dimu,)
            buffer_shapes["r"] = (1,)
            if self.dimg != 0:  # for multigoal environment - or states that do not change over episodes.
                buffer_shapes["ag"] = (self.dimg,)
                buffer_shapes["g"] = (self.dimg,)
                buffer_shapes["ag_2"] = (self.dimg,)
                buffer_shapes["g_2"] = (self.dimg,)
            for key, val in self.input_dims.items():
                if key.startswith("info"):
                    buffer_shapes[key] = tuple([val]) if val > 0 else tuple()
            buffer_shapes["done"] = (1,)  # need the "done" signal for restarting from training
            self.demo_buffer = RingReplayBuffer(buffer_shapes, self.buffer_size)
            # policy buffer
            policy_buffer_shapes = buffer_shapes.copy()
            policy_buffer_shapes["pr"] = (1,)
            policy_buffer_shapes["pv"] = (1,)
            policy_buffer_shapes["pv_2"] = (1,)
            self.policy_buffer = RingReplayBuffer(policy_buffer_shapes, self.buffer_size)

    def _create_network(self):
        # Inputs to generator and value function
        self.policy_inputs_tf = {}
        self.policy_inputs_tf["o"] = tf.placeholder(tf.float32, shape=(None, self.dimo))
        if self.dimg != 0:
            self.policy_inputs_tf["g"] = tf.placeholder(tf.float32, shape=(None, self.dimg))
        self.policy_inputs_tf["u"] = tf.placeholder(tf.float32, shape=(None, self.dimu))
        self.policy_inputs_tf["atarg"] = tf.placeholder(tf.float32, shape=(None, 1))
        self.policy_inputs_tf["ret"] = tf.placeholder(tf.float32, shape=(None, 1))

        # Inputs to discriminator
        self.disc_inputs_tf = {}
        self.disc_inputs_tf["po"] = tf.placeholder(tf.float32, shape=(None, self.dimo))
        self.disc_inputs_tf["do"] = tf.placeholder(tf.float32, shape=(None, self.dimo))
        if self.dimg != 0:
            self.disc_inputs_tf["pg"] = tf.placeholder(tf.float32, shape=(None, self.dimg))
            self.disc_inputs_tf["dg"] = tf.placeholder(tf.float32, shape=(None, self.dimg))
        self.disc_inputs_tf["pu"] = tf.placeholder(tf.float32, shape=(None, self.dimu))
        self.disc_inputs_tf["du"] = tf.placeholder(tf.float32, shape=(None, self.dimu))

        # Normalizers
        self.policy_o_stats = Normalizer(self.dimo, self.norm_eps, self.norm_clip, sess=self.sess)
        self.policy_g_stats = Normalizer(self.dimg, self.norm_eps, self.norm_clip, sess=self.sess)
        self.disc_o_stats = Normalizer(self.dimo, self.norm_eps, self.norm_clip, sess=self.sess)
        self.disc_g_stats = Normalizer(self.dimg, self.norm_eps, self.norm_clip, sess=self.sess)

        # policy inputs (with normalization)
        policy_input_o_tf = self.policy_o_stats.normalize(self.policy_inputs_tf["o"])
        policy_input_g_tf = self.policy_g_stats.normalize(self.policy_inputs_tf["g"]) if self.dimg != 0 else None
        policy_input_u_tf = self.policy_inputs_tf["u"]
        policy_input_atarg_tf = self.policy_inputs_tf["atarg"]
        policy_input_ret_tf = self.policy_inputs_tf["ret"]
        # disc inputs (with normalization)
        disc_input_po_tf = self.disc_o_stats.normalize(self.disc_inputs_tf["po"])
        disc_input_do_tf = self.disc_o_stats.normalize(self.disc_inputs_tf["do"])
        disc_input_pg_tf = self.disc_g_stats.normalize(self.disc_inputs_tf["pg"]) if self.dimg != 0 else None
        disc_input_dg_tf = self.disc_g_stats.normalize(self.disc_inputs_tf["dg"]) if self.dimg != 0 else None
        disc_input_pu_tf = self.disc_inputs_tf["pu"]
        disc_input_du_tf = self.disc_inputs_tf["du"]

        # Models
        self.generator = Generator(
            dimo=self.dimo, dimg=self.dimg, dimu=self.dimu, max_u=self.max_u, layer_sizes=self.gen_layer_sizes
        )
        self.generator_old = Generator(
            dimo=self.dimo, dimg=self.dimg, dimu=self.dimu, max_u=self.max_u, layer_sizes=self.gen_layer_sizes
        )
        self.discriminator = Discriminator(
            dimo=self.dimo, dimg=self.dimg, dimu=self.dimu, max_u=self.max_u, layer_sizes=self.disc_layer_sizes
        )
        self.value_net = ValueNet(dimo=self.dimo, dimg=self.dimg, layer_sizes=self.gen_layer_sizes)

        # Discriminator loss
        self.disc_gen_logit_tf = self.discriminator(o=disc_input_po_tf, g=disc_input_pg_tf, u=disc_input_pu_tf)
        self.disc_demo_logit_tf = self.discriminator(o=disc_input_do_tf, g=disc_input_dg_tf, u=disc_input_du_tf)
        self.disc_gen_sigmoid_tf = tf.sigmoid(self.disc_gen_logit_tf)
        self.disc_demo_sigmoid_tf = tf.sigmoid(self.disc_demo_logit_tf)
        self.disc_gen_acc_tf = tf.reduce_mean(tf.cast(self.disc_gen_sigmoid_tf < 0.5, tf.float32))
        self.disc_demo_acc_tf = tf.reduce_mean(tf.cast(self.disc_demo_sigmoid_tf > 0.5, tf.float32))
        # log likelihood loss
        self.disc_gen_loss_tf = tf.reduce_mean(-tf.log(1.0 - self.disc_gen_sigmoid_tf))
        self.disc_demo_loss_tf = tf.reduce_mean(-tf.log(self.disc_demo_sigmoid_tf))
        # entropy loss
        disc_logit_tf = tf.concat([self.disc_gen_logit_tf, self.disc_demo_logit_tf], axis=0)
        disc_ent_tf = (1.0 - tf.sigmoid(disc_logit_tf)) * disc_logit_tf - tf.log_sigmoid(disc_logit_tf)
        self.disc_ent_loss_tf = tf.reduce_mean(-self.disc_ent_coeff * disc_ent_tf)
        # total loss
        self.disc_loss_tf = self.disc_gen_loss_tf + self.disc_demo_loss_tf + self.disc_ent_loss_tf

        # Generator Loss
        gen_output_dist_tf = self.generator.get_output_dist(o=policy_input_o_tf, g=policy_input_g_tf)
        gen_old_output_dist_tf = self.generator_old.get_output_dist(o=policy_input_o_tf, g=policy_input_g_tf)
        # kl divergence
        self.gen_kl_div_tf = gen_old_output_dist_tf.kl_divergence(gen_output_dist_tf)
        # entropy
        self.gen_ent_tf = gen_output_dist_tf.entropy()
        # policy gradient loss (importance sampling)
        ratio_tf = tf.exp(
            gen_output_dist_tf.log_prob(policy_input_u_tf) - gen_old_output_dist_tf.log_prob(policy_input_u_tf)
        )
        self.gen_surrgain_loss_tf = tf.reduce_mean(ratio_tf * policy_input_atarg_tf)
        self.gen_mean_ent_tf = tf.reduce_mean(self.gen_ent_tf)
        self.gen_ent_loss_tf = self.gen_ent_coeff * self.gen_mean_ent_tf
        # total loss
        self.gen_loss_tf = self.gen_surrgain_loss_tf + self.gen_ent_loss_tf
        # kl constraint
        self.gen_mean_kl_tf = tf.reduce_mean(self.gen_kl_div_tf)

        # Value Function Loss
        self.val_loss_tf = tf.reduce_mean(
            tf.square(self.value_net(o=policy_input_o_tf, g=policy_input_g_tf) - policy_input_ret_tf)
        )

        # Calculate rewards
        self.disc_reward_tf = -tf.log(1.0 - self.disc_gen_sigmoid_tf + 1e-8)
        # Calculate value
        self.val_output_tf = self.value_net(o=policy_input_o_tf, g=policy_input_g_tf)
        # Get Output
        self.generator_output_tf = self.generator(o=policy_input_o_tf, g=policy_input_g_tf)

        # Optimizers
        self.disc_update_op = tf.train.AdamOptimizer(learning_rate=3e-4).minimize(
            self.disc_loss_tf, var_list=self.discriminator.trainable_variables
        )
        self.val_update_op = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(
            self.val_loss_tf, var_list=self.value_net.trainable_variables
        )
        # TRPO update for generator
        self.get_gen_var_flat = GetFlat(self.generator.trainable_variables)
        self.set_gen_var_flat = SetFromFlat(self.generator.trainable_variables)
        kl_grads_tf = tf.gradients(self.gen_mean_kl_tf, self.generator.trainable_variables)

        self.flat_tangent = tf.placeholder(dtype=tf.float32, shape=[None], name="flat_tan")
        shapes = [var.get_shape().as_list() for var in self.generator.trainable_variables]
        start = 0
        tangents = []
        for shape in shapes:
            sz = intprod(shape)
            tangents.append(tf.reshape(self.flat_tangent[start : start + sz], shape))
            start += sz
        gvp_tf = tf.add_n([tf.reduce_sum(g * tangent) for (g, tangent) in zip(kl_grads_tf, tangents)])
        self.gen_fvp_tf = flatgrad(gvp_tf, self.generator.trainable_variables)
        self.gen_grad_tf = flatgrad(self.gen_loss_tf, self.generator.trainable_variables)

        # generator old <- generator new
        self.update_generator_op = list(
            map(lambda v: v[0].assign(v[1]), zip(self.generator_old.variables, self.generator.variables))
        )

        # Initialize all variables
        self.sess.run(tf.global_variables_initializer())
        # self.training_step = self.sess.run(self.training_step_tf)

    def __getstate__(self):
        """
        Our policies can be loaded from pkl, but after unpickling you cannot continue training.
        """
        state = {k: v for k, v in self.init_args.items() if not k == "self"}
        state["tf"] = self.sess.run([x for x in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)])
        return state

    def __setstate__(self, state):
        stored_vars = state.pop("tf")
        self.__init__(**state)
        vars = [x for x in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)]
        assert len(vars) == len(stored_vars)
        node = [tf.assign(var, val) for var, val in zip(vars, stored_vars)]
        self.sess.run(node)


# compute conjugate gradient
def cg(f_Ax, b, cg_iters=10, callback=None, verbose=False, residual_tol=1e-10):
    """
    Demmel p 312
    """
    p = b.copy()
    r = b.copy()
    x = np.zeros_like(b)
    rdotr = r.dot(r)

    fmtstr = "%10i %10.3g %10.3g"
    titlestr = "%10s %10s %10s"
    if verbose:
        print(titlestr % ("iter", "residual norm", "soln norm"))

    for i in range(cg_iters):
        if callback is not None:
            callback(x)
        if verbose:
            print(fmtstr % (i, rdotr, np.linalg.norm(x)))
        z = f_Ax(p)
        v = rdotr / p.dot(z)
        x += v * p
        r -= v * z
        newrdotr = r.dot(r)
        mu = newrdotr / rdotr
        p = r + mu * p

        rdotr = newrdotr
        if rdotr < residual_tol:
            break

    if callback is not None:
        callback(x)
    if verbose:
        print(fmtstr % (i + 1, rdotr, np.linalg.norm(x)))  # pylint: disable=W0631
    return x
