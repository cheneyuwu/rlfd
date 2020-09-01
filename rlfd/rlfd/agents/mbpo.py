import os
import pickle
osp = os.path

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfk = tf.keras
import pdb

from rlfd import logger, memory, normalizer, policies
from rlfd.agents import agent, sac, sac_networks, mbpo_networks

def default_termination(o, u, o_2):
    #array of false values (ie assume we don't stop rollouts unless told otherwise)
    return np.full((o.shape[0], 1), False)

class MBPO(sac.SAC):
    def __init__(
        self,
        dims,
        max_u,
        eps_length,
        gamma,
        online_batch_size,
        offline_batch_size,
        fix_T,
        norms_obs_online,
        norms_obs_offline,
        norm_eps,
        norm_clip,
        layer_sizes,
        q_lr,
        pi_lr,
        action_l2,
        auto_alpha,
        alpha,
        soft_target_tau,
        target_update_freq,
        use_pretrained_actor,
        use_pretrained_critic,
        use_pretrained_alpha,
        online_data_strategy,
        bc_params,
        buffer_size,
        info,

        #MBPO-specifc parameters

        model_layer_size, #size of the layers (exept the output layer) for the model
        model_weight_decay, #weight decay terms for regularization loss
        num_networks, #number of individual networks in the ensemble
        num_elites, #number of "elites" (models with lowest holdout loss) to be selected from the ensemble
        model_lr, #learning rate for the model gradient descent update
        model_train_freq, #frequency of updates to the model, in terms of gradient updates to the actor and critic
        rollout_batch_size, #number of real samples to rollout in the model
        real_ratio, #percentage of "fake" data to include in actor and critic training batch
        rollout_schedule, #increases in number of model steps as epochs increase
        termination_function = default_termination #compute done bit for model rollouts

        ):

        super(MBPO, self).__init__(
            dims,
            max_u,
            eps_length,
            gamma,
            online_batch_size,
            offline_batch_size,
            fix_T,
            norms_obs_online,
            norms_obs_offline,
            norm_eps,
            norm_clip,
            layer_sizes,
            q_lr,
            pi_lr,
            action_l2,
            auto_alpha,
            alpha,
            soft_target_tau,
            target_update_freq,
            use_pretrained_actor,
            use_pretrained_critic,
            use_pretrained_alpha,
            online_data_strategy,
            bc_params,
            buffer_size,
            info
        )

        self.dynamics_model = BNNEnsemble(self.dimo, self.dimu, model_layer_size, model_weight_decay, num_networks, num_elites, model_lr)
        self.model_train_freq = model_train_freq
        self.real_ratio = real_ratio
        self.rollout_schedule = rollout_schedule
        self.termination_func = termination_function
        self.rollout_length = 1 #set default rollout length to 1
        self.rollout_batch_size = rollout_batch_size

    def _create_memory(self):
        #modify to create a buffer to model rollouts
        buffer_shapes = dict(o=self.dimo,
                         o_2=self.dimo,
                         u=self.dimu,
                         r=(1,),
                         ag=self.dimg,
                         ag_2=self.dimg,
                         g=self.dimg,
                         g_2=self.dimg,
                         done=(1,))
        if self.fix_T:
            buffer_shapes = {k: (self.eps_length,) + v for k, v in buffer_shapes.items()}
            self.online_buffer = memory.EpisodeBaseReplayBuffer(buffer_shapes, self.buffer_size, self.eps_length)
            self.offline_buffer = memory.EpisodeBaseReplayBuffer(buffer_shapes, self.buffer_size, self.eps_length)
            self.model_buffer = memory.EpisodeBaseReplayBuffer(buffer_shapes, self.buffer_size, self.eps_length)
        else:
            self.online_buffer = memory.StepBaseReplayBuffer(buffer_shapes, self.buffer_size)
            self.offline_buffer = memory.StepBaseReplayBuffer(buffer_shapes, self.buffer_size)
            self.model_buffer = memory.StepBaseReplayBuffer(buffer_shapes, self.buffer_size)
    
    def set_rollout_length(self):
        min_step, max_step, min_length, max_length = self.rollout_schedule
        if self.online_training_step < min_step:
            rollout_length = min_length
        else:
            dx = (self.online_training_step - min_step) / (max_step - min_step)
            dx = min(1, dx)
            rollout_length = dx * (max_length - min_length) + min_length
        self.rollout_length = int(rollout_length)

    def rollout_model(self):
        self.set_rollout_length()
        self.model_buffer.clear_buffer()
        batch = self.online_buffer.sample(self.rollout_batch_size)
        o_tf = tf.convert_to_tensor(batch["o"], dtype=tf.float32)
        g_tf = tf.convert_to_tensor(batch["g"], dtype=tf.float32) #useless junk

        #useless junk that shouldn't be here
        ag = batch["ag"]
        ag_2 = batch["ag_2"]
        g_2 = batch["g_2"]

        for _ in range(self.rollout_length):
            u_tf, _ = self._actor([self._actor_o_norm(o_tf), self._actor_g_norm(g_tf)]) #sample next action from policy
            o_2_tf, r_tf = self.dynamics_model.predict(o_tf, u_tf) #predict next_obs and reward
            o = tf.make_ndarray(o_tf)
            o_2 = tf.make_ndarray(o_2_tf)
            g = tf.make_ndarray(g_tf)
            u = tf.make_ndarray(u_tf)
            r = tf.make_ndarray(r_tf)
            done = self.termination_func(o, u, o_2)
            samples = {"o": o, "o_2": o_2, "r": r, "done": done, "g": g, "g_2": g_2, "u": u, "ag": ag, "ag_2": ag_2}
            self.model_buffer.store(samples)

            not_done_mask = ~done.squeeze(-1)
            o_tf = tf.convert_to_tensor(o_2[not_done_mask]) #make the next obs the current obs

            #useless junk
            g_tf = tf.convert_to_tensor(g[not_done_mask])
            g_2 = g_2[not_done_mask]
            ag = ag[not_done_mask]
            ag_2 = ag_2[not_done_mask]
            n = n[not_done_mask]

    def sample_batch(self):
        #modify to sample from model buffer according to real ratio
        if self.real_ratio == 0:
            batch = self.online_buffer.sample(self.online_batch_size)
        else:
            model_batch_size = self.online_batch_size * self.real_ratio
            real_batch_size = self.online_batch_size - model_batch_size
            model_batch = self.model_buffer.sample(model_batch_size)
            real_batch = self.model_buffer.sample(real_batch_size)
            batch = self._merge_batch_experiences(real_batch, model_batch)
        
        return batch
    
    def train_online(self):
        with tf.summary.record_if(lambda: self.online_training_step % 200 == 0):
            batch = self.sample_batch()

            o_tf = tf.convert_to_tensor(batch["o"], dtype=tf.float32)
            g_tf = tf.convert_to_tensor(batch["g"], dtype=tf.float32)
            o_2_tf = tf.convert_to_tensor(batch["o_2"], dtype=tf.float32)
            g_2_tf = tf.convert_to_tensor(batch["g_2"], dtype=tf.float32)
            u_tf = tf.convert_to_tensor(batch["u"], dtype=tf.float32)
            r_tf = tf.convert_to_tensor(batch["r"], dtype=tf.float32)
            n_tf = tf.convert_to_tensor(batch["n"], dtype=tf.float32)
            done_tf = tf.convert_to_tensor(batch["done"], dtype=tf.float32)

            self._train_online_graph(o_tf, g_tf, o_2_tf, g_2_tf, u_tf, r_tf, n_tf, done_tf)

            if self.online_training_step % self.model_train_freq == 0:
                model_batch = self.online_buffer(batch_size=-1)
                o_tf = tf.convert_to_tensor(model_batch["o"], dtype=tf.float32)
                o_2_tf = tf.convert_to_tensor(model_batch["o_2"], dtype=tf.float32)
                u_tf = tf.convert_to_tensor(model_batch["u"], dtype=tf.float32)
                r_tf = tf.convert_to_tensor(model_batch["r"], dtype=tf.float32)
                done_tf = tf.convert_to_tensor(model_batch["done"], dtype=tf.float32)

                self.dynamics_model.train(o_tf, u_tf, o_2_tf, r_tf)
                self.rollout_model()

            if self.online_training_step % self.target_update_freq == 0:
                self._copy_weights(self._criticq1, self._criticq1_target)
                self._copy_weights(self._criticq2, self._criticq2_target)





        