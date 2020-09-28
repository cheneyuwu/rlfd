import os
import pickle
osp = os.path

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfk = tf.keras
import pdb

from rlfd import  memory, normalizer, policies
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
        online_sample_ratio,
        offline_batch_size,
        fix_T,
        norm_obs_online,
        norm_obs_offline,
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
        info,
        termination_function = default_termination #compute done bit for model rollouts

        ):
        
        super(MBPO, self).__init__(
            dims,
            max_u,
            eps_length,
            gamma,
            offline_batch_size,
            online_batch_size,
            online_sample_ratio,
            fix_T,
            norm_obs_online,
            norm_obs_offline,
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
        agent.Agent.__init__(self, locals())
        self.dynamics_model = mbpo_networks.BNNEnsemble(self.dimo[0], self.dimu[0], model_layer_size, model_weight_decay, num_networks, num_elites, model_lr)
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

        for _ in range(self.rollout_length):
            u_tf, _ = self._actor([self._actor_o_norm(o_tf)]) #sample next action from policy
            o_2_tf, r_tf = self.dynamics_model.predict(o_tf, u_tf) #predict next_obs and reward
            o = o_tf.numpy()
            o_2 = o_2_tf.numpy()
            u = u_tf.numpy()
            r = r_tf.numpy()
            done = self.termination_func(o, u, o_2)
            samples = {"o": o, "o_2": o_2, "r": r, "done": done, "u": u}
            self.model_buffer.store(samples)

            not_done_mask = ~done.squeeze(-1)
            o_tf = tf.convert_to_tensor(o_2[not_done_mask]) #make the next obs the current obs

    def sample_batch(self):
        #modify to sample from model buffer according to real ratio
        if self.real_ratio == 1:
            batch = self.online_buffer.sample(self.online_batch_size)
        else:
            model_batch_size = int((self.online_batch_size * self.real_ratio))
            real_batch_size = self.online_batch_size - model_batch_size
            model_batch = self.model_buffer.sample(model_batch_size)
            real_batch = self.model_buffer.sample(real_batch_size)
            batch = self._merge_batch_experiences(real_batch, model_batch)
        
        return batch
    
    def train_online(self):
        with tf.summary.record_if(lambda: self.online_training_step % 200 == 0):

            if self.online_training_step % self.model_train_freq == 0:
                model_batch = self.online_buffer.sample(self.online_buffer.stored_steps)
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
        
            batch = self.sample_batch()

            o_tf = tf.convert_to_tensor(batch["o"], dtype=tf.float32)
            o_2_tf = tf.convert_to_tensor(batch["o_2"], dtype=tf.float32)
            u_tf = tf.convert_to_tensor(batch["u"], dtype=tf.float32)
            r_tf = tf.convert_to_tensor(batch["r"], dtype=tf.float32)
            done_tf = tf.convert_to_tensor(batch["done"], dtype=tf.float32)

            self._train_online_graph(o_tf, o_2_tf, u_tf, r_tf, done_tf)





        
