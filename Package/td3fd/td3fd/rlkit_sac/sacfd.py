from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import TorchTrainer

from rlkit.torch.core import np_to_pytorch_batch


class SACFDTrainer(TorchTrainer):
    def __init__(
            self,
            env,
            policy,
            qf1,
            qf2,
            target_qf1,
            target_qf2,

            discount=0.99,
            reward_scale=1.0,

            policy_lr=1e-3,
            qf_lr=1e-3,
            optimizer_class=optim.Adam,

            soft_target_tau=1e-2,
            target_update_period=1,
            plotter=None,
            render_eval_paths=False,

            use_automatic_entropy_tuning=True,
            target_entropy=None,

            demo_strategy="none",
            demo_batch_size=128,
            prm_loss_weight=1.0,
            aux_loss_weight=1.0,
            q_filter=True,
            demo_replay_buffer=None,
            shaping=None,
    ):
        super().__init__()
        self.env = env
        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.target_qf1 = target_qf1
        self.target_qf2 = target_qf2
        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period

        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            if target_entropy:
                self.target_entropy = target_entropy
            else:
                self.target_entropy = -np.prod(self.env.action_space.shape).item()  # heuristic value from Tuomas
            self.log_alpha = ptu.zeros(1, requires_grad=True)
            self.alpha_optimizer = optimizer_class(
                [self.log_alpha],
                lr=policy_lr,
            )

        self.plotter = plotter
        self.render_eval_paths = render_eval_paths

        self.demo_strategy = demo_strategy
        self.demo_batch_size = demo_batch_size
        self.prm_loss_weight = prm_loss_weight
        self.aux_loss_weight = aux_loss_weight
        self.q_filter = q_filter
        self.demo_replay_buffer = demo_replay_buffer
        self.shaping = shaping

        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
        )
        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(),
            lr=qf_lr,
        )
        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(),
            lr=qf_lr,
        )

        self.discount = discount
        self.reward_scale = reward_scale
        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True

    def train_from_torch(self, batch):
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        if self.demo_strategy != "none":
            demo_batch = self.demo_replay_buffer.random_batch(self.demo_batch_size) # todo: make this a hyper parameter
            demo_batch = np_to_pytorch_batch(demo_batch)
            demo_rewards = demo_batch['rewards']
            demo_terminals = demo_batch['terminals']
            demo_obs = demo_batch['observations']
            demo_actions = demo_batch['actions']
            demo_next_obs = demo_batch['next_observations']

            comb_obs = torch.cat((obs, demo_obs))
        else:
            comb_obs = obs

        """
        Policy and Alpha Loss
        """
        new_obs_actions, policy_mean, policy_log_std, log_pi, *_ = self.policy(
            comb_obs, reparameterize=True, return_log_prob=True,
        )
        if self.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha = self.log_alpha.exp()
        else:
            alpha_loss = 0
            alpha = 1

        q_new_actions = torch.min(
            self.qf1(comb_obs, new_obs_actions),
            self.qf2(comb_obs, new_obs_actions),
        )
        policy_loss = (alpha*log_pi - q_new_actions).mean()

        # extra bc loss on demonstration data
        if self.demo_strategy in ["gan", "nf"]:
            policy_loss += -self.shaping.potential(comb_obs, None, new_obs_actions).mean()
        if self.demo_strategy == "bc":
            new_demo_obs_actions, *_ = self.policy(
                demo_obs, reparameterize=True, return_log_prob=True,
            )
            if self.q_filter:
                q_demo_actions = torch.min(
                    self.qf1(demo_obs, demo_actions),
                    self.qf2(demo_obs, demo_actions),
                ).detach()
                q_new_demo_actions = torch.min(
                    self.qf1(demo_obs, new_demo_obs_actions),
                    self.qf2(demo_obs, new_demo_obs_actions),
                ).detach()
                masked_bc_loss = torch.where(
                    q_demo_actions > q_new_demo_actions, 
                    (new_demo_obs_actions - demo_actions) ** 2, 
                    torch.zeros_like(demo_actions)
                )
                bc_loss = torch.mean(masked_bc_loss)
            else: 
                bc_loss = torch.mean((new_demo_obs_actions - demo_actions) ** 2)
            policy_loss = self.aux_loss_weight * bc_loss + self.prm_loss_weight * policy_loss # TODO

        """
        QF Loss
        """
        q1_pred = self.qf1(obs, actions)
        q2_pred = self.qf2(obs, actions)
        # Make sure policy accounts for squashing functions like tanh correctly!
        new_next_actions, _, _, new_log_pi, *_ = self.policy(
            next_obs, reparameterize=True, return_log_prob=True,
        )
        target_q_values = torch.min(
            self.target_qf1(next_obs, new_next_actions),
            self.target_qf2(next_obs, new_next_actions),
        ) - alpha * new_log_pi

        q_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_q_values
        if self.demo_strategy in ["gan", "nf"]:
            q_target += self.shaping.reward(obs, None, actions, next_obs, None, new_next_actions)
        qf1_loss = self.qf_criterion(q1_pred, q_target.detach())
        qf2_loss = self.qf_criterion(q2_pred, q_target.detach())

        # extra logging
        if self.demo_strategy in ["gan", "nf"]:
            p_pred = self.shaping.potential(obs, None, actions)
            qp1_pred = p_pred + q1_pred
            qp2_pred = p_pred + q2_pred

        """
        Update networks
        """
        self.qf1_optimizer.zero_grad()
        qf1_loss.backward()
        self.qf1_optimizer.step()

        self.qf2_optimizer.zero_grad()
        qf2_loss.backward()
        self.qf2_optimizer.step()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        """
        Soft Updates
        """
        if self._n_train_steps_total % self.target_update_period == 0:
            ptu.soft_update_from_to(
                self.qf1, self.target_qf1, self.soft_target_tau
            )
            ptu.soft_update_from_to(
                self.qf2, self.target_qf2, self.soft_target_tau
            )

        """
        Save some statistics for eval
        """
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            policy_loss = (log_pi - q_new_actions).mean()

            self.eval_statistics['QF1 Loss'] = np.mean(ptu.get_numpy(qf1_loss))
            self.eval_statistics['QF2 Loss'] = np.mean(ptu.get_numpy(qf2_loss))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            if self.demo_strategy in ["gan", "nf"]:
                self.eval_statistics.update(create_stats_ordered_dict(
                    'P Predictions',
                    ptu.get_numpy(p_pred),
                ))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'P plus Q1 Predictions',
                    ptu.get_numpy(qp1_pred),
                ))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'P plus Q2 Predictions',
                    ptu.get_numpy(qp2_pred),
                ))                
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q1 Predictions',
                ptu.get_numpy(q1_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q2 Predictions',
                ptu.get_numpy(q2_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Targets',
                ptu.get_numpy(q_target),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(log_pi),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy mu',
                ptu.get_numpy(policy_mean),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy log std',
                ptu.get_numpy(policy_log_std),
            ))
            if self.use_automatic_entropy_tuning:
                self.eval_statistics['Alpha'] = alpha.item()
                self.eval_statistics['Alpha Loss'] = alpha_loss.item()
        self._n_train_steps_total += 1

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        return [
            self.policy,
            self.qf1,
            self.qf2,
            self.target_qf1,
            self.target_qf2,
        ]

    def get_snapshot(self):
        return dict(
            policy=self.policy,
            qf1=self.qf1,
            qf2=self.qf2,
            target_qf1=self.qf1,
            target_qf2=self.qf2,
            shaping=self.shaping,
        )

