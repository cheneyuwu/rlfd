"""
This is a simple wrapper on top of the shaping code in td3fd, for debugging
"""
import numpy as np
import torch
from td3fd.td3.shaping import GANShaping, NFShaping
from td3fd.memory import iterbatches

from rlkit.core.logging import logger

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

shaping_cls = {"nf": NFShaping, "gan": GANShaping}


class EnsembleRewardShapingWrapper:
    def __init__(self, num_ensembles, *args, **kwargs):
        self.shapings = [RewardShaping(*args, **kwargs) for _ in range(num_ensembles)]

    def train(self, *args, **kwargs):
        for i, shaping in enumerate(self.shapings):
            logger.log("Training shaping function #{}...".format(i))
            shaping.train(*args, **kwargs)

    def evaluate(self, *args, **kwargs):
        for i, shaping in enumerate(self.shapings):
            logger.log("Evaluating shaping function #{}...".format(i))
            shaping.evaluate(*args, **kwargs)

    def potential(self, *args, **kwargs):
        potentials = [x.potential(*args, **kwargs) for x in self.shapings]
        return torch.mean(torch.stack(potentials), dim=0)

    def reward(self, *args, **kwargs):
        rewards = [x.reward(*args, **kwargs) for x in self.shapings]
        return torch.mean(torch.stack(rewards), dim=0)

    def __getstate__(self):
        """
        We only save the shaping class. after reloading, only potential and reward functions can be used.
        """
        state = {"shaping": self.shapings}
        return state

    def __setstate__(self, state):
        self.shapings = state["shaping"]


class RewardShaping:
    def __init__(self, env, demo_strategy, discount, num_epochs, batch_size, **shaping_params):

        if demo_strategy not in shaping_cls.keys():
            self.shaping = None
            return

        self.shaping_params = shaping_params[demo_strategy]

        self.num_epochs = num_epochs
        self.batch_size = batch_size

        dims = {
            # "o": env.observation_space["observation"].shape,  # for td3fd
            # "g": env.observation_space["desired_goal"].shape,
            "o": env.observation_space.shape,  # for rlkit
            "g": (0,),
            #
            "u": env.action_space.shape,
        }
        max_u = env.action_space.high[0]
        assert max_u == 1.0  # for rlkit
        # Update parameters
        self.shaping_params.update(
            {
                "dims": dims,  # agent takes an input observations
                "max_u": max_u,
                "gamma": discount,
                "norm_obs": shaping_params["norm_obs"],
                "norm_eps": shaping_params["norm_eps"],
                "norm_clip": shaping_params["norm_clip"],
            }
        )
        self.shaping = shaping_cls[demo_strategy](**self.shaping_params)

    def _convert_rlkit_to_td3fd(self, demo_data):
        converted_demo_data = dict()
        keys = demo_data[0].keys()
        for path in demo_data:
            for key in keys:
                if key in converted_demo_data.keys():
                    if type(path[key]) == list:
                        converted_demo_data[key] += path[key]
                    else:
                        converted_demo_data[key] = np.concatenate((converted_demo_data[key], path[key]), axis=0)
                else:
                    converted_demo_data[key] = path[key]
        demo_data = converted_demo_data

        demo_data["o"] = demo_data["observations"]
        demo_data["u"] = demo_data["actions"]
        assert len(demo_data["o"].shape) == 2
        demo_data["g"] = np.empty((demo_data["o"].shape[0], 0))
        return demo_data

    def train(self, demo_data):
        # for rlkit
        demo_data = self._convert_rlkit_to_td3fd(demo_data)
        #
        self.shaping.update_stats(demo_data)

        for epoch in range(self.num_epochs):
            losses = np.empty(0)
            for (o, g, u) in iterbatches((demo_data["o"], demo_data["g"], demo_data["u"]), batch_size=self.batch_size):
                batch = {"o": o, "g": g, "u": u}
                d_loss, g_loss = self.shaping.train(batch)
                losses = np.append(losses, d_loss.cpu().data.numpy())
            if epoch % (self.num_epochs / 100) == (self.num_epochs / 100 - 1):
                logger.log("epoch: {} demo shaping loss: {}".format(epoch, np.mean(losses)))
                # print("epoch: {} demo shaping loss: {}".format(epoch, np.mean(losses)))
                mean_pot = self.shaping.evaluate(batch)
                logger.log("epoch: {} mean potential on demo data: {}".format(epoch, mean_pot))

        # Subtract mean, do this if needed
        # self.shaping.update_potential_mean(demo_data)

    def evaluate(self):
        return
        # input data - used for both training and test set
        dim1 = 0
        dim2 = 1
        num_point = 24
        ls = np.linspace(-1.0, 1.0, num_point)
        o_1, o_2 = np.meshgrid(ls, ls)
        o_r = 0.0 * np.ones((num_point ** 2, 4))  # TODO change this dimension
        o_r[..., dim1 : dim1 + 1] = o_1.reshape(-1, 1)
        o_r[..., dim2 : dim2 + 1] = o_2.reshape(-1, 1)
        u_r = 1.0 * np.ones((num_point ** 2, 2))  # TODO change this dimension

        o_tc = torch.tensor(o_r, dtype=torch.float).to(device)
        g_tc = torch.empty((o_tc.shape[0], 0)).to(device)
        u_tc = torch.tensor(u_r, dtype=torch.float).to(device)

        p = self.shaping.potential(o_tc, g_tc, u_tc).cpu().data.numpy()

        res = {"o": (o_1, o_2), "surf": p.reshape((num_point, num_point))}

        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        from td3fd.td3.debug import visualize_query

        plt.figure(0)
        gs = gridspec.GridSpec(1, 1)
        ax = plt.subplot(gs[0, 0], projection="3d")
        ax.clear()
        visualize_query.visualize_potential_surface(ax, res)
        plt.show(block=True)

    def potential(self, o, g, u):
        # for rlkit
        assert len(o.shape) == 2
        g = torch.empty((o.shape[0], 0)).to(device)
        #
        potential = self.shaping.potential(o, g, u)
        assert not any(torch.isnan(potential)), "Found NaN in potential {}.".format(potential)
        return potential

    def reward(self, o, g, u, o_2, g_2, u_2):
        # for rlkit
        assert len(o.shape) == 2
        g = torch.empty((o.shape[0], 0)).to(device)
        g_2 = torch.empty((o_2.shape[0], 0)).to(device)
        #
        return self.shaping.reward(o, g, u, o_2, g_2, u_2)

    def __getstate__(self):
        """
        We only save the shaping class. after reloading, only potential and reward functions can be used.
        """
        state = {"shaping": self.shaping}
        return state

    def __setstate__(self, state):
        self.shaping = state["shaping"]
