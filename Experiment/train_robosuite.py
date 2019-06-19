import os
from collections import OrderedDict
import numpy as np
from yw.util.cmd_util import Command
from train import Experiment, Demo, Train, Display, Plot


class Render(Experiment):
    """For rendering the environment
    """

    def __init__(self):
        super().__init__()
        self.run = OrderedDict([("python", os.path.join(self.flow_dir, "render/robosuite_render.py"))])

    @Command.execute
    def render(self, **override):
        command = self.run.copy()
        command["--policy_file"] = self.result_dir + "RLDemoCriticPolicy/rl/policy_latest.pkl"
        command["--env_arg"] = {
            "has_renderer": "bool:*",
            "use_object_obs": "bool:*",
            "use_camera_obs": "bool:",
            "ignore_done": "bool:*",
            "control_freq": "int:50",
        }
        command["--num_itr"] = 10
        # command["--eps_length"] = 1000
        return command


if __name__ == "__main__":

    demo_exp = Demo()
    train_exp = Train()
    display_exp = Display()
    plot_exp = Plot()
    render_exp = Render()

    environment = "SawyerLift"
    train_exp.env = environment
    train_exp.num_cpu = 3
    train_exp.update()

    train_rl_epochs = 50
    seed = 1

    for i in range(1):
        seed += i * 100

        # We can change the result directory without updating
        exp_dir = os.getenv("EXPERIMENT")
        result_dir = os.path.join(exp_dir, "Result/Temp/Exp" + str(i) + "/")
        demo_exp.result_dir = result_dir
        train_exp.result_dir = result_dir

        assert not train_exp.rl_only(
            rl_scope="rl_only",
            n_cycles=20,
            seed=seed + 10,
            rl_num_sample=1,
            rl_batch_size=256,
            train_rl_epochs=train_rl_epochs,
            rollout_batch_size=4,
            n_test_rollouts=5,  # for evaluation only, cannot make this larger because we will run out of memory
            env_arg={
                "has_renderer": "bool:",  # no on-screen renderer
                "has_offscreen_renderer": "bool:",  # no off-screen renderer
                "use_object_obs": "bool:*",  # use object-centric feature
                "use_camera_obs": "bool:",  # no camera observations)
                "reward_shaping": "bool:*",  # use dense rewards
                "control_freq": "int:20",
            },
        )

    # Plot the training result
    # assert not plot_exp.plot(dir=plot_exp.result_dir)

    # Render a policy result (calls robosuite_render.py).
    assert not render_exp.render(
        policy_file=render_exp.result_dir + "../Jun15RoboSuite/RLNoDemo/rl/policy_latest.pkl"
    )
