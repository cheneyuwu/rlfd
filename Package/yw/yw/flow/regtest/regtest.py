"""RL Project Yuchen WU

This file is for end to end testing.

"""

from yw.util.cmd_util import Command
from collections import OrderedDict
import os

class RegTest:
    """Directories of sample directories
    """

    def __init__(self):
        # Environment variables
        self.root_dir = os.getenv("PROJECT")
        self.flow_dir = os.path.join(self.root_dir, "Package/yw/yw/flow/")
        self.data_dir = os.path.join(self.root_dir, "Package/yw/yw/flow/regtest/data/")
        self.exp_dir = os.getenv("TEMPDIR")
        self.result_dir = os.path.join(self.exp_dir, "RegResult/")

class Demo(RegTest):
    """For the generate demo process
    """

    def __init__(self):
        super().__init__()
        self.flow = os.path.join(self.flow_dir, "generate_demo.py")
        self.run = OrderedDict([("python", self.flow), ("--loglevel", 1)])

    @Command.execute
    def generate_demo(self):
        command = self.run.copy()
        command["--policy_file"] = self.data_dir + "rl_policy_latest"
        command["--store_dir"] = self.result_dir + "DemoData/"
        command["--num_itr"] = 10
        return command


class Train(RegTest):
    """For the training process
    """

    def __init__(self):
        super().__init__()
        self.run = OrderedDict(
            [
                ("python", os.path.join(self.flow_dir, "train_ddpg_main.py")),
                ("--loglevel", 1),
                ("--env", "Reach2D"),
                ("--eps_length", 2),
                ("--seed", 0),
                ("--train_rl_epochs", 1),
                ("--debug_params", 1),
            ]
        )

    @Command.execute
    def rl_only(self):
        command = self.run.copy()
        command["--logdir"] = self.result_dir + "RLNoDemo/"
        command["--save_path"] = self.result_dir + "RLNoDemo/"
        command["--train_rl"] = 1
        return command

if __name__ == "__main__":

    train_test = Train()
    demo_test = Demo()

    assert not train_test.rl_only(), "RL Only flow failed!"
    assert not demo_test.generate_demo(), "Demonstration generation failed!"

