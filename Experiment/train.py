"""Script for running experiments and analyzing results.
"""

import os
from collections import OrderedDict

import numpy as np

from yw.util.cmd_util import Command


class Experiment:
    """Directories of sample directories
    """

    def __init__(self):
        # Environment variables
        self.root_dir = os.getenv("PROJECT")
        self.flow_dir = os.path.join(self.root_dir, "Package/yw/yw/flow/")  # Cannot change this directly!
        self.exp_dir = os.getenv("EXPERIMENT")
        self.result_dir = os.path.join(self.exp_dir, "Result/Temp/")


class Demo(Experiment):
    """For the generate demo process
    """

    def __init__(self):
        super().__init__()
        self.run = OrderedDict([("python", os.path.join(self.flow_dir, "generate_demo.py")), ("--loglevel", 2)])
        self.tested = []

    @Command.execute
    def generate_demo(self, **override):
        command = self.run.copy()
        command["--policy_file"] = self.result_dir + "RLHERNoDemo/rl/policy_latest.pkl"
        command["--store_dir"] = self.result_dir + "DemoData/"
        command["--num_itr"] = 128
        command["--entire_eps"] = 1
        command["--seed"] = 0
        return command


class Train(Experiment):
    """For the training process
    """

    def __init__(self):
        super().__init__()
        self.env = "Reach2D"
        self.num_cpu = 1
        self.update()

    def update(self):
        """Call this function when you change flowdir or env
        """
        self.run = OrderedDict(
            [
                ("mpirun -np " + str(self.num_cpu) + " python", os.path.join(self.flow_dir, "train_ddpg_main.py")),
                ("--loglevel", 2),
                ("--save_interval", 2),
                ("--env", self.env),
                ("--r_scale", 1.0),
                ("--r_shift", 0.0),
                ("--seed", 0),
                ("--train_rl_epochs", 20),
                ("--rl_num_sample", 1),
            ]
        )

    @Command.execute
    def rl_her_only(self, **kwargs):
        command = self.run.copy()
        command["--logdir"] = self.result_dir + "RLHER/"
        command["--save_path"] = self.result_dir + "RLHER/"
        command["--rl_replay_strategy"] = "her"
        return command

    @Command.execute
    def rl_only(self, **kwargs):
        command = self.run.copy()
        command["--logdir"] = self.result_dir + "RL/"
        command["--save_path"] = self.result_dir + "RL/"
        command["--rl_replay_strategy"] = "none"
        return command

    @Command.execute
    def rl_with_shaping(self, **kwargs):
        command = self.run.copy()
        command["--logdir"] = self.result_dir + "RLDemoShaping"
        command["--save_path"] = self.result_dir + "RLDemoShaping/"
        command["--demo_critic"] = "shaping"
        return command


class Display(Experiment):
    """For running the agent
    """

    def __init__(self):
        super().__init__()
        self.run = OrderedDict([("python", os.path.join(self.flow_dir, "run_agent.py"))])

    @Command.execute
    def display(self, **override):
        command = self.run.copy()
        command["--policy_file"] = self.result_dir + "RLDemoCriticPolicy/rl/policy_latest.pkl"
        return command


class Plot(Experiment):
    """For result plotting
    """

    def __init__(self):
        super().__init__()
        self.run = OrderedDict([("python", os.path.join(self.flow_dir, "plot.py"))])

    @Command.execute
    def plot(self, **override):
        command = self.run.copy()
        command["--dir"] = [self.result_dir + "/RLNoDemo"]
        command["--xy"] = ["epoch:test/success_rate"]
        return command


if __name__ == "__main__":
    demo_exp = Demo()
    train_exp = Train()
    display_exp = Display()
    plot_exp = Plot()

    # Quick checks
    ###################################################################################################################

    # exp_dir = os.getenv("EXPERIMENT")
    # result_dir = os.path.join(exp_dir, "Result/Temp/")

    # display_exp.display(policy_file=result_dir + "/RLDemoCriticPolicy/rl/policy_latest.pkl")
    # assert not uncertainty.check()
    # assert not animation.plot_animation(load_dir=os.path.join(result_dir, "../Ap28Reach2DFO/Demo256/RLNoDemo/ac_output"))

    # exit()