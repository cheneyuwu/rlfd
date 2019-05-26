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
        self.update()

    def update(self):
        """Call this function when you change flowdir or env
        """
        self.run = OrderedDict(
            [
                ("python", os.path.join(self.flow_dir, "train_ddpg_main.py")),
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
        command["--logdir"] = self.result_dir + "RLHERNoDemo/"
        command["--save_path"] = self.result_dir + "RLHERNoDemo/"
        command["--her_strategy"] = "future"
        return command

    @Command.execute
    def rl_only(self, **kwargs):
        command = self.run.copy()
        command["--logdir"] = self.result_dir + "RLNoDemo/"
        command["--save_path"] = self.result_dir + "RLNoDemo/"
        command["--her_strategy"] = "none"
        return command

    @Command.execute
    def rl_with_demo_critic_rb(self, **kwargs):
        command = self.run.copy()
        command["--logdir"] = self.result_dir + "RLDemoCriticReplBuffer"
        command["--save_path"] = self.result_dir + "RLDemoCriticreplBuffer/"
        command["--demo_critic"] = "rb"
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
        command["--dirs"] = self.result_dir + "/RLNoDemo"
        return command


class Animation(Experiment):
    """For result plotting
    """

    def __init__(self):
        super().__init__()
        self.run = OrderedDict([("python", os.path.join(self.flow_dir, "query/query_animation.py"))])

    @Command.execute
    def plot_animation(self, **override):
        command = self.run.copy()
        command["--load_dir"] = self.result_dir + "/RLDemoCriticPolicy/query"
        command["--save"] = 1
        return command


class CompareQ(Experiment):
    """For result plotting
    """

    def __init__(self):
        super().__init__()
        self.run = OrderedDict([("python", os.path.join(self.flow_dir, "query/query_compare_q.py"))])

    @Command.execute
    def compare(self, **override):
        command = self.run.copy()
        command["--query_file"] = self.result_dir + "/RLNoDemo/critic_q/query_latest.npz"
        command["--policy_file"] = self.result_dir + "/RLNoDemo/rl/policy_latest.pkl"
        command["--store_dir"] = self.result_dir
        return command


class Uncertainty(Experiment):
    """For result plotting
    """

    def __init__(self):
        super().__init__()
        self.run = OrderedDict([("python", os.path.join(self.flow_dir, "query/query_uncertainty.py"))])

    @Command.execute
    def check(self, **override):
        command = self.run.copy()
        command["--load_dir"] = self.result_dir + "RLDemoCriticPolicy/uncertainty/"
        command["--save"] = 1
        return command


if __name__ == "__main__":
    demo_exp = Demo()
    train_exp = Train()
    display_exp = Display()
    plot_exp = Plot()
    animation = Animation()
    compare_q = CompareQ()
    uncertainty = Uncertainty()

    # Quick checks
    ###################################################################################################################
    # exp_dir = os.getenv("EXPERIMENT")
    # result_dir = os.path.join(exp_dir, "Result/Temp/")

    # display_exp.display(policy_file=result_dir + "/RLDemoCriticPolicy/rl/policy_latest.pkl")
    # assert not uncertainty.check()
    # assert not animation.plot_animation(load_dir=os.path.join(result_dir, "../Ap28Reach2DFO/Demo256/RLNoDemo/ac_output"))
    # exit(0)

    # # Used for openai environments
    # ###################################################################################################################

    # train_exp.env = "FetchPush-v1"
    # train_exp.update()

    # # We can change the result directory without updating
    # exp_dir = os.getenv("EXPERIMENT")
    # demo_exp.result_dir = os.path.join(exp_dir, "Result/Temp2/")
    # train_exp.result_dir = os.path.join(exp_dir, "Result/Temp2/")
    # display_exp.result_dir = os.path.join(exp_dir, "Result/Temp2/")
    # plot_exp.result_dir = os.path.join(exp_dir, "Result/Temp2/")
    # animation.result_dir = os.path.join(exp_dir, "Result/Temp2/")
    # compare_q.result_dir = os.path.join(exp_dir, "Result/Temp2/")
    # uncertainty.result_dir = os.path.join(exp_dir, "Result/Temp2/")

    # demo_data_size = 10240
    # train_rl_epochs = 100

    # # Train the RL without demonstration
    # train_exp.rl_only(
    #     rl_scope="critic_demo",
    #     seed=0,
    #     rl_num_sample=1,
    #     train_rl_epochs=train_rl_epochs,
    # )

    # # Generate demonstration data
    # demo_exp.generate_demo(
    #     num_itr=demo_data_size,
    #     entire_eps=1,
    # )

    # # Train the demonstration network.
    # # Method 1
    # # train_exp.demo_nn_critic_only(demo_file=train_exp.result_dir + "/RLNoDemo/critic_q/query_latest.npz", demo_test_file=train_exp.result_dir + "/RLNoDemo/critic_q/query_latest.npz")
    # # Method 2
    # train_exp.demo_nn_critic_only(
    #     train_demo_epochs=100,
    #     demo_num_sample=12,
    #     demo_data_size=demo_data_size,
    #     demo_batch_size=int(np.ceil(demo_data_size / 16)),
    # )

    # # Train the RL with demonstration
    # train_exp.rl_with_demo_critic_nn(
    #     seed=13,
    #     rl_num_sample=1,
    #     train_rl_epochs=train_rl_epochs,
    #     # Method 1
    #     # demo_policy_file=os.path.join(train_exp.result_dir, "RLNoDemo/rl/policy_latest.pkl"),
    #     # Method 2
    #     demo_policy_file=os.path.join(train_exp.result_dir, "DemoCriticOnly/rl_demo_critic/policy_latest.pkl"),
    # )

    # # Plot the training result
    # plot_exp.plot(dirs=plot_exp.result_dir)

    # exit()

    # Used for the Reach2D environment
    ###################################################################################################################

    environment = "Reach2D"
    train_exp.env = environment
    train_exp.update()

    demo_data_size = 128
    train_rl_epochs = 50
    # seed = 0
    # for i in range(4):
    #     seed += i*100

    #     # We can change the result directory without updating
    #     exp_dir = os.getenv("EXPERIMENT")
    #     result_dir = os.path.join(exp_dir, "Result/Temp/Exp"+str(i)+"/")
    #     demo_exp.result_dir = result_dir
    #     train_exp.result_dir = result_dir
    #     display_exp.result_dir = result_dir
    #     plot_exp.result_dir = result_dir
    #     animation.result_dir = result_dir
    #     compare_q.result_dir = result_dir
    #     uncertainty.result_dir = result_dir

    #     # Train the RL without demonstration
    #     assert not train_exp.rl_only(
    #         r_scale=1.0,
    #         r_shift=0.0,
    #         rl_action_l2=0.5,
    #         rl_scope="critic_demo",
    #         n_cycles=10,
    #         seed=seed+10,
    #         rl_num_sample=1,
    #         rl_batch_size=256,
    #         train_rl_epochs=train_rl_epochs,
    #     )
    #     assert not train_exp.rl_her_only(
    #         r_scale=1.0,
    #         r_shift=0.0,
    #         rl_action_l2=0.5,
    #         rl_scope="critic_demo",
    #         n_cycles=10,
    #         seed=seed+20,
    #         rl_num_sample=1,
    #         rl_batch_size=256,
    #         train_rl_epochs=train_rl_epochs,
    #     )

    #     # Generate demonstration data
    #     assert not demo_exp.generate_demo(seed=seed+30, num_itr=demo_data_size, entire_eps=1, shuffle=0)

    #     # Train the RL with demonstration
    #     assert not train_exp.rl_with_demo_critic_rb(
    #         r_scale=1.0,
    #         r_shift=0.0,
    #         rl_action_l2=0.5,
    #         n_cycles=10,
    #         seed=seed+40,
    #         rl_num_sample=1,
    #         rl_batch_size=512,
    #         rl_batch_size_demo=256, 
    #         train_rl_epochs=train_rl_epochs,
    #         demo_file=result_dir + "DemoData/" + environment + ".npz",
    #         rl_num_demo=demo_data_size,
    #         her_strategy="none",
    #     )

    # Plot the training result
    assert not plot_exp.plot(dirs=plot_exp.result_dir)

    # Plot the query result
    # assert not animation.plot_animation(load_dir=os.path.join(animation.result_dir, "RLNoDemo/ac_output"))
    # assert not animation.plot_animation(load_dir=os.path.join(animation.result_dir, "RLDemoCriticPolicy/ac_output"))

    # Compare the real q and critic q from arbitrary initial state and action pair.
    # assert not compare_q.compare()

    # Check the uncertainty output of the demonstration output
    # assert not uncertainty.check()

    # Display a policy result (calls run_agent).
    # assert not display_exp.display(policy_file=display_exp.result_dir + "RLDemoCriticReplBuffer/rl/policy_latest.pkl")

    exit()

    ###################################################################################################################
    # Used for checking the uncertainty output from the demonstration neural net on simple data

    # Other useful commands
    ###################################################################################################################

    # Plot the collected query result
    # animation.plot_animation()

    # Plot the training result
    # plot_exp.plot()

    # Display a policy result (calls run_agent).
    # display_exp.display()

    # A complete test set as example.
    # train_exp.rl_only( rl_action_l2 = 0.5, n_cycles=10, seed=0)
    # demo_exp.generate_demo()
    # train_exp.demo_nn_critic_only()
    # train_exp.rl_with_demo_critic_nn( rl_action_l2 = 0.5, n_cycles=10, seed=0)
    # train_exp.rl_with_demo_critic( rl_action_l2 = 0.1, n_cycles=20, seed=3)
