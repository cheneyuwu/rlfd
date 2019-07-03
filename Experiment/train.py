"""Script for running experiments and analyzing results.
"""

import os
import inspect

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

from yw.flow import plot, train_ddpg_main, generate_demo, run_agent


class Experiment:
    """Directories of sample directories
    """

    def __init__(self):
        # Environment variables
        self.root_dir = os.getenv("PROJECT")
        self.exp_dir = os.getenv("EXPERIMENT")
        self.result_dir = os.path.join(self.exp_dir, "TempResult")  # Cannot change this directly!
        # Overwrite the following variables
        self.use_mpi = False  # whether or not to run with multiple threads
        self.shared_cmd = {}
        self.launch_function = None
        self.parser = None

    def set_shared_cmd(self, **kwargs):
        self.shared_cmd.update(kwargs)

    @staticmethod
    def execute(func):
        def wrapper(self, **override):
            cmd = {}
            cmd.update({"--" + k: func(self)[k] for k in func(self).keys()})
            cmd.update({"--" + k: override[k] for k in override.keys()})

            # Call the target function directly
            run = []
            for key in cmd:
                if type(cmd[key]) is dict:
                    for v in cmd[key].keys():
                        run.append(str(key))
                        run.append(str(v) + ":" + str(cmd[key][v]))
                elif type(cmd[key]) is list:
                    for v in cmd[key]:
                        run.append(str(key))
                        run.append(str(v))
                else:  # string
                    run.append(str(key))
                    run.append(str(cmd[key]))
            is_main_thd = not MPI.COMM_WORLD.Get_rank() if MPI != None else True
            if is_main_thd:
                print("\n\nTo launch from command line:")
                print("============================")
                print("python", inspect.getsourcefile(self.launch_function), " ".join(run))
                print("============================")
            if is_main_thd or self.use_mpi:
                self.parser.parse(run)
                self.launch_function(**self.parser.get_dict())

        return wrapper


class Demo(Experiment):
    """For the generate demo process
    """

    def __init__(self):
        super().__init__()
        self.launch_function = generate_demo.main
        self.parser = generate_demo.ap
        # command common to all sub exps
        self.use_mpi = True
        self.set_shared_cmd(loglevel=2, entire_eps=1, shuffle=0)

    @Experiment.execute
    def generate_demo(self, **override):
        command = self.shared_cmd.copy()
        command["num_itr"] = 128
        command["store_dir"] = os.path.join(self.result_dir, "DemoData")
        return command


class Train(Experiment):
    """For the training process
    """

    def __init__(self):
        super().__init__()
        self.launch_function = train_ddpg_main.main
        self.parser = train_ddpg_main.ap
        self.use_mpi = True
        # command common to all sub exps
        self.set_shared_cmd(
            loglevel=2,
            save_interval=2,
            env="Reach2D",
            r_scale=1.0,
            r_shift=0.0,
            seed=0,
            train_rl_epochs=20,
            rl_num_sample=1,
        )


    @Experiment.execute
    def rl_only_dense(self, **kwargs):
        command = self.shared_cmd.copy()
        command["logdir"] = os.path.join(self.result_dir, "RLDense")
        command["save_path"] = os.path.join(self.result_dir, "RLDense")
        command["rl_replay_strategy"] = "none"
        return command

    @Experiment.execute
    def rl_her_only(self, **kwargs):
        command = self.shared_cmd.copy()
        command["logdir"] = os.path.join(self.result_dir, "RLHER")
        command["save_path"] = os.path.join(self.result_dir, "RLHER")
        command["rl_replay_strategy"] = "her"
        return command

    @Experiment.execute
    def rl_only(self, **kwargs):
        command = self.shared_cmd.copy()
        command["logdir"] = os.path.join(self.result_dir, "RL")
        command["save_path"] = os.path.join(self.result_dir, "RL")
        command["rl_replay_strategy"] = "none"
        return command

    @Experiment.execute
    def rl_her_with_shaping(self, **kwargs):
        command = self.shared_cmd.copy()
        command["logdir"] = os.path.join(self.result_dir, "RLHERDemoShaping")
        command["save_path"] = os.path.join(self.result_dir, "RLDemoShaping")
        command["rl_replay_strategy"] = "her"
        command["demo_critic"] = "nf"
        return command

    @Experiment.execute
    def rl_with_shaping(self, **kwargs):
        command = self.shared_cmd.copy()
        command["logdir"] = os.path.join(self.result_dir, "RLDemoShaping")
        command["save_path"] = os.path.join(self.result_dir, "RLDemoShaping")
        command["rl_replay_strategy"] = "none"
        command["demo_critic"] = "nf"
        return command

    @Experiment.execute
    def rl_her_with_bc(self, **kwargs):
        command = self.shared_cmd.copy()
        command["logdir"] = os.path.join(self.result_dir, "RLHERDemoBC")
        command["save_path"] = os.path.join(self.result_dir, "RLHERDemoBC")
        command["rl_replay_strategy"] = "her"
        command["demo_actor"] = "bc"
        return command

    @Experiment.execute
    def rl_with_bc(self, **kwargs):
        command = self.shared_cmd.copy()
        command["logdir"] = os.path.join(self.result_dir, "RLDemoBC")
        command["save_path"] = os.path.join(self.result_dir, "RLDemoBC")
        command["rl_replay_strategy"] = "none"
        command["demo_actor"] = "bc"
        return command


class Display(Experiment):
    """For running the agent
    """

    def __init__(self):
        super().__init__()
        self.launch_function = run_agent.main
        self.parser = run_agent.ap
        self.use_mpi = True

    @Experiment.execute
    def display(self, **override):
        command = self.shared_cmd.copy()
        command["policy_file"] = os.path.join(self.result_dir, "RL/rl/policy_latest.pkl")
        return command


class Plot(Experiment):
    """For result plotting
    """

    def __init__(self):
        super().__init__()
        self.launch_function = plot.main
        self.parser = plot.ap

    @Experiment.execute
    def plot(self, **override):
        command = self.shared_cmd.copy()
        command["dir"] = [os.path.join(self.result_dir, "RLNoDemo")]
        command["xy"] = ["epoch:test/success_rate"]
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
