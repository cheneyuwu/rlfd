import os
from collections import OrderedDict

from yw.util.cmd_util import Command
from train import Experiment, Demo, Train, Display, Plot


class Uncertainty(Experiment):
    """For result plotting
    """

    def __init__(self):
        super().__init__()
        self.run = OrderedDict([("python", os.path.join(self.flow_dir, "query/query_uncertainty_2.py"))])

    @Command.execute
    def check(self, **override):
        command = self.run.copy()
        command["--load_dir"] = self.result_dir + "RLNoDemo/uncertainty/"
        command["--save"] = 1
        return command

if __name__ == "__main__":

    demo_exp = Demo()
    train_exp = Train()
    display_exp = Display()
    plot_exp = Plot()
    uncertainty_exp = Uncertainty()


    environment = "Reach1DFirstOrderSparse"
    train_exp.env = environment
    train_exp.num_cpu = 1
    train_exp.update()

    demo_data_size = 128
    train_rl_epochs = 32
    seed = 2
    for i in range(1):
        seed += i * 100

        # We can change the result directory without updating
        exp_dir = os.getenv("EXPERIMENT")
        result_dir = os.path.join(exp_dir, "Result/Temp/")
        demo_exp.result_dir = result_dir
        train_exp.result_dir = result_dir

        # Train the RL without demonstration
        # assert not train_exp.rl_her_only(
        #     r_scale=1.0,
        #     r_shift=0.0,
        #     rl_action_l2=0.5,
        #     rl_scope="critic_demo",
        #     n_cycles=10,
        #     seed=seed + 20,
        #     rl_num_sample=1,
        #     rl_batch_size=256,
        #     train_rl_epochs=train_rl_epochs,
        # )
        assert not train_exp.rl_only(
            rl_action_l2=0.5,
            rl_scope="rl_only",
            n_cycles=4,
            n_batches=8,
            seed=seed + 10,
            rl_num_sample=1,
            rl_batch_size=32,
            train_rl_epochs=train_rl_epochs,
        )

        # Generate demonstration data
        # assert not demo_exp.generate_demo(
        #     policy_file=demo_exp.result_dir + "RLNoDemo/rl/policy_latest.pkl",
        #     seed=seed + 30,
        #     num_itr=demo_data_size,
        #     entire_eps=1,
        #     shuffle=0,
        # )

        assert not train_exp.rl_with_shaping(
            rl_action_l2=0.5,
            rl_scope="rl_only",
            n_cycles=4,
            n_batches=8,
            seed=seed + 10,
            rl_num_sample=1,
            rl_batch_size=32,
            train_rl_epochs=train_rl_epochs,
            demo_critic="shaping",
        )

    # Plot the training result
    assert not plot_exp.plot(dir=plot_exp.result_dir, xy=["epoch:test/success_rate", "epoch:test/total_reward", "epoch:test/mean_Q"])

    # Check the uncertainty output of the demonstration output
    # assert not uncertainty_exp.check(load_dir=uncertainty_exp.result_dir + "RL/uncertainty/", save=0)
    # assert not uncertainty_exp.check(load_dir=uncertainty_exp.result_dir + "RLDemoShaping/uncertainty/", save=0)

    # Display a policy result (calls run_agent).
    # assert not display_exp.display(policy_file=display_exp.result_dir + "RLDemoShaping/rl/policy_latest.pkl", num_itr=3)
