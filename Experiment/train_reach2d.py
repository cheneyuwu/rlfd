import os
import sys

from train import Experiment, Demo, Train, Display, Plot
from train import exp_parser

from yw.flow import visualize_query


class Query(Experiment):
    """For result plotting
    """

    def __init__(self):
        super().__init__()
        self.launch_function = visualize_query.main
        self.parser = visualize_query.ap

    @Experiment.execute
    def query(self, **override):
        command = self.shared_cmd.copy()
        command["save"] = 0
        command["load_dir"] = [
            # os.path.join(self.result_dir, "RLDense"),
            # os.path.join(self.result_dir, "RLDemoNegDistance"),
            # os.path.join(self.result_dir, "RLDemoL2DistanceToAllDemo"),
            # os.path.join(self.result_dir, "RLDemoL2DistanceToClosestDemo"),
            # os.path.join(self.result_dir, "RLDemoMAFPlus1Trick"),
            # os.path.join(self.result_dir, "RLDemoMAFClip1000"),
            # os.path.join(self.result_dir, "RL"),
            # temp
            os.path.join(self.result_dir, "*"),
        ]
        return command


# take cmd line args passed with --target
exp_parser.parse(sys.argv)
target = exp_parser.get_dict()["targets"]

demo_exp = Demo()
train_exp = Train()
display_exp = Display()
plot_exp = Plot()
query_exp = Query()

# Common result directory
result_dir = os.path.join(os.getenv("EXPERIMENT"), "TempResult/Temp")
train_exp.result_dir = result_dir
demo_exp.result_dir = result_dir
display_exp.result_dir = result_dir
plot_exp.result_dir = result_dir
query_exp.result_dir = result_dir

# Specify the environment and reward type.
# If you want to use dense reward, add "Dense" to the reward name and make sure the env manager recognizes that.
# Please follow this convention so that you can plot the same env with different reward types in the same graph.
environment = "Reach2DFDense"
demo_data_size = 32
seed = 0

train_exp.set_shared_cmd(
    env=environment,
    n_cycles=10,
    train_rl_epochs=60,
    rl_action_l2=0.5,
)
demo_exp.set_shared_cmd(num_demo=demo_data_size)

for i in range(1):

    # Change seed value in each iteration
    seed += i * 100

    # Change the result directory so that different seed goes to different directory
    # train_exp.result_dir = os.path.join(result_dir, "Exp" + str(i))
    # demo_exp.result_dir = os.path.join(result_dir, "Exp" + str(i))

    # Train the RL without demonstration using dense reward.
    if "rldense" in target:
        train_exp.rl_only_dense(
            env="Reach2DFDense", 
            r_shift=0.0,
            seed=seed + 0,
        )

    # Generate demonstration data
    if "demo" in target:
        demo_exp.generate_demo(
            policy_file=os.path.join(demo_exp.result_dir, "RLDense/rl/policy_latest.pkl"), seed=seed + 20
        )

    # Train the RL without demonstration using sparse reward.
    if "rlsparse" in target:
        train_exp.rl_only(seed=seed + 10)

    # Train the RL with demonstration through BC
    if "bc" in target:
        train_exp.rl_with_bc(
            seed=seed + 30,
            rl_batch_size_demo=128,
            num_demo=demo_data_size,
            demo_file=os.path.join(train_exp.result_dir, "DemoData", environment.replace("Dense", "") + ".npz"),
        )

    # Train the RL with demonstration through shaping
    if "norm" in target:
        train_exp.rl_with_shaping(
            logdir=os.path.join(train_exp.result_dir, "RLDemoNorm"),
            save_path=os.path.join(train_exp.result_dir, "RLDemoNorm"),
            seed=seed + 40,
            num_demo=demo_data_size,
            demo_file=os.path.join(train_exp.result_dir, "DemoData", environment.replace("Dense", "") + ".npz"),
            demo_critic="norm",
            # shaping_policy=os.path.join(train_exp.result_dir, "RLDemoShaping", "shaping/shaping_latest.ckpt"),
        )

    # Train the RL with demonstration through shaping
    if "maf" in target:
        train_exp.rl_with_shaping(
            logdir=os.path.join(train_exp.result_dir, "RLDemoMAF"),
            save_path=os.path.join(train_exp.result_dir, "RLDemoMAF"),
            seed=seed + 40,
            num_demo=demo_data_size,
            demo_file=os.path.join(train_exp.result_dir, "DemoData", environment.replace("Dense", "") + ".npz"),
            demo_critic="maf",
            # shaping_policy=os.path.join(train_exp.result_dir, "RLDemoShaping", "shaping/shaping_latest.ckpt"),
        )

    # Train the RL with demonstration through shaping
    if "manual" in target:
        train_exp.rl_with_shaping(
            logdir=os.path.join(train_exp.result_dir, "RLDemoManual"),
            save_path=os.path.join(train_exp.result_dir, "RLDemoManual"),
            seed=seed + 40,
            num_demo=demo_data_size,
            demo_file=os.path.join(train_exp.result_dir, "DemoData", environment.replace("Dense", "") + ".npz"),
            demo_critic="manual",
            # shaping_policy=os.path.join(train_exp.result_dir, "RLDemoShaping", "shaping/shaping_latest.ckpt"),
        )


# Plot the training result
if "plot" in target:
    plot_exp.plot(
        dir=plot_exp.result_dir,
        xy=[
            "epoch:test/success_rate",
            "epoch:test/total_shaping_reward",
            "epoch:test/total_reward",
            "epoch:test/mean_Q",
        ],
    )

# Display a policy result (calls run_agent).
if "display" in target:
    display_exp.display(
        policy_file=os.path.join(display_exp.result_dir, "RLDense/rl/policy_latest.pkl"), 
        num_itr=10
    )

if "query" in target:
    query_exp.query()
