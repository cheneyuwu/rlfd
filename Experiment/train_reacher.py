import os
import sys

from train import Experiment, Demo, Train, Display, Plot
from train import exp_parser

from yw.flow.query import generate_query, visualize_query


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
        command["mode"] = "plot"
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
            os.path.join(self.result_dir, "*")
        ]
        return command


class QueryPolicy(Experiment):
    """For result plotting
    """

    def __init__(self):
        super().__init__()
        self.launch_function = generate_query.main
        self.parser = generate_query.ap

    @Experiment.execute
    def query(self, **override):
        command = self.shared_cmd.copy()
        command["save"] = 1
        command["directory"] = [
            # os.path.join(self.result_dir, "RLDemoMAF"),
            os.path.join(self.result_dir, "*")  # need this
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
query_policy_exp = QueryPolicy()

# Common result directory
result_dir = os.path.join(os.getenv("EXPERIMENT"), "TempResult/Temp")
train_exp.result_dir = result_dir
demo_exp.result_dir = result_dir
display_exp.result_dir = result_dir
plot_exp.result_dir = result_dir

query_exp.result_dir = result_dir
query_policy_exp.result_dir = result_dir

# Train the RL without demonstration using dense reward.
if "rldense" in target:
    train_exp.launch(root_dir=os.path.join(train_exp.result_dir, "RLDense"))

# Generate demonstration data
if "demo" in target:
    demo_exp.generate_demo(
        num_demo=100, policy_file=os.path.join(demo_exp.result_dir, "RLDense/rl/policy_latest.pkl"), seed=seed + 20
    )

# Train the RL without demonstration using sparse reward.
if "rlsparse" in target:
    train_exp.launch(root_dir=os.path.join(train_exp.result_dir, "RL"))

# Train the RL with demonstration through BC
if "bc" in target:
    train_exp.launch(root_dir=os.path.join(train_exp.result_dir, "RLDemoBC"))


# Train the RL with demonstration through shaping
if "norm" in target:
    train_exp.launch(root_dir=os.path.join(train_exp.result_dir, "RLDemoNorm"))


# Train the RL with demonstration through shaping
if "maf" in target:
    train_exp.launch(root_dir=os.path.join(train_exp.result_dir, "RLDemoMAF"))


# Train the RL with demonstration through shaping
if "manual" in target:
    train_exp.launch(logdir=os.path.join(train_exp.result_dir, "RLDemoManual"))


# Plot the training result
if "plot" in target:
    plot_exp.plot(
        dir=plot_exp.result_dir,
        xy=[
            "epoch:test/success_rate",
            "epoch:test/total_shaping_reward",
            "epoch:test/total_reward",
            "epoch:test/mean_Q",
            "epoch:test/mean_Q_plus_P",
        ],
    )

# Display a policy result (calls run_agent).
if "display" in target:
    display_exp.display(policy_file=os.path.join(display_exp.result_dir, "RLDense/rl/policy_latest.pkl"), num_itr=10)

if "query_policy" in target:
    query_policy_exp.query()

if "query" in target:
    query_exp.query()
