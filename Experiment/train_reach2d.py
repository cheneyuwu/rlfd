import os
import sys

from train import Demo, Train, Display, Plot
from train import exp_parser

# take cmd line args passed with --target
exp_parser.parse(sys.argv)
target = exp_parser.get_dict()["targets"]
print("Using target: ", target)

demo_exp = Demo()
train_exp = Train()
display_exp = Display()
plot_exp = Plot()

# Common result directory
result_dir = os.path.join(os.getenv("EXPERIMENT"), "TempResult/Reacher2D")
train_exp.result_dir = result_dir
demo_exp.result_dir = result_dir
display_exp.result_dir = result_dir
plot_exp.result_dir = result_dir

# Specify the environment and reward type.
# If you want to use dense reward, add "Dense" to the reward name and make sure the env manager recognizes that.
# Please follow this convention so that you can plot the same env with different reward types in the same graph.
environment = "Reach2DSparse"
demo_data_size = 64
seed = 2

train_exp.set_shared_cmd(
    env=environment,
    n_cycles=10,
    rl_num_sample=1,
    rl_batch_size=256,
    train_rl_epochs=500,
)

demo_exp.set_shared_cmd(
    num_demo=demo_data_size,
)

for i in range(1):

    # Change seed value in each iteration
    seed += i * 100

    # We can change the result directory without updating
    exp_dir = os.getenv("EXPERIMENT")
    result_dir = os.path.join(exp_dir, "Result/Temp/")
    demo_exp.result_dir = result_dir
    train_exp.result_dir = result_dir

    # Train the RL without demonstration using dense reward.
    if "rldense" in target:
        train_exp.rl_only_dense(
            env="Reach2DDense", # use env with dense reward
            r_shift=0.0,
            seed=seed + 0,
        )

    # Generate demonstration data
    if "demo" in target:
        demo_exp.generate_demo(
            policy_file=os.path.join(demo_exp.result_dir, "RLDense/rl/policy_latest.pkl"),
            seed=seed + 20,
        )

    # Train the RL without demonstration using sparse reward.
    if "rlsparse" in target:
        train_exp.rl_only(
            seed=seed+10,
        )

    # Train the RL with demonstration through BC
    if "bc" in target:
        train_exp.rl_with_bc(
            seed=seed + 30,
            rl_batch_size_demo=128,
            num_demo=demo_data_size,
            demo_file=os.path.join(train_exp.result_dir, "DemoData", environment+".npz"),
        )

    # Train the RL with demonstration through shaping
    if "shaping" in target:
        train_exp.rl_with_shaping(
            seed=seed + 40,
            num_demo=demo_data_size,
            demo_file=os.path.join(train_exp.result_dir, "DemoData", environment+".npz"),
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
        policy_file=os.path.join(display_exp.result_dir, "Exp0/RLDense/rl/policy_latest.pkl"), num_itr=10
    )