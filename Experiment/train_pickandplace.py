import os
import sys

from train import Demo, Train, Display, Plot
from yw.util.cmd_util import ArgParser

# A arg parser that takes options for which sub exp to run
ap = ArgParser()
ap.parser.add_argument(
    "--target",
    help="which script to run",
    type=str,
    choices=["rlsparse", "rldense", "bc", "shaping", "plot", "display"],
    action="append",
    default=None,
    dest="targets",
)
ap.parse(sys.argv)
target = ap.get_dict()["targets"]
print("Using target: ", target)

if __name__ == "__main__":


    demo_exp = Demo()
    train_exp = Train()
    display_exp = Display()
    plot_exp = Plot()

    # Common result directory
    exp_dir = os.getenv("EXPERIMENT")
    result_dir = os.path.join(exp_dir, "Result/OpenAIPickPlaceSingleGoal")
    train_exp.result_dir = result_dir
    demo_exp.result_dir = result_dir
    display_exp.result_dir = result_dir
    plot_exp.result_dir = result_dir

    # Specify the environment and reward type.
    # If you want to use dense reward, add "Dense" to the reward name and make sure the env manager recognizes that.
    # Please follow this convention so that you can plot the same env with different reward types in the same graph.
    environment = "FetchPickAndPlace-v1"
    demo_data_size = 128
    seed = 0  # change seed value inside the for loop

    train_exp.set_shared_cmd(
        env=environment,
        r_shift=1.0,  # use dense reward for all now
        n_cycles=10,
        rl_num_sample=1,
        rl_batch_size=256,
        train_rl_epochs=10000,
    )

    demo_exp.set_shared_cmd(num_demo=demo_data_size)

    for i in range(4):

        # Change seed value in each iteration
        seed += i * 100

        # Change the result directory so that different seed goes to different directory
        train_exp.result_dir = os.path.join(result_dir, "Exp" + str(i))
        demo_exp.result_dir = os.path.join(result_dir, "Exp" + str(i))

        if "rldense" in target:
            # Train the RL without demonstration using dense reward.
            train_exp.rl_only_dense(
                env="FetchPickAndPlaceDense-v1", r_shift=0.0, seed=seed + 0  # use env with dense reward
            )

        if "rlsparse" in target:
            # Train the RL without demonstration using sparse reward.
            train_exp.rl_only(seed=seed + 10)
            # train_exp.rl_her_only(
            #     seed=seed+10,
            # )

        # Generate demonstration data
        # demo_exp.generate_demo(
        #     policy_file=os.path.join(demo_exp.result_dir, "RLDense/rl/policy_latest.pkl"),
        #     seed=seed + 20,
        # )

        if "bc" in target:
            # Train the RL with demonstration through BC
            train_exp.rl_with_bc(
                seed=seed + 30,
                rl_batch_size_demo=128,
                num_demo=demo_data_size,
                demo_file=os.path.join(train_exp.result_dir, "../DemoData", environment + ".npz"),
            )

        if "shaping" in target:
            # Train the RL with demonstration through shaping
            train_exp.rl_with_shaping(
                seed=seed + 40,
                num_demo=demo_data_size,
                demo_file=os.path.join(train_exp.result_dir, "../DemoData", environment + ".npz"),
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
