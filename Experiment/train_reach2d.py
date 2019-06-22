import os
from train import Demo, Train, Display, Plot

if __name__ == "__main__":

    demo_exp = Demo()
    train_exp = Train()
    display_exp = Display()
    plot_exp = Plot()


    environment = "Reach2DSparse"
    train_exp.env = environment
    train_exp.num_cpu = 1
    train_exp.update()

    demo_data_size = 64
    train_rl_epochs = 64
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
        #     rl_action_l2=0.5,
        #     rl_scope="critic_demo",
        #     n_cycles=10,
        #     seed=seed + 20,
        #     rl_num_sample=1,
        #     rl_batch_size=256,
        #     train_rl_epochs=train_rl_epochs,
        # )
        # assert not train_exp.rl_only(
        #     rl_action_l2=0.5,
        #     rl_scope="rl_only",
        #     n_cycles=10,
        #     seed=seed + 10,
        #     rl_num_sample=1,
        #     rl_batch_size=256,
        #     train_rl_epochs=train_rl_epochs,
        # )

        # Generate demonstration data
        # assert not demo_exp.generate_demo(
        #     policy_file=demo_exp.result_dir + "RL/rl/policy_latest.pkl",
        #     seed=seed + 30,
        #     num_itr=demo_data_size,
        #     entire_eps=1,
        #     shuffle=0,
        # )

        # Train the RL with demonstration
        assert not train_exp.rl_with_shaping(
            rl_action_l2=0.5,
            rl_scope="rl_with_shaping",
            n_cycles=10,
            seed=seed + 10,
            rl_num_sample=1,
            rl_batch_size=256,
            train_rl_epochs=train_rl_epochs,
            demo_critic="shaping",
            num_demo=demo_data_size,
            demo_file=train_exp.result_dir+"/DemoData/Reach2DSparse.npz"
        )

    # Plot the training result
    assert not plot_exp.plot(dir=plot_exp.result_dir, xy=["epoch:test/success_rate", "epoch:test/total_reward", "epoch:test/mean_Q"])

    # Display a policy result (calls run_agent).
    # assert not display_exp.display(policy_file=display_exp.result_dir + "Exp0/RLNoDemo/rl/policy_latest.pkl", num_itr=3)
