import os
from train import Demo, Train, Display, Plot

demo_exp = Demo()
train_exp = Train()
display_exp = Display()
plot_exp = Plot()


environment = "Reach2D"
train_exp.env = environment
train_exp.update()

demo_data_size = 512
train_rl_epochs = 100
seed = 2
for i in range(6):
    seed += i * 100

    # We can change the result directory without updating
    exp_dir = os.getenv("EXPERIMENT")
    result_dir = os.path.join(exp_dir, "Result/Temp/Exp" + str(i) + "/")
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
    #     rl_num_sample=4,
    #     rl_batch_size=256,
    #     train_rl_epochs=train_rl_epochs,
    # )
    # assert not train_exp.rl_prioritized_only(
    #     r_scale=1.0,
    #     r_shift=0.0,
    #     rl_action_l2=0.5,
    #     rl_scope="critic_demo",
    #     n_cycles=10,
    #     seed=seed + 10,
    #     rl_num_sample=4,
    #     rl_batch_size=256,
    #     train_rl_epochs=train_rl_epochs,
    # )
    assert not train_exp.rl_only(
        r_scale=1.0,
        r_shift=0.0,
        rl_action_l2=0.5,
        rl_scope="critic_demo",
        n_cycles=10,
        seed=seed + 10,
        rl_num_sample=4,
        rl_batch_size=256,
        train_rl_epochs=train_rl_epochs,
    )

    # Generate demonstration data
    assert not demo_exp.generate_demo(seed=seed + 30, num_itr=demo_data_size, entire_eps=1, shuffle=0)

    # Train the RL with demonstration
    # assert not train_exp.rl_with_demo_critic_rb(
    #     r_scale=1.0,
    #     r_shift=0.0,
    #     rl_action_l2=0.5,
    #     n_cycles=10,
    #     seed=seed + 40,
    #     rl_num_sample=4,
    #     rl_batch_size=512,
    #     rl_batch_size_demo=256,
    #     train_rl_epochs=train_rl_epochs,
    #     demo_file=result_dir + "DemoData/" + environment + ".npz",
    #     rl_num_demo=demo_data_size,
    #     rl_replay_strategy="none",
    # )

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