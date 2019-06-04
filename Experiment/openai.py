import os
from train import Demo, Train, Display, Plot

demo_exp = Demo()
train_exp = Train()
display_exp = Display()
plot_exp = Plot()

environment = "FetchPickAndPlace-v1"
train_exp.env = environment
train_exp.num_cpu = 4
train_exp.update()

demo_data_size = 1024
train_rl_epochs = 50

seed = 1
for i in range(3):
    seed += i*100

    # We can change the result directory without updating
    exp_dir = os.getenv("EXPERIMENT")
    result_dir = os.path.join(exp_dir, "Result/Temp/Exp"+str(i)+"/")
    demo_exp.result_dir = result_dir
    train_exp.result_dir = result_dir

    # Train the RL without demonstration
    assert not train_exp.rl_her_only(
        rl_scope="rl_only",
        n_cycles=50,
        seed=seed+20,
        rl_num_sample=1,
        rl_batch_size=256,
        train_rl_epochs=train_rl_epochs,
    )
    assert not train_exp.rl_only(
        rl_scope="rl_only",
        n_cycles=50,
        seed=seed+10,
        rl_num_sample=1,
        rl_batch_size=256,
        train_rl_epochs=train_rl_epochs,
    )

    # Generate demonstration data
    assert not demo_exp.generate_demo(seed=seed+30, num_itr=demo_data_size, entire_eps=1, shuffle=0)

    # Train the RL with demonstration
    assert not train_exp.rl_with_demo_critic_rb(
        n_cycles=50,
        seed=seed + 40,
        rl_num_sample=1,
        rl_batch_size=512,
        rl_batch_size_demo=256,
        rl_num_demo=demo_data_size,
        rl_replay_strategy="none",
        demo_file=result_dir + "DemoData/" + environment + ".npz",
        train_rl_epochs=train_rl_epochs,
    )

# Plot the training result
plot_exp.plot(dirs=plot_exp.result_dir)