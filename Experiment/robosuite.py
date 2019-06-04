import os
from train import Demo, Train, Display, Plot

demo_exp = Demo()
train_exp = Train()
display_exp = Display()
plot_exp = Plot()

environment = "SawyerLift"
train_exp.env = environment
train_exp.num_cpu = 6
train_exp.update()

train_rl_epochs = 50
seed = 1

for i in range(3):
    seed += i*100

    # We can change the result directory without updating
    exp_dir = os.getenv("EXPERIMENT")
    result_dir = os.path.join(exp_dir, "Result/Temp/Exp"+str(i)+"/")
    demo_exp.result_dir = result_dir
    train_exp.result_dir = result_dir

    assert not train_exp.rl_only(
        rl_scope="rl_only",
        n_cycles=50,
        seed=seed+10,
        rl_num_sample=1,
        rl_batch_size=256,
        train_rl_epochs=train_rl_epochs,
        n_test_rollouts=5,
    )

# Plot the training result
plot_exp.plot(dirs=plot_exp.result_dir)

# Display a policy result (calls run_agent).
assert not display_exp.display(policy_file=display_exp.result_dir + "Exp0/RLNoDemo/rl/policy_latest.pkl")