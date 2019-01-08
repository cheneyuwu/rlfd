from subprocess import call
from collections import OrderedDict

root_dir = "/home/yuchen/Desktop/FlorianResearch/RLProject/"
result_dir = root_dir + "Result/"
exp_dir = root_dir + "experiment/"
num_cpu = 4
n_epochs = 6
seed = [0,1,2,3,4]
config = ["td3", "ddpg"]
replay_strategy = ["none", "future"]
env = ["FetchReachDense-v1"]
log_dir = [result_dir + x + "-" + y + "-" + z + "/" for x in config for y in replay_strategy for z in env]
runs = []
for w in seed:
    for x in config:
        for y in replay_strategy:
            for z in env:
                run = OrderedDict()
                run["python"] = exp_dir+"train_"+x+".py"
                run["--logdir"] = result_dir + x + "-" + y + "-" + z + "-" + str(w) + "/"
                run["--num_cpu"] = str(num_cpu)
                run["--replay_strategy"] = y
                run["--env"] = z
                run["--save_path"] = run["--logdir"]+"policy/"
                run["--n_epochs"] = str(n_epochs)
                run["--seed"] = str(w)
                runs.append(run)

for run in runs:
    cmd = []
    for key in run:
        cmd.append(key)
        cmd.append(run[key])
    print(cmd)
    call(cmd)