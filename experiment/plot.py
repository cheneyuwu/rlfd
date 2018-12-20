import os
import matplotlib
matplotlib.use('TkAgg') # Can change to 'Agg' for non-interactive mode
import matplotlib.pyplot as plt
import numpy as np
import json
import seaborn as sns; sns.set()
import argparse


def smooth_reward_curve(x, y):
    halfwidth = int(np.ceil(len(x) / 60))  # Halfwidth of our smoothing convolution
    k = halfwidth
    xsmoo = x
    ysmoo = np.convolve(y, np.ones(2 * k + 1), mode='same') / np.convolve(np.ones_like(y), np.ones(2 * k + 1),
        mode='same')
    return xsmoo, ysmoo


def load_results(file):
    if not os.path.exists(file):
        return None
    with open(file, 'r') as f:
        lines = [line for line in f]
    if len(lines) < 2:
        return None
    keys = [name.strip() for name in lines[0].split(',')]
    data = np.genfromtxt(file, delimiter=',', skip_header=1, filling_values=0.)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    assert data.ndim == 2
    assert data.shape[-1] == len(keys)
    result = {}
    for idx, key in enumerate(keys):
        result[key] = data[:, idx]
    return result


def pad(xs, value=np.nan):
    maxlen = np.max([len(x) for x in xs])

    padded_xs = []
    for x in xs:
        if x.shape[0] >= maxlen:
            padded_xs.append(x)

        padding = np.ones((maxlen - x.shape[0],) + x.shape[1:]) * value
        x_padded = np.concatenate([x, padding], axis=0)
        assert x_padded.shape[1:] == x.shape[1:]
        assert x_padded.shape[0] == maxlen
        padded_xs.append(x_padded)
    return np.array(padded_xs)


# parser = argparse.ArgumentParser()
# parser.add_argument('dir', type=str)
# parser.add_argument('--smooth', type=int, default=1)
# args = parser.parse_args()

# Load all data.
data = {}
paths = ['/home/yuchen/Desktop/FlorianResearch/RLProject/temp/log/']
for curr_path in paths:
    if not os.path.isdir(curr_path):
        continue
    results = load_results(os.path.join(curr_path, 'progress.csv'))
    if not results:
        print('skipping {}'.format(curr_path))
        continue
    print('loading {} ({})'.format(curr_path, len(results['epoch'])))
    with open(os.path.join(curr_path, 'params.json'), 'r') as f:
        params = json.load(f)

    success_rate = np.array(results['test/success_rate'])
    epoch = np.array(results['epoch']) + 1
    env_id = params['env_name']
    replay_strategy = params['replay_strategy']

    if replay_strategy == 'future':
        config = 'her'
    else:
        config = 'ddpg'
    if 'Dense' in env_id:
        config += '-dense'
    else:
        config += '-sparse'
    env_id = env_id.replace('Dense', '')

    # Process and smooth data.
    assert success_rate.shape == epoch.shape
    x = epoch
    y = success_rate
    if args.smooth:
        x, y = smooth_reward_curve(epoch, success_rate)
    assert x.shape == y.shape

    if env_id not in data:
        data[env_id] = {}
    if config not in data[env_id]:
        data[env_id][config] = []
    data[env_id][config].append((x, y))

# Plot data.
for env_id in sorted(data.keys()):
    print('exporting {}'.format(env_id))
    plt.clf()

    for config in sorted(data[env_id].keys()):
        xs, ys = zip(*data[env_id][config])
        xs, ys = pad(xs), pad(ys)
        assert xs.shape == ys.shape

        plt.plot(xs[0], np.nanmedian(ys, axis=0), label=config)
        plt.fill_between(xs[0], np.nanpercentile(ys, 25, axis=0), np.nanpercentile(ys, 75, axis=0), alpha=0.25)
    plt.title(env_id)
    plt.xlabel('Epoch')
    plt.ylabel('Median Success Rate')
    plt.legend()
    plt.savefig(os.path.join(args.dir, 'fig_{}.png'.format(env_id)))


# import numpy as np
# import matplotlib
# matplotlib.use('TkAgg') # Can change to 'Agg' for non-interactive mode

# import matplotlib.pyplot as plt
# plt.rcParams['svg.fonttype'] = 'none'

# from yw import plot_util

# X_TIMESTEPS = 'timesteps'
# X_EPISODES = 'episodes'
# X_WALLTIME = 'walltime_hrs'
# Y_REWARD = 'reward'
# Y_TIMESTEPS = 'timesteps'
# POSSIBLE_X_AXES = [X_TIMESTEPS, X_EPISODES, X_WALLTIME]
# EPISODES_WINDOW = 100
# COLORS = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'pink',
#         'brown', 'orange', 'teal', 'coral', 'lightblue', 'lime', 'lavender', 'turquoise',
#         'darkgreen', 'tan', 'salmon', 'gold', 'darkred', 'darkblue']

# def rolling_window(a, window):
#     shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
#     strides = a.strides + (a.strides[-1],)
#     return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

# def window_func(x, y, window, func):
#     yw = rolling_window(y, window)
#     yw_func = func(yw, axis=-1)
#     return x[window-1:], yw_func

# def ts2xy(ts, xaxis, yaxis):
#     if xaxis == X_TIMESTEPS:
#         x = np.cumsum(ts.l.values)
#     elif xaxis == X_EPISODES:
#         x = np.arange(len(ts))
#     elif xaxis == X_WALLTIME:
#         x = ts.t.values / 3600.
#     else:
#         raise NotImplementedError
#     if yaxis == Y_REWARD:
#         y = ts.r.values
#     elif yaxis == Y_TIMESTEPS:
#         y = ts.l.values
#     else:
#         raise NotImplementedError
#     return x, y

# def plot_curves(xy_list, xaxis, yaxis, title):
#     fig = plt.figure(figsize=(8,2))
#     maxx = max(xy[0][-1] for xy in xy_list)
#     minx = 0
#     for (i, (x, y)) in enumerate(xy_list):
#         color = COLORS[i % len(COLORS)]
#         plt.scatter(x, y, s=2)
#         x, y_mean = window_func(x, y, EPISODES_WINDOW, np.mean) #So returns average of last EPISODE_WINDOW episodes
#         plt.plot(x, y_mean, color=color)
#     plt.xlim(minx, maxx)
#     plt.title(title)
#     plt.xlabel(xaxis)
#     plt.ylabel(yaxis)
#     plt.tight_layout()
#     fig.canvas.mpl_connect('resize_event', lambda event: plt.tight_layout())
#     plt.grid(True)


# def split_by_task(taskpath):
#     return taskpath['dirname'].split('/')[-1].split('-')[0]

# def plot_results(dirs, num_timesteps=10e6, xaxis=X_TIMESTEPS, yaxis=Y_REWARD, title='', split_fn=split_by_task):
#     results = plot_util.load_results(dirs)
#     plot_util.plot_results(results, xy_fn=lambda r: ts2xy(r['monitor'], xaxis, yaxis), split_fn=split_fn, average_group=True, resample=int(1e6))

# # Example usage in jupyter-notebook
# # from baselines.results_plotter import plot_results
# # %matplotlib inline
# # plot_results("./log")
# # Here ./log is a directory containing the monitor.csv files

# def main():
#     import argparse
#     import os
#     parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#     parser.add_argument('--dirs', help='List of log directories', nargs = '*', default=['./log'])
#     parser.add_argument('--num_timesteps', type=int, default=int(10e6))
#     parser.add_argument('--xaxis', help = 'Varible on X-axis', default = X_TIMESTEPS)
#     parser.add_argument('--yaxis', help = 'Varible on Y-axis', default = Y_REWARD)
#     parser.add_argument('--task_name', help = 'Title of plot', default = 'Breakout')
#     args = parser.parse_args()
#     args.dirs = [os.path.abspath(dir) for dir in args.dirs]
#     plot_results(args.dirs, args.num_timesteps, args.xaxis, args.yaxis, args.task_name)
#     plt.show()

# if __name__ == '__main__':
#     main()
