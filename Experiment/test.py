import numpy as np
import robosuite as suite

# create an environment for learning on states
env = suite.make(
    "SawyerLift",
    has_renderer=False,           # no on-screen renderer
    has_offscreen_renderer=False, # no off-screen renderer
    use_object_obs=True,          # use object-centric feature
    use_camera_obs=False,         # no camera observations
)

# reset the environment
env.reset()

for i in range(60):
    action = np.random.randn(env.dof)  # sample random action
    obs, reward, done, info = env.step(action)  # take action in the environment

print(obs["robot-state"].shape)
print(obs["object-state"].shape)
print(done)
print(info)