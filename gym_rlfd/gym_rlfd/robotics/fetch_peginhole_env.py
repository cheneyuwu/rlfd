import numpy as np

from gym.envs.robotics import rotations, utils
from gym_rlfd.robotics import robot_env


class FetchEnv(robot_env.RobotEnv):
    """Superclass for all Fetch environments.
    """

    def __init__(
        self,
        model_path,
        n_substeps,
        gripper_extra_height,
        distance_threshold,
        initial_qpos,
        reward_type,
        rand_init,
        no_y,
        pixel,
    ):
        """Initializes a new Fetch environment.

        Args:
            model_path (string): path to the environments XML file
            n_substeps (int): number of substeps the simulation runs on every call to step
            gripper_extra_height (float): additional height above the table when positioning the gripper
            target_range (float): range of a uniform distribution for sampling a target
            distance_threshold (float): the threshold after which a goal is considered achieved
            initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
            reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
        """
        self.gripper_extra_height = gripper_extra_height
        self.distance_threshold = distance_threshold
        self.reward_type = reward_type
        self.rand_init = rand_init
        self.no_y = no_y
        self.pixel = pixel

        super(FetchEnv, self).__init__(
            model_path=model_path, n_substeps=n_substeps, n_actions=4, initial_qpos=initial_qpos
        )

    # GoalEnv methods
    # ----------------------------

    def compute_goal_distance(self, goal_a, goal_b):
        assert goal_a.shape == goal_b.shape
        return np.linalg.norm(goal_a - goal_b, axis=-1)

    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        d = self.compute_goal_distance(achieved_goal, goal)
        # if has object, add an extra reward for reaching the object
        if self.reward_type == "sparse":
            return -(d > self.distance_threshold).astype(np.float32)
        return -d

    # RobotEnv methods
    # ----------------------------
    def _step_callback(self):
        pass

    def _set_action(self, action):
        assert action.shape == (4,)
        action = action.copy()  # ensure that we don't change the action outside of this scope
        if self.no_y:
            action[1] = 0.0
        pos_ctrl, _ = action[:3], action[3]

        pos_ctrl *= 0.05  # limit maximum change in position
        rot_ctrl = [1.0, 0.0, 1.0, 0.0]  # fixed rotation of the end effector, expressed as a quaternion
        gripper_ctrl = np.array([-1.0, -1.0])
        assert gripper_ctrl.shape == (2,)
        action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])

        # Apply action to simulation.
        utils.ctrl_set_action(self.sim, action)
        utils.mocap_set_action(self.sim, action)

    def _get_obs(self):

        # positions
        grip_pos = self.sim.data.get_site_xpos("robot0:grip")
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp("robot0:grip") * dt
        robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)

        gripper_state = robot_qpos[-2:]
        gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

        achieved_goal = grip_pos.copy()

        state = np.concatenate([grip_pos, grip_velp])

        # pixel = self.render(mode="rgb_array")

        return {
            "observation": state.copy(),
            "achieved_goal": np.zeros((0,)),
            "desired_goal": np.zeros((0,)),
            # "_pixel": pixel.copy(),
            "_state": state.copy(),
            "_achieved_goal": achieved_goal.copy(),
            "_desired_goal": self.goal.copy(),
        }

    def _viewer_setup(self):
        body_id = self.sim.model.body_name2id("robot0:gripper_link")
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 1.0
        self.viewer.cam.azimuth = 90.0
        self.viewer.cam.elevation = -45.0

    def _render_callback(self):
        self.sim.forward()

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)
        self.sim.forward()
        if self.rand_init:
            gripper_target = np.array(
                (np.random.uniform(-0.05, 0.05), np.random.uniform(-0.2, 0.2), np.random.uniform(-0.02, 0.02))
            ) + self.sim.data.get_site_xpos("robot0:grip")
            self._env_setup_helper(gripper_target, -1.0)
        else:
            gripper_target = np.array((-0.15, 0.0, -0.1)) + self.sim.data.get_site_xpos("robot0:grip")
            self._env_setup_helper(gripper_target, -1.0)
        self.sim.forward()
        return True

    def _sample_goal(self):
        return self.goal

    def _is_success(self, achieved_goal, desired_goal):
        d = self.compute_goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    def _shaping_reward(self, achieved_goal, desired_goal):
        # Compute distance between goal and the achieved goal.
        d = self.compute_goal_distance(achieved_goal, desired_goal)
        return -d

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        utils.reset_mocap_welds(self.sim)
        self.sim.forward()
        # Move end effector into position.
        self.sim.data.set_mocap_quat("robot0:mocap", np.array([1.0, 0.0, 1.0, 0.0]))
        gripper_target = np.array([-0.59, 0.0, -0.3 + self.gripper_extra_height]) + self.sim.data.get_site_xpos(
            "robot0:grip"
        )
        self._env_setup_helper(gripper_target, 1.0)
        gripper_target = np.array([0.0, 0.0, -0.07]) + self.sim.data.get_site_xpos("robot0:grip")
        self._env_setup_helper(gripper_target, 1.0)
        gripper_target = np.array([0.0, 0.0, 0.0]) + self.sim.data.get_site_xpos("robot0:grip")
        self._env_setup_helper(gripper_target, -1.0)
        gripper_target = np.array([0.0, 0.0, -0.05]) + self.sim.data.get_site_xpos("robot0:grip")
        self._env_setup_helper(gripper_target, -1.0)
        # Extract information for sampling goals.
        self.goal = self.sim.data.get_site_xpos("hole").copy() + np.array((0.0, 0.0, 0.0))
        # self.goal = np.array([1.45, 0.75 ,0.5])

    def _env_setup_helper(self, gripper_target, gripper):
        self.sim.data.set_mocap_pos("robot0:mocap", gripper_target)
        for i in range(2):
            idx = self.sim.model.jnt_qposadr[self.sim.model.actuator_trnid[i, 0]]
            self.sim.data.ctrl[i] = self.sim.data.qpos[idx] + gripper
        for _ in range(10):
            self.sim.step()
            # self._get_viewer("human").render()
