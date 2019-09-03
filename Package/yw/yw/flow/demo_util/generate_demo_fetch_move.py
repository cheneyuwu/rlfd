from yw.flow.demo_util.generate_demo_fetch_policy import FetchDemoGenerator

class MoveDemoGenerator(FetchDemoGenerator):
    def __init__(self, env, policy, system_noise_level, variance_level, sub_opt_level, render):
        super().__init__(self, env, policy, system_noise_level, variance_level, sub_opt_level, render)

    def generate_move(self):
        self._reset()
        for i in range(self.num_object):
            goal_dim = 3 * i
            obj_pos_dim = 10 + 15 * i
            obj_rel_pos_dim = 10 + 15 * i + 3
            # move to the above of the object
            self._move_to_object(obj_pos_dim, obj_rel_pos_dim, offset=0.05, gripper_open=True)
            # grab the object
            self._move_to_object(obj_pos_dim, obj_rel_pos_dim, offset=0.0, gripper_open=False)
            # move to the goal
            self._move_to_goal(obj_pos_dim, goal_dim)
            # open the gripper
            self._move_to_object(obj_pos_dim, obj_rel_pos_dim, offset=0.05, gripper_open=True)
            # # move back to initial state
            # self._move_back()
        # stay until the end
        self._stay()

        self.num_itr += 1
        assert self.episode_info[-1]["is_success"]
        return self.episode_obs, self.episode_act, self.episode_rwd, self.episode_info

    def generate_move_auto_place(self):
        self._reset()
        for i in range(self.num_object):
            goal_dim = 3 * i
            obj_pos_dim = 10 + 15 * i
            obj_rel_pos_dim = 10 + 15 * i + 3
            # move to the above of the object
            self._move_to_object(obj_pos_dim, obj_rel_pos_dim, offset=0.05, gripper_open=True)
            # grab the object
            self._move_to_object(obj_pos_dim, obj_rel_pos_dim, offset=0.0, gripper_open=False)
            # move to the goal
            self._move_to_goal(obj_pos_dim, goal_dim)
            # open the gripper
            self._move_to_object(obj_pos_dim, obj_rel_pos_dim, offset=0.05, gripper_open=True)
            # move back to initial state
            self._move_back()
        # stay until the end
        self._stay()

        self.num_itr += 1
        assert self.episode_info[-1]["is_success"]
        return self.episode_obs, self.episode_act, self.episode_rwd, self.episode_info



