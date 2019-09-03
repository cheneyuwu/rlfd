from yw.flow.demo_util.generate_demo_fetch_policy import FetchDemoGenerator

class PickAndPlaceDemoGenerator(FetchDemoGenerator):
    def __init__(self, env, policy, system_noise_level, variance_level, sub_opt_level, render):
        super().__init__(self, env, policy, system_noise_level, variance_level, sub_opt_level, render)

    def generate_pick_place(self):

        goal_dim = 0
        obj_pos_dim = 10
        obj_rel_pos_dim = 13

        self._reset()
        # move to the above of the object
        self._move_to_object(obj_pos_dim, obj_rel_pos_dim, offset=0.05, gripper_open=True)
        # grab the object
        self._move_to_object(obj_pos_dim, obj_rel_pos_dim, offset=0.0, gripper_open=False)
        # move to the goal
        sub1 = 0.5 + self.sub_opt_level
        sub2 = 0.5 - self.sub_opt_level
        assert self.sub_opt_level <= 0.5
        assert self.variance_level <= 0.2
        if self.num_itr % 2 == 0:
            weight = np.array((sub1, sub2, 0.0)) + np.random.normal(scale=self.variance_level, size=3)
        elif self.num_itr % 2 == 1:
            weight = np.array((sub2, sub1, 0.0)) + np.random.normal(scale=self.variance_level, size=3)
        else:
            assert False
        self._move_to_interm_goal(obj_pos_dim, goal_dim, weight)
        self._move_to_goal(obj_pos_dim, goal_dim)
        # open the gripper
        self._move_to_object(obj_pos_dim, obj_rel_pos_dim, offset=0.03, gripper_open=True)
        # stay until the end
        self._stay()

        self.num_itr += 1
        assert self.episode_info[-1]["is_success"]
        return self.episode_obs, self.episode_act, self.episode_rwd, self.episode_info
        