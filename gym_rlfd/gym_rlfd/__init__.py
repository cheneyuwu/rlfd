from gym.envs.registration import register

def _merge(a, b):
    a.update(b)
    return a

for reward_type in ['sparse', 'dense']:
    suffix = 'Dense' if reward_type == 'dense' else ''
    kwargs = {
        'reward_type': reward_type,
    }

    register(
        id='YWFetchPegInHole{}-v0'.format(suffix),
        entry_point='gym_rlfd.robotics:FetchPegInHoleEnv',
        kwargs=kwargs,
        max_episode_steps=40,
    )

    register(
        id='YWFetchPegInHole2D{}-v0'.format(suffix),
        entry_point='gym_rlfd.robotics:FetchPegInHoleEnv',
        kwargs=_merge({'no_y': True}, kwargs),
        max_episode_steps=40,
    )

    register(
        id='YWFetchPegInHoleRandInit{}-v0'.format(suffix),
        entry_point='gym_rlfd.robotics:FetchPegInHoleEnv',
        kwargs=_merge({'rand_init': True}, kwargs),
        max_episode_steps=40,
    )

    register(
        id='YWFetchPickAndPlaceRandInit{}-v0'.format(suffix),
        entry_point='gym_rlfd.robotics:FetchPickAndPlaceEnv',
        kwargs=_merge({'rand_init': True}, kwargs),
        max_episode_steps=40,
    )