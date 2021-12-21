import gym

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

policy_1 = {
    0: RIGHT,
    1: RIGHT, 
    2: RIGHT,
    3: DOWN, 
    4: RIGHT,
    5: RIGHT,
    6: RIGHT,
    7: DOWN,
    9: RIGHT,
    10: RIGHT,
}

policy_2 = {
    0: DOWN,
    1: DOWN, 
    2: DOWN,
    3: DOWN, 
    4: RIGHT,
    5: DOWN,
    6: DOWN,
    7: DOWN,
    9: RIGHT,
    10: RIGHT,
}

policy_3 = {
    0: RIGHT,
    1: RIGHT, 
    2: RIGHT,
    3: DOWN, 
    4: RIGHT,
    5: DOWN,
    6: LEFT,
    7: LEFT,
    9: RIGHT,
    10: RIGHT,
}

policy_4 = {
    0: DOWN,
    1: LEFT, 
    2: LEFT,
    3: LEFT, 
    4: DOWN,
    5: LEFT,
    6: LEFT,
    7: LEFT,
    9: LEFT,
    10: LEFT,
}

class RewardWrapper(gym.RewardWrapper):
    def __init__(self, env, step_penalty=0.01):
        self.step_penalty = step_penalty
        super().__init__(env)
    
    def reward(self, rew):
        # modify rew
        if rew == 0:
            rew = -self.step_penalty
        return rew
    
class ResetWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def reset(self, start_pos=0):
        super().reset()
        self.env.env.env.s = start_pos
        return start_pos

def get_frozenlake_env(is_slippery, step_penalty=0.01, custom_map = ['SFFF', 'FFFF', 'HFFG']):
    env = gym.make("FrozenLake-v1", desc=custom_map, is_slippery=is_slippery)
    env = RewardWrapper(env, step_penalty=step_penalty)
    env = ResetWrapper(env)
    return env

