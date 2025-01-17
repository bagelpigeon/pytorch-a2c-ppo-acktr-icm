import os

import gym
from gym.spaces.box import Box

from baselines import bench
from baselines.common.atari_wrappers import make_atari, wrap_deepmind

import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

try:
    import dm_control2gym
except ImportError:
    pass

try:
    import pybullet_envs
    import roboschool
except ImportError:
    pass


def make_env(env_id, seed, rank, log_dir):
    def _thunk():
        if "mario" in env_id:
            env = gym_super_mario_bros.make('SuperMarioBros-v0')
            env = JoypadSpace(env, SIMPLE_MOVEMENT)
        elif env_id.startswith("dm"):
            _, domain, task = env_id.split('.')
            env = dm_control2gym.make(domain_name=domain, task_name=task)
        else:
            env = gym.make(env_id)
        is_atari = hasattr(gym.envs, 'atari') and isinstance(env.unwrapped, gym.envs.atari.atari_env.AtariEnv)
        if is_atari:
            env = make_atari(env_id)
        env.seed(seed + rank)
        if log_dir is not None:
            env = bench.Monitor(env, os.path.join(log_dir, str(rank)))
        if is_atari:
            env = wrap_deepmind(env, clip_rewards=False)
        # If the input has shape (W,H,3), wrap for PyTorch convolutions
        obs_shape = env.observation_space.shape
        if "mario" in env_id:
            env = wrap_deepmind(env, clip_rewards=False, episode_life=False)
            env = WrapPyTorchMario(env)
        elif len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
            env = WrapPyTorch(env)

        return env

    return _thunk

class WrapPyTorchMario(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(WrapPyTorchMario, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0,0,0],
            self.observation_space.high[0,0,0],
            [obs_shape[2], obs_shape[1], obs_shape[0]],
            self.observation_space.dtype,
        )

    def observation(self, observation):
        return observation.transpose(2, 1, 0)

class WrapPyTorch(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(WrapPyTorch, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0,0,0],
            self.observation_space.high[0,0,0],
            [obs_shape[2], obs_shape[1], obs_shape[0]],
            self.observation_space.dtype,
        )

    def observation(self, observation):
        return observation.transpose(2, 0, 1)
