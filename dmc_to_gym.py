import gym
import numpy as np
from dm_env import StepType, specs

class DMCGymWrapper(gym.Env):
    """
    Wraps a dm_env.Environment (your dmc.make(...) env) into a Gym Env
    so that stable-baselines3 can use it.
    Assumes that reset/step return an ExtendedTimeStep with:
        .observation (np.ndarray),
        .reward (float),
        .discount (float),
        .step_type (StepType),
        .physics (np.ndarray)
    """

    metadata = {"render.modes": []}

    def __init__(self, dmc_env):
        super().__init__()
        self._env = dmc_env

        # Array(shape=(24,), dtype='float32'
        obs_spec = dmc_env.observation_spec()
        # BoundedArray(shape=(6,), dtype='float32', minimum=-1.0, maximum=1.0)
        act_spec = dmc_env.action_spec()

        if isinstance(obs_spec, specs.BoundedArray):
            low = np.broadcast_to(obs_spec.minimum, obs_spec.shape)
            high = np.broadcast_to(obs_spec.maximum, obs_spec.shape)
        else:
            low = -np.inf * np.ones(obs_spec.shape, dtype=np.float32)
            high = np.inf * np.ones(obs_spec.shape, dtype=np.float32)

        # print("Low", low)
        # print("High", high)
        # raise ZeroDivisionError
        self.observation_space = gym.spaces.Box(
            low=low.astype(np.float32),
            high=high.astype(np.float32),
            dtype=np.float32,
        )

        # actions are BoundedArray
        assert isinstance(act_spec, specs.BoundedArray), act_spec
        self.action_space = gym.spaces.Box(
            low=act_spec.minimum.astype(np.float32),
            high=act_spec.maximum.astype(np.float32),
            shape=act_spec.shape,
            dtype=np.float32,
        )

    def reset(self):
        ts = self._env.reset()          # ExtendedTimeStep
        obs = np.array(ts.observation, copy=False)
        return obs

    def step(self, action):
        # SB3 will give numpy float32 actions in [-1, 1] already
        ts = self._env.step(action)
        obs = np.array(ts.observation, copy=False)
        reward = float(ts.reward)
        done = bool(ts.step_type == StepType.LAST)
        info = {
            "discount": float(ts.discount),
            "physics": ts.physics,
            "step_type": ts.step_type,
        }
        return obs, reward, done, info
