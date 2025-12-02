import gym
import numpy as np
from dm_env import StepType, specs

class DMCGymWrapper(gym.Env):
    """
    Wraps a dm_env.Environment into a Gym Env
    Assumes that reset/step return an ExtendedTimeStep with:
        .observation (np.ndarray),
        .reward (float),
        .discount (float),
        .step_type (StepType),
        .physics (np.ndarray)
    (This is because the dmc module wraps envs in an ExtendedTimestepWrapper)
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

        self.observation_space = gym.spaces.Box(
            low=low.astype(np.float32),
            high=high.astype(np.float32),
            dtype=np.float32,
        )

        # actions are BoundedArray
        assert isinstance(act_spec, specs.BoundedArray), act_spec
        act_low = np.full(act_spec.shape, act_spec.minimum, dtype=np.float32)
        act_high = np.full(act_spec.shape, act_spec.maximum, dtype=np.float32)
        self.action_space = gym.spaces.Box(
            low=act_low,
            high=act_high,
            shape=act_spec.shape,
            dtype=np.float32,
        )

    def reset(self):
        ts = self._env.reset() # TimeStep
        obs = np.array(ts.observation, copy=False)
        return obs

    def step(self, action):
        # SB3 give numpy float32 actions in [-1, 1]
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
