import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os

os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MUJOCO_GL"] = "disable"

from pathlib import Path

import hydra
import numpy as np
import torch

import dmc
# Having this line before import dmc won't work
# I think import numpy resets warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import utils
from stable_baselines3 import PPO
from dmc_to_gym import DMCGymWrapper

from stable_baselines3.common.env_checker import check_env

# This links it to eval.yaml
@hydra.main(config_path=".", config_name="finetune_ppo")
def main(cfg):

    work_dir = Path.cwd()
    print(f"workspace: {work_dir}")

    utils.set_seed_everywhere(cfg.seed)
    device = torch.device(cfg.device)

    # Create dmc env and convert to gym (Not gymnasium)
    dmc_env = dmc.make(cfg.task, seed=cfg.seed)
    env = DMCGymWrapper(dmc_env)

    check_env(env)
    print("Passed the check")

    model = PPO("MlpPolicy", env, verbose=1)

    model.learn(total_timesteps=10)
    print("Ended successfully")

if __name__ == "__main__":
    main()