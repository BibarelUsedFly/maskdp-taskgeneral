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
from stable_baselines3.common.monitor import Monitor # This wraps gym envs
# This allows to make a custom callback class to collect training data
from stable_baselines3.common.callbacks import BaseCallback 

from stable_baselines3.common.env_checker import check_env

torch.backends.cudnn.benchmark = True


def get_dir(cfg):
    resume_dir = Path(cfg.resume_dir)
    snapshot = resume_dir / f"snapshot_{cfg.resume_step}.pt"
    print("loading from", snapshot)
    return snapshot

def get_domain(task):
    if task.startswith("point_mass_maze"):
        return "point_mass_maze"
    return task.split("_", 1)[0]

# This links it to eval.yaml
@hydra.main(config_path=".", config_name="finetune_ppo")
def main(cfg):

    work_dir = Path.cwd()
    print(f"workspace: {work_dir}")

    utils.set_seed_everywhere(cfg.seed)
    device = torch.device(cfg.device)

    # Create a snapshot directory for the task
    domain = get_domain(cfg.task)
    snapshot_dir = work_dir / Path(cfg.snapshot_dir) / \
                   domain / str(cfg.seed) / cfg.algorithm
    snapshot_dir.mkdir(exist_ok=True, parents=True)

    # Create dmc env and convert to gym (Not gymnasium)
    dmc_env = dmc.make(cfg.task, seed=cfg.seed)
    env = DMCGymWrapper(dmc_env)
    check_env(env) # This verifies env actually complies
    env = Monitor(env) # Wraps for SB3

    # agent = hydra.utils.instantiate(
    #     cfg.agent,
    #     obs_shape=env.observation_spec().shape,
    #     action_shape=env.action_spec().shape)

    if cfg.resume:
        # pretrained = torch.load(get_dir(cfg), map_location=cfg.device)
        # agent.model.load_state_dict(pretrained["model"], strict=True)
        model = PPO.load(str(get_dir(cfg)), env=env, device=device)
        reset_num_timesteps = False
        print(f"Loaded pretrained weights from: {get_dir(cfg)}")

    raise ZeroDivisionError

    model.learn(
        total_timesteps=cfg.num_grad_steps,
        callback=callback,
        reset_num_timesteps=reset_num_timesteps)

    # model = PPO("MlpPolicy", env, verbose=1)
    print("Ended successfully")

if __name__ == "__main__":
    main()