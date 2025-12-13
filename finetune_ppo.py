import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os

os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MUJOCO_GL"] = "disable"

from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn as nn

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
# This is for the custom policy using MaskDP as an actor
from stable_baselines3.common.policies import ActorCriticPolicy, MlpExtractor
from stable_baselines3.common.env_checker import check_env

# Actor expects observation sequence of size (batch, timesteps, dimension)
from agent.mdp_rl import Actor

torch.backends.cudnn.benchmark = True

class MaskDPActorPPOPolicy(ActorCriticPolicy):
    """
    PPO policy that:
      - uses default feature extractor and value network from SB3,
      - but replaces the actor head with a pre-trained MaskDP Actor.
    """

    def __init__(self, *args,
                 maskdp_actor: Actor, fixed_std: float=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.maskdp_actor = maskdp_actor
        self.fixed_std = fixed_std

    def _build_mlp_extractor(self):
        # 𝜋 - Identity policy network beacuse we're using MaskDP
        # V - Two-layer 64-neuron network, as is standard for actor-critic
        net_arch = [dict(pi=[], vf=[64, 64])]
        self.mlp_extractor = MlpExtractor(
            self.features_dim,
            net_arch=net_arch,
            activation_fn=self.activation_fn)

    def _get_action_dist_from_latent(self, latent_pi, latent_sde=None):

        # (batch, state_dim) 
        # barch is always 1 here though
        state = latent_pi.unsqueeze(1)  # (batch, 1, state_dim)

        # MaskDP Actor returns a distribution
        # (batch, 1, action_dim)
        dist = self.maskdp_actor(state, std=self.fixed_std)
        mean_actions = dist.mean[:, -1, :]  # (batch, action_dim)

        # return SB3's Gaussian distribution
        return self.action_dist.proba_distribution(mean_actions, self.log_std)


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

    agent = hydra.utils.instantiate(
        cfg.agent,
        obs_shape=dmc_env.observation_spec().shape,
        action_shape=dmc_env.action_spec().shape)

    if cfg.resume:
        pretrained = torch.load(get_dir(cfg), map_location=cfg.device)
        agent.model.load_state_dict(pretrained["model"], strict=True)

    # === Build MaskDP Actor for PPO ===
    obs_dim = env.observation_space.shape[0] # 24
    action_dim = env.action_space.shape[0]   # 6

    # Number of tokens is (2 * trajectory length) because each step in
    # the trajectory has one token for state and one for action
    attn_len = cfg.agent.transformer_cfg.traj_length * 2
    # finetune doesn't actually get used, so it can be whatever
    maskdp_actor = Actor(
        obs_dim=obs_dim,
        action_dim=action_dim,
        attention_length=attn_len,
        finetune='whatever',
        config=cfg.agent.transformer_cfg,
    ).to(device)

    # Load weights
    maskdp_actor.mdp.load_state_dict(agent.model.state_dict())

    policy_kwargs = dict(
        maskdp_actor=maskdp_actor,
        fixed_std=cfg.fixed_std if "fixed_std" in cfg else 0.1)

    model = PPO(
        policy=MaskDPActorPPOPolicy,
        env=env,
        device=device,
        policy_kwargs=policy_kwargs,
        verbose=1)

    model.learn(total_timesteps=10)#cfg.num_grad_steps)

    # model.learn(
    #     total_timesteps=cfg.num_grad_steps,
    #     callback=callback,
    #     reset_num_timesteps=reset_num_timesteps)

    print("Ended successfully")

if __name__ == "__main__":
    main()