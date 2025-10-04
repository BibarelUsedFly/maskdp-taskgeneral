#!/usr/bin/env bash
python pretrain.py \
    agent=mdp \
    agent.batch_size=384 \
    agent.transformer_cfg.traj_length=64 \
    agent.transformer_cfg.loss="total" \
    agent.transformer_cfg.n_embd=256 \
    agent.transformer_cfg.n_head=4 \
    agent.transformer_cfg.n_enc_layer=3 \
    agent.transformer_cfg.n_dec_layer=2 \
    agent.transformer_cfg.norm='l2' \
    num_grad_steps=400010 \
    task=cheetah_run \
    snapshot_dir=snapshot \
    resume=false\
    project=final_mt_mdp \
    use_wandb=True