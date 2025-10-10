#!/bin/bash
#SBATCH --job-name=maskdp_eval_walker_walk       # Job name
#SBATCH --mail-type=BEGIN,END,FAIL       # Mail (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=fivillagran@uc.cl    # El mail del usuario
#SBATCH --output=logs/%x-%j.out          # Log file (%x=job-name, %j=job-ID)
#SBATCH --error=logs/%x-%j.err           # Error log                    
#SBATCH --gres=gpu:1                     # Number of GPUs
#SBATCH --cpus-per-task=16               # CPU cores
#SBATCH --nodelist=peteroa
#SBATCH --partition=debug
#SBATCH --account=defaultacc             
#SBATCH --qos=normal 
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --chdir=/home/bibarel/workspace

# --- Environment setup ---
source "./miniconda3/etc/profile.d/conda.sh"
conda activate maskdp
cd "./maskdp-taskgeneral"
pwd
echo "Evaluating walker_walk on goal_reaching..."

python eval_goal_novideo.py \
    agent=mdp_goal \
    agent.batch_size=384 \
    seed=3 \
    num_eval_episodes=300 \
    task=walker_walk \
    snapshot_base_dir=/home/bibarel/workspace/maskdp_models/output/2025.10.01 \
    goal_buffer_dir=/home/bibarel/workspace/maskdp_data/maskdp_eval/expert \
    snapshot_ts=400000 \
    project=eval-single-goal \
    replan=false \
    use_wandb=True