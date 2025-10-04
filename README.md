## File Descriptions

`conda_env.yml`: Conda YAML specifying Python & package dependencies. Used to create the "maskdp" conda environment.

```bash
conda env create -f conda_env.yml
conda activate maskdp
```

`datachecker.py`: File for checking the structure of a .npz data file in the dataset.

`dmc.py`: Wrapper for DeepMind Control

`agent/`: Contains agent definitions for MaskDP.

- `mdp`: Masked DP agent 

`custom_dmc_tasks/`: Contains custom DM Control tasks. 
