git clone --recurse-submodules git@github.com:boazlavon/eg_cfg.git
conda env create -f environment.yml -n eg-cfg-env
conda activate eg-cfg-env
python scripts/redirect_env_to_submodules.py $PWD/submodules/
