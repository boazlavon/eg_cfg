git clone --recurse-submodules git@github.com:boazlavon/eg_cfg.git
conda env create -f environment.yml -n eg-cfg-env
conda activate eg-cfg-env
python scripts/redirect_env_to_submodules.py $PWD/submodules/

## monitor
python eg_cfg/eg_cfg_monitor.py --aggregate-dir trials/inference_endpoint_results/mbpp/deepseek-ai_DeepSeek-V3-0324/ --model "deepseek-ai/DeepSeek-V3-0324" --gammas 0.0 0.5 1.0 3.0
python eg_cfg/eg_cfg_monitor.py --aggregate-dir trials/local_results/mbpp/deepseek-ai_deepseek-coder-1.3b-instruct/  --model "deepseek-ai/deepseek-coder-1.3b-instruct" --gammas 0.0 0.5 1.0 3.0
