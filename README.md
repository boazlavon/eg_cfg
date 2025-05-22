# EG-CFG - Execution-Guided Line-by-Line Code Generation 

EG-CFG is a decoding-time algorithm for code generation that incorporates real-time **execution feedback** into LLM inference. By injecting dynamic signals during generation, EG-CFG guides the model toward correct and executable solutions — achieving state-of-the-art performance on the MBPP benchmark using open-source models only.

---

## 🚀 Highlights

- 📈 **New SOTA on MBPP** using open models (96.6% with DeepSeek-V3-0324)
- ⚡ Real-time execution feedback integrated during decoding
- 🛠️ Fully configurable pipeline: local or endpoint inference
- 🔁 Reproducible and extensible for code generation research

---
## 🧠 Models

EG-CFG supports any causal language model that provides token-level log probabilities. In our experiments, we use two models from the **DeepSeek** family:

### 🔹 [DeepSeek-Coder-1.3B-Instruct](https://huggingface.co/deepseek-ai/deepseek-coder-1.3b-instruct)
- 1.3B parameter instruction-tuned model
- Suitable for local inference
- Efficient yet surprisingly strong for Python code generation

### 🔹 [DeepSeek-V3-0324](https://huggingface.co/deepseek-ai/DeepSeek-V3-0324)
- Large-scale foundation model
- Used via inference endpoint
- Achieves 96.6% on MBPP with EG-CFG, setting a new state-of-the-art

---
## 📊 Benchmark Results

| Model                 | Method        | Accuracy (%) | RSR (%) |
|----------------------|---------------|--------------|---------|
| DeepSeek-Coder 1.3B  | EG-CFG        | 83.2         | 66.79   |
| DeepSeek-V3-0324     | **EG-CFG**    | **96.6**     | 80.23   |
| Claude-Sonnet-3.5    | QualityFlow   | 94.2         | –       |
| GPT-4                | MetaGPT       | 87.7         | –       |

> See full tables and ablations in the [paper](link) or `output/`.

---

## 🧱 Project Structure

```
eg_cfg/           # Core implementation (EG-CFG inference loop, CFG, prompts)
traces_dumper/    # Trace extraction tools for partial execution feedback
scripts/          # Entry points for launching and monitoring experiments
configs/          # Configuration files
trials/           # Stores generated results from inference runs
output/           # Stores stdout outputs of inference runs
data/             # Data used for inference runs like prompts and baseline results
submodules/       # Local modules (e.g., xpython, trepan, transformers)
environment.yml   # Conda environment
```

---

## ⚡ Quickstart

```bash
git clone --recurse-submodules git@github.com:OUR_REPO/eg_cfg.git
cd eg_cfg
conda env create -f environment.yml -n eg-cfg-env
conda activate eg-cfg-env
python scripts/redirect_env_to_submodules.py $PWD/submodules/
```

---

## Launch Inference Jobs

```bash
./scripts/job_runners/inference_sbatch.local.sh
# Or monitor in watch mode
./scripts/job_runners/inference_sbatch.local.sh watch
```

---

## Monitor and Aggregate Results

```bash
# DeepSeek-Coder-1.3B (local)
python eg_cfg/eg_cfg_monitor.py \
  --aggregate-dir trials/local_results/mbpp/deepseek-ai_deepseek-coder-1.3b-instruct/ \
  --model "deepseek-ai/deepseek-coder-1.3b-instruct" --gammas 0.0 0.5 1.0 3.0

# DeepSeek-V3-0324 (inference endpoint)
python eg_cfg/eg_cfg_monitor.py \
  --aggregate-dir trials/inference_endpoint_results/mbpp/deepseek-ai_DeepSeek-V3-0324/ \
  --model "deepseek-ai/DeepSeek-V3-0324" --gammas 0.0 0.5 1.0 3.0
```

---

## 📘 Configuration Guide

### 🔧 dynamic_signals_params.json

Defines the sampling and guidance sweep:

```json
{
  "t": [0.7, 0.75],         // Sampling temperatures
  "s": [3],                 // Number of candidates (beam size)
  "d": [2, 3],              // Completion horizon (lines)
  "prompt_type": ["deepseek_instruct", "long_code"]
}
```

### 🔧 session_config.local.json / session_config.inference_endpoint.json

Defines runtime setup per session:

| Field                      | Description                                                  |
|----------------------------|--------------------------------------------------------------|
| `model_name`              | Model to use (local path or HuggingFace hub name)            |
| `gammas`                  | CFG guidance strengths                                       |
| `deployment_type`         | `"local"` or `"inference_endpoint"`                         |
| `results_dir`             | Root directory for saving results                            |
| `inference_endpoint_url`  | (if endpoint) API URL for inference                          |
| `inference_endpoint_api_key` | (if endpoint) API key for Fireworks                        |
| `use_global_cache`        | Avoid recomputing same completions                           |
| `debug_mode`              | Enable logging/debug information                             |
| `is_prod`                 | Run in production mode (disable debug/test toggles)          |


---


## 📁 Results Directory Structure

Each trial is written under the path defined by `results_dir` in your session config.
For example:

```json
{
  "results_dir": "trials/local_results",
  "model_name": "deepseek-ai/deepseek-coder-1.3b-instruct",
  "deployment_type": "local",
  ...
}
```

This results in directories like:

```
trials/local_results/mbpp/deepseek-ai_deepseek-coder-1.3b-instruct/ns2t0.75d2_ln/
```

The folder name encodes the run configuration:
- `s2` → 2 candidates
- `t0.75` → temperature 0.75
- `d2` → horizon 2 lines
- `_ln` or `_lci_ln` suffix → prompt type (`deepseek_instruct` or `long_code`)

Each config directory contains:
- One JSON per task and gamma (e.g. `task_id=395_gamma=1.0.json`)

### 🧪 JSON file format

Each file includes:
```json
{
  "code": "...",                 // Model-generated Python code
  "results": {
    "assert ...": {
      "result": true/false,
      "time": <float>,           // Execution time
      "error": null / string     // Any runtime error
    },
    ...
  },
  "passed": true/false,          // Did all test cases pass?
  "accuracy": 1.0 / 0.0 / ...,
  "stats": {
    "start_time": "...",
    "end_time": "...",
    "input_tokens": <int>,
    "output_tokens": <int>,
    "duration": "HH:MM:SS"
  },
  ...
}
```

A successful solution is:
- `passed = true`
- `accuracy = 1.0`

These fields are used for filtering and reporting.

---

## 🔧 Submodules and Custom Modifications

Some core functionality in EG-CFG relies on **custom extensions of external libraries**, which are included as Git submodules and redirected into the conda environment via symlinks.

### 🛠️ Modified `transformers/` Library

In local inference mode, we extend the internal decoding loop of the HuggingFace `transformers` library to support our method’s execution-aware guidance.  
This modification enables token-level integration of execution feedback (as described in Section 3 of the paper), ensuring the model conditions on runtime traces dynamically during generation.

### 🧪 Execution Tracing via `trepan-xpy`

We use the `trepan-xpy` debugger to execute partially completed code and extract traces during inference.  
To support our framework, we modified the debugger to emit execution traces in a **canonical form** — consistent structure regardless of success, failure, or runtime errors.

> These are included in `submodules/` and linked into `site-packages/` using:
> ```bash
> python scripts/redirect_env_to_submodules.py $PWD/submodules/
> ```

---

## 📚 Data

We evaluate EG-CFG on the **MBPP (Mostly Basic Python Problems)** benchmark [Austin et al., 2021] — a widely used dataset of Python programming tasks. Each task includes a natural language description, a target function name, and a set of unit tests.

### 🧾 Prompt Format

To ensure consistency with prior work, we adopt the **official evaluation prompt formats introduced by DeepSeek-Coder** [Guo et al., 2024], including:

- **Few-shot instruction prompts** with multiple solved examples  
- **Instruction-only prompts** designed for line-by-line execution and debugging

These templates align with the evaluation procedure described in the DeepSeek-Coder paper and are used in both baseline and EG-CFG runs.

## 📜 Citation

```bibtex
@inproceedings{anonymous2025egcfg,
  title={Execution-Guided Line-by-Line Code Generation},
  author={Anonymous},
  booktitle={NeurIPS 2025},
  year={2025}
}
@article{guo2024deepseek,
  title={DeepSeek-Coder: When the Large Language Model Meets Programming--The Rise of Code Intelligence},
  author={Guo, Daya and Zhu, Qihao and Yang, Dejian and Xie, Zhenda and Dong, Kai and Zhang, Wentao and Chen, Guanting and Bi, Xiao and Wu, Yu and Li, YK and others},
  journal={arXiv preprint arXiv:2401.14196},
  year={2024}
}
@article{austin2021program,
  title={Program synthesis with large language models},
  author={Austin, Jacob and Odena, Augustus and Nye, Maxwell and Bosma, Maarten and Michalewski, Henryk and Dohan, David and Jiang, Ellen and Cai, Carrie and Terry, Michael and Le, Quoc and others},
  journal={arXiv preprint arXiv:2108.07732},
  year={2021}
}
@article{liu2024deepseekv3,
  title={DeepSeek-V3 Technical Report},
  author={Liu, Aixin and Feng, Bei and Xue, Bing and Wang, Bingxuan and others},
  journal={arXiv preprint arXiv:2412.19437},
  year={2024}
}
```

---

## ✅ ML Code Checklist

- [x] Dependency spec: `environment.yml`
- [x] Inference + Analysis code
- [x] Evaluation scripts and commands
- [x] Result tables + reproducibility

