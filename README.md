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

| Model                   | Method                      | Accuracy (%) | RSR (%)   |
|------------------------|-----------------------------|--------------|-----------|
| DeepSeek-Coder 1.3B    | Baseline LLM                | 50.8         | 0.00      |
| DeepSeek-Coder 1.3B    | EG-CFG (Ours)               | 84.8         | 69.11     |
| DeepSeek-Coder 1.3B    | MapCoder                    | 55.2         | 8.94      |
| DeepSeek-Coder 1.3B    | MGDebugger                  | 70.4         | 39.84     |
| DeepSeek-V3-0324       | Baseline LLM                | 84.2         | 0.00      |
| **DeepSeek-V3-0324**   | **EG-CFG (Ours)**           | **98.4**     | **89.87** |
| DeepSeek-V3-0324       | MapCoder                    | 74.23        | -63.10    |
| DeepSeek-V3-0324       | MGDebugger                  | 77.0         | -45.57    |
| GPT-4                  | Baseline LLM                | 68.3         | –         |
| GPT-4                  | Self-Collaboration          | 78.9         | –         |
| GPT-4                  | Self-Debugging              | 80.6         | –         |
| GPT-4                  | MetaGPT                     | 87.7         | –         |
| GPT-4                  | MapCoder                    | 83.1         | –         |
| CodeQwen1.5            | MGDebugger                  | 80.8         | –         |
| DeepSeek-Coder-V2-Lite | MGDebugger                  | 80.0         | –         |
| Claude-Sonnet-3.5      | Baseline LM                 | 88.7         | –         |
| Claude-Sonnet-3.5      | QualityFlow                 | 94.2         | –         |

> See full tables and ablations in the [paper](link).

---

## 🧱 Project Structure

```
eg_cfg/           # Core implementation (EG-CFG inference loop, CFG logic, and prompts)
traces_dumper/    # Tools for extracting execution traces
scripts/          # Entry points for launching and monitoring experiments
configs/          # Configuration files
trials/           # Generated results from inference runs
output/           # Stdout logs from inference runs
data/             # Input data for inference, such as prompts and baseline results
submodules/       # Local submodules (e.g., xpython, trepan, transformers)
environment.yml   # Conda environment definition
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

## 🚀 Launch Inference Jobs
To maximize throughput, we encourage launching this script **multiple times—once per available node**. The pipeline supports full synchronization across jobs, so no manual coordination is needed. Just launch as many instances as you have nodes, and they’ll run in parallel seamlessly.
```bash
./scripts/job_runners/inference_sbatch.local.sh
# Or monitor in watch mode
./scripts/job_runners/inference_sbatch.local.sh watch
```

---

## 📈 Monitor and Aggregate Results

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

Defines the parameters used to generate dynamic execution signals.
```json
{
  "t": [0.7, 0.75],         # Sampling temperatures
  "s": [3],                 # Number of candidates (beam size)
  "d": [2, 3],              # Completion horizon (lines)
  "prompt_type": ["deepseek_instruct", "long_code"]
}
```

### 🔧 session_config.local.json / session_config.inference_endpoint.json

Defines runtime setup per session:

| Field                      | Description                                                  |
|----------------------------|--------------------------------------------------------------|
| `model_name`              | Model to use (local path or HuggingFace hub name)            |
| `gammas`                  | CFG guidance strengths                                       |
| `deployment_type`         | `"local"` or `"inference_endpoint"`                          |
| `results_dir`             | Root directory for saving results                            |
| `inference_endpoint_url`  | (if endpoint) API URL for inference                          |
| `inference_endpoint_api_key` | (if endpoint) API key for Fireworks                       |
| `use_global_cache`        | Avoid recomputing same completions                           |
| `debug_mode`              | Enable logging/debug information                             |
| `is_prod`                 | Run in production mode (disable debug/test toggles)          |
| `minimal_trace`           | Use final-state-only traces instead of full step-by-step traces |


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
  "code": "...",  # Model-generated Python code
  "results": {
    "assert ...": {
      "result": true,           # Whether test case passed
      "time": 0.123,            # Execution time in seconds
      "error": null             # Any runtime error (or null)
    }
  },
  "passed": true,              # True if all test cases passed
  "accuracy": 1.0,             # Fraction of passed test cases
  "general_error": null,       # Top-level failure unrelated to test cases
  "has_testcase_error": false, # True if any test case raised an exception
  "stats": {
    "start_time": "...",
    "end_time": "...",
    "input_tokens": 1234,      # Total prompt tokens
    "output_tokens": 456,      # Total generated tokens
    "duration": "00:01:23"     # Inference wall-time duration
  }
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

In local inference mode, we extend the internal decoding loop of the HuggingFace `transformers` library to support execution-aware generation.
Specifically, our modifications in `transformers/generation/utils.py` enable token-level integration of runtime feedback, allowing the model to dynamically condition on execution traces as described in Section 3 of the paper.
This integration is essential for realizing EG-CFG's line-by-line guidance mechanism during inference.

### 🛠️ Execution Tracing via `trepan-xpy`
We use the `trepan-xpy` debugger to execute partially completed code and extract execution traces during inference.
To support our framework, we extended the debugger to emit canonicalized traces — a consistent structure that captures all relevant runtime signals, regardless of whether the execution succeeds or fails.
This includes not only variable values and function calls, but also bytecode-level events such as instruction execution, enabling fine-grained introspection.
The canonical format allows us to easily manipulate the trace to retain only the information most relevant for guiding generation.

> These are included in `submodules/` and linked into `site-packages/` using:
> ```bash
> python scripts/redirect_env_to_submodules.py $PWD/submodules/
> ```

---

## 📚 Data

We evaluate EG-CFG on the **MBPP (Mostly Basic Python Problems)** benchmark [Austin et al., 2021] — a widely used dataset of Python programming tasks. Each task includes a natural language description, a target function name, and a set of unit tests.

### 🧾 Prompt Format

We use two prompt types to ensure broad and reproducible evaluation:

#### 🔹 Official Few-Shot Prompt (DeepSeek-Coder)
We adopt the **official evaluation prompt** provided by DeepSeek-Coder’s GitHub [Guo et al., 2024]:
- Includes 3 few-shot examples before each target problem
- Matches the DeepSeek-Coder evaluation setting  
- Source: [deepseek-ai/DeepSeek-Coder GitHub](https://github.com/deepseek-ai/DeepSeek-Coder)

#### 🔹 Long-Code Prompt (ours)
In addition, we introduce a **long-code instruction-only prompt** that:
- Encourages line-by-line, traceable completions
- Follows stylistic constraints aligned with dynamic execution trace extraction
- Designed for EG-CFG’s runtime-guided generation  
- Detailed in Appendix A of our paper

---

### ☁️ Inference Endpoint

For large-scale model inference (e.g., using DeepSeek-V3-0324), we use [Fireworks.ai](https://fireworks.ai/) as the inference endpoint provider.
Fireworks supports **token-level log probabilities**, which are essential for performing Classifier-Free Guidance (CFG) during decoding.

No local GPU is required—all inference runs remotely on Fireworks infrastructure.

> Endpoint access is configured via `session_config.inference_endpoint.json` using your Fireworks API key and endpoint URL.

---

## 📚 Related Work Citations

We gratefully acknowledge the authors of the following works for their implementations and publicly available models. If you find this repository helpful, please consider citing their papers as well.

```bibtex
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

## 📜 Citation

```bibtex
@inproceedings{anonymous2025egcfg,
  title={Execution-Guided Line-by-Line Code Generation},
  author={Anonymous},
  booktitle={NeurIPS 2025},
  year={2025}
}
```


---

## ✅ ML Code Checklist

- [x] Dependency spec: `environment.yml`
- [x] Inference + Analysis code
- [x] Evaluation scripts and commands
- [x] Result tables + reproducibility

