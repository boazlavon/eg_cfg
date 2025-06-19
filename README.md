# EG-CFG: Execution-Guided Line-by-Line Code Generation 

**EG-CFG** is an inference-time algorithm for code generation that injects real-time execution feedback directly into the model’s decoding loop. By incorporating dynamic runtime signals during generation, it steers the model toward solutions that are not only syntactically valid, but also functionally correct and executable.

**SOTA performance on top code generation benchmarks**: from foundational tasks (*MBPP*, *HumanEval*) to extended evaluations (*MBPP-ET*, *HumanEval-ET*) and challenging competitive programming problems (*CodeContests*) - all using open-source models only.

[![arXiv](https://img.shields.io/badge/arXiv-2506.10948-b31b1b)](https://arxiv.org/abs/2506.10948)
[![Video](https://img.shields.io/badge/YouTube-Video-red)](https://youtu.be/YgBcDUQg7As?si=SYyKIyPTdKPNDmO4)
[![Papers with Code](https://img.shields.io/badge/Papers%20with%20Code-View-blue)](https://paperswithcode.com/paper/execution-guided-line-by-line-code-generation)

---

## 🚀 Highlights

📈 **New state-of-the-art (SOTA) results**:

- **MBPP**: 96.6%
- **MBPP-ET**: 73.0%
- **HumanEval-ET**: 87.19%
- **CodeContests**: 58.18%

✅ Achieved using **open-source models only** (DeepSeek-V3-0324)<br>
⚡ Real-time execution feedback integrated during decoding<br>
🛠️ Fully configurable pipeline — supports both local and endpoint inference<br>
🔁 Reproducible and extensible framework for code generation research<br>

## 🧠 Models

EG-CFG supports any causal language model that provides token-level log probabilities. In our experiments, we use two models from the **DeepSeek** family:

### 🔹 [DeepSeek-Coder-1.3B-Instruct](https://huggingface.co/deepseek-ai/deepseek-coder-1.3b-instruct)
- 1.3B parameter instruction-tuned model
- Suitable for local inference
- Efficient yet surprisingly strong for Python code generation

### 🔹 [DeepSeek-V3-0324](https://huggingface.co/deepseek-ai/DeepSeek-V3-0324)
- Large-scale foundation model
- Used via inference endpoint

---
## 📊 Benchmark Results
### MBPP and MBPP-ET

| Model               | Method            | MBPP (%) | MBPP-ET (%) | RSR (MBPP) | RSR (MBPP-ET) |
| ------------------- | ----------------- | -------- | ----------- | ---------- | ------------- |
| DeepSeek-Coder 1.3B | Baseline LLM      | 49.4     | 42.6        | 0.0        | 0.0           |
| DeepSeek-Coder 1.3B | EG-CFG (Ours)     | 83.2     | 59.8        | 66.79      | 29.96         |
| DeepSeek-V3-0324    | Baseline LLM      | 82.8     | 64.8        | 0.0        | 0.0           |
| DeepSeek-V3-0324    | **EG-CFG (Ours)** | **96.6** | **73.0**    | **80.23**  | **23.30**     |
| GPT-4o              | LPW               | 84.4     | 65.3        | N/A        | N/A           |
| Claude-Sonnet-3.5   | QualityFlow       | 94.2     | N/A         | N/A        | N/A           |
| GPT-4               | MetaGPT           | 87.7     | N/A         | N/A        | N/A           |

### HumanEval and HumanEval-ET

| Model            | Method            | HumanEval (%) | HumanEval-ET (%) | RSR (HE)  | RSR (HE-ET) |
| ---------------- | ----------------- | ------------- | ---------------- | --------- | ----------- |
| DeepSeek-V3-0324 | Baseline LLM      | 82.92         | 79.20            | 0.0       | 0.0         |
| DeepSeek-V3-0324 | **EG-CFG (Ours)** | **96.95**     | **87.19**        | **78.54** | **38.56**   |
| DeepSeek-V3-0324 | MapCoder          | 96.95         | 81.70            | 81.88     | 12.02       |
| DeepSeek-V3-0324 | MGDebugger        | 87.20         | 81.09            | 25.39     | 9.44        |
| DeepSeek-V3-0324 | LPW               | 95.12         | 84.74            | 68.02     | 26.89       |
| GPT-4o           | LPW               | 98.2          | 84.8             | N/A       | N/A         |

### CodeContests
| Model            | Method           | Accuracy (%) | RSR (%)  |
|------------------|------------------|--------------|----------|
| DeepSeek-V3-0324 | Baseline LLM     | 41.81        | 0.00     |
| DeepSeek-V3-0324 | **EG-CFG (Ours)**| **58.18**    | **28.13**|
| DeepSeek-V3-0324 | MapCoder         | 50.30        | 14.59    |
| GPT-4o           | LPW              | 34.7         | N/A      |
| GPT-4o           | LDB              | 29.3         | N/A      |
| GPT-4            | CodeSim          | 29.1         | N/A      |
| GPT-4            | MapCoder         | 28.5         | N/A      |
| GPT-3.5 Turbo    | CodeSim          | 16.4         | N/A      |
| GPT-3.5 Turbo    | MapCoder         | 12.7         | N/A      |
| MoTCoder-15B     | MoTCoder         | 26.34        | N/A      |

> RSR: Relative Success Rate = Accuracy gain over baseline normalized to full success (see paper Sec. 4.1)
> See full tables and ablations in the [paper](https://arxiv.org/abs/2506.10948).

### Evaluation Limitations

We manually reviewed all 17 MBPP tasks that were not solved by DeepSeek-V3-0324 and found that 9 contain invalid unit tests, with some also having incorrect reference solutions. In these cases, the model-generated code is correct but marked as failed due to flawed benchmark tests.  Full details are available in the [`mbpp_analysis/`](./mbpp_analysis/) directory.

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
| `dataset`                 | `"mbpp"` or `"humaneval"`                                    |
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
  "dataset": "mbpp",
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

### 🛠️ Modified `transformers` Library

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
We evaluate EG-CFG on three widely used Python code generation benchmarks:

🔹 MBPP

The MBPP (Mostly Basic Python Problems) benchmark [Austin et al., 2021] includes 500 Python tasks, each with a natural language description, function name, and 3 unit tests. It is a popular dataset for evaluating basic code generation.

🔹 HumanEval

The HumanEval benchmark [Chen et al., 2021] consists of 164 hand-written Python programming tasks with hidden test cases. Each task defines a function signature and problem description, designed to measure functional correctness.

🔹 MBPP-ET & HumanEval-ET

We also evaluate on MBPP-ET and HumanEval-ET, extended test suites proposed in CodeScore [Dong et al., 2025]. These enhancements add more challenging edge cases and improve coverage, offering better estimates of real-world generalization.

🔹 CodeContests

The CodeContests benchmark [Li et al., 2022] is a suite of competitive programming problems designed to evaluate advanced algorithmic reasoning and problem-solving skills. Each task includes a problem description and multiple hidden test cases. Solutions are evaluated using the [ExecEval framework](https://github.com/ntunlp/ExecEval) [Khan et al., 2024]. Performance on CodeContests reflects a model’s robustness and problem-solving depth under competitive constraints.

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
## 📖 Citation

If you use EG-CFG or find our work helpful, please consider citing:

```bibtex
@article{lavon2025execution,
  title={Execution Guided Line-by-Line Code Generation},
  author={Lavon, Boaz and Katz, Shahar and Wolf, Lior},
  journal={arXiv preprint arXiv:2506.10948},
  year={2025}
}
```
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
@article{liu2024deepseekv3,
  title={DeepSeek-V3 Technical Report},
  author={Liu, Aixin and Feng, Bei and Xue, Bing and Wang, Bingxuan and others},
  journal={arXiv preprint arXiv:2412.19437},
  year={2024}
}
@article{austin2021program,
  title={Program synthesis with large language models},
  author={Austin, Jacob and Odena, Augustus and Nye, Maxwell and Bosma, Maarten and Michalewski, Henryk and Dohan, David and Jiang, Ellen and Cai, Carrie and Terry, Michael and Le, Quoc and others},
  journal={arXiv preprint arXiv:2108.07732},
  year={2021}
}
@article{chen2021evaluating,
  title={Evaluating large language models trained on code},
  author={Chen, Mark and Tworek, Jerry and Jun, Heewoo and Yuan, Qiming and Pinto, Henrique Ponde de Oliveira and Kaplan, Jared and Edwards, Harri and Burda, Yuri and Joseph, Nicholas and Brockman, Greg and others},
  journal={arXiv preprint arXiv:2107.03374},
  year={2021}
}
@article{dong2025codescore,
  title={CodeScore: Evaluating Code Generation by Learning Code Execution},
  author={Dong, Yihong and Ding, Jiazheng and Jiang, Xue and Li, Ge and Li, Zhuo and Jin, Zhi},
  journal={ACM Transactions on Software Engineering and Methodology},
  volume={34},
  number={3},
  pages={1--22},
  year={2025}
}
@article{li2022alphacode,
  title={Competition-level code generation with AlphaCode},
  author={Li, Yujia and Choi, David and Chung, Junyoung and Kushman, Nate and Schrittwieser, Julian and others},
  journal={Science},
  volume={378},
  number={6624},
  pages={1092--1097},
  year={2022}
}
@inproceedings{khan2024xcodeeval,
  title={XCodeEval: An Execution-Based Large Scale Multilingual Multitask Benchmark for Code Understanding, Generation, Translation and Retrieval},
  author={Khan, Mohammad Abdullah Matin and Bari, M Saiful and Long, Do and Wang, Weishi and Parvez, Md Rizwan and Joty, Shafiq},
  booktitle={Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={6766--6805},
  year={2024}
}
```
---

## ✅ ML Code Checklist

- [x] Dependency spec: `environment.yml`
- [x] Inference + Analysis code
- [x] Evaluation scripts and commands
- [x] Result tables + reproducibility

## 🧾 License
This repository is licensed under the [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license.
This software is provided for non-commercial use only.
For commercial use, you must obtain a commercial license by contacting Ramot - Technology Transfer Company of Tel Aviv University (yair.eran@ramot.org).
