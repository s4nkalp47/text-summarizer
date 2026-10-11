# Build Plan: Dialogue Summarizer

This file plans the build phase by phase for Claude Code. Each phase has three parts:

- **You do:** manual steps for you
- **Prompt:** text to paste into Claude Code
- **Done when:** checks that must pass before you move on

## Progress

- [ ] Phase 0: Setup (you)
- [x] Phase 1: Project skeleton and config plumbing
- [x] Phase 2: Data ingestion (Stage 1)
- [ ] Phase 3: Data transformation (Stage 2)
- [ ] Phase 4: Model fine-tuning (Stage 3), with a local smoke test and Colab training
- [ ] Phase 5: Model evaluation (Stage 4)
- [ ] Phase 6: Prediction pipeline and API (Stage 5)
- [ ] Phase 7: Docker, README and deployment

| Phase | Builds | Runs on | Rough time |
|---|---|---|---|
| 0 | Tools, venv, git, accounts | Laptop | 30 min |
| 1 | Folder structure, config, logging, `main.py` | Laptop | 1 hr |
| 2 | Dataset download and length analysis | Laptop | 30 min |
| 3 | Tokenization | Laptop | 45 min |
| 4 | Training code + Colab notebook | Laptop (smoke) / Colab T4 (full) | 1 hr + 1–2 hr of GPU time |
| 5 | ROUGE evaluation and base-vs-fine-tuned comparison | Laptop (smoke) / Colab | 1 hr |
| 6 | `/summarize` API and a small web page | Laptop | 1 hr |
| 7 | Dockerfile, README, CI, Hugging Face Space | Laptop + HF | 1–2 hr |

---

## How to work through each phase with Claude Code

1. **Start clean.** Open a new session or type `/clear`. `CLAUDE.md` loads automatically, so Claude always knows the project rules.
2. **Plan first.** Switch to **Plan mode** and paste the phase prompt. Read the plan Claude proposes, question anything you don't understand or didn't ask for, then approve it.
3. **Let it build.** Claude writes the code and runs the checks. Read the output yourself as well.
4. **Understand it.** Review the diff. Ask things like "explain `model_trainer.py` for a beginner." Understanding the code is the point of the project.
5. **Verify and commit.** Run the **Done when** checks yourself, then commit (or ask Claude to commit):
   ```
   git add -A
   git commit -m "Phase N: <what you built>"
   ```

### Useful prompts at any point

```
This failed with the error below. Find the root cause before changing anything, then explain it to me.
<paste the full error>
```
```
Review the Phase N changes against CLAUDE.md conventions and list anything that breaks them. Don't fix yet.
```
```
Explain @src/textSummarizer/components/<file>.py step by step, as if I'm new to Hugging Face.
```

Type `/compact` if a session gets long. Run long jobs yourself in the terminal so you can watch them.

---

## Phase 0: Setup (you do all of this)

**You do:**
1. Install **Python 3.11** (from python.org; tick "Add python.exe to PATH") and **Git**.
2. Create a project folder (for example `Documents\text-summarizer`), copy `CLAUDE.md` and `PLAN.md` into it, and open that folder in Claude Code.
3. In a terminal in that folder:
   ```
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   git init
   ```
   If PowerShell blocks `Activate.ps1`, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or use Git Bash with `source .venv/Scripts/activate`.
4. Create an empty **GitHub** repo and link it with `git remote add origin <url>`. Colab clones the project from it in Phase 4. A public repo is easiest to clone.
5. Accounts: a **Google** account for Colab (required) and a free **Hugging Face** account for hosting the model in Phase 7 (optional).
6. If your laptop has an NVIDIA GPU, edit the **Hardware** line in `CLAUDE.md`.

**Done when:** `python --version` shows 3.11.x, the venv is active, `git status` works, and Claude Code is open in the folder.

---

## Phase 1: Project skeleton and config plumbing

**Prompt:**
```
Read @CLAUDE.md and @PLAN.md. Implement Phase 1 only: the project skeleton and config plumbing. No ML logic yet.

Create:
- Package src/textSummarizer/ with subpackages components, pipeline, config, entity, utils, logging, constants (each with __init__.py), installable with pyproject.toml (`pip install -e .`).
- requirements.txt: torch, transformers, datasets, evaluate, rouge_score, sentencepiece, accelerate, pandas, pyyaml, python-box, fastapi, uvicorn, pytest, httpx, matplotlib. Install them into .venv, then pin the versions that got installed.
- logging/__init__.py: a `logger` that writes to the console and to logs/running_logs.log.
- utils/common.py: read_yaml(path) -> ConfigBox (raise a clear error on an empty file), create_directories(paths).
- constants: paths to config/config.yaml and params.yaml.
- config/config.yaml: artifacts_root, plus one section per stage (data_ingestion, data_transformation, model_trainer, model_evaluation, prediction) with root_dir and the paths each stage reads and writes. Dataset: knkarthick/samsum. Model checkpoint: google/pegasus-cnn_dailymail.
- params.yaml:
  - training: num_train_epochs 1, per_device_train_batch_size 1, gradient_accumulation_steps 16, learning_rate 5e-5, warmup_steps 500, weight_decay 0.01, logging_steps 10, fp16 false, optim adamw_torch, gradient_checkpointing false
  - tokenization: max_input_length and max_target_length (placeholders, set in Phase 2)
  - generation: num_beams 8, length_penalty 0.8, max_new_tokens 128
  - evaluation: num_samples 100, batch_size 4
  - smoke_test: enabled true, model_ckpt t5-small, num_samples 64, max_steps 5
- entity/: one frozen dataclass per stage config.
- config/configuration.py: ConfigurationManager that reads both YAML files, creates the artifact dirs, and returns each stage's dataclass. When smoke mode is on, it substitutes the smoke model, sample limits and max_steps.
- main.py: argparse with --stage {ingestion,transformation,training,evaluation,all} (default all) and --smoke/--no-smoke to override params.yaml. For now each stage only logs "not implemented".
- .gitignore: .venv, artifacts/, logs/, __pycache__, *.egg-info, .ipynb_checkpoints.
- tests/test_common.py covering read_yaml and create_directories.

Run pytest and `python main.py` and show me the output. Then explain in a few sentences how config flows from the YAML files to a component.
```

**Done when:**
- [x] `pip install -e .` succeeds
- [x] `python main.py` logs all four stages and `logs/running_logs.log` exists
- [x] `python main.py --no-smoke` shows that the Pegasus checkpoint was selected
- [x] `pytest -q` passes

---

## Phase 2: Data ingestion (Stage 1)

**Prompt:**
```
Implement Phase 2 (Stage 1: data ingestion) from @PLAN.md.

- components/data_ingestion.py: DataIngestion(config) loads the dataset named in config with datasets.load_dataset and saves it with save_to_disk under artifacts/data_ingestion/. If it's already saved, skip the download and log that.
- pipeline/stage_01_data_ingestion.py wires config to the component. Hook it into main.py.
- research/eda.py: load the saved dataset and print the split sizes, 3 example dialogue/summary pairs, and token-length percentiles (50/90/95/99/max) for dialogues and summaries, measured with the Pegasus tokenizer (tokenizer only, a small download). Save a histogram to research/length_hist.png.
- From the 99th percentiles, propose max_input_length and max_target_length and write them to params.yaml (round up to a multiple of 64; input at most 1024).

Run `python main.py --stage ingestion` and research/eda.py, show me the stats, and explain what these lengths mean for training (truncation, memory).
```

**Done when:**
- [x] `artifacts/data_ingestion/` has train (~14.7k), validation (818) and test (819) splits
- [x] Running ingestion again skips the download
- [x] `params.yaml` has real max lengths (expect about 512 for input and 128 for target)
- [ ] You've looked at `research/length_hist.png` and a few raw dialogues

---

## Phase 3: Data transformation (Stage 2)

**Prompt:**
```
Implement Phase 3 (Stage 2: data transformation) from @PLAN.md.

- components/data_transformation.py: load the tokenizer for the configured checkpoint (the smoke model when smoke mode is on). A batched map function tokenizes `dialogue` as inputs (max_input_length, truncation) and `summary` via text_target= as labels (max_target_length). If the checkpoint is a T5 model, prefix the inputs with "summarize: ". Remove the original columns and save to artifacts/data_transformation/.
- In smoke mode, keep only smoke num_samples rows per split.
- Add the stage pipeline and hook it into main.py.
- tests/test_data_transformation.py: build a 3-row in-memory Dataset, run the map function with the t5-small tokenizer, and assert the output columns and that no sequence exceeds the max lengths.

Run pytest and `python main.py --stage transformation`. Print one transformed example decoded back to text, and explain input_ids, attention_mask and labels.
```

**Done when:**
- [ ] The saved dataset has only `input_ids`, `attention_mask` and `labels` columns
- [ ] The decoded example reads like the original dialogue and summary
- [ ] `pytest -q` passes

---

## Phase 4: Model fine-tuning (Stage 3)

### 4a: Training code with a local smoke test

**Prompt:**
```
Implement Phase 4a (Stage 3: model training) from @PLAN.md.

- components/model_trainer.py: load AutoModelForSeq2SeqLM and the tokenizer for the configured checkpoint. Use DataCollatorForSeq2Seq and Seq2SeqTrainer, with Seq2SeqTrainingArguments built from params.yaml. Pick the device automatically. Keep fp16 configurable but off by default, because Pegasus can produce NaN loss in fp16. Pass through the optim and gradient_checkpointing params for low-memory GPUs. In smoke mode, use max_steps from smoke_test.
- Evaluate on the validation split at the end of training and log the eval loss.
- Save the model to artifacts/model_trainer/model and the tokenizer to artifacts/model_trainer/tokenizer.
- Add the stage pipeline and hook it into main.py.

Run `python main.py --stage training` in smoke mode. Show me the logged loss and the saved files, and explain what the data collator, gradient accumulation and warmup do.
```

**Done when:**
- [ ] The smoke training finishes in a few minutes on CPU and the loss is logged
- [ ] `artifacts/model_trainer/model` and `artifacts/model_trainer/tokenizer` exist

### 4b: Full training on Colab (free T4 GPU)

**Prompt:**
```
Implement Phase 4b from @PLAN.md: create research/colab_train.ipynb for full training on a free Colab T4 GPU. Cells:
1. Check the GPU (nvidia-smi, torch.cuda.is_available()).
2. Mount Google Drive.
3. Clone my GitHub repo (REPO_URL variable at the top) and pip install requirements plus `pip install -e .`.
4. Run ingestion, then transformation and training with --no-smoke.
5. Copy artifacts/model_trainer/ to MyDrive/text-summarizer/model_trainer so it survives a disconnect.
6. Sanity check: summarize 2 test dialogues with the trained model.
Add a markdown cell before each step explaining it, plus a "CUDA out of memory?" cell: set optim: adafactor and gradient_checkpointing: true, or lower max_input_length.

Don't run the notebook. Afterwards, list exactly what I need to do in Colab, step by step.
```

**You do:**
1. Commit, then `git push -u origin main`.
2. Open colab.research.google.com, choose **File → Upload notebook** and pick `research/colab_train.ipynb`.
3. Choose **Runtime → Change runtime type → T4 GPU**, then **Run all**. Training takes about 1–2 hours. Keep the tab open.
4. Download `MyDrive/text-summarizer/model_trainer/` and put it in your local `artifacts/model_trainer/`.

**Plan B (no training):** set the model path in `config.yaml` to the public, already fine-tuned `transformersbook/pegasus-samsum` and continue with Phase 5.

**Done when:**
- [ ] The Colab run finished with a decreasing loss and sensible sample summaries
- [ ] The fine-tuned model is in `artifacts/model_trainer/` on your laptop (or you're using Plan B)

---

## Phase 5: Model evaluation (Stage 4)

**Prompt:**
```
Implement Phase 5 (Stage 4: model evaluation) from @PLAN.md.

- components/model_evaluation.py: load the raw test split from artifacts/data_ingestion and the model to evaluate (path from config). Generate summaries in batches with the generation params, and replace "<n>" with a space. Compute rouge1, rouge2, rougeL and rougeLsum with `evaluate`. Use evaluation.num_samples test samples (10 in smoke mode).
- Save the metrics to artifacts/model_evaluation/metrics.csv and 10 example rows (dialogue, reference, prediction) to samples.csv.
- research/compare_models.py: evaluate the base checkpoint and the fine-tuned model on the same samples and write comparison.csv.
- Add the stage pipeline and hook it into main.py. Add an evaluation + comparison cell to research/colab_train.ipynb so it can run on the GPU.

Run the evaluation in smoke mode locally and show me the metrics. Explain what ROUGE-1, ROUGE-2 and ROUGE-L measure and what they miss.
```

**You do:** run the comparison cell in Colab and download `comparison.csv`.

**Done when:**
- [ ] `metrics.csv` and `samples.csv` exist
- [ ] The fine-tuned model clearly beats the base model (fine-tuned ROUGE-1 is typically around 0.40 or higher; the base model scores noticeably lower)
- [ ] You've read `samples.csv` and judged the summaries yourself

---

## Phase 6: Prediction pipeline and API (Stage 5)

**Prompt:**
```
Implement Phase 6 (Stage 5: prediction pipeline and API) from @PLAN.md.

- pipeline/prediction.py: PredictionPipeline loads the model and tokenizer once from config prediction.model_path (a local folder or a Hugging Face model id) and exposes summarize(dialogue: str) -> str using the generation params, with "<n>" stripped.
- app.py (FastAPI): GET /health; POST /summarize takes {"dialogue": str} and returns {"summary": str}. Reject empty input, and input over a configured character limit, with a 422. Load the model once at startup, not per request.
- GET / serves static/index.html: a textarea, a Summarize button and the result, using fetch. Plain HTML/CSS/JS, no frameworks.
- tests/test_api.py: use FastAPI TestClient with the pipeline replaced by a fake. Tests must not load a real model.

Run pytest. Then start uvicorn, call /summarize with a sample SAMSum dialogue, and show me the response.
```

**You do:** set `prediction.model_path` to your fine-tuned model folder (or the Plan B id), run `uvicorn app:app --port 8080`, open http://localhost:8080 and try your own chats.

**Done when:**
- [ ] http://localhost:8080/docs works and `/summarize` returns a good summary
- [ ] The web page works
- [ ] `pytest -q` passes without loading a real model

---

## Phase 7: Docker, README and deployment

**Prompt:**
```
Implement Phase 7 from @PLAN.md: packaging and docs.

- Dockerfile: python:3.11-slim. Install CPU-only torch from the PyTorch CPU index to keep the image small, then install the package. Run uvicorn on port 7860 (the Hugging Face Spaces default), overridable with a PORT env var. Add a .dockerignore that excludes .venv, artifacts, logs, research and .git.
- The container gets its model from prediction.model_path set to a Hugging Face Hub id. Don't bake weights into the image.
- research/push_to_hub.py: upload the model and tokenizer in artifacts/model_trainer to a Hub repo name given as an argument. I'll log in myself first.
- README.md: what the project does, the 5 stages, how to run each stage, API usage with a curl example, the ROUGE results table from Phase 5, and licensing (Pegasus: Apache-2.0; SAMSum: CC BY-NC-ND 4.0, non-commercial).
- .github/workflows/tests.yml runs pytest on every push.

Don't build or push anything. Give me the exact commands to build and run the container, and to deploy to a Hugging Face Space.
```

**You do:**
1. Run `hf auth login` (or `huggingface-cli login` on older versions) and paste your HF access token yourself.
2. Run `python research/push_to_hub.py <your-username>/pegasus-samsum`, then set `prediction.model_path` to that id.
3. Optional, needs Docker Desktop: `docker build -t text-summarizer .`, then `docker run -p 7860:7860 text-summarizer`.
4. Optional, free hosting: on huggingface.co create a **Space** with SDK **Docker** and hardware **CPU basic (free)**, then push the repo to it.

**Done when:**
- [ ] The container serves `/summarize` (if you used Docker)
- [ ] The README is complete, with real ROUGE numbers
- [ ] CI is green on GitHub
- [ ] Optional: the Space is live and you have a public demo link

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: textSummarizer` | Activate the venv, then `pip install -e .` |
| `Samsung/samsum` fails to load | Use `knkarthick/samsum` (the default), or `pip install py7zr` |
| CUDA out of memory in Colab | `optim: adafactor`, `gradient_checkpointing: true`, `max_input_length: 512`, batch size 1 |
| Loss is `nan` | Turn `fp16` off |
| `<n>` in summaries | Replace `<n>` with a space after decoding |
| Hugging Face symlink warning on Windows | Harmless. Hide it with the env var `HF_HUB_DISABLE_SYMLINKS_WARNING=1` |
| Colab disconnected mid-training | Rerun. Ingestion is cached, and the saved model is copied to Drive only at the end, so consider adding `save_steps` checkpoints to Drive |
| Summaries look generic | Check that you're loading the **fine-tuned** model path, not the base checkpoint |
