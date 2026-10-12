---
title: Dialogue Summarizer
emoji: 💬
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
short_description: Pegasus fine-tuned on SAMSum to summarize chats
---

# Dialogue Summarizer

Summarizes messenger-style chats in one or two sentences. Google's [Pegasus](https://huggingface.co/google/pegasus-cnn_dailymail)
is fine-tuned on the [SAMSum](https://huggingface.co/datasets/knkarthick/samsum) dialogue dataset, evaluated with ROUGE
and a hand-labeled faithfulness check, and served through a FastAPI endpoint with a small web page.

```text
Rita: I'm so bloody tired. Falling asleep at work. :-(
Tina: I know what you mean.
Tina: I keep on nodding off at my keyboard hoping that the boss doesn't notice..
Rita: I keep on looking at the clock and there's still 4 hours of this drudgery to go.
Tina: Times like these I really hate my work.

→ "Rita is bored at work. Tina hates her work."
```

## Results

**Fine-tuning vs. the base model** (100 SAMSum test dialogues, same decoding settings):

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | ROUGE-Lsum |
|---|---|---|---|---|
| `google/pegasus-cnn_dailymail` (news summarizer, no fine-tuning) | 0.295 | 0.087 | 0.227 | 0.270 |
| **Fine-tuned on SAMSum (this project)** | **0.492** | **0.253** | **0.415** | **0.452** |

The base model treats a chat like a news article and copies lines from it ("Eric: Machine! Rob: Machine! …");
the fine-tuned model writes a third-person summary. **Final test scores** with the tuned decoding settings
(4 beams): ROUGE-1 **0.493**, ROUGE-2 **0.255**, ROUGE-L **0.418**, ROUGE-Lsum **0.456**.

**Training:** 1 epoch (921 optimizer steps, effective batch size 16) on a free Colab T4 in 46 minutes.
Training loss fell from 2.69 to 1.49; validation loss was 1.44 (perplexity ≈ 4.2).

### Decoding tuning: ROUGE and faithfulness disagree

Decoding settings were tuned on 100 **validation** dialogues (never on test), and 30 summaries per setting
were labeled by hand as OK / OMIT (misses the main point) / PAD (filler) / WRONG (states something false):

| Setting | ROUGE-1 (val) | Avg. words | Hand labels (30 dialogues) |
|---|---|---|---|
| `min_length=0` (chosen) | 0.503 | 14.2 | 20 OK · 8 OMIT · 0 PAD · **2 WRONG** |
| `min_length=15` | **0.529** | 16.8 | 16 OK · 6 OMIT · 1 PAD · **7 WRONG** |
| `num_beams=4` (chosen) | 0.512 | 13.7 | same quality, **42% faster** than 8 beams |

Forcing longer summaries *raised* ROUGE but more than tripled false statements: when the model has nothing
left to say, it invents plausible sentences that still share words with the reference. The checkpoint's
built-in default (`min_length=32`, a news-summary length) made this worse, which is why `params.yaml` sets
every generation parameter explicitly.

### Limitations

- **Faithfulness:** roughly 1 in 15 to 1 in 4 summaries contains a false statement, depending on the sample
  (2/30 in the labeled validation set, 3/10 in a test spot-check). Most errors are negation ("Megan will buy milk"
  when she *needn't*) and who-did-what confusions. No decoding setting fixes these. Summaries are machine-written:
  check important details against the chat.
- **Omissions:** about 1 in 4 summaries misses the chat's main point.
- **Small evaluation sets:** 100 test and 30 labeled dialogues show direction, not precision (ROUGE-1 ≈ ±0.02).
- **Input length:** dialogues are truncated at 512 tokens (0.65% of SAMSum dialogues are longer).
- **English only**, and SAMSum's chats were written by linguists, so real chats may differ.

Next steps that target these errors: a shorter warmup (the run spent 500 of 921 steps warming up),
2–3 epochs with early stopping, keeping turn boundaries visible to the model, and comparing BART-large and an
LLM baseline.

## How it works

Five stages, each a component class in `src/textSummarizer/components/` with a thin wrapper in
`src/textSummarizer/pipeline/`. Each stage reads only the previous stage's output folder and writes only to its own.

| # | Stage | Reads | Writes |
|---|---|---|---|
| 1 | Data ingestion | Hugging Face Hub (`knkarthick/samsum`) | `artifacts/data_ingestion/` |
| 2 | Tokenization | `artifacts/data_ingestion/` | `artifacts/data_transformation/` |
| 3 | Training | `artifacts/data_transformation/` | `artifacts/model_trainer/` |
| 4 | Evaluation (ROUGE) | raw test split + trained model | `artifacts/model_evaluation/` |
| 5 | Prediction API | trained model | HTTP (`app.py`) |

**Configuration** is never hardcoded: `config/config.yaml` holds paths and model names, `params.yaml` holds
hyperparameters, and `ConfigurationManager` turns them into one frozen dataclass per stage.

**Smoke mode** (`smoke_test.enabled: true` in `params.yaml`) runs the whole pipeline with `t5-small`, 64 examples
and 5 training steps in about 2 minutes on a CPU, writing to separate `smoke/` folders so it never overwrites the
real model. Full Pegasus training needs a GPU: see `research/colab_train.ipynb`.

## Run it

Requires Python 3.11+. Commands are for Windows PowerShell; on macOS/Linux activate with `source .venv/bin/activate`.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
pytest -q
```

**Pipeline stages:**

```powershell
python main.py                                    # all stages; smoke mode follows params.yaml
python main.py --stage ingestion                  # ingestion | transformation | training | evaluation | all
python main.py --stage training --no-smoke        # full Pegasus training (GPU: use the Colab notebook)
python main.py --stage evaluation --no-smoke      # ROUGE on the test split
python research/compare_models.py                 # base vs fine-tuned on the same test dialogues
python research/decoding_sweep.py                 # compare decoding settings on validation dialogues
```

**Full training on a free GPU:** open `research/colab_train.ipynb` in Colab (Runtime → T4 GPU → Run all).
It clones this repo, trains, evaluates, and saves the model to Google Drive (or a zip to download).

## API

Start the server, then open http://localhost:8080 (web page) or http://localhost:8080/docs (interactive API docs):

```powershell
uvicorn app:app --port 8080
```

| Endpoint | Description |
|---|---|
| `GET /` | Web page: paste a chat, get a summary |
| `GET /health` | `{"status": "ok", "model": "...", "max_input_chars": 6000}` |
| `POST /summarize` | `{"dialogue": "..."}` → `{"summary": "..."}`; `422` for empty or too-long input (over 6,000 characters) |

```bash
curl -X POST http://localhost:8080/summarize \
  -H "Content-Type: application/json" \
  -d '{"dialogue": "Amanda: I baked cookies. Do you want some?\nJerry: Sure!\nAmanda: I will bring you some tomorrow :-)"}'
```

In Windows PowerShell (where `curl` means something else):

```powershell
Invoke-RestMethod -Method Post -Uri http://localhost:8080/summarize -ContentType "application/json" `
  -Body '{"dialogue": "Amanda: I baked cookies. Do you want some?\nJerry: Sure!"}'
```

The model loads once at startup (about 5 seconds); each summary takes about 1–1.5 seconds on a laptop CPU.

## Docker and deployment

The image installs CPU-only PyTorch and only the serving dependencies. It doesn't contain model weights:
set `MODEL_ID` to a Hugging Face Hub repo and the model is downloaded at startup.

**1. Upload the model to the Hub** (once; private by default):

```powershell
hf auth login
python research/push_to_hub.py <your-username>/pegasus-samsum
```

**2. Run the container locally:**

```powershell
docker build -t text-summarizer .
docker run -p 7860:7860 -e MODEL_ID=<your-username>/pegasus-samsum -e HF_TOKEN=<read-token> text-summarizer
```

`HF_TOKEN` is only needed while the model repo is private. Then open http://localhost:7860.

**3. Deploy to a Hugging Face Space** (free CPU hardware):

```powershell
hf repos create <your-username>/dialogue-summarizer --type space --sdk docker
hf upload <your-username>/dialogue-summarizer . . --type space `
  --include Dockerfile --include README.md --include requirements.txt --include pyproject.toml `
  --include params.yaml --include app.py --include "config/*" --include "src/*" --include "static/*" `
  --exclude "*__pycache__*" --exclude "*.egg-info*" --commit-message "Deploy dialogue summarizer"
```

In the Space's **Settings → Variables and secrets**, add the variable `MODEL_ID` = `<your-username>/pegasus-samsum`
and, if the model repo is private, the secret `HF_TOKEN` (a read token). The Space rebuilds and serves the web page
on its URL. Free CPU Spaces are slower than a laptop, so expect a few seconds per summary.

## Project structure

```text
├── app.py                      FastAPI app: /, /health, /summarize
├── main.py                     runs pipeline stages (--stage, --smoke/--no-smoke)
├── config/config.yaml          paths and model names
├── params.yaml                 hyperparameters and decoding settings
├── src/textSummarizer/
│   ├── components/             one class per stage
│   ├── pipeline/               stage wrappers + PredictionPipeline
│   ├── config/configuration.py ConfigurationManager
│   ├── entity/                 frozen config dataclasses
│   └── utils/                  YAML helpers, shared generation code
├── static/index.html           web page (plain HTML/CSS/JS)
├── research/                   EDA, Colab notebook, model comparison, decoding sweep, Hub upload
├── tests/                      pytest suite (fakes and t5-small only: no large models)
└── Dockerfile
```

Tests run on every push via GitHub Actions (`.github/workflows/tests.yml`) on Python 3.11 and 3.12.

## License

- **Code:** written for this project.
- **Pegasus** (`google/pegasus-cnn_dailymail`): Apache-2.0.
- **SAMSum dataset:** [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/), **non-commercial**.
  The fine-tuned model inherits this restriction: don't use it commercially.
