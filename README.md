# Gemma 4 Vision Studio

> Built autonomously by [NEO](https://heyneo.com) — your fully autonomous AI coding agent. &nbsp; [![NEO for VS Code](https://img.shields.io/badge/VS%20Code-NEO%20Extension-5C2D91?logo=visual-studio-code&logoColor=white)](https://marketplace.visualstudio.com/items?itemName=NeoResearchInc.heyneo)

A web application that combines four vision AI capabilities in one interface — image analysis, object detection, screenshot-to-HTML conversion, and a structured element pipeline — all powered by Gemma 4.

---

## What It Does

```
Upload Image  ──►  Choose Mode  ──►  Get Results
                        │
         ┌──────────────┼──────────────┬──────────────┐
         ▼              ▼              ▼              ▼
   Analyze Image    Detect Objects  Screenshot      Pipeline
   (Gemma 4)        (DETR model)    → HTML          (structured
                                    (Gemma 4)        JSON output)
```

### Four Modes

| Tab | What it does |
|-----|-------------|
| **Analyze** | Describe an image in natural language, answer questions about it |
| **Detect** | Run Facebook's DETR model locally — returns bounding boxes + labels drawn on the image |
| **Screenshot → HTML** | Upload any UI screenshot and get clean, semantic HTML/CSS back |
| **Pipeline** | Structured analysis: description, detected UI elements with types/confidence, categories, extracted text, and optional Q&A |

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `torch` and `transformers` are included for the DETR object detection model. If you only need the Gemma vision features (no local DETR), you can skip those two packages — the app will automatically fall back to Gemma for object detection.

### 2. Set your API key

```bash
cp .env.example .env
# Open .env and add your key (see options below)
```

#### Option A — OpenRouter (recommended)

OpenRouter provides access to Gemma and 100+ other models through a single OpenAI-compatible API. Sign up at **https://openrouter.ai** and add your key:

```env
OPENROUTER_API_KEY=sk-or-...
```

The default model is `google/gemma-4-31b-it` (31B, full vision, 262K context). Override with:

```env
OPENROUTER_MODEL=google/gemma-4-31b-it
```

#### Option B — Google AI Studio

Get a free key at **https://aistudio.google.com/app/apikey**:

```env
GEMINI_API_KEY=AIza...
```

#### Option C — Mock mode (no key needed)

If neither key is set, the app starts in **mock mode** — all endpoints return realistic demo responses so you can build and test the UI without any API account.

#### Option D — Local inference via Ollama

Install [Ollama](https://ollama.com) and pull Gemma 4:

```bash
ollama run gemma4
```

Then point the app at your local Ollama endpoint:

```env
OPENROUTER_API_KEY=ollama
OPENROUTER_BASE_URL=http://localhost:11434/v1
OPENROUTER_MODEL=gemma4
```

### 3. Start the server

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Open **http://localhost:8000** in your browser.

---

## Architecture

```
Browser (index.html)
    │  drag-drop upload, 4-tab UI
    │
    ▼
FastAPI (app.py) ─── /analyze            ──► GemmaClient → OpenRouter or Google AI
                 ─── /segment            ──► ImageSegmenter (DETR, local)
                 ─── /screenshot-to-html ──► ScreenshotToHTML → GemmaClient
                 ─── /pipeline           ──► PipelineProcessor → GemmaClient
                 ─── /health
                 ─── / (serves UI)
```

---

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENROUTER_API_KEY` | *(empty)* | OpenRouter key — **recommended provider** |
| `OPENROUTER_MODEL` | `google/gemma-4-31b-it` | Model to use via OpenRouter |
| `GEMINI_API_KEY` | *(empty)* | Google AI Studio key — alternative provider |
| `GEMINI_MODEL` | `gemma-4-31b-it` | Model to use via Google AI |
| `MAX_IMAGE_SIZE_MB` | `10` | Maximum upload size |
| `MOCK_MODE` | auto | `True` when no API key is set |

If both `OPENROUTER_API_KEY` and `GEMINI_API_KEY` are set, OpenRouter is used.

---

## API Endpoints

| Endpoint | Method | Body | Description |
|----------|--------|------|-------------|
| `GET /` | GET | — | Serves the web UI |
| `GET /health` | GET | — | Health check + mock mode status |
| `POST /analyze` | POST | `file` (image), `prompt` (optional) | Analyse image with Gemma |
| `POST /segment` | POST | `file` (image) | Detect objects with DETR (Gemma fallback) |
| `POST /screenshot-to-html` | POST | `file` (image) | Generate HTML from screenshot |
| `POST /pipeline` | POST | `file` (image), `question` (optional) | Structured JSON analysis + Q&A |

---

## Project Structure

```
gemma4-vision-studio/
├── app.py                # FastAPI app + all endpoints
├── gemma_client.py       # OpenRouter / Google AI client with mock fallback
├── segmenter.py          # DETR object detection (local, no API needed)
├── screenshot_to_html.py # Screenshot → HTML converter
├── pipeline_processor.py # Structured analysis pipeline
├── config.py             # Settings and environment variables
├── requirements.txt
├── .env.example          # API key template — copy to .env
├── static/
│   └── index.html        # Dark-themed 4-tab frontend
└── tests/
    └── test_app.py       # 19 pytest tests (all endpoints mocked)
```

---

## Running Tests

```bash
pytest tests/ -v
```

19 tests covering all endpoints and core classes. No API key or GPU required — all external calls are mocked.

---

## License

MIT
