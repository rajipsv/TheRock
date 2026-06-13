# Document Comparison Agent (AGENTS_003)

Policy & document comparison for legacy vs modernized PDFs: structural diff, semantic analysis, and regulatory impact summaries.

Originally built for the AMD TCS Hackathon; lives under TheRock `agents/` alongside change-impact and log-analysis.

## Pipeline

```
legacy PDF + modernized PDF
  → Parser (pdfplumber text + tables)
  → Section alignment (title match, then content similarity)
  → Structural diff (add/remove/modify per aligned pair)
  → LLM chunk diff fallback (low alignment / unstructured docs)
  → Semantic comparison (LLM or rule-based fallback)
  → Regulatory impact (executive summary + impact items)
  → JSON response (+ format_warnings, alignment_score)
```

## Setup

```powershell
cd TheRock
pip install -r agents/document-comparison-agent/requirements.txt
copy agents\document-comparison-agent\.env.example agents\document-comparison-agent\.env
```

Optional notebook extras: `pip install -r agents/document-comparison-agent/requirements-notebook.txt`

Generate local sample PDFs:

```powershell
python agents/document-comparison-agent/scripts/make_sample_pdfs.py
python agents/document-comparison-agent/scripts/make_mismatch_sample_pdfs.py
```

Download GitHub GDPR dataset (MIT):

```powershell
python agents/document-comparison-agent/scripts/download_github_dataset.py
```

## CLI (no server)

```powershell
python agents/document-comparison-agent/scripts/compare_cli.py `
  agents/document-comparison-agent/data/samples/legacy_policy.pdf `
  agents/document-comparison-agent/data/samples/modernized_policy.pdf `
  --pretty -o agents/document-comparison-agent/out/result.json

python agents/document-comparison-agent/scripts/compare_github_cli.py --download --list-pairs
python agents/document-comparison-agent/scripts/compare_github_cli.py --pair-id europe-brazil --pretty
```

## API server

vLLM on port **8000**; API on **8080**:

```powershell
cd agents/document-comparison-agent
uvicorn app.main:app --reload --host 0.0.0.0 --port 8080
```

OpenAPI: http://localhost:8080/docs

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Service status and LLM config |
| `POST` | `/api/compare` | Upload `legacy_pdf` + `modernized_pdf` |
| `POST` | `/api/compare/paths` | Compare PDFs by file path |
| `GET` | `/api/dataset/status` | GitHub dataset cache status |
| `POST` | `/api/dataset/download` | Download PDFs from GitHub |
| `GET` | `/api/dataset/pairs` | List policy comparison pairs |
| `POST` | `/api/compare/dataset` | Compare a GitHub pair by `pair_id` |

## vLLM (optional, MI300)

```bash
VLLM_USE_TRITON_FLASH_ATTN=0 \
vllm serve Qwen/Qwen3-30B-A3B \
  --served-model-name Qwen3-30B-A3B \
  --api-key abc-123 \
  --port 8000 \
  --trust-remote-code
```

Set `USE_LLM=false` for rule-based fallback when vLLM is unavailable.

## Notebook

`notebook/document_comparison_demo.ipynb` — vLLM verify, Pydantic AI smoke test, comparison walkthrough.

## Tests

```powershell
cd agents/document-comparison-agent
python -m pytest tests/ -q
```

## Layout

```
agents/document-comparison-agent/
  app/                 # FastAPI app + pipeline services
  scripts/             # CLI + sample PDF generators
  tests/
  data/samples/        # generated demo PDFs
  data/github/         # downloaded dataset cache (gitignored)
  notebook/
  out/                 # CLI JSON output (gitignored)
```
