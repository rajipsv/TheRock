from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse

from app.config import settings
from app.llm.client import LLMClient
from app.models import CompareResponse
from app.services.github_regulatory import dataset_status, download_dataset, get_pair, list_pairs
from app.services.pipeline import run_comparison, run_comparison_from_github

app = FastAPI(
    title="Document Comparison Agent (TheRock AGENTS_003)",
    description="Compare legacy vs modernized policy PDFs with structural diff, semantic analysis, and regulatory impact.",
    version="0.3.0",
)


@app.get("/health")
async def health():
    llm_ok, llm_message = await LLMClient().ping()
    return {
        "status": "ok",
        "use_llm": settings.use_llm,
        "llm_model": settings.llm_model,
        "llm_base_url": settings.llm_base_url,
        "llm_reachable": llm_ok,
        "llm_message": llm_message,
        "github_dataset": f"https://github.com/{settings.github_repo}",
    }


@app.get("/api/dataset/status")
async def github_dataset_status():
    return dataset_status()


@app.post("/api/dataset/download")
async def github_download_dataset(force: bool = Query(False)):
    try:
        paths = download_dataset(force=force)
        return {
            "status": "success",
            "downloaded": [str(p) for p in paths],
            "count": len(paths),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/dataset/pairs")
async def github_list_pairs():
    pairs = list_pairs()
    return {
        "count": len(pairs),
        "pairs": [p.__dict__ for p in pairs],
    }


@app.get("/api/dataset/pairs/{pair_id}")
async def github_get_pair(pair_id: str):
    try:
        pair = get_pair(pair_id)
        return pair.__dict__
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/compare/dataset", response_model=CompareResponse)
async def compare_dataset_pair(
    pair_id: str = Query(..., description="Policy pair id, e.g. europe-brazil or europe-india"),
):
    try:
        result = await run_comparison_from_github(pair_id)
        return CompareResponse(status="success", result=result)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content=CompareResponse(status="error", message=str(exc)).model_dump(),
        )


@app.post("/api/compare", response_model=CompareResponse)
async def compare_policies(
    legacy_pdf: UploadFile = File(..., description="Legacy policy PDF"),
    modernized_pdf: UploadFile = File(..., description="Modernized policy PDF"),
):
    if not legacy_pdf.filename or not legacy_pdf.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="legacy_pdf must be a PDF file")
    if not modernized_pdf.filename or not modernized_pdf.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="modernized_pdf must be a PDF file")

    legacy_bytes = await legacy_pdf.read()
    modernized_bytes = await modernized_pdf.read()

    if not legacy_bytes or not modernized_bytes:
        raise HTTPException(status_code=400, detail="Both PDF files must be non-empty")

    try:
        result = await run_comparison(
            legacy_pdf.filename,
            legacy_bytes,
            modernized_pdf.filename,
            modernized_bytes,
        )
        return CompareResponse(status="success", result=result)
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content=CompareResponse(status="error", message=str(exc)).model_dump(),
        )


@app.post("/api/compare/paths", response_model=CompareResponse)
async def compare_policy_paths(
    legacy_path: str = Query(..., description="Absolute or relative path to legacy PDF"),
    modernized_path: str = Query(..., description="Absolute or relative path to modernized PDF"),
):
    """Compare two PDFs on disk — useful for CLI and automated tests."""
    leg = Path(legacy_path)
    mod = Path(modernized_path)

    if not leg.is_file() or leg.suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail=f"Invalid legacy path: {legacy_path}")
    if not mod.is_file() or mod.suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail=f"Invalid modernized path: {modernized_path}")

    try:
        result = await run_comparison(leg.name, leg.read_bytes(), mod.name, mod.read_bytes())
        return CompareResponse(status="success", result=result)
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content=CompareResponse(status="error", message=str(exc)).model_dump(),
        )
