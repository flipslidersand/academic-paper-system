"""FastAPI server for academic paper ingestion and retrieval."""

import json
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import httpx
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.staticfiles import StaticFiles

from academic_paper.chunker import chunk_pages
from academic_paper.config import settings
from academic_paper.db import (
    get_all_papers_for_scoring,
    get_connection,
    get_paper,
    get_summary,
    init_db,
    list_papers_filtered,
    list_summaries,
    save_chunks,
    save_paper,
    save_summary,
    search_fts,
    update_paper_score,
    update_paper_status,
)
from academic_paper.embedder import EmbedderClient
from academic_paper.extractor import extract_text, hash_file
from academic_paper.hybrid import rrf_merge
from academic_paper.jobs import job_store
from academic_paper.llm import get_llm_client
from academic_paper.nugget import extract_nuggets, split_sentences
from academic_paper.scorer import compute_score
from academic_paper.summarizer import RAGSummarizer
from academic_paper.telemetry import get_tracer, setup_telemetry
from academic_paper.vector_store import QdrantStore, make_qdrant_id

tracer = get_tracer()


def _parse_list_field(value: str | None) -> list[str] | None:
    """Parse a JSON array string or comma-separated string into a list."""
    if not value:
        return None
    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    except (json.JSONDecodeError, ValueError):
        pass
    return [x.strip() for x in value.split(",") if x.strip()]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database and services on startup."""
    setup_telemetry(app, settings.otel_endpoint)
    init_db(settings.academic_db)
    job_store.init(settings.academic_db)
    app.state.embedder = EmbedderClient()
    app.state.vector_store = QdrantStore()
    llm_client = get_llm_client()
    app.state.llm = llm_client
    if llm_client is not None:
        app.state.summarizer = RAGSummarizer(llm_client, app.state.vector_store)
    else:
        app.state.summarizer = None
    yield


app = FastAPI(title="Academic Paper System", lifespan=lifespan)


@app.post("/papers/ingest")
async def ingest_paper(
    file: UploadFile = File(...),
    title: str | None = Form(None),
    authors: str | None = Form(None),
    categories: str | None = Form(None),
    published_date: str | None = Form(None),
    source: str | None = Form(None),
):
    """Ingest a PDF paper with optional metadata.

    Args:
        file: PDF file to ingest.
        title: Paper title (optional; overrides extracted title).
        authors: Authors as JSON array or comma-separated string (optional).
        categories: Categories as JSON array or comma-separated string (optional).
        published_date: Publication date string YYYY-MM-DD (optional).
        source: Source identifier e.g. 'arxiv' (optional).

    Returns:
        JSON with paper_id, file_name, chunks, status.

    Raises:
        HTTPException 409: File already ingested.
        HTTPException 400: Extraction / chunking / embedding error.
    """
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        file_hash = hash_file(tmp_path)
        conn = get_connection(settings.academic_db)

        cursor = conn.cursor()
        cursor.execute("SELECT id FROM papers WHERE file_hash = ?", (file_hash,))
        if cursor.fetchone():
            conn.close()
            raise HTTPException(status_code=409, detail="File already ingested")

        with tracer.start_as_current_span("pdf.extract"):
            pages = extract_text(tmp_path)
        if not pages:
            conn.close()
            raise HTTPException(status_code=400, detail="No text extracted from PDF")

        chunks_list = chunk_pages(pages, chunk_size=settings.chunk_size, overlap=settings.chunk_overlap)
        if not chunks_list:
            conn.close()
            raise HTTPException(status_code=400, detail="No chunks generated")

        paper_id = save_paper(
            conn,
            file.filename or "unknown.pdf",
            file_hash,
            title=title,
            authors=_parse_list_field(authors),
            categories=_parse_list_field(categories),
            published_date=published_date or None,
            source=source or None,
        )

        try:
            chunk_texts = [chunk["text"] for chunk in chunks_list]
            with tracer.start_as_current_span("embed.batch"):
                embeddings = await app.state.embedder.embed(chunk_texts, mode="index")

            app.state.vector_store.ensure_collection()

            points = []
            for idx, (chunk, embedding) in enumerate(zip(chunks_list, embeddings)):
                qdrant_id = make_qdrant_id(file_hash, idx)
                chunk["qdrant_id"] = qdrant_id
                points.append({
                    "id": qdrant_id,
                    "vector": embedding,
                    "payload": {
                        "paper_id": paper_id,
                        "chunk_index": idx,
                        "text": chunk["text"],
                        "file_name": file.filename or "unknown.pdf",
                    },
                })

            with tracer.start_as_current_span("qdrant.upsert"):
                app.state.vector_store.upsert(points)

            save_chunks(conn, paper_id, chunks_list)
            update_paper_status(conn, paper_id, "indexed")
            conn.close()

            return {
                "paper_id": paper_id,
                "file_name": file.filename or "unknown.pdf",
                "chunks": len(chunks_list),
                "status": "indexed",
            }

        except Exception as e:
            update_paper_status(conn, paper_id, "failed")
            conn.close()
            raise HTTPException(status_code=400, detail=f"Embedding or Qdrant error: {str(e)}")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/papers")
def list_papers_endpoint(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    author: str | None = Query(None, description="Filter by author name (substring match)"),
    category: str | None = Query(None, description="Filter by category code e.g. cs.AI"),
    sort: Literal["ingested_at", "score"] = Query("ingested_at", description="Sort order"),
):
    """List papers with pagination, optional filters, and sort."""
    conn = get_connection(settings.academic_db)
    total, papers = list_papers_filtered(
        conn, limit=limit, offset=offset, author=author, category=category, sort=sort
    )
    conn.close()
    return {"total": total, "papers": papers}


@app.get("/papers/{paper_id}")
def get_paper_endpoint(paper_id: int):
    """Get paper details by ID."""
    conn = get_connection(settings.academic_db)
    paper = get_paper(conn, paper_id)
    conn.close()
    if paper is None:
        raise HTTPException(status_code=404, detail="Paper not found")
    return paper


@app.get("/papers/{paper_id}/summary")
async def get_summary_endpoint(paper_id: int, force: bool = Query(False)):
    """Get summary of a paper (cached unless force=true)."""
    conn = get_connection(settings.academic_db)
    paper = get_paper(conn, paper_id)

    if paper is None:
        conn.close()
        raise HTTPException(status_code=404, detail="Paper not found")

    if not force:
        cached_summary = get_summary(conn, paper_id)
        if cached_summary is not None:
            conn.close()
            return {
                "paper_id": paper_id,
                "model": cached_summary["model"],
                "objective": cached_summary["objective"],
                "method": cached_summary["method"],
                "results": cached_summary["results"],
                "limitations": cached_summary["limitations"],
                "keywords": cached_summary["keywords"],
                "cached": True,
            }

    if app.state.llm is None:
        conn.close()
        raise HTTPException(status_code=503, detail="LLM not configured")

    if app.state.summarizer is None:
        conn.close()
        raise HTTPException(status_code=503, detail="Summarizer not initialized")

    try:
        with tracer.start_as_current_span("summarize") as span:
            span.set_attribute("paper_id", paper_id)
            summary = await app.state.summarizer.summarize(paper_id, paper["file_hash"])

        llm_class_name = app.state.llm.__class__.__name__
        if llm_class_name == "GeminiClient":
            model = "gemini-2.0-flash"
        elif llm_class_name == "OllamaClient":
            model = f"ollama/{app.state.llm.model}"
        else:
            model = llm_class_name

        save_summary(conn, paper_id, model, summary)
        conn.close()

        return {
            "paper_id": paper_id,
            "model": model,
            "objective": summary.get("objective", ""),
            "method": summary.get("method", ""),
            "results": summary.get("results", ""),
            "limitations": summary.get("limitations", ""),
            "keywords": summary.get("keywords", []),
            "cached": False,
        }

    except Exception as e:
        conn.close()
        raise HTTPException(status_code=400, detail=f"Summarization error: {str(e)}")


@app.post("/papers/score-all")
def score_all_papers():
    """Compute and store relevance scores for all papers.

    Score = freshness (30-day half-life, 0–0.5) + category match (0–0.5).
    Preferred categories are configured via PREFERRED_CATEGORIES env var.

    Returns:
        JSON with total and scored counts.
    """
    preferred = settings.preferred_categories_list
    conn = get_connection(settings.academic_db)
    papers = get_all_papers_for_scoring(conn)
    scored = 0
    for paper in papers:
        score = compute_score(paper, preferred)
        update_paper_score(conn, paper["id"], score)
        scored += 1
    conn.close()
    return {"total": len(papers), "scored": scored, "preferred_categories": preferred}


@app.post("/papers/{paper_id}/score")
def score_paper(paper_id: int):
    """Compute and store relevance score for a single paper."""
    conn = get_connection(settings.academic_db)
    paper = get_paper(conn, paper_id)
    if paper is None:
        conn.close()
        raise HTTPException(status_code=404, detail="Paper not found")
    preferred = settings.preferred_categories_list
    score = compute_score(paper, preferred)
    update_paper_score(conn, paper_id, score)
    conn.close()
    return {"paper_id": paper_id, "score": score}


@app.get("/summaries")
def list_summaries_endpoint(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    """List all paper summaries with associated paper metadata."""
    conn = get_connection(settings.academic_db)
    total, summaries = list_summaries(conn, limit=limit, offset=offset)
    conn.close()
    return {"total": total, "summaries": summaries}


async def _run_summarize_all(job_id: str) -> None:
    """Background task: summarize all indexed papers without a cached summary."""
    job = job_store.get(job_id)
    if job is None:
        return

    job.status = "running"
    job_store.persist(job)
    try:
        conn = get_connection(settings.academic_db)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT p.id, p.file_hash
            FROM papers p
            LEFT JOIN summaries s ON p.id = s.paper_id
            WHERE s.paper_id IS NULL AND p.status = 'indexed'
        """)
        rows = cursor.fetchall()
        conn.close()

        job.total = len(rows)

        llm_class_name = app.state.llm.__class__.__name__
        if llm_class_name == "GeminiClient":
            model = "gemini-2.0-flash"
        elif llm_class_name == "OllamaClient":
            model = f"ollama/{app.state.llm.model}"
        else:
            model = llm_class_name

        for row in rows:
            paper_id = row[0]
            file_hash = row[1]
            try:
                summary = await app.state.summarizer.summarize(paper_id, file_hash)
                conn = get_connection(settings.academic_db)
                save_summary(conn, paper_id, model, summary)
                conn.close()
                job.processed += 1
            except Exception as e:
                job.failed += 1
                job.errors.append(f"paper_id={paper_id}: {e}")

        job.status = "done"
    except Exception as e:
        job.status = "failed"
        job.errors.append(str(e))
    finally:
        job.finished_at = time.time()
        job_store.persist(job)


@app.post("/jobs/summarize-all", status_code=202)
async def start_summarize_all(background_tasks: BackgroundTasks):
    """Start a background job to summarize all papers without a cached summary."""
    if job_store.has_running():
        raise HTTPException(status_code=409, detail="A summarize-all job is already running")
    if app.state.llm is None or app.state.summarizer is None:
        raise HTTPException(status_code=503, detail="LLM not configured")

    job = job_store.create()
    background_tasks.add_task(_run_summarize_all, job.id)
    return {"job_id": job.id, "status": job.status}


@app.get("/jobs/{job_id}")
def get_job_endpoint(job_id: str):
    """Get status of a background job by ID."""
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job.to_dict()


@app.get("/jobs")
def list_jobs_endpoint():
    """List all background jobs."""
    return {"jobs": [j.to_dict() for j in job_store.list_all()]}


@app.get("/search")
async def search(
    q: str = Query(..., min_length=1),
    mode: str = Query("hybrid", pattern="^(vector|keyword|hybrid|nugget)$"),
    limit: int = Query(10, ge=1, le=100),
    paper_id: int | None = Query(None),
    nuggets_per_chunk: int = Query(3, ge=1, le=10, description="Sentences per chunk (nugget mode only)"),
    nugget_embed_weight: float = Query(
        0.7,
        ge=0.0,
        le=1.0,
        description="Embedding weight in nugget hybrid scoring: 0=BM25-only, 1=embed-only "
        "(nugget mode only; nugget-rag-eval #11 found 0.7 optimal)",
    ),
):
    """Search papers using vector, keyword, hybrid, or nugget mode.

    nugget mode: runs hybrid search then extracts the top-N most query-relevant
    sentences from each chunk instead of returning the full snippet. Reduces
    context length by ~68% while maintaining Recall@5.
    """
    try:
        conn = get_connection(settings.academic_db)
        cursor = conn.cursor()

        if mode == "keyword":
            fts_results = search_fts(conn, query=q, limit=limit, paper_id=paper_id)
            results = []
            for rank, result in enumerate(fts_results, start=1):
                chunk_id = result["chunk_id"]
                paper_id_res = result["paper_id"]
                cursor.execute("SELECT page_start FROM chunks WHERE id = ?", (chunk_id,))
                row = cursor.fetchone()
                page_start = row["page_start"] if row else None
                results.append({
                    "rank": rank,
                    "score": result["rank"],
                    "paper_id": paper_id_res,
                    "chunk_index": result.get("chunk_index", 0),
                    "page_start": page_start,
                    "snippet": result["text"][:200],
                })
            conn.close()
            return {"mode": mode, "query": q, "results": results}

        elif mode == "vector":
            with tracer.start_as_current_span("embed.query"):
                query_vector = await app.state.embedder.embed_single(q, mode="search")
            search_results = app.state.vector_store.search(
                query_vector=query_vector, limit=limit, paper_id_filter=paper_id
            )
            results = []
            for rank, result in enumerate(search_results, start=1):
                qdrant_id = result["id"]
                payload = result["payload"]
                cursor.execute("SELECT page_start FROM chunks WHERE qdrant_id = ?", (qdrant_id,))
                row = cursor.fetchone()
                results.append({
                    "rank": rank,
                    "score": result["score"],
                    "paper_id": payload["paper_id"],
                    "chunk_index": payload["chunk_index"],
                    "page_start": row["page_start"] if row else None,
                    "snippet": payload["text"][:200],
                })
            conn.close()
            return {"mode": mode, "query": q, "results": results}

        else:  # hybrid or nugget (same retrieval, different snippet)
            fts_results = search_fts(conn, query=q, limit=limit, paper_id=paper_id)
            for fts_result in fts_results:
                cursor.execute("SELECT chunk_index FROM chunks WHERE id = ?", (fts_result["chunk_id"],))
                row = cursor.fetchone()
                if row:
                    fts_result["chunk_index"] = row["chunk_index"]

            with tracer.start_as_current_span("embed.query"):
                query_vector = await app.state.embedder.embed_single(q, mode="search")
            vector_results = app.state.vector_store.search(
                query_vector=query_vector, limit=limit, paper_id_filter=paper_id
            )

            for vec_result in vector_results:
                if "chunk_id" not in vec_result["payload"]:
                    cursor.execute("SELECT id FROM chunks WHERE qdrant_id = ?", (vec_result["id"],))
                    row = cursor.fetchone()
                    if row:
                        vec_result["payload"]["chunk_id"] = row["id"]

            merged = rrf_merge(fts_results, vector_results)
            results = []
            for rank, result in enumerate(merged[:limit], start=1):
                cursor.execute("SELECT page_start FROM chunks WHERE id = ?", (result["chunk_id"],))
                row = cursor.fetchone()
                full_text = result["text"]
                if mode == "nugget":
                    sentence_vecs = None
                    if nugget_embed_weight > 0.0:
                        sentences = split_sentences(full_text)
                        if sentences:
                            with tracer.start_as_current_span("embed.nuggets"):
                                sentence_vecs = await app.state.embedder.embed(sentences, mode="search")
                    snippet = extract_nuggets(
                        q,
                        full_text,
                        top_k=nuggets_per_chunk,
                        embed_weight=nugget_embed_weight,
                        query_vec=query_vector,
                        sentence_vecs=sentence_vecs,
                    )
                else:
                    snippet = full_text[:200]
                results.append({
                    "rank": rank,
                    "score": result["rrf_score"],
                    "paper_id": result["paper_id"],
                    "chunk_index": result["chunk_index"],
                    "page_start": row["page_start"] if row else None,
                    "snippet": snippet,
                })
            conn.close()
            return {"mode": mode, "query": q, "results": results}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Search error: {str(e)}")


@app.get("/health")
async def health():
    """Qdrant / embedding-svc 痎通チェック"""
    status = {"qdrant": "ok", "embedding_svc": "ok"}
    overall = "ok"

    try:
        app.state.vector_store.client.get_collections()
    except Exception:
        status["qdrant"] = "error"
        overall = "degraded"

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(f"{settings.embedding_svc_url}/health")
            if resp.status_code >= 500:
                raise Exception()
    except Exception:
        status["embedding_svc"] = "error"
        overall = "degraded"

    return {"status": overall, **status}


@app.get("/stats")
def stats():
    """DB統計情報"""
    conn = get_connection(settings.academic_db)
    try:
        papers = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
        chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        try:
            info = app.state.vector_store.client.get_collection(settings.qdrant_collection)
            qdrant_points = info.points_count
        except Exception:
            qdrant_points = -1
        return {"papers": papers, "chunks": chunks, "qdrant_points": qdrant_points, "db": settings.academic_db}
    finally:
        conn.close()


# Serve frontend at /ui — must be mounted after all API routes
_frontend_dir = Path(__file__).parent.parent / "frontend"
if _frontend_dir.exists():
    app.mount("/ui", StaticFiles(directory=str(_frontend_dir), html=True), name="ui")
