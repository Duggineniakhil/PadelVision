"""
app/main.py — FastAPI application entry point

Run with:
    uvicorn app.main:app --reload --port 8000

From inside the backend/ directory.
"""

import os
import sys
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from jobs.store import init_db
from routes.upload import router as upload_router
from routes.analysis import router as analysis_router
from routes.video import router as health_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ── Directories ───────────────────────────────────────────────────────────────
BACKEND_DIR = os.path.dirname(os.path.dirname(__file__))
OUTPUTS_DIR = os.path.join(BACKEND_DIR, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)


# ── Lifespan (replaces deprecated @app.on_event) ──────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initialising SQLite job store...")
    init_db()
    logger.info("PadelVision API ready.")
    yield
    logger.info("PadelVision API shutting down.")


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="PadelVision",
    description="AI-powered padel match video analysis API",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS (allow Next.js dev server) ───────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static file serving ────────────────────────────────────────────────────────
# Serves processed videos and images at /outputs/{job_id}/...
app.mount("/outputs", StaticFiles(directory=OUTPUTS_DIR), name="outputs")

# ── Routes ────────────────────────────────────────────────────────────────────
app.include_router(upload_router, prefix="/api", tags=["Upload"])
app.include_router(analysis_router, prefix="/api", tags=["Analysis"])
app.include_router(health_router, prefix="/api", tags=["Health"])


@app.get("/")
def root():
    return {
        "name": "PadelVision API",
        "version": "1.0.0",
        "docs": "/docs",
    }
