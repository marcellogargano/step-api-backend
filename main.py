"""
STEP File Analysis API

FastAPI backend for:
- STEP geometry analysis
- B-Rep face analysis
- topology validation
- CPU orthographic projection from Open CASCADE geometry
"""

import logging
import os
import tempfile
from typing import Any, Dict, Optional

from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from step_processor import STEPProcessor
from cad_renderer_cpu import render_step_svg


# ------------------------------------------------------------------
# LOGGING
# ------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# FASTAPI APPLICATION
# ------------------------------------------------------------------

app = FastAPI(
    title="STEP File Analysis API",
    description=(
        "Open CASCADE API for STEP B-Rep analysis, "
        "face identification and orthographic CAD projection"
    ),
    version="1.1.0",
)


# ------------------------------------------------------------------
# CORS
# ------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------
# STEP PROCESSOR
# ------------------------------------------------------------------

processor = STEPProcessor()


# ------------------------------------------------------------------
# UTILITY FUNCTIONS
# ------------------------------------------------------------------

def validate_step_filename(filename: Optional[str]) -> str:
    """
    Validate a STEP/STP filename and return the file extension.
    """

    if not filename:
        raise HTTPException(
            status_code=400,
            detail="No filename provided",
        )

    file_ext = os.path.splitext(filename)[1].lower()

    if file_ext not in [".step", ".stp"]:
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid file type. "
                f"Expected .step or .stp, got {file_ext or 'no extension'}"
            ),
        )

    return file_ext


def safe_unlink(file_path: Optional[str]) -> None:
    """
    Delete a temporary file without raising an
