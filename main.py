"""
STEP File Analysis API

FastAPI backend for STEP B-Rep analysis and CPU orthographic projection.
"""

import logging
import os
import tempfile
from typing import Any, Dict, Optional

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from step_processor import STEPProcessor
from cad_renderer_cpu import render_step_svg

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="STEP File Analysis API",
    description="Open CASCADE API for STEP B-Rep analysis, face identification and orthographic CAD projection",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

processor = STEPProcessor()


def validate_step_filename(filename: Optional[str]) -> str:
    if not filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    file_ext = os.path.splitext(filename)[1].lower()
    if file_ext not in [".step", ".stp"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Expected .step or .stp, got {file_ext or 'no extension'}",
        )
    return file_ext


def safe_unlink(file_path: Optional[str]) -> None:
    if file_path and os.path.exists(file_path):
        try:
            os.unlink(file_path)
        except Exception as exc:
            logger.warning(f"Could not delete temporary file {file_path}: {exc}")


def parse_face_ids(face_text: str) -> list:
    if not face_text:
        return []
    return [value.strip().upper() for value in face_text.split(",") if value.strip()]


@app.get("/")
async def root():
    return {
        "status": "healthy",
        "service": "STEP File Analysis API",
        "version": "1.1.0",
        "rendering": "Open CASCADE B-Rep CPU projection",
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "occt_available": processor.is_available(),
        "supported_formats": ["STEP", "STP"],
        "available_operations": ["geometry", "topology", "faces", "validation", "cpu_projection"],
    }


@app.post("/analyze")
async def analyze_step_file(file: UploadFile = File(...)) -> Dict[str, Any]:
    file_ext = validate_step_filename(file.filename)
    temp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        logger.info(f"Processing complete STEP analysis: {file.filename} ({len(content)} bytes)")
        result = processor.analyze_file(temp_file_path)
        result["file_info"]["original_filename"] = file.filename
        return result
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Complete STEP analysis failed: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing STEP file: {exc}")
    finally:
        safe_unlink(temp_file_path)


@app.post("/analyze/geometry")
async def analyze_geometry_only(file: UploadFile = File(...)) -> Dict[str, Any]:
    file_ext = validate_step_filename(file.filename)
    temp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        logger.info(f"Processing STEP geometry: {file.filename} ({len(content)} bytes)")
        return processor.get_geometric_properties(temp_file_path)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Geometry analysis failed: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Geometry analysis failed: {exc}")
    finally:
        safe_unlink(temp_file_path)


@app.post("/analyze/topology")
async def analyze_topology_only(file: UploadFile = File(...)) -> Dict[str, Any]:
    file_ext = validate_step_filename(file.filename)
    temp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        logger.info(f"Processing STEP topology: {file.filename} ({len(content)} bytes)")
        return processor.get_topology_info(temp_file_path)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Topology analysis failed: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Topology analysis failed: {exc}")
    finally:
        safe_unlink(temp_file_path)


@app.post("/analyze/faces")
async def analyze_faces(file: UploadFile = File(...)) -> Dict[str, Any]:
    file_ext = validate_step_filename(file.filename)
    temp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        logger.info(f"Processing B-Rep faces: {file.filename} ({len(content)} bytes)")
        result = processor.get_faces(temp_file_path)
        result["original_filename"] = file.filename
        return result
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"B-Rep face analysis failed: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"B-Rep face analysis failed: {exc}")
    finally:
        safe_unlink(temp_file_path)


@app.post("/render-cpu")
async def render_step_cpu(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    view_x: float = Form(0.0),
    view_y: float = Form(0.0),
    view_z: float = Form(-1.0),
    up_x: float = Form(0.0),
    up_y: float = Form(1.0),
    up_z: float = Form(0.0),
    highlight_faces: str = Form(""),
    highlight_color: str = Form("#FF0000"),
    base_color: str = Form("#BFC5CC"),
    edge_color: str = Form("#202020"),
    background_color: str = Form("#FFFFFF"),
    width: int = Form(1600),
    height: int = Form(1000),
    linear_deflection: float = Form(1.0),
    angular_deflection: float = Form(0.35),
):
    file_ext = validate_step_filename(file.filename)
    if width < 200 or width > 4000:
        raise HTTPException(status_code=400, detail="Width must be between 200 and 4000 pixels")
    if height < 200 or height > 4000:
        raise HTTPException(status_code=400, detail="Height must be between 200 and 4000 pixels")
    if linear_deflection <= 0 or angular_deflection <= 0:
        raise HTTPException(status_code=400, detail="Mesh deflections must be greater than zero")

    step_path = None
    svg_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_step:
            content = await file.read()
            temp_step.write(content)
            step_path = temp_step.name
        with tempfile.NamedTemporaryFile(delete=False, suffix=".svg") as temp_svg:
            svg_path = temp_svg.name

        selected_faces = parse_face_ids(highlight_faces)
        logger.info(
            f"CPU rendering STEP B-Rep: {file.filename} ({len(content)} bytes), "
            f"view=[{view_x}, {view_y}, {view_z}], highlighted_faces={len(selected_faces)}"
        )

        render_step_svg(
            file_path=step_path,
            output_path=svg_path,
            view_direction=[view_x, view_y, view_z],
            up_direction=[up_x, up_y, up_z],
            highlight_faces=selected_faces,
            highlight_color=highlight_color,
            base_color=base_color,
            edge_color=edge_color,
            background_color=background_color,
            width=width,
            height=height,
            linear_deflection=linear_deflection,
            angular_deflection=angular_deflection,
        )

        background_tasks.add_task(safe_unlink, step_path)
        background_tasks.add_task(safe_unlink, svg_path)
        return FileResponse(
            path=svg_path,
            media_type="image/svg+xml",
            filename="step_projection.svg",
            background=background_tasks,
        )
    except HTTPException:
        safe_unlink(step_path)
        safe_unlink(svg_path)
        raise
    except Exception as exc:
        logger.error(f"CPU STEP rendering failed: {exc}", exc_info=True)
        safe_unlink(step_path)
        safe_unlink(svg_path)
        raise HTTPException(status_code=500, detail=f"CPU STEP rendering failed: {exc}")


@app.post("/validate")
async def validate_step_file(file: UploadFile = File(...)) -> Dict[str, Any]:
    file_ext = validate_step_filename(file.filename)
    temp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        logger.info(f"Validating STEP: {file.filename} ({len(content)} bytes)")
        return processor.validate_file(temp_file_path)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"STEP validation failed: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"STEP validation failed: {exc}")
    finally:
        safe_unlink(temp_file_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

