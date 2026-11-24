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
from xcaf_processor import XCAFProcessor
from cad_renderer_cpu import render_step_svg, render_step_section_svg


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = FastAPI(
    title="STEP File Analysis API",
    description=(
        "Open CASCADE API for STEP B-Rep analysis, "
        "XCAF assembly analysis, face identification "
        "and orthographic CAD projection"
    ),
    version="1.4.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


processor = STEPProcessor()
xcaf_processor = XCAFProcessor()


def validate_step_filename(
    filename: Optional[str]
) -> str:

    if not filename:
        raise HTTPException(
            status_code=400,
            detail="No filename provided",
        )

    file_ext = os.path.splitext(
        filename
    )[1].lower()

    if file_ext not in [
        ".step",
        ".stp",
    ]:
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid file type. "
                "Expected .step or .stp, got {}".format(
                    file_ext or "no extension"
                )
            ),
        )

    return file_ext


def safe_unlink(
    file_path: Optional[str]
) -> None:

    if (
        file_path
        and os.path.exists(file_path)
    ):
        try:
            os.unlink(
                file_path
            )

        except Exception as exc:
            logger.warning(
                "Could not delete temporary file %s: %s",
                file_path,
                exc,
            )


def parse_face_ids(
    face_text: str
) -> list:

    if not face_text:
        return []

    return [
        value.strip().upper()
        for value in face_text.split(",")
        if value.strip()
    ]


@app.get("/")
async def root():

    return {
        "status": "healthy",
        "service": "STEP File Analysis API",
        "version": "1.4.0",
        "rendering": (
            "Open CASCADE B-Rep "
            "CPU projection with HLR"
        ),
        "xcaf": (
            "isolated XCAF assembly analysis"
        ),
    }


@app.get("/health")
async def health_check():

    return {
        "status": "healthy",
        "occt_available": (
            processor.is_available()
        ),
        "xcaf_available": (
            xcaf_processor.is_available()
        ),
        "supported_formats": [
            "STEP",
            "STP",
        ],
        "available_operations": [
            "geometry",
            "topology",
            "faces",
            "assembly",
            "references",
            "section",
            "validation",
            "cpu_projection",
            "hidden_line_removal",
        ],
    }


@app.post("/analyze")
async def analyze_step_file(
    file: UploadFile = File(...)
) -> Dict[str, Any]:

    file_ext = validate_step_filename(
        file.filename
    )

    temp_file_path = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=file_ext,
        ) as temp_file:

            content = await file.read()

            temp_file.write(
                content
            )

            temp_file_path = (
                temp_file.name
            )

        logger.info(
            "Processing complete STEP analysis: "
            "%s (%s bytes)",
            file.filename,
            len(content),
        )

        result = processor.analyze_file(
            temp_file_path
        )

        result[
            "file_info"
        ][
            "original_filename"
        ] = file.filename

        return result

    except HTTPException:
        raise

    except Exception as exc:
        logger.error(
            "Complete STEP analysis failed: %s",
            exc,
            exc_info=True,
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Error processing STEP file: "
                "{}".format(exc)
            ),
        )

    finally:
        safe_unlink(
            temp_file_path
        )


@app.post("/analyze/geometry")
async def analyze_geometry_only(
    file: UploadFile = File(...)
) -> Dict[str, Any]:

    file_ext = validate_step_filename(
        file.filename
    )

    temp_file_path = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=file_ext,
        ) as temp_file:

            content = await file.read()

            temp_file.write(
                content
            )

            temp_file_path = (
                temp_file.name
            )

        logger.info(
            "Processing STEP geometry: "
            "%s (%s bytes)",
            file.filename,
            len(content),
        )

        return processor.get_geometric_properties(
            temp_file_path
        )

    except HTTPException:
        raise

    except Exception as exc:
        logger.error(
            "Geometry analysis failed: %s",
            exc,
            exc_info=True,
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Geometry analysis failed: "
                "{}".format(exc)
            ),
        )

    finally:
        safe_unlink(
            temp_file_path
        )


@app.post("/analyze/topology")
async def analyze_topology_only(
    file: UploadFile = File(...)
) -> Dict[str, Any]:

    file_ext = validate_step_filename(
        file.filename
    )

    temp_file_path = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=file_ext,
        ) as temp_file:

            content = await file.read()

            temp_file.write(
                content
            )

            temp_file_path = (
                temp_file.name
            )

        logger.info(
            "Processing STEP topology: "
            "%s (%s bytes)",
            file.filename,
            len(content),
        )

        return processor.get_topology_info(
            temp_file_path
        )

    except HTTPException:
        raise

    except Exception as exc:
        logger.error(
            "Topology analysis failed: %s",
            exc,
            exc_info=True,
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Topology analysis failed: "
                "{}".format(exc)
            ),
        )

    finally:
        safe_unlink(
            temp_file_path
        )


@app.post("/analyze/faces")
async def analyze_faces(
    file: UploadFile = File(...)
) -> Dict[str, Any]:

    file_ext = validate_step_filename(
        file.filename
    )

    temp_file_path = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=file_ext,
        ) as temp_file:

            content = await file.read()

            temp_file.write(
                content
            )

            temp_file_path = (
                temp_file.name
            )

        logger.info(
            "Processing B-Rep faces: "
            "%s (%s bytes)",
            file.filename,
            len(content),
        )

        result = processor.get_faces(
            temp_file_path
        )

        result[
            "original_filename"
        ] = file.filename

        return result

    except HTTPException:
        raise

    except Exception as exc:
        logger.error(
            "B-Rep face analysis failed: %s",
            exc,
            exc_info=True,
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "B-Rep face analysis failed: "
                "{}".format(exc)
            ),
        )

    finally:
        safe_unlink(
            temp_file_path
        )


@app.post("/analyze/assembly")
async def analyze_assembly(
    file: UploadFile = File(...)
) -> Dict[str, Any]:

    file_ext = validate_step_filename(
        file.filename
    )

    temp_file_path = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=file_ext,
        ) as temp_file:

            content = await file.read()

            temp_file.write(
                content
            )

            temp_file_path = (
                temp_file.name
            )

        logger.info(
            "Processing XCAF assembly: "
            "%s (%s bytes)",
            file.filename,
            len(content),
        )

        result = xcaf_processor.analyze_assembly(
            temp_file_path
        )

        result[
            "original_filename"
        ] = file.filename

        return result

    except HTTPException:
        raise

    except Exception as exc:
        logger.error(
            "XCAF assembly analysis failed: %s",
            exc,
            exc_info=True,
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "XCAF assembly analysis failed: "
                "{}".format(exc)
            ),
        )

    finally:
        safe_unlink(
            temp_file_path
        )


@app.post("/analyze/references")
async def analyze_references(file: UploadFile = File(...)) -> Dict[str, Any]:
    file_ext = validate_step_filename(file.filename)
    temp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        logger.info("Processing STEP references: %s (%s bytes)", file.filename, len(content))
        result = xcaf_processor.analyze_references(temp_file_path)
        result["original_filename"] = file.filename
        return result
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("STEP reference analysis failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="STEP reference analysis failed: {}".format(exc))
    finally:
        safe_unlink(temp_file_path)


@app.post("/section-cpu")
async def section_step_cpu(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    origin_x: float = Form(0.0), origin_y: float = Form(0.0), origin_z: float = Form(0.0),
    normal_x: float = Form(0.0), normal_y: float = Form(0.0), normal_z: float = Form(1.0),
    up_x: float = Form(0.0), up_y: float = Form(1.0), up_z: float = Form(0.0),
    fill_color: str = Form("#D9D9D9"), line_color: str = Form("#000000"),
    background_color: str = Form("#FFFFFF"), width: int = Form(1600), height: int = Form(1000),
):
    file_ext = validate_step_filename(file.filename)
    step_path = None
    svg_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_step:
            content = await file.read(); temp_step.write(content); step_path = temp_step.name
        with tempfile.NamedTemporaryFile(delete=False, suffix=".svg") as temp_svg:
            svg_path = temp_svg.name
        render_step_section_svg(
            step_path, svg_path, [origin_x, origin_y, origin_z], [normal_x, normal_y, normal_z],
            [up_x, up_y, up_z], width, height, line_color, fill_color, background_color,
        )
        background_tasks.add_task(safe_unlink, step_path)
        background_tasks.add_task(safe_unlink, svg_path)
        return FileResponse(path=svg_path, media_type="image/svg+xml", filename="step_section.svg", background=background_tasks)
    except HTTPException:
        safe_unlink(step_path); safe_unlink(svg_path); raise
    except Exception as exc:
        logger.error("STEP section failed: %s", exc, exc_info=True)
        safe_unlink(step_path); safe_unlink(svg_path)
        raise HTTPException(status_code=500, detail="STEP section failed: {}".format(exc))


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
    edge_color: str = Form("#000000"),
    background_color: str = Form("#FFFFFF"),

    show_hidden_edges: bool = Form(False),

    hidden_edge_color: str = Form("#808080"),

    width: int = Form(1600),
    height: int = Form(1000),

    linear_deflection: float = Form(1.0),
    angular_deflection: float = Form(0.35),
):

    file_ext = validate_step_filename(
        file.filename
    )

    if (
        width < 200
        or width > 4000
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "Width must be between "
                "200 and 4000 pixels"
            ),
        )

    if (
        height < 200
        or height > 4000
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "Height must be between "
                "200 and 4000 pixels"
            ),
        )

    if (
        linear_deflection <= 0
        or angular_deflection <= 0
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "Mesh deflections must "
                "be greater than zero"
            ),
        )

    step_path = None
    svg_path = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=file_ext,
        ) as temp_step:

            content = await file.read()

            temp_step.write(
                content
            )

            step_path = (
                temp_step.name
            )

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".svg",
        ) as temp_svg:

            svg_path = (
                temp_svg.name
            )

        selected_faces = parse_face_ids(
            highlight_faces
        )

        logger.info(
            "CPU rendering STEP B-Rep: "
            "%s (%s bytes), "
            "view=[%s, %s, %s], "
            "highlighted_faces=%s, "
            "show_hidden_edges=%s",
            file.filename,
            len(content),
            view_x,
            view_y,
            view_z,
            len(selected_faces),
            show_hidden_edges,
        )

        render_step_svg(
            file_path=step_path,
            output_path=svg_path,

            view_direction=[
                view_x,
                view_y,
                view_z,
            ],

            up_direction=[
                up_x,
                up_y,
                up_z,
            ],

            highlight_faces=(
                selected_faces
            ),

            highlight_color=(
                highlight_color
            ),

            base_color=(
                base_color
            ),

            edge_color=(
                edge_color
            ),

            background_color=(
                background_color
            ),

            width=width,
            height=height,

            linear_deflection=(
                linear_deflection
            ),

            angular_deflection=(
                angular_deflection
            ),

            show_hidden_edges=(
                show_hidden_edges
            ),

            hidden_edge_color=(
                hidden_edge_color
            ),
        )

        background_tasks.add_task(
            safe_unlink,
            step_path,
        )

        background_tasks.add_task(
            safe_unlink,
            svg_path,
        )

        return FileResponse(
            path=svg_path,
            media_type="image/svg+xml",
            filename="step_projection.svg",
            background=background_tasks,
        )

    except HTTPException:

        safe_unlink(
            step_path
        )

        safe_unlink(
            svg_path
        )

        raise

    except Exception as exc:

        logger.error(
            "CPU STEP rendering failed: %s",
            exc,
            exc_info=True,
        )

        safe_unlink(
            step_path
        )

        safe_unlink(
            svg_path
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "CPU STEP rendering failed: "
                "{}".format(exc)
            ),
        )


@app.post("/validate")
async def validate_step_file(
    file: UploadFile = File(...)
) -> Dict[str, Any]:

    file_ext = validate_step_filename(
        file.filename
    )

    temp_file_path = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=file_ext,
        ) as temp_file:

            content = await file.read()

            temp_file.write(
                content
            )

            temp_file_path = (
                temp_file.name
            )

        logger.info(
            "Validating STEP: "
            "%s (%s bytes)",
            file.filename,
            len(content),
        )

        return processor.validate_file(
            temp_file_path
        )

    except HTTPException:
        raise

    except Exception as exc:

        logger.error(
            "STEP validation failed: %s",
            exc,
            exc_info=True,
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "STEP validation failed: "
                "{}".format(exc)
            ),
        )

    finally:
        safe_unlink(
            temp_file_path
        )


if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
    )
