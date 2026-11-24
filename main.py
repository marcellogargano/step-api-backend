"""
STEP File Analysis API
FastAPI backend for processing STEP files and extracting geometric and topology data
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi import BackgroundTasks
import tempfile
import os
from typing import Dict, Any
import logging

from step_processor import STEPProcessor
from cad_renderer import render_step_view

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="STEP File Analysis API",
    description="API for analyzing STEP files and extracting geometric, topology, and metadata",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize STEP processor
processor = STEPProcessor()


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "STEP File Analysis API",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "occt_available": processor.is_available(),
        "supported_formats": ["STEP", "STP"]
    }


@app.post("/analyze")
async def analyze_step_file(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Analyze a STEP file and return comprehensive data
    
    Args:
        file: Uploaded STEP file (.step or .stp)
    
    Returns:
        JSON with geometric properties, topology, metadata, and validation results
    """
    # Validate file extension
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in ['.step', '.stp']:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Expected .step or .stp, got {file_ext}"
        )
    
    # Create temporary file to save upload
    temp_file = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        logger.info(f"Processing file: {file.filename} ({len(content)} bytes)")
        
        # Process the STEP file
        result = processor.analyze_file(temp_file_path)
        
        # Add original filename to result
        result["file_info"]["original_filename"] = file.filename
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing STEP file: {str(e)}"
        )
    
    finally:
        # Clean up temporary file
        if temp_file and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                logger.warning(f"Failed to delete temp file: {e}")


@app.post("/analyze/geometry")
async def analyze_geometry_only(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Extract only geometric properties from STEP file
    
    Args:
        file: Uploaded STEP file
    
    Returns:
        Geometric properties (volume, surface area, bounding box)
    """
    # Validate file
    if not file.filename or not file.filename.lower().endswith(('.step', '.stp')):
        raise HTTPException(status_code=400, detail="Invalid STEP file")
    
    temp_file = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.step') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        result = processor.get_geometric_properties(temp_file_path)
        return result
        
    except Exception as e:
        logger.error(f"Error processing geometry: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        if temp_file and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

@app.post("/analyze/topology")
async def analyze_topology_only(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Extract only topology information from STEP file
    
    Args:
        file: Uploaded STEP file
    
    Returns:
        Topology counts (solids, shells, faces, edges, vertices)
    """
    if not file.filename or not file.filename.lower().endswith(('.step', '.stp')):
        raise HTTPException(status_code=400, detail="Invalid STEP file")
    
    temp_file = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.step') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        result = processor.get_topology_info(temp_file_path)
        return result
        
    except Exception as e:
        logger.error(f"Error processing topology: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        if temp_file and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

@app.post("/analyze/faces")
async def analyze_faces(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Extract all B-Rep faces from STEP file.

    Returns exact Open CASCADE face data including:
    face ID, surface type, area, centroid and bounding box.
    """

    if not file.filename or not file.filename.lower().endswith(('.step', '.stp')):
        raise HTTPException(status_code=400, detail="Invalid STEP file")

    temp_file = None
    temp_file_path = None

    try:
        file_ext = os.path.splitext(file.filename)[1].lower()

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=file_ext
        ) as temp_file:

            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        logger.info(
            f"Processing faces: {file.filename} ({len(content)} bytes)"
        )

        result = processor.get_faces(temp_file_path)

        result["original_filename"] = file.filename

        return result

    except Exception as e:
        logger.error(
            f"Error processing faces: {str(e)}",
            exc_info=True
        )

        raise HTTPException(
            status_code=500,
            detail=f"Error processing STEP faces: {str(e)}"
        )

    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                logger.warning(
                    f"Failed to delete temp file: {e}"
                )

@app.post("/render")
async def render_step_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Generate an orthographic CAD rendering from the real STEP B-Rep.

    Initial validation configuration:
    - view direction: -Z
    - up direction: +Y
    - orthographic projection
    - no highlighted faces
    """

    if not file.filename or not file.filename.lower().endswith(('.step', '.stp')):
        raise HTTPException(
            status_code=400,
            detail="Invalid STEP file"
        )

    step_path = None
    image_path = None

    try:
        file_ext = os.path.splitext(file.filename)[1].lower()

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=file_ext
        ) as temp_step:

            content = await file.read()
            temp_step.write(content)
            step_path = temp_step.name

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".png"
        ) as temp_image:

            image_path = temp_image.name

        logger.info(
            f"Rendering STEP B-Rep: {file.filename} "
            f"({len(content)} bytes)"
        )

        render_step_view(
            file_path=step_path,
            output_path=image_path,

            # XY view looking along -Z
            view_direction=[0.0, 0.0, -1.0],

            # +Y is vertical in the resulting image
            up_direction=[0.0, 1.0, 0.0],

            highlight_faces=[],

            base_color="#B8BEC7",
            background_color="#FFFFFF",

            width=1600,
            height=1000
        )

        background_tasks.add_task(
            os.unlink,
            step_path
        )

        background_tasks.add_task(
            os.unlink,
            image_path
        )

        return FileResponse(
            image_path,
            media_type="image/png",
            filename="step_projection_XY_minusZ.png",
            background=background_tasks
        )

    except Exception as e:

        logger.error(
            f"STEP rendering error: {str(e)}",
            exc_info=True
        )

        if step_path and os.path.exists(step_path):
            os.unlink(step_path)

        if image_path and os.path.exists(image_path):
            os.unlink(image_path)

        raise HTTPException(
            status_code=500,
            detail=f"STEP rendering failed: {str(e)}"
        )

@app.post("/validate")
async def validate_step_file(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Validate STEP file quality and structure
    
    Args:
        file: Uploaded STEP file
    
    Returns:
        Validation results and quality metrics
    """
    if not file.filename or not file.filename.lower().endswith(('.step', '.stp')):
        raise HTTPException(status_code=400, detail="Invalid STEP file")
    
    temp_file = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.step') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        result = processor.validate_file(temp_file_path)
        return result
        
    except Exception as e:
        logger.error(f"Error validating file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        if temp_file and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
