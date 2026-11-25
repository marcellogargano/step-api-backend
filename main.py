"""STEP File Analysis API V10."""
import json
import logging
import os
import tempfile
from typing import Any, Dict, Optional

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from cad_renderer_cpu import render_step_section_svg, render_step_svg
from step_processor import STEPProcessor
from xcaf_processor import XCAFProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = FastAPI(title="STEP File Analysis API", version="10.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
processor = STEPProcessor()
xcaf_processor = XCAFProcessor()


def validate_step_filename(filename: Optional[str]) -> str:
    if not filename:
        raise HTTPException(400, "No filename provided")
    extension = os.path.splitext(filename)[1].lower()
    if extension not in (".step", ".stp"):
        raise HTTPException(400, "Expected a .step or .stp file")
    return extension


def safe_unlink(path):
    if path and os.path.exists(path):
        try:
            os.unlink(path)
        except Exception as exc:
            logger.warning("Temporary-file cleanup failed for %s: %s", path, exc)


def csv_list(text):
    return [item.strip() for item in (text or "").split(",") if item.strip()]


def json_map(text, field_name):
    if not (text or "").strip():
        return {}
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise HTTPException(400, "{} must be a JSON object".format(field_name)) from exc
    if not isinstance(value, dict):
        raise HTTPException(400, "{} must be a JSON object".format(field_name))
    return {str(key): str(item) for key, item in value.items()}


def standard_view(name, manual_view, manual_up):
    table = {
        "FRONT": ((0, -1, 0), (0, 0, 1)),
        "REAR": ((0, 1, 0), (0, 0, 1)),
        "LEFT": ((1, 0, 0), (0, 0, 1)),
        "RIGHT": ((-1, 0, 0), (0, 0, 1)),
        "TOP": ((0, 0, -1), (0, 1, 0)),
        "BOTTOM": ((0, 0, 1), (0, 1, 0)),
        "ISOMETRIC": ((-1, -1, -1), (0, 0, 1)),
        "MANUAL": (manual_view, manual_up),
    }
    key = (name or "MANUAL").strip().upper()
    if key not in table:
        raise HTTPException(400, "Unsupported standard_view: {}".format(name))
    return table[key]


def select_references(items, selectors):
    wanted = {item.upper() for item in selectors}
    if "ALL" in wanted:
        return items
    type_map = {"ALL_PLANES": "DATUM_PLANE", "ALL_AXES": "DATUM_AXIS", "ALL_CSYS": "DATUM_CSYS", "ALL_POINTS": "DATUM_POINT"}
    wanted_types = {value for key, value in type_map.items() if key in wanted}
    selected = []
    for item in items:
        names = {
            str(item.get("name", "")).upper(),
            str(item.get("display_name", "")).upper(),
            str(item.get("id", "")).upper(),
        }
        names.update(str(alias).upper() for alias in item.get("aliases", []))
        if names.intersection(wanted) or item.get("type") in wanted_types:
            selected.append(item)
    return selected


@app.get("/")
async def root():
    return {"status": "healthy", "service": "STEP File Analysis API", "version": "10.0.0"}


@app.get("/health")
async def health():
    return {"status": "healthy", "occt_available": processor.is_available(), "xcaf_available": xcaf_processor.is_available(), "version": "10.0.0"}


async def save_upload(file):
    extension = validate_step_filename(file.filename)
    with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as stream:
        stream.write(await file.read())
        return stream.name


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)) -> Dict[str, Any]:
    path = await save_upload(file)
    try:
        result = processor.analyze_file(path)
        result.setdefault("file_info", {})["original_filename"] = file.filename
        return result
    finally:
        safe_unlink(path)


@app.post("/analyze/geometry")
async def geometry(file: UploadFile = File(...)):
    path = await save_upload(file)
    try: return processor.get_geometric_properties(path)
    finally: safe_unlink(path)


@app.post("/analyze/topology")
async def topology(file: UploadFile = File(...)):
    path = await save_upload(file)
    try: return processor.get_topology_info(path)
    finally: safe_unlink(path)


@app.post("/analyze/faces")
async def faces(file: UploadFile = File(...)):
    path = await save_upload(file)
    try: return processor.get_faces(path)
    finally: safe_unlink(path)


@app.post("/analyze/assembly")
async def assembly(file: UploadFile = File(...)):
    path = await save_upload(file)
    try: return xcaf_processor.analyze_assembly(path)
    finally: safe_unlink(path)


@app.post("/analyze/references")
async def references(file: UploadFile = File(...)):
    path = await save_upload(file)
    try: return xcaf_processor.analyze_references(path)
    finally: safe_unlink(path)


@app.post("/analyze/colors")
async def colors(file: UploadFile = File(...)):
    path = await save_upload(file)
    try: return xcaf_processor.analyze_colors(path)
    finally: safe_unlink(path)


@app.post("/analyze/instance-colors")
async def instance_colors(file: UploadFile = File(...)):
    path = await save_upload(file)
    try: return xcaf_processor.analyze_instance_colors(path)
    finally: safe_unlink(path)


@app.post("/validate")
async def validate(file: UploadFile = File(...)):
    path = await save_upload(file)
    try: return processor.validate_file(path)
    finally: safe_unlink(path)


@app.post("/section-cpu")
async def section_cpu(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    origin_x: float = Form(0), origin_y: float = Form(0), origin_z: float = Form(0),
    normal_x: float = Form(0), normal_y: float = Form(0), normal_z: float = Form(1),
    up_x: float = Form(0), up_y: float = Form(1), up_z: float = Form(0),
    section_reference: str = Form(""),
    fill_color: str = Form("#D9D9D9"), line_color: str = Form("#000000"), background_color: str = Form("#FFFFFF"),
    width: int = Form(1600), height: int = Form(1000),
):
    step_path = await save_upload(file)
    svg_path = tempfile.NamedTemporaryFile(delete=False, suffix=".svg").name
    try:
        origin, normal = [origin_x, origin_y, origin_z], [normal_x, normal_y, normal_z]
        if section_reference.strip():
            data = xcaf_processor.analyze_references(step_path)
            target = next((item for item in data.get("references", []) if item.get("name", "").upper() == section_reference.upper()), None)
            if not target or target.get("type") != "DATUM_PLANE":
                raise HTTPException(400, "section_reference must identify a DATUM_PLANE")
            origin_data, normal_data = target["origin"], target["normal"]
            origin = [origin_data[key] for key in ("x","y","z")]
            normal = [normal_data[key] for key in ("x","y","z")]
        render_step_section_svg(step_path, svg_path, origin, normal, [up_x,up_y,up_z], width, height, line_color, fill_color, background_color)
        background_tasks.add_task(safe_unlink, step_path); background_tasks.add_task(safe_unlink, svg_path)
        return FileResponse(svg_path, media_type="image/svg+xml", filename="step_section.svg", background=background_tasks)
    except Exception:
        safe_unlink(step_path); safe_unlink(svg_path); raise


@app.post("/render-cpu")
async def render_cpu(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    standard_view_name: str = Form("MANUAL"),
    view_x: float = Form(0), view_y: float = Form(0), view_z: float = Form(-1),
    up_x: float = Form(0), up_y: float = Form(1), up_z: float = Form(0),
    display_mode: str = Form("SHADED"),
    highlight_faces: str = Form(""), highlight_color: str = Form("#FF0000"),
    face_colors: str = Form("{}"), component_colors: str = Form("{}"),
    use_step_colors: bool = Form(True),
    base_color: str = Form("#BFC5CC"), edge_color: str = Form("#000000"), background_color: str = Form("#FFFFFF"),
    show_hidden_edges: bool = Form(False), hidden_edge_color: str = Form("#808080"),
    show_references: bool = Form(False), reference_names: str = Form(""), reference_color: str = Form("#606060"), reference_scale: float = Form(1.0),
    show_reference_labels: bool = Form(False), reference_label_names: str = Form(""),
    width: int = Form(1600), height: int = Form(1000),
    linear_deflection: float = Form(1.0), angular_deflection: float = Form(0.35),
):
    step_path = await save_upload(file)
    svg_path = tempfile.NamedTemporaryFile(delete=False, suffix=".svg").name
    try:
        view, up = standard_view(standard_view_name, (view_x,view_y,view_z), (up_x,up_y,up_z))
        requested = csv_list(reference_names)
        entities = []
        if show_references and requested:
            entities = select_references(xcaf_processor.analyze_references(step_path).get("references", []), requested)

        # STEP face colors are the base. Explicit user face overrides win.
        resolved_face_colors = {}
        if use_step_colors:
            resolved_face_colors.update(
                xcaf_processor.analyze_colors(step_path).get("face_colors", {})
            )
        resolved_face_colors.update(json_map(face_colors, "face_colors"))

        render_step_svg(
            step_path, svg_path, list(view), list(up),
            highlight_faces=csv_list(highlight_faces), highlight_color=highlight_color,
            base_color=base_color, edge_color=edge_color, background_color=background_color,
            width=width, height=height, linear_deflection=linear_deflection, angular_deflection=angular_deflection,
            show_hidden_edges=show_hidden_edges, hidden_edge_color=hidden_edge_color,
            show_references=show_references, reference_entities=entities, reference_color=reference_color, reference_scale=reference_scale,
            show_reference_labels=show_reference_labels, reference_label_names=csv_list(reference_label_names),
            face_colors=resolved_face_colors, component_colors=json_map(component_colors, "component_colors"),
            display_mode=display_mode,
        )
        background_tasks.add_task(safe_unlink, step_path); background_tasks.add_task(safe_unlink, svg_path)
        return FileResponse(svg_path, media_type="image/svg+xml", filename="step_projection.svg", background=background_tasks)
    except Exception:
        safe_unlink(step_path); safe_unlink(svg_path); raise


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
