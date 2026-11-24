"""
CPU CAD renderer.

Geometry comes exclusively from the real Open CASCADE B-Rep.
No OpenGL, VTK, OSMesa or GPU rendering is used.

Open CASCADE:
    STEP -> TopoDS_Shape -> BRepMesh triangulation

CPU renderer:
    3D vertices -> orthographic projection -> SVG

B-Rep face IDs remain consistent with /analyze/faces.
"""

import logging
import math
import os
from typing import List, Optional, Tuple

from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_REVERSED
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.TopoDS import topods


logger = logging.getLogger(__name__)


def _normalize(
    vector: List[float]
) -> Tuple[float, float, float]:
    """Normalize a three-dimensional vector."""

    if len(vector) != 3:
        raise ValueError(
            "Direction vector must contain exactly three values"
        )

    x = float(vector[0])
    y = float(vector[1])
    z = float(vector[2])

    magnitude = math.sqrt(
        x * x +
        y * y +
        z * z
    )

    if magnitude < 1.0e-12:
        raise ValueError(
            "Direction vector cannot be zero"
        )

    return (
        x / magnitude,
        y / magnitude,
        z / magnitude
    )


def _cross(a, b):
    """Cross product of two three-dimensional vectors."""

    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    )


def _dot(a, b):
    """Dot product of two three-dimensional vectors."""

    return (
        a[0] * b[0] +
        a[1] * b[1] +
        a[2] * b[2]
    )


def _validate_color(
    value: str,
    name: str
) -> str:
    """Validate and normalize an SVG hexadecimal color."""

    value = value.strip()

    if (
        len(value) != 7
        or not value.startswith("#")
    ):
        raise ValueError(
            f"{name} must use #RRGGBB format"
        )

    try:
        int(value[1:], 16)

    except ValueError as exc:
        raise ValueError(
            f"{name} must use #RRGGBB format"
        ) from exc

    return value.upper()


def render_step_svg(
    file_path: str,
    output_path: str,
    view_direction: List[float],
    up_direction: List[float],
    highlight_faces: Optional[List[str]] = None,
    highlight_color: str = "#FF0000",
    base_color: str = "#BFC5CC",
    edge_color: str = "#202020",
    background_color: str = "#FFFFFF",
    width: int = 1600,
    height: int = 1000,
    linear_deflection: float = 1.0,
    angular_deflection: float = 0.35
) -> str:
    """
    Generate an orthographic SVG view from a real STEP B-Rep.

    Face numbering follows the same TopExp traversal used by
    the /analyze/faces endpoint.

    Open CASCADE is used for:
    - STEP import
    - TopoDS B-Rep geometry
    - face enumeration
    - BRep triangulation

    The final SVG projection is calculated on the CPU.
    """

    # ------------------------------------------------------------
    # INPUT VALIDATION
    # ------------------------------------------------------------

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"STEP file not found: {file_path}"
        )

    if width < 200 or height < 200:
        raise ValueError(
            "SVG width and height must be at least 200 pixels"
        )

    if linear_deflection <= 0:
        raise ValueError(
            "linear_deflection must be greater than zero"
        )

    if angular_deflection <= 0:
        raise ValueError(
            "angular_deflection must be greater than zero"
        )

    highlight_color = _validate_color(
        highlight_color,
        "highlight_color"
    )

    base_color = _validate_color(
        base_color,
        "base_color"
    )

    edge_color = _validate_color(
        edge_color,
        "edge_color"
    )

    background_color = _validate_color(
        background_color,
        "background_color"
    )

    # ------------------------------------------------------------
    # READ STEP WITH OPEN CASCADE
    # ------------------------------------------------------------

    logger.info(
        "CPU RENDER 1: reading STEP"
    )

    reader = STEPControl_Reader()

    status = reader.ReadFile(
        file_path
    )

    if status != IFSelect_RetDone:
        raise RuntimeError(
            "Open CASCADE STEP read failed"
        )

    transferred_roots = reader.TransferRoots()

    if transferred_roots <= 0:
        raise RuntimeError(
            "Open CASCADE did not transfer any STEP roots"
        )

    shape = reader.OneShape()

    if shape.IsNull():
        raise RuntimeError(
            "Open CASCADE returned a null shape"
        )

    logger.info(
        "CPU RENDER 2: STEP loaded"
    )

    # ------------------------------------------------------------
    # B-REP TRIANGULATION
    # ------------------------------------------------------------

    logger.info(
        "CPU RENDER 3: starting BRep meshing"
    )

    mesher = BRepMesh_IncrementalMesh(
        shape,
        linear_deflection,
        False,
        angular_deflection,
        True
    )

    mesher.Perform()

    if not mesher.IsDone():
        raise RuntimeError(
            "Open CASCADE B-Rep meshing failed"
        )

    logger.info(
        "CPU RENDER 4: BRep meshing completed"
    )

    # ------------------------------------------------------------
    # ORTHOGRAPHIC CAMERA BASIS
    # ------------------------------------------------------------

    view = _normalize(
        view_direction
    )

    requested_up = _normalize(
        up_direction
    )

    right_vector = _cross(
        view,
        requested_up
    )

    right_magnitude = math.sqrt(
        _dot(
            right_vector,
            right_vector
        )
    )

    if right_magnitude < 1.0e-12:
        raise ValueError(
            "view_direction and up_direction cannot be parallel"
        )

    right = _normalize(
        list(right_vector)
    )

    up = _normalize(
        list(
            _cross(
                right,
                view
            )
        )
    )

    # ------------------------------------------------------------
    # FACE SELECTION
    # ------------------------------------------------------------

    highlight_set = {
        str(face_id).strip().upper()
        for face_id in (
            highlight_faces or []
        )
        if str(face_id).strip()
    }

    # ------------------------------------------------------------
    # TRIANGLE EXTRACTION AND PROJECTION
    # ------------------------------------------------------------

    triangles = []

    projected_x_min = float("inf")
    projected_x_max = float("-inf")

    projected_y_min = float("inf")
    projected_y_max = float("-inf")

    face_explorer = TopExp_Explorer(
        shape,
        TopAbs_FACE
    )

    face_index = 1

    while face_explorer.More():

        face = topods.Face(
            face_explorer.Current()
        )

        face_id = (
            f"FACE_{face_index}"
        )

        location = TopLoc_Location()

        triangulation = BRep_Tool.Triangulation(
            face,
            location
        )

        if triangulation is not None:

            transformation = (
                location.Transformation()
            )

            reversed_face = (
                face.Orientation()
                == TopAbs_REVERSED
            )

            if face_id in highlight_set:
                face_color = highlight_color
            else:
                face_color = base_color

            triangle_total = (
                triangulation.NbTriangles()
            )

            for triangle_index in range(
                1,
                triangle_total + 1
            ):

                triangle = triangulation.Triangle(
                    triangle_index
                )

                node_1, node_2, node_3 = (
                    triangle.Get()
                )

                if reversed_face:
                    node_2, node_3 = (
                        node_3,
                        node_2
                    )

                projected_points = []
                depth_sum = 0.0

                for node_index in (
                    node_1,
                    node_2,
                    node_3
                ):

                    point = triangulation.Node(
                        node_index
                    ).Transformed(
                        transformation
                    )

                    point_3d = (
                        point.X(),
                        point.Y(),
                        point.Z()
                    )

                    projected_x = _dot(
                        point_3d,
                        right
                    )

                    projected_y = _dot(
                        point_3d,
                        up
                    )

                    depth = _dot(
                        point_3d,
                        view
                    )

                    projected_points.append(
                        (
                            projected_x,
                            projected_y
                        )
                    )

                    depth_sum += depth

                    projected_x_min = min(
                        projected_x_min,
                        projected_x
                    )

                    projected_x_max = max(
                        projected_x_max,
                        projected_x
                    )

                    projected_y_min = min(
                        projected_y_min,
                        projected_y
                    )

                    projected_y_max = max(
                        projected_y_max,
                        projected_y
                    )

                triangles.append(
                    {
                        "face_id": face_id,
                        "points": projected_points,
                        "depth": depth_sum / 3.0,
                        "color": face_color
                    }
                )

        face_index += 1
        face_explorer.Next()

    if not triangles:
        raise RuntimeError(
            "Open CASCADE did not generate any display triangles"
        )

    logger.info(
        "CPU RENDER 5: "
        f"extracted {len(triangles)} triangles"
    )

    # ------------------------------------------------------------
    # PROJECTED MODEL EXTENTS
    # ------------------------------------------------------------

    projected_width = (
        projected_x_max
        - projected_x_min
    )

    projected_height = (
        projected_y_max
        - projected_y_min
    )

    if (
        projected_width <= 1.0e-12
        or projected_height <= 1.0e-12
    ):
        raise RuntimeError(
            "Invalid projected model dimensions"
        )

    # ------------------------------------------------------------
    # FIT PROJECTION INTO SVG
    # ------------------------------------------------------------

    margin = 40.0

    usable_width = max(
        1.0,
        float(width) - 2.0 * margin
    )

    usable_height = max(
        1.0,
        float(height) - 2.0 * margin
    )

    scale = min(
        usable_width / projected_width,
        usable_height / projected_height
    )

    rendered_width = (
        projected_width * scale
    )

    rendered_height = (
        projected_height * scale
    )

    x_offset = (
        float(width) - rendered_width
    ) / 2.0

    y_offset = (
        float(height) - rendered_height
    ) / 2.0

    def image_coordinates(
        projected_x,
        projected_y
    ):
        image_x = (
            x_offset
            + (
                projected_x
                - projected_x_min
            ) * scale
        )

        # SVG coordinates increase downward on the Y-axis.
        image_y = (
            float(height)
            - (
                y_offset
                + (
                    projected_y
                    - projected_y_min
                ) * scale
            )
        )

        return (
            image_x,
            image_y
        )

    # ------------------------------------------------------------
    # DEPTH SORTING
    # ------------------------------------------------------------

    # Far triangles are written first.
    # Near triangles are written later.
    #
    # This affects only graphic visibility.
    # Measurements and face selection remain B-Rep based.

    triangles.sort(
        key=lambda item: item["depth"],
        reverse=True
    )

    # ------------------------------------------------------------
    # WRITE SVG
    # ------------------------------------------------------------

    logger.info(
        "CPU RENDER 6: writing SVG"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as svg:

        svg.write(
            '<?xml version="1.0" '
            'encoding="UTF-8"?>\n'
        )

        svg.write(
            f'<svg '
            f'xmlns="http://www.w3.org/2000/svg" '
            f'width="{width}" '
            f'height="{height}" '
            f'viewBox="0 0 {width} {height}">\n'
        )

        svg.write(
            f'<rect '
            f'x="0" '
            f'y="0" '
            f'width="{width}" '
            f'height="{height}" '
            f'fill="{background_color}"/>\n'
        )

        for triangle in triangles:

            image_point_1 = image_coordinates(
                triangle["points"][0][0],
                triangle["points"][0][1]
            )

            image_point_2 = image_coordinates(
                triangle["points"][1][0],
                triangle["points"][1][1]
            )

            image_point_3 = image_coordinates(
                triangle["points"][2][0],
                triangle["points"][2][1]
            )

            polygon_points = (
                f"{image_point_1.3f},"
                f"{image_point_1.3f} "

                f"{image_point_2.3f},"
                f"{image_point_2.3f} "

                f"{image_point_3.3f},"
                f"{image_point_3.3f}"
            )

            svg.write(
                f'<polygon '
                f'points="{polygon_points}" '
                f'fill="{triangle["color"]}" '
                f'stroke="none"/>\n'
            )

        svg.write(
            "</svg>\n"
        )

    # ------------------------------------------------------------
    # VERIFY OUTPUT
    # ------------------------------------------------------------

    if not os.path.exists(
        output_path
    ):
        raise RuntimeError(
            "SVG file was not generated"
        )

    if os.path.getsize(
        output_path
    ) <= 0:
        raise RuntimeError(
            "Generated SVG file is empty"
        )

    logger.info(
        "CPU RENDER 7: "
        f"SVG generated: {output_path}"
    )

    return output_path
