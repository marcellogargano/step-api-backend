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
    """Normalize a 3D vector."""

    if len(vector) != 3:
        raise ValueError(
            "Direction vector must contain exactly 3 values"
        )

    x = float(vector[0])
    y = float(vector[1])
    z = float(vector[2])

    magnitude = math.sqrt(
        x * x + y * y + z * z
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
    """Cross product."""

    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    )


def _dot(a, b):
    """Dot product."""

    return (
        a[0] * b[0]
        + a[1] * b[1]
        + a[2] * b[2]
    )


def _validate_color(
    value: str,
    name: str
) -> str:
    """Validate #RRGGBB color."""

    value = value.strip().upper()

    if len(value) != 7:
        raise ValueError(
            "{} must use #RRGGBB format".format(name)
        )

    if not value.startswith("#"):
        raise ValueError(
            "{} must use #RRGGBB format".format(name)
        )

    try:
        int(value[1:], 16)

    except ValueError as exc:
        raise ValueError(
            "{} must use #RRGGBB format".format(name)
        ) from exc

    return value


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
    Generate an orthographic SVG projection from a STEP B-Rep.

    Geometry and triangulation come from Open CASCADE.

    Face IDs follow the same TopExp traversal used by
    /analyze/faces.
    """

    # ---------------------------------------------------------
    # INPUT VALIDATION
    # ---------------------------------------------------------

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            "STEP file not found: {}".format(file_path)
        )

    if width < 200:
        raise ValueError(
            "Width must be at least 200 pixels"
        )

    if height < 200:
        raise ValueError(
            "Height must be at least 200 pixels"
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

    # ---------------------------------------------------------
    # READ STEP
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # OPEN CASCADE B-REP MESH
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # CAMERA BASIS
    # ---------------------------------------------------------

    view = _normalize(
        view_direction
    )

    requested_up = _normalize(
        up_direction
    )

    right_raw = _cross(
        view,
        requested_up
    )

    right_length = math.sqrt(
        _dot(
            right_raw,
            right_raw
        )
    )

    if right_length < 1.0e-12:
        raise ValueError(
            "view_direction and up_direction cannot be parallel"
        )

    right = _normalize(
        list(right_raw)
    )

    up = _normalize(
        list(
            _cross(
                right,
                view
            )
        )
    )

    # ---------------------------------------------------------
    # HIGHLIGHTED FACE IDs
    # ---------------------------------------------------------

    highlight_set = set()

    if highlight_faces:

        for face_id in highlight_faces:

            clean_face_id = (
                str(face_id)
                .strip()
                .upper()
            )

            if clean_face_id:
                highlight_set.add(
                    clean_face_id
                )

    # ---------------------------------------------------------
    # EXTRACT AND PROJECT TRIANGLES
    # ---------------------------------------------------------

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

    logger.info(
        "CPU RENDER 5: extracting B-Rep face triangulations"
    )

    while face_explorer.More():

        face = topods.Face(
            face_explorer.Current()
        )

        face_id = "FACE_{}".format(
            face_index
        )

        location = TopLoc_Location()

        triangulation = (
            BRep_Tool.Triangulation(
                face,
                location
            )
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

            number_of_triangles = (
                triangulation.NbTriangles()
            )

            for triangle_index in range(
                1,
                number_of_triangles + 1
            ):

                triangle = (
                    triangulation.Triangle(
                        triangle_index
                    )
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

                    point = (
                        triangulation.Node(
                            node_index
                        ).Transformed(
                            transformation
                        )
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

                    projected_depth = _dot(
                        point_3d,
                        view
                    )

                    projected_points.append(
                        (
                            projected_x,
                            projected_y
                        )
                    )

                    depth_sum += (
                        projected_depth
                    )

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

    if len(triangles) == 0:
        raise RuntimeError(
            "Open CASCADE did not generate any display triangles"
        )

    logger.info(
        "CPU RENDER 6: extracted {} triangles".format(
            len(triangles)
        )
    )

    # ---------------------------------------------------------
    # PROJECTED EXTENTS
    # ---------------------------------------------------------

    projected_width = (
        projected_x_max
        - projected_x_min
    )

    projected_height = (
        projected_y_max
        - projected_y_min
    )

    if projected_width <= 1.0e-12:
        raise RuntimeError(
            "Invalid projected model width"
        )

    if projected_height <= 1.0e-12:
        raise RuntimeError(
            "Invalid projected model height"
        )

    # ---------------------------------------------------------
    # FIT PROJECTION INTO SVG CANVAS
    # ---------------------------------------------------------

    margin = 40.0

    usable_width = max(
        1.0,
        float(width)
        - 2.0 * margin
    )

    usable_height = max(
        1.0,
        float(height)
        - 2.0 * margin
    )

    scale_x = (
        usable_width
        / projected_width
    )

    scale_y = (
        usable_height
        / projected_height
    )

    scale = min(
        scale_x,
        scale_y
    )

    rendered_width = (
        projected_width
        * scale
    )

    rendered_height = (
        projected_height
        * scale
    )

    x_offset = (
        float(width)
        - rendered_width
    ) / 2.0

    y_offset = (
        float(height)
        - rendered_height
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

    # ---------------------------------------------------------
    # DEPTH SORT
    # ---------------------------------------------------------

    triangles.sort(
        key=lambda item: item["depth"],
        reverse=True
    )

    # ---------------------------------------------------------
    # WRITE SVG
    # ---------------------------------------------------------

    logger.info(
        "CPU RENDER 7: writing SVG"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as svg:

        svg.write(
            '<?xml version="1.0" encoding="UTF-8"?>\n'
        )

        svg.write(
            '<svg '
            'xmlns="http://www.w3.org/2000/svg" '
            'width="{}" '
            'height="{}" '
            'viewBox="0 0 {} {}">\n'.format(
                width,
                height,
                width,
                height
            )
        )

        svg.write(
            '<rect '
            'x="0" '
            'y="0" '
            'width="{}" '
            'height="{}" '
            'fill="{}"/>\n'.format(
                width,
                height,
                background_color
            )
        )

        for triangle in triangles:

            image_point_1 = (
                image_coordinates(
                    triangle["points"][0][0],
                    triangle["points"][0][1]
                )
            )

            image_point_2 = (
                image_coordinates(
                    triangle["points"][1][0],
                    triangle["points"][1][1]
                )
            )

            image_point_3 = (
                image_coordinates(
                    triangle["points"][2][0],
                    triangle["points"][2][1]
                )
            )

            polygon_points = (
                "{:.3f},{:.3f} "
                "{:.3f},{:.3f} "
                "{:.3f},{:.3f}"
            ).format(
                image_point_1[0],
                image_point_1[1],
                image_point_2[0],
                image_point_2[1],
                image_point_3[0],
                image_point_3[1]
            )

            svg.write(
                '<polygon '
                'points="{}" '
                'fill="{}" '
                'stroke="none"/>\n'.format(
                    polygon_points,
                    triangle["color"]
                )
            )

        svg.write(
            "</svg>\n"
        )

    # ---------------------------------------------------------
    # VERIFY SVG
    # ---------------------------------------------------------

    if not os.path.exists(
        output_path
    ):
        raise RuntimeError(
            "SVG file was not generated"
        )

    output_size = os.path.getsize(
        output_path
    )

    if output_size <= 0:
        raise RuntimeError(
            "Generated SVG file is empty"
        )

    logger.info(
        "CPU RENDER 8: SVG generated, size={} bytes".format(
            output_size
        )
    )

    return output_path
