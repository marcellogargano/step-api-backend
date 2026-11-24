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

import math
import os
import logging
from typing import List, Optional, Tuple

from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_REVERSED
from OCC.Core.TopoDS import topods
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.TopLoc import TopLoc_Location


logger = logging.getLogger(__name__)


def _normalize(v: List[float]) -> Tuple[float, float, float]:
    if len(v) != 3:
        raise ValueError(
            "Direction vector must contain exactly 3 values"
        )

    x, y, z = map(float, v)

    n = math.sqrt(
        x * x +
        y * y +
        z * z
    )

    if n < 1e-12:
        raise ValueError(
            "Zero direction vector"
        )

    return (
        x / n,
        y / n,
        z / n
    )


def _cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    )


def _dot(a, b):
    return (
        a[0] * b[0] +
        a[1] * b[1] +
        a[2] * b[2]
    )


def _validate_color(
    value: str,
    name: str
) -> str:

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
):

    # ------------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------------

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"STEP file not found: {file_path}"
        )

    if width < 200 or height < 200:
        raise ValueError(
            "SVG width and height must be at least 200"
        )

    if (
        linear_deflection <= 0
        or angular_deflection <= 0
    ):
        raise ValueError(
            "Mesh deflections must be greater than zero"
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
    # READ STEP USING OPEN CASCADE
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

    reader.TransferRoots()

    shape = reader.OneShape()

    if shape.IsNull():
        raise RuntimeError(
            "Open CASCADE returned null shape"
        )

    logger.info(
        "CPU RENDER 2: STEP loaded"
    )

    # ------------------------------------------------------------
    # CREATE REAL OPEN CASCADE B-REP TRIANGULATION
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
            "Open CASCADE meshing failed"
        )

    logger.info(
        "CPU RENDER 4: BRep meshing completed"
    )

    # ------------------------------------------------------------
    # CAMERA BASIS
    # ------------------------------------------------------------

    view = _normalize(
        view_direction
    )

    up_input = _normalize(
        up_direction
    )

    cross = _cross(
        view,
        up_input
    )

    cross_size = math.sqrt(
        _dot(cross, cross)
    )

    if cross_size < 1e-12:
        raise ValueError(
            "view_direction and up_direction "
            "cannot be parallel"
        )

    right = _normalize(
        list(cross)
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
    # EXTRACT FACE TRIANGULATION
    # ------------------------------------------------------------

    triangles = []

    xmin = float("inf")
    xmax = float("-inf")

    ymin = float("inf")
    ymax = float("-inf")

    explorer = TopExp_Explorer(
        shape,
        TopAbs_FACE
    )

    face_index = 1

    while explorer.More():

        face = topods.Face(
            explorer.Current()
        )

        face_id = (
            f"FACE_{face_index}"
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
                color = highlight_color
            else:
                color = base_color

            for i in range(
                1,
                triangulation.NbTriangles() + 1
            ):

                triangle = (
                    triangulation.Triangle(i)
                )

                n1, n2, n3 = (
                    triangle.Get()
                )

                if reversed_face:
                    n2, n3 = n3, n2

                projected = []

                depth_sum = 0.0

                for node_index in (
                    n1,
                    n2,
                    n3
                ):

                    p = (
                        triangulation.Node(
                            node_index
                        ).Transformed(
                            transformation
                        )
                    )

                    point = (
                        p.X(),
                        p.Y(),
                        p.Z()
                    )

                    px = _dot(
                        point,
                        right
                    )

                    py = _dot(
                        point,
                        up
                    )

                    pz = _dot(
                        point,
                        view
                    )

                    projected.append(
                        (px, py)
                    )

                    depth_sum += pz

                    xmin = min(
                        xmin,
                        px
                    )

                    xmax = max(
                        xmax,
                        px
                    )

                    ymin = min(
                        ymin,
                        py
                    )

                    ymax = max(
                        ymax,
                        py
                    )

                triangles.append(
                    {
                        "face_id": face_id,
                        "points": projected,
                        "depth": (
                            depth_sum / 3.0
                        ),
                        "color": color
                    }
                )

        face_index += 1

        explorer.Next()

    if not triangles:
        raise RuntimeError(
            "No B-Rep triangles generated"
        )

    logger.info(
        "CPU RENDER 5: "
        f"extracted {len(triangles)} triangles"
    )

    # ------------------------------------------------------------
    # PROJECTED BOUNDING BOX
    # ------------------------------------------------------------

    dx = xmax - xmin
    dy = ymax - ymin

    if (
        dx <= 1e-12
        or dy <= 1e-12
    ):
        raise RuntimeError(
            "Invalid projected model dimensions"
        )

    # ------------------------------------------------------------
    # FIT MODEL TO IMAGE
    # ------------------------------------------------------------

    margin = 40.0

    available_width = max(
        1.0,
        width - 2.0 * margin
    )

    available_height = max(
        1.0,
        height - 2.0 * margin
    )

    scale = min(
        available_width / dx,
        available_height / dy
    )

    projected_width = (
        dx * scale
    )

    projected_height = (
        dy * scale
    )

    offset_x = (
        width - projected_width
    ) / 2.0

    offset_y = (
        height - projected_height
    ) / 2.0

    def image_coordinates(x, y):

        ix = (
            offset_x
            + (x - xmin) * scale
        )

        # SVG Y-axis points downward.
        iy = (
            height
            - (
                offset_y
                + (y - ymin) * scale
            )
        )

        return (
            ix,
            iy
        )

    # ------------------------------------------------------------
    # PAINTER DEPTH SORT
    # ------------------------------------------------------------

    # Far triangles are written first.
    # Near triangles are written last.
    #
    # IMPORTANT:
    # This is a visualization algorithm only.
    # Geometric measurements and face selection remain B-Rep based.

    triangles.sort(
        key=lambda item: item["depth"],
        reverse=True
    )

    logger.info(
        "CPU RENDER 6: writing SVG"
    )

    # ------------------------------------------------------------
    # WRITE SVG
    # ------------------------------------------------------------

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

            p1 = image_coordinates(
                triangle["points"][0][0],
                triangle["points"][0][1]
            )

            p2 = image_coordinates(
                triangle["points"][1][0],
                triangle["points"][1][1]
            )

            p3 = image_coordinates(
                triangle["points"][2][0],
                triangle["points"][2][1]
            )

            points = (
                f"{p1.3f},"
                f"{p1.3f} "

                f"{p2.3f},"
                f"{p2.3f} "

                f"{p3.3f},"
                f"{p3.3f}"
            )

            svg.write(
                f'<polygon '
                f'points="{points}" '
                f'fill="{triangle["color"]}" '
                f'stroke="none"/>\n'
            )

        svg.write(
            "</svg>\n"
        )

    # ------------------------------------------------------------
    # VERIFY OUTPUT
    # ------------------------------------------------------------

    if (
        not os.path.exists(output_path)
        or os.path.getsize(output_path) == 0
    ):
        raise RuntimeError(
            "SVG file was not generated"
        )

    logger.info(
        "CPU RENDER 7: "
        f"SVG generated: {output_path}"
    )

    return output_path
