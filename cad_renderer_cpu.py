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

    x = float(v[0])
    y = float(v[1])
    z = float(v[2])

    n = math.sqrt(x*x + y*y + z*z)

    if n < 1e-12:
        raise ValueError("Zero direction vector")

    return x/n, y/n, z/n


def _cross(a, b):

    return (
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0]
    )


def _dot(a, b):

    return (
        a[0]*b[0] +
        a[1]*b[1] +
        a[2]*b[2]
    )


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

    logger.info("CPU RENDER 1: reading STEP")

    reader = STEPControl_Reader()

    status = reader.ReadFile(file_path)

    if status != IFSelect_RetDone:
        raise RuntimeError("Open CASCADE STEP read failed")

    reader.TransferRoots()

    shape = reader.OneShape()

    if shape.IsNull():
        raise RuntimeError("Open CASCADE returned null shape")

    logger.info("CPU RENDER 2: STEP loaded")


    # ------------------------------------------------------------
    # CREATE REAL OCCT TRIANGULATION
    # ------------------------------------------------------------

    logger.info("CPU RENDER 3: starting BRep meshing")

    mesher = BRepMesh_IncrementalMesh(
        shape,
        linear_deflection,
        False,
        angular_deflection,
        True
    )

    mesher.Perform()

    if not mesher.IsDone():
        raise RuntimeError("Open CASCADE meshing failed")

    logger.info("CPU RENDER 4: BRep meshing completed")


    # ------------------------------------------------------------
    # CAMERA BASIS
    # ------------------------------------------------------------

    view = _normalize(view_direction)

    up_input = _normalize(up_direction)

    right = _normalize(_cross(view, up_input))

    up = _normalize(_cross(right, view))


    # ------------------------------------------------------------
    # EXTRACT EXACT FACE TRIANGULATION
    # ------------------------------------------------------------

    highlight_set = set(highlight_faces or [])

    triangles = []

    explorer = TopExp_Explorer(
        shape,
        TopAbs_FACE
    )

    face_index = 1

    while explorer.More():

        face = topods.Face(
            explorer.Current()
        )

        face_id = f"FACE_{face_index}"

        location = TopLoc_Location()

        triangulation = BRep_Tool.Triangulation(
            face,
            location
        )

        if triangulation is not None:

            transformation = location.Transformation()

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

                triangle = triangulation.Triangle(i)

                n1, n2, n3 = triangle.Get()

                if reversed_face:
                    n2, n3 = n3, n2

                nodes = []

                for node_index in (n1, n2, n3):

                    p = triangulation.Node(
                        node_index
                    ).Transformed(
                        transformation
                    )

                    nodes.append(
                        (
                            p.X(),
                            p.Y(),
                            p.Z()
                        )
                    )


                # ------------------------------------------------
                # ORTHOGRAPHIC PROJECTION
                # ------------------------------------------------

                projected = []

                depth_sum = 0.0

                for p in nodes:

                    px = _dot(p, right)
                    py = _dot(p, up)
                    pz = _dot(p, view)

                    projected.append(
                        (px, py)
                    )

                    depth_sum += pz

                depth = depth_sum / 3.0

                triangles.append(
                    {
                        "face_id": face_id,
                        "points": projected,
                        "depth": depth,
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
        f"CPU RENDER 5: extracted {len(triangles)} triangles"
    )


    # ------------------------------------------------------------
    # PROJECTED BOUNDING BOX
    # ------------------------------------------------------------

    all_x = []
    all_y = []

    for triangle in triangles:

        for x, y in triangle["points"]:

            all_x.append(x)
            all_y.append(y)


    xmin = min(all_x)
    xmax = max(all_x)

    ymin = min(all_y)
    ymax = max(all_y)


    dx = xmax - xmin
    dy = ymax - ymin


    if dx <= 0 or dy <= 0:
        raise RuntimeError(
            "Invalid projected model dimensions"
        )


    # ------------------------------------------------------------
    # FIT MODEL TO IMAGE
    # ------------------------------------------------------------

    margin = 40.0

    scale_x = (
        (width - 2*margin) / dx
    )

    scale_y = (
        (height - 2*margin) / dy
    )

    scale = min(
        scale_x,
        scale_y
    )


    projected_width = dx * scale
    projected_height = dy * scale


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

        # SVG Y axis points downward.
        iy = (
            height
            - (
                offset_y
                + (y - ymin) * scale
            )
        )

        return ix, iy


    # ------------------------------------------------------------
    # PAINTER DEPTH SORT
    # ------------------------------------------------------------

    triangles.sort(
        key=lambda item: item["depth"],
        reverse=True
    )


    logger.info(
        "CPU RENDER 6: writing SVG"
    )


    # ------------------------------------------------------------
    # WRITE VECTOR SVG
    # ------------------------------------------------------------

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as svg:

        svg.write(
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">'
        )

        svg.write(
            f'<rect width="100%" height="100%" '
            f'fill="{background_color}"/>'
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
                f"{p1[0]:.3f},{p1[1]:.3f} "
                f"{p2[0]:.3f},{p2[1]:.3f} "
                f"{p3[0]:.3f},{p3[1]:.3f}"
            )


            svg.write(
                f'<polygon '
                f'points="{points}" '
                f'fill="{triangle["color"]}" '
                f'stroke="none"/>'
            )


        # SVG ending
        svg.write("</svg>")


    if not os.path.exists(output_path):
        raise RuntimeError(
            "SVG file was not generated"
        )


    logger.  f"CPU RENDER 7: SVG generated: {output_path}"
    )

    return output_path
