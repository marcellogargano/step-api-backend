"""
Memory-efficient CPU CAD renderer with Open CASCADE hidden-line removal.

Geometry comes exclusively from the real Open CASCADE B-Rep.
No OpenGL, VTK, OSMesa or GPU rendering is used.

Open CASCADE:
    STEP -> TopoDS_Shape -> BRepMesh triangulation
    STEP -> HLRBRep_Algo -> visible and hidden projected edges

CPU renderer:
    faces -> compact NumPy arrays -> orthographic SVG fills
    visible edges -> exact Open CASCADE HLR -> SVG lines
    hidden edges -> omitted by default, optional dashed lines

B-Rep face IDs remain consistent with /analyze/faces.
"""

import logging
import math
import os
from typing import List, Optional, Tuple

import numpy as np

from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.GeomAbs import GeomAbs_Line
from OCC.Core.HLRAlgo import HLRAlgo_Projector
from OCC.Core.HLRBRep import HLRBRep_Algo, HLRBRep_HLRToShape
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_REVERSED
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.TopoDS import topods
from OCC.Core.gp import gp_Ax2, gp_Dir, gp_Pnt


logger = logging.getLogger(__name__)


def _normalize(
    vector: List[float]
) -> Tuple[float, float, float]:

    if len(vector) != 3:
        raise ValueError(
            "Direction vector must contain exactly 3 values"
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

    value = value.strip().upper()

    if (
        len(value) != 7
        or not value.startswith("#")
    ):
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


def _read_and_mesh_step(
    file_path: str,
    linear_deflection: float,
    angular_deflection: float
):

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

    transferred_roots = (
        reader.TransferRoots()
    )

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

    return shape


def _count_triangles(shape) -> int:

    total = 0

    explorer = TopExp_Explorer(
        shape,
        TopAbs_FACE
    )

    while explorer.More():

        face = topods.Face(
            explorer.Current()
        )

        location = TopLoc_Location()

        triangulation = (
            BRep_Tool.Triangulation(
                face,
                location
            )
        )

        if triangulation is not None:
            total += (
                triangulation.NbTriangles()
            )

        explorer.Next()

    return total


def _sample_hlr_edge(
    edge,
    curve_samples: int = 64
):

    if BRep_Tool.Degenerated(
        edge
    ):
        return None

    adaptor = BRepAdaptor_Curve(
        edge
    )

    first_parameter = (
        adaptor.FirstParameter()
    )

    last_parameter = (
        adaptor.LastParameter()
    )

    if (
        not math.isfinite(
            first_parameter
        )
        or not math.isfinite(
            last_parameter
        )
    ):
        return None

    if abs(
        last_parameter
        - first_parameter
    ) < 1.0e-14:
        return None

    if adaptor.GetType() == GeomAbs_Line:
        sample_count = 2
    else:
        sample_count = curve_samples

    parameters = np.linspace(
        first_parameter,
        last_parameter,
        sample_count,
        dtype=np.float64
    )

    projected_points = []

    for parameter in parameters:

        point = adaptor.Value(
            float(parameter)
        )

        candidate = (
            point.X(),
            point.Y()
        )

        if not projected_points:

            projected_points.append(
                candidate
            )

            continue

        previous_point = (
            projected_points[-1]
        )

        if (
            abs(
                candidate[0]
                - previous_point[0]
            ) > 1.0e-10
            or
            abs(
                candidate[1]
                - previous_point[1]
            ) > 1.0e-10
        ):
            projected_points.append(
                candidate
            )

    if len(
        projected_points
    ) < 2:
        return None

    return projected_points


def _extract_edges_from_hlr_shape(
    hlr_shape
):

    projected_edges = []

    if (
        hlr_shape is None
        or hlr_shape.IsNull()
    ):
        return projected_edges

    explorer = TopExp_Explorer(
        hlr_shape,
        TopAbs_EDGE
    )

    while explorer.More():

        edge = topods.Edge(
            explorer.Current()
        )

        try:

            projected_points = (
                _sample_hlr_edge(
                    edge
                )
            )

            if projected_points:

                projected_edges.append(
                    projected_points
                )

        except Exception as exc:

            logger.warning(
                "Could not sample one HLR edge: {}".format(
                    exc
                )
            )

        explorer.Next()

    return projected_edges


def _extract_hlr_edge_sets(
    shape,
    view,
    right,
    show_hidden_edges: bool
):

    logger.info(
        "CPU RENDER 9: running exact Open CASCADE HLR"
    )

    # view_direction points from camera toward the model.
    # The HLR projection-axis normal points toward the camera.

    eye_direction = (
        -view[0],
        -view[1],
        -view[2]
    )

    projection_axis = gp_Ax2(
        gp_Pnt(
            0.0,
            0.0,
            0.0
        ),
        gp_Dir(
            eye_direction[0],
            eye_direction[1],
            eye_direction[2]
        ),
        gp_Dir(
            right[0],
            right[1],
            right[2]
        )
    )

    algorithm = HLRBRep_Algo()

    algorithm.Add(
        shape
    )

    projector = HLRAlgo_Projector(
        projection_axis
    )

    algorithm.Projector(
        projector
    )

    algorithm.Update()

    algorithm.Hide()

    hlr_result = HLRBRep_HLRToShape(
        algorithm
    )

    visible_shape = (
        hlr_result.VCompound()
    )

    visible_edges = (
        _extract_edges_from_hlr_shape(
            visible_shape
        )
    )

    hidden_edges = []

    if show_hidden_edges:

        hidden_shape = (
            hlr_result.HCompound()
        )

        hidden_edges = (
            _extract_edges_from_hlr_shape(
                hidden_shape
            )
        )

    logger.info(
        "CPU RENDER 10: HLR extracted "
        "{} visible and {} hidden edges".format(
            len(
                visible_edges
            ),
            len(
                hidden_edges
            )
        )
    )

    return (
        visible_edges,
        hidden_edges
    )


def render_step_svg(
    file_path: str,
    output_path: str,
    view_direction: List[float],
    up_direction: List[float],
    highlight_faces: Optional[List[str]] = None,
    highlight_color: str = "#FF0000",
    base_color: str = "#BFC5CC",
    edge_color: str = "#000000",
    background_color: str = "#FFFFFF",
    width: int = 1600,
    height: int = 1000,
    linear_deflection: float = 1.0,
    angular_deflection: float = 0.35,
    show_hidden_edges: bool = False,
    hidden_edge_color: str = "#808080"
) -> str:

    # ---------------------------------------------------------
    # INPUT VALIDATION
    # ---------------------------------------------------------

    if not os.path.exists(
        file_path
    ):
        raise FileNotFoundError(
            "STEP file not found: {}".format(
                file_path
            )
        )

    if width < 200 or height < 200:
        raise ValueError(
            "SVG width and height must be at least 200 pixels"
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

    hidden_edge_color = _validate_color(
        hidden_edge_color,
        "hidden_edge_color"
    )

    background_color = _validate_color(
        background_color,
        "background_color"
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
        list(
            right_raw
        )
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
    # FACE SELECTION
    # ---------------------------------------------------------

    highlight_set = {
        str(face_id).strip().upper()
        for face_id in (
            highlight_faces or []
        )
        if str(face_id).strip()
    }

    # ---------------------------------------------------------
    # READ AND MESH STEP
    # ---------------------------------------------------------

    shape = _read_and_mesh_step(
        file_path,
        linear_deflection,
        angular_deflection
    )

    # ---------------------------------------------------------
    # COUNT TRIANGLES
    # ---------------------------------------------------------

    logger.info(
        "CPU RENDER 5: counting B-Rep triangles"
    )

    triangle_count = _count_triangles(
        shape
    )

    if triangle_count <= 0:
        raise RuntimeError(
            "Open CASCADE did not generate any display triangles"
        )

    logger.info(
        "CPU RENDER 6: allocating compact arrays "
        "for {} triangles".format(
            triangle_count
        )
    )

    # ---------------------------------------------------------
    # COMPACT TRIANGLE ARRAYS
    # ---------------------------------------------------------

    points = np.empty(
        (
            triangle_count,
            3,
            2
        ),
        dtype=np.float32
    )

    depths = np.empty(
        triangle_count,
        dtype=np.float32
    )

    highlighted = np.zeros(
        triangle_count,
        dtype=np.uint8
    )

    projected_x_min = float(
        "inf"
    )

    projected_x_max = float(
        "-inf"
    )

    projected_y_min = float(
        "inf"
    )

    projected_y_max = float(
        "-inf"
    )

    # ---------------------------------------------------------
    # PROJECT FACE TRIANGLES
    # ---------------------------------------------------------

    explorer = TopExp_Explorer(
        shape,
        TopAbs_FACE
    )

    face_index = 1
    write_index = 0

    logger.info(
        "CPU RENDER 7: extracting projected triangles"
    )

    while explorer.More():

        face = topods.Face(
            explorer.Current()
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
                is_highlighted = 1
            else:
                is_highlighted = 0

            triangle_total = (
                triangulation.NbTriangles()
            )

            for triangle_index in range(
                1,
                triangle_total + 1
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

                depth_sum = 0.0

                for (
                    local_index,
                    node_index
                ) in enumerate(
                    (
                        node_1,
                        node_2,
                        node_3
                    )
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

                    points[
                        write_index,
                        local_index,
                        0
                    ] = projected_x

                    points[
                        write_index,
                        local_index,
                        1
                    ] = projected_y

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

                depths[
                    write_index
                ] = depth_sum / 3.0

                highlighted[
                    write_index
                ] = is_highlighted

                write_index += 1

        face_index += 1

        explorer.Next()

    if write_index <= 0:
        raise RuntimeError(
            "No projected B-Rep triangles were extracted"
        )

    if write_index != triangle_count:

        points = points[
            :write_index
        ]

        depths = depths[
            :write_index
        ]

        highlighted = highlighted[
            :write_index
        ]

        triangle_count = write_index

    logger.info(
        "CPU RENDER 8: extracted {} triangles".format(
            triangle_count
        )
    )

    # ---------------------------------------------------------
    # EXACT HIDDEN-LINE REMOVAL
    # ---------------------------------------------------------

    visible_edges, hidden_edges = (
        _extract_hlr_edge_sets(
            shape,
            view,
            right,
            show_hidden_edges
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

    if (
        projected_width <= 1.0e-12
        or projected_height <= 1.0e-12
    ):
        raise RuntimeError(
            "Invalid projected model dimensions"
        )

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

    scale = min(
        usable_width / projected_width,
        usable_height / projected_height
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

    # ---------------------------------------------------------
    # CONVERT FACE POINTS TO IMAGE COORDINATES
    # ---------------------------------------------------------

    points[:, :, 0] = (
        x_offset
        + (
            points[:, :, 0]
            - projected_x_min
        ) * scale
    )

    points[:, :, 1] = (
        float(height)
        - (
            y_offset
            + (
                points[:, :, 1]
                - projected_y_min
            ) * scale
        )
    )

    logger.info(
        "CPU RENDER 11: sorting triangles by depth"
    )

    draw_order = np.argsort(
        depths
    )[::-1]

    def hlr_to_image(
        projected_point
    ):

        image_x = (
            x_offset
            + (
                projected_point[0]
                - projected_x_min
            ) * scale
        )

        image_y = (
            float(height)
            - (
                y_offset
                + (
                    projected_point[1]
                    - projected_y_min
                ) * scale
            )
        )

        return (
            image_x,
            image_y
        )

    # ---------------------------------------------------------
    # WRITE SVG
    # ---------------------------------------------------------

    logger.info(
        "CPU RENDER 12: writing SVG"
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

        # -----------------------------------------------------
        # FILLED FACE TRIANGLES
        # -----------------------------------------------------

        face_buffer = []
        buffer_limit = 2000

        for ordered_index in draw_order:

            triangle_points = points[
                ordered_index
            ]

            if highlighted[
                ordered_index
            ]:
                color = highlight_color
            else:
                color = base_color

            polygon_points = (
                "{:.3f},{:.3f} "
                "{:.3f},{:.3f} "
                "{:.3f},{:.3f}"
            ).format(
                float(
                    triangle_points[0, 0]
                ),
                float(
                    triangle_points[0, 1]
                ),
                float(
                    triangle_points[1, 0]
                ),
                float(
                    triangle_points[1, 1]
                ),
                float(
                    triangle_points[2, 0]
                ),
                float(
                    triangle_points[2, 1]
                )
            )

            face_buffer.append(
                '<polygon '
                'points="{}" '
                'fill="{}" '
                'stroke="none"/>\n'.format(
                    polygon_points,
                    color
                )
            )

            if len(
                face_buffer
            ) >= buffer_limit:

                svg.write(
                    "".join(
                        face_buffer
                    )
                )

                face_buffer.clear()

        if face_buffer:

            svg.write(
                "".join(
                    face_buffer
                )
            )

        # -----------------------------------------------------
        # OPTIONAL HIDDEN EDGES
        # -----------------------------------------------------

        if (
            show_hidden_edges
            and hidden_edges
        ):

            svg.write(
                '<g '
                'fill="none" '
                'stroke="{}" '
                'stroke-width="1.0" '
                'stroke-dasharray="6,6" '
                'stroke-linecap="round" '
                'stroke-linejoin="round">\n'.format(
                    hidden_edge_color
                )
            )

            for edge_points in hidden_edges:

                svg_points = []

                for point in edge_points:

                    image_x, image_y = (
                        hlr_to_image(
                            point
                        )
                    )

                    svg_points.append(
                        "{:.3f},{:.3f}".format(
                            image_x,
                            image_y
                        )
                    )

                if len(
                    svg_points
                ) >= 2:

                    svg.write(
                        '<polyline '
                        'points="{}"/>\n'.format(
                            " ".join(
                                svg_points
                            )
                        )
                    )

            svg.write(
                "</g>\n"
            )

        # -----------------------------------------------------
        # VISIBLE EDGES ONLY
        # -----------------------------------------------------

        svg.write(
            '<g '
            'fill="none" '
            'stroke="{}" '
            'stroke-width="1.5" '
            'stroke-linecap="round" '
            'stroke-linejoin="round">\n'.format(
                edge_color
            )
        )

        for edge_points in visible_edges:

            svg_points = []

            for point in edge_points:

                image_x, image_y = (
                    hlr_to_image(
                        point
                    )
                )

                svg_points.append(
                    "{:.3f},{:.3f}".format(
                        image_x,
                        image_y
                    )
                )

            if len(
                svg_points
            ) >= 2:

                svg.write(
                    '<polyline '
                    'points="{}"/>\n'.format(
                        " ".join(
                            svg_points
                        )
                    )
                )

        svg.write(
            "</g>\n"
        )

        svg.write(
            "</svg>\n"
        )

    # ---------------------------------------------------------
    # VERIFY OUTPUT
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
        "CPU RENDER 13: "
        "SVG generated, size={} bytes".format(
            output_size
        )
    )

    return output_path
