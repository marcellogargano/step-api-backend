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

import html
import logging
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from OCC.Core.BRep import BRep_Builder, BRep_Tool
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.GeomAbs import GeomAbs_Line
from OCC.Core.HLRAlgo import HLRAlgo_Projector
from OCC.Core.HLRBRep import HLRBRep_Algo, HLRBRep_HLRToShape
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_REVERSED, TopAbs_SOLID
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.TopoDS import TopoDS_Compound, topods
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

    # HLR must receive only real solid geometry. Free datum axes, datum planes,
    # construction wires and isolated edges are intentionally excluded here.
    # Named references are rendered separately only when explicitly requested.
    solid_compound = TopoDS_Compound()
    solid_builder = BRep_Builder()
    solid_builder.MakeCompound(solid_compound)
    solid_explorer = TopExp_Explorer(shape, TopAbs_SOLID)
    solid_count = 0

    while solid_explorer.More():
        solid_builder.Add(solid_compound, solid_explorer.Current())
        solid_count += 1
        solid_explorer.Next()

    hlr_shape = solid_compound if solid_count > 0 else shape

    algorithm = HLRBRep_Algo()

    algorithm.Add(
        hlr_shape
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

    # A complete engineering view is the union of real visible sharp edges
    # and view-dependent apparent contours (silhouettes). OutLineVCompound
    # supplies cylinder/sphere/torus contours that are not real B-Rep edges.
    visible_edges = []
    visible_edges.extend(_extract_edges_from_hlr_shape(hlr_result.VCompound()))
    visible_edges.extend(_extract_edges_from_hlr_shape(hlr_result.OutLineVCompound()))

    hidden_edges = []

    if show_hidden_edges:
        hidden_edges.extend(_extract_edges_from_hlr_shape(hlr_result.HCompound()))
        hidden_edges.extend(_extract_edges_from_hlr_shape(hlr_result.OutLineHCompound()))

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
    hidden_edge_color: str = "#808080",
    show_references: bool = False,
    reference_names: Optional[List[str]] = None,
    reference_entities: Optional[List[Dict[str, Any]]] = None,
    reference_color: str = "#606060",
    reference_scale: float = 1.0
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

    if reference_scale <= 0.0:
        raise ValueError("reference_scale must be greater than zero")

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

    reference_color = _validate_color(
        reference_color,
        "reference_color"
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

        # Named datum/reference geometry is a separate graphic layer. Hidden
        # edges never enable datum geometry. References appear only on request.
        if show_references and reference_entities:
            selected = list(reference_entities)
            reference_length = reference_scale * 0.15 * max(projected_width, projected_height)
            plane_half_size = reference_scale * 0.12 * max(projected_width, projected_height)
            svg.write(
                '<g fill="none" stroke="{}" stroke-width="1.2" '
                'stroke-dasharray="7,5" stroke-linecap="round">\n'.format(reference_color)
            )

            def project_reference_point(point_3d):
                projected_x = _dot(point_3d, right)
                projected_y = _dot(point_3d, up)
                return (
                    x_offset + (projected_x - projected_x_min) * scale,
                    float(height) - (y_offset + (projected_y - projected_y_min) * scale),
                )

            def vector_tuple(data):
                if not data or not all(key in data for key in ("x", "y", "z")):
                    return None
                return (float(data["x"]), float(data["y"]), float(data["z"]))

            for item in selected:
                origin_data = item.get("origin") or {}
                if not all(key in origin_data for key in ("x", "y", "z")):
                    continue
                origin_3d = (float(origin_data["x"]), float(origin_data["y"]), float(origin_data["z"]))
                ox, oy = project_reference_point(origin_3d)
                item_type = item.get("type", "")

                if item_type == "DATUM_PLANE":
                    normal = vector_tuple(item.get("normal"))
                    if normal:
                        helper = (1.0, 0.0, 0.0) if abs(normal[0]) < 0.9 else (0.0, 1.0, 0.0)
                        plane_x = _normalize(list(_cross(normal, helper)))
                        plane_y = _normalize(list(_cross(normal, plane_x)))
                        corners_3d = []
                        for sx, sy in ((-1,-1),(1,-1),(1,1),(-1,1)):
                            corners_3d.append((
                                origin_3d[0] + plane_half_size * (sx * plane_x[0] + sy * plane_y[0]),
                                origin_3d[1] + plane_half_size * (sx * plane_x[1] + sy * plane_y[1]),
                                origin_3d[2] + plane_half_size * (sx * plane_x[2] + sy * plane_y[2]),
                            ))
                        corners = [project_reference_point(point) for point in corners_3d]
                        svg.write('<polygon points="{}" fill="{}" fill-opacity="0.10"/>\n'.format(
                            " ".join("{:.3f},{:.3f}".format(x, y) for x, y in corners), reference_color
                        ))
                elif item_type == "DATUM_CSYS":
                    for key in ("x_axis", "y_axis", "z_axis"):
                        vector = vector_tuple(item.get(key))
                        if not vector:
                            continue
                        endpoint = tuple(origin_3d[index] + vector[index] * reference_length for index in range(3))
                        ex, ey = project_reference_point(endpoint)
                        svg.write('<line x1="{:.3f}" y1="{:.3f}" x2="{:.3f}" y2="{:.3f}"/>\n'.format(ox, oy, ex, ey))
                elif item_type == "DATUM_AXIS":
                    vector = vector_tuple(item.get("direction") or item.get("z_axis"))
                    if vector:
                        first = tuple(origin_3d[index] - vector[index] * reference_length for index in range(3))
                        second = tuple(origin_3d[index] + vector[index] * reference_length for index in range(3))
                        x1, y1 = project_reference_point(first); x2, y2 = project_reference_point(second)
                        svg.write('<line x1="{:.3f}" y1="{:.3f}" x2="{:.3f}" y2="{:.3f}"/>\n'.format(x1, y1, x2, y2))
                else:
                    svg.write('<line x1="{:.3f}" y1="{:.3f}" x2="{:.3f}" y2="{:.3f}"/>\n'.format(ox - 6, oy, ox + 6, oy))
                    svg.write('<line x1="{:.3f}" y1="{:.3f}" x2="{:.3f}" y2="{:.3f}"/>\n'.format(ox, oy - 6, ox, oy + 6))

                label = html.escape(str(item.get("name", "")))
                svg.write('<text x="{:.3f}" y="{:.3f}" fill="{}" stroke="none" font-size="14">{}</text>\n'.format(ox + 8, oy - 8, reference_color, label))

            svg.write("</g>\n")

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


def render_step_section_svg(
    file_path,
    output_path,
    plane_origin,
    plane_normal,
    up_direction,
    width=1600,
    height=1000,
    line_color="#000000",
    fill_color="#D9D9D9",
    background_color="#FFFFFF",
    curve_samples=64,
):
    """Create a filled orthographic SVG of a true B-Rep planar section."""
    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Section
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
    from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
    from OCC.Core.GeomAbs import GeomAbs_Line
    from OCC.Core.gp import gp_Dir, gp_Pln, gp_Pnt

    for value, name in ((line_color, "line_color"), (fill_color, "fill_color"), (background_color, "background_color")):
        _validate_color(value, name)
    normal = _normalize(list(plane_normal))
    requested_up = _normalize(list(up_direction))
    right_raw = _cross(normal, requested_up)
    if math.sqrt(_dot(right_raw, right_raw)) < 1.0e-12:
        raise ValueError("plane_normal and up_direction cannot be parallel")
    right = _normalize(list(right_raw))
    up = _normalize(list(_cross(right, normal)))
    shape = _read_and_mesh_step(file_path, 1.0, 0.35)
    plane = gp_Pln(gp_Pnt(*[float(v) for v in plane_origin]), gp_Dir(*normal))
    plane_face = BRepBuilderAPI_MakeFace(plane).Face()
    section = BRepAlgoAPI_Section(shape, plane_face, False)
    section.Approximation(True)
    section.Build()
    if not section.IsDone():
        raise RuntimeError("Open CASCADE section operation failed")
    section_shape = section.Shape()
    polylines = []
    explorer = TopExp_Explorer(section_shape, TopAbs_EDGE)
    while explorer.More():
        edge = topods.Edge(explorer.Current())
        adaptor = BRepAdaptor_Curve(edge)
        first = adaptor.FirstParameter()
        last = adaptor.LastParameter()
        if math.isfinite(first) and math.isfinite(last) and abs(last - first) > 1e-14:
            sample_count = 2 if adaptor.GetType() == GeomAbs_Line else curve_samples
            points_2d = []
            for parameter in np.linspace(first, last, sample_count):
                point = adaptor.Value(float(parameter))
                point_3d = (point.X(), point.Y(), point.Z())
                points_2d.append((_dot(point_3d, right), _dot(point_3d, up)))
            if len(points_2d) >= 2:
                polylines.append(points_2d)
        explorer.Next()
    if not polylines:
        raise RuntimeError("The section plane does not intersect the STEP model")
    all_points = [point for polyline in polylines for point in polyline]
    xmin = min(point[0] for point in all_points); xmax = max(point[0] for point in all_points)
    ymin = min(point[1] for point in all_points); ymax = max(point[1] for point in all_points)
    span_x = max(xmax - xmin, 1e-9); span_y = max(ymax - ymin, 1e-9)
    margin = 40.0
    scale = min((width - 2 * margin) / span_x, (height - 2 * margin) / span_y)
    xoff = (width - span_x * scale) / 2.0
    yoff = (height - span_y * scale) / 2.0
    def image(point):
        return (xoff + (point[0] - xmin) * scale, height - (yoff + (point[1] - ymin) * scale))
    paths = []
    for polyline in polylines:
        converted = [image(point) for point in polyline]
        command = "M " + " L ".join("{:.3f} {:.3f}".format(x, y) for x, y in converted)
        if math.hypot(converted[0][0] - converted[-1][0], converted[0][1] - converted[-1][1]) < 2.0:
            command += " Z"
        paths.append(command)
    with open(output_path, "w", encoding="utf-8") as svg:
        svg.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        svg.write('<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}">\n'.format(width, height, width, height))
        svg.write('<rect width="{}" height="{}" fill="{}"/>\n'.format(width, height, background_color))
        svg.write('<path d="{}" fill="{}" fill-rule="evenodd" stroke="{}" stroke-width="1.5"/>\n'.format(" ".join(paths), fill_color, line_color))
        svg.write('</svg>\n')
    return output_path
