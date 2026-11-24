"""
CAD rendering from real Open CASCADE B-Rep geometry.

Open CASCADE reads and triangulates the STEP B-Rep.
VTK is used only to render the resulting face triangulations.
Face selection and face numbering remain based on TopoDS_Face.
"""

import math
import os
from typing import List, Optional, Tuple

from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_REVERSED
from OCC.Core.TopoDS import topods
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib


def _hex_to_rgb(color: str) -> Tuple[int, int, int]:
    """Convert #RRGGBB to integer RGB."""

    value = color.strip().lstrip("#")

    if len(value) != 6:
        raise ValueError(
            f"Invalid color '{color}'. Expected format #RRGGBB."
        )

    return (
        int(value[0:2], 16),
        int(value[2:4], 16),
        int(value[4:6], 16)
    )


def _normalize(vector: List[float]) -> Tuple[float, float, float]:
    """Normalize a 3D vector."""

    if len(vector) != 3:
        raise ValueError("A direction vector must contain three values.")

    x = float(vector[0])
    y = float(vector[1])
    z = float(vector[2])

    magnitude = math.sqrt(x * x + y * y + z * z)

    if magnitude <= 1.0e-12:
        raise ValueError("Direction vector cannot be zero.")

    return (
        x / magnitude,
        y / magnitude,
        z / magnitude
    )


def render_step_view(
    file_path: str,
    output_path: str,
    view_direction: List[float],
    up_direction: List[float],
    highlight_faces: Optional[List[str]] = None,
    highlight_color: str = "#FF0000",
    base_color: str = "#B8BEC7",
    background_color: str = "#FFFFFF",
    width: int = 1600,
    height: int = 1000,
    linear_deflection: float = 1.0,
    angular_deflection: float = 0.35
) -> str:
    """
    Render an orthographic view of a STEP model.

    Geometry source:
        Real Open CASCADE B-Rep from the STEP file.

    Visualization source:
        Open CASCADE face triangulation.

    Parameters:
        view_direction:
            Direction from the camera toward the model.

            Examples:
            [0, 0, -1] = XY view along -Z
            [0, 0,  1] = XY view along +Z
            [-1, 0, 0] = YZ view along -X
            [0, -1, 0] = XZ view along -Y
            [1, 1, -1] = isometric-type direction

        highlight_faces:
            Sequential B-Rep face IDs, for example:
            ["FACE_10", "FACE_25"].
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"STEP file not found: {file_path}")

    if width < 200 or height < 200:
        raise ValueError("Image dimensions must be at least 200 pixels.")

    view_dir = _normalize(view_direction)
    view_up = _normalize(up_direction)

    dot_product = (
        view_dir[0] * view_up[0]
        + view_dir[1] * view_up[1]
        + view_dir[2] * view_up[2]
    )

    if abs(dot_product) > 0.999:
        raise ValueError(
            "view_direction and up_direction cannot be parallel."
        )

    highlight_set = set(highlight_faces or [])

    # Import VTK only when rendering is requested.
    import vtk

    # Read the STEP file with Open CASCADE.
    reader = STEPControl_Reader()
    status = reader.ReadFile(file_path)

    if status != IFSelect_RetDone:
        raise ValueError(f"Open CASCADE could not read: {file_path}")

    reader.TransferRoots()
    shape = reader.OneShape()

    if shape.IsNull():
        raise ValueError("The STEP file produced a null Open CASCADE shape.")

    # Triangulate the Open CASCADE B-Rep.
    mesher = BRepMesh_IncrementalMesh(
        shape,
        linear_deflection,
        False,
        angular_deflection,
        True
    )

    mesher.Perform()

    if not mesher.IsDone():
        raise RuntimeError("Open CASCADE B-Rep meshing failed.")

    vtk_points = vtk.vtkPoints()
    vtk_cells = vtk.vtkCellArray()

    cell_colors = vtk.vtkUnsignedCharArray()
    cell_colors.SetName("FaceColors")
    cell_colors.SetNumberOfComponents(3)

    default_rgb = _hex_to_rgb(base_color)
    highlight_rgb = _hex_to_rgb(highlight_color)

    face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
    face_index = 1
    triangle_count = 0

    while face_explorer.More():

        face = topods.Face(face_explorer.Current())
        face_id = f"FACE_{face_index}"

        location = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation(face, location)

        if triangulation is not None:

            transformation = location.Transformation()
            reverse_face = face.Orientation() == TopAbs_REVERSED

            color = (
                highlight_rgb
                if face_id in highlight_set
                else default_rgb
            )

            for triangle_index in range(
                1,
                triangulation.NbTriangles() + 1
            ):
                triangle = triangulation.Triangle(triangle_index)

                node_1, node_2, node_3 = triangle.Get()

                if reverse_face:
                    node_2, node_3 = node_3, node_2

                point_1 = triangulation.Node(node_1).Transformed(
                    transformation
                )
                point_2 = triangulation.Node(node_2).Transformed(
                    transformation
                )
                point_3 = triangulation.Node(node_3).Transformed(
                    transformation
                )

                point_ids = []

                for point in (point_1, point_2, point_3):
                    point_ids.append(
                        vtk_points.InsertNextPoint(
                            point.X(),
                            point.Y(),
                            point.Z()
                        )
                    )

                vtk_cells.InsertNextCell(3)

                for point_id in point_ids:
                    vtk_cells.InsertCellPoint(point_id)

                cell_colors.InsertNextTuple3(
                    color[0],
                    color[1],
                    color[2]
                )

                triangle_count += 1

        face_index += 1
        face_explorer.Next()

    if triangle_count == 0:
        raise RuntimeError(
            "Open CASCADE did not generate any display triangles."
        )

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(vtk_points)
    polydata.SetPolys(vtk_cells)
    polydata.GetCellData().SetScalars(cell_colors)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    mapper.SetScalarModeToUseCellData()
    mapper.SetColorModeToDirectScalars()
    mapper.ScalarVisibilityOn()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    # Avoid showing internal tessellation edges.
    actor.GetProperty().EdgeVisibilityOff()
    actor.GetProperty().SetInterpolationToPhong()

    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)

    background_rgb = _hex_to_rgb(background_color)

    renderer.SetBackground(
        background_rgb[0] / 255.0,
        background_rgb[1] / 255.0,
        background_rgb[2] / 255.0
    )

    # Exact model bounding box from the B-Rep.
    bounding_box = Bnd_Box()
    brepbndlib.Add(shape, bounding_box)

    xmin, ymin, zmin, xmax, ymax, zmax = bounding_box.Get()

    center_x = 0.5 * (xmin + xmax)
    center_y = 0.5 * (ymin + ymax)
    center_z = 0.5 * (zmin + zmax)

    size_x = xmax - xmin
    size_y = ymax - ymin
    size_z = zmax - zmin

    model_size = max(size_x, size_y, size_z)

    if model_size <= 1.0e-9:
        raise ValueError("The STEP model has a null bounding box.")

    camera_distance = 3.0 * model_size

    camera = vtk.vtkCamera()

    # view_direction is defined from camera toward model.
    camera.SetPosition(
        center_x - view_dir[0] * camera_distance,
        center_y - view_dir[1] * camera_distance,
        center_z - view_dir[2] * camera_distance
    )

    camera.SetFocalPoint(
        center_x,
        center_y,
        center_z
    )

    camera.SetViewUp(
        view_up[0],
        view_up[1],
        view_up[2]
    )

    camera.ParallelProjectionOn()
    camera.SetParallelScale(0.57 * model_size)

    renderer.SetActiveCamera(camera)
    renderer.ResetCameraClippingRange()

    render_window = vtk.vtkRenderWindow()
    render_window.SetOffScreenRendering(1)
    render_window.SetSize(width, height)
    render_window.AddRenderer(renderer)
    render_window.Render()

    window_to_image = vtk.vtkWindowToImageFilter()
    window_to_image.SetInput(render_window)
    window_to_image.SetInputBufferTypeToRGBA()
    window_to_image.ReadFrontBufferOff()
    window_to_image.Update()

    writer = vtk.vtkPNGWriter()
    writer.SetFileName(output_path)
    writer.SetInputConnection(window_to_image.GetOutputPort())
    writer.Write()

    render_window.Finalize()

    if not os.path.exists(output_path):
        raise RuntimeError("The PNG rendering was not generated.")

    return output_path
