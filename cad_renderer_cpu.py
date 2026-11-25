"""CPU-only Open CASCADE STEP renderer with SVG HLR, sections and datum overlays."""
import html
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from OCC.Core.BRep import BRep_Builder, BRep_Tool
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Section
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
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
from OCC.Core.gp import gp_Ax2, gp_Dir, gp_Pln, gp_Pnt


def _normalize(v):
    if len(v) != 3:
        raise ValueError("A 3D vector requires three values")
    m = math.sqrt(sum(float(x) ** 2 for x in v))
    if m < 1e-12:
        raise ValueError("Direction vector cannot be zero")
    return tuple(float(x) / m for x in v)


def _dot(a, b):
    return sum(a[i] * b[i] for i in range(3))


def _cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def _color(value, name):
    value = value.strip().upper()
    if len(value) != 7 or not value.startswith("#"):
        raise ValueError("{} must use #RRGGBB".format(name))
    int(value[1:], 16)
    return value


def _read_shape(path, linear_deflection, angular_deflection):
    reader = STEPControl_Reader()
    if reader.ReadFile(path) != IFSelect_RetDone:
        raise RuntimeError("Open CASCADE could not read the STEP file")
    if reader.TransferRoots() <= 0:
        raise RuntimeError("No STEP roots were transferred")
    shape = reader.OneShape()
    if shape.IsNull():
        raise RuntimeError("The STEP file produced a null shape")
    mesh = BRepMesh_IncrementalMesh(shape, linear_deflection, False, angular_deflection, True)
    mesh.Perform()
    if not mesh.IsDone():
        raise RuntimeError("Open CASCADE B-Rep meshing failed")
    return shape


def _hlr_edges(shape, view, right, hidden):
    solid_compound = TopoDS_Compound()
    builder = BRep_Builder()
    builder.MakeCompound(solid_compound)
    ex = TopExp_Explorer(shape, TopAbs_SOLID)
    count = 0
    while ex.More():
        builder.Add(solid_compound, ex.Current())
        count += 1
        ex.Next()
    hlr_shape = solid_compound if count else shape
    axis = gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(-view[0], -view[1], -view[2]), gp_Dir(*right))
    algo = HLRBRep_Algo()
    algo.Add(hlr_shape)
    algo.Projector(HLRAlgo_Projector(axis))
    algo.Update()
    algo.Hide()
    result = HLRBRep_HLRToShape(algo)

    def extract(compounds):
        output = []
        for compound in compounds:
            if compound is None or compound.IsNull():
                continue
            edge_ex = TopExp_Explorer(compound, TopAbs_EDGE)
            while edge_ex.More():
                edge = topods.Edge(edge_ex.Current())
                if not BRep_Tool.Degenerated(edge):
                    adaptor = BRepAdaptor_Curve(edge)
                    first, last = adaptor.FirstParameter(), adaptor.LastParameter()
                    if math.isfinite(first) and math.isfinite(last) and abs(last-first) > 1e-14:
                        n = 2 if adaptor.GetType() == GeomAbs_Line else 64
                        pts = []
                        for parameter in np.linspace(first, last, n):
                            point = adaptor.Value(float(parameter))
                            pts.append((point.X(), point.Y()))
                        output.append(pts)
                edge_ex.Next()
        return output

    visible = extract([result.VCompound(), result.OutLineVCompound()])
    hidden_edges = extract([result.HCompound(), result.OutLineHCompound()]) if hidden else []
    return visible, hidden_edges


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
    reference_entities: Optional[List[Dict[str, Any]]] = None,
    reference_color: str = "#606060",
    reference_scale: float = 1.0,
    show_reference_labels: bool = False,
    reference_label_names: Optional[List[str]] = None,
    face_colors: Optional[Dict[str, str]] = None,
    component_colors: Optional[Dict[str, str]] = None,
    display_mode: str = "SHADED",
) -> str:
    del component_colors  # Reserved for native XCAF component-to-shape mapping.
    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)
    if width < 200 or height < 200 or width > 4000 or height > 4000:
        raise ValueError("Image dimensions must be between 200 and 4000")
    if reference_scale <= 0:
        raise ValueError("reference_scale must be greater than zero")
    display_mode = display_mode.strip().upper()
    if display_mode not in {"SHADED", "WIREFRAME", "HIDDEN_LINE", "HIDDEN_LINE_DASHED"}:
        raise ValueError("Unsupported display_mode")
    if display_mode == "HIDDEN_LINE_DASHED":
        show_hidden_edges = True
    highlight_color = _color(highlight_color, "highlight_color")
    base_color = _color(base_color, "base_color")
    edge_color = _color(edge_color, "edge_color")
    background_color = _color(background_color, "background_color")
    hidden_edge_color = _color(hidden_edge_color, "hidden_edge_color")
    reference_color = _color(reference_color, "reference_color")
    face_colors = {k.strip().upper(): _color(v, "face_colors") for k, v in (face_colors or {}).items()}
    highlight_set = {x.strip().upper() for x in (highlight_faces or [])}
    label_set = {x.strip().upper() for x in (reference_label_names or [])}

    view = _normalize(view_direction)
    requested_up = _normalize(up_direction)
    right = _normalize(_cross(view, requested_up))
    up = _normalize(_cross(right, view))
    shape = _read_shape(file_path, linear_deflection, angular_deflection)

    triangles, depths, tri_colors = [], [], []
    xmin = ymin = float("inf")
    xmax = ymax = float("-inf")
    ex = TopExp_Explorer(shape, TopAbs_FACE)
    face_index = 1
    while ex.More():
        face = topods.Face(ex.Current())
        face_id = "FACE_{}".format(face_index)
        location = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation(face, location)
        if triangulation is not None:
            transform = location.Transformation()
            reversed_face = face.Orientation() == TopAbs_REVERSED
            current_color = highlight_color if face_id in highlight_set else face_colors.get(face_id, base_color)
            for index in range(1, triangulation.NbTriangles()+1):
                nodes = list(triangulation.Triangle(index).Get())
                if reversed_face:
                    nodes[1], nodes[2] = nodes[2], nodes[1]
                projected, depth = [], 0.0
                for node in nodes:
                    point = triangulation.Node(node).Transformed(transform)
                    xyz = (point.X(), point.Y(), point.Z())
                    px, py = _dot(xyz, right), _dot(xyz, up)
                    projected.append((px, py))
                    depth += _dot(xyz, view)
                    xmin, xmax, ymin, ymax = min(xmin, px), max(xmax, px), min(ymin, py), max(ymax, py)
                triangles.append(projected)
                depths.append(depth/3.0)
                tri_colors.append(current_color)
        face_index += 1
        ex.Next()
    if not triangles:
        raise RuntimeError("No B-Rep display triangles were extracted")

    visible, hidden = _hlr_edges(shape, view, right, show_hidden_edges)
    span_x, span_y = max(xmax-xmin, 1e-9), max(ymax-ymin, 1e-9)
    margin = 40.0
    scale = min((width-2*margin)/span_x, (height-2*margin)/span_y)
    xoff, yoff = (width-span_x*scale)/2, (height-span_y*scale)/2

    def image2(point):
        return (xoff+(point[0]-xmin)*scale, height-(yoff+(point[1]-ymin)*scale))

    def image3(point):
        return image2((_dot(point, right), _dot(point, up)))

    with open(output_path, "w", encoding="utf-8") as svg:
        svg.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        svg.write('<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}">\n'.format(width,height,width,height))
        svg.write('<rect width="{}" height="{}" fill="{}"/>\n'.format(width,height,background_color))
        if display_mode == "SHADED":
            for idx in np.argsort(np.asarray(depths))[::-1]:
                pts = [image2(p) for p in triangles[int(idx)]]
                svg.write('<polygon points="{}" fill="{}" stroke="none"/>\n'.format(" ".join("{:.3f},{:.3f}".format(*p) for p in pts), tri_colors[int(idx)]))
        elif display_mode in {"HIDDEN_LINE", "HIDDEN_LINE_DASHED"}:
            for idx in np.argsort(np.asarray(depths))[::-1]:
                pts = [image2(p) for p in triangles[int(idx)]]
                svg.write('<polygon points="{}" fill="#FFFFFF" stroke="none"/>\n'.format(" ".join("{:.3f},{:.3f}".format(*p) for p in pts)))
        if show_hidden_edges and hidden:
            svg.write('<g fill="none" stroke="{}" stroke-width="1" stroke-dasharray="6,6">\n'.format(hidden_edge_color))
            for edge in hidden:
                svg.write('<polyline points="{}"/>\n'.format(" ".join("{:.3f},{:.3f}".format(*image2(p)) for p in edge)))
            svg.write('</g>\n')
        svg.write('<g fill="none" stroke="{}" stroke-width="1.5">\n'.format(edge_color))
        for edge in visible:
            svg.write('<polyline points="{}"/>\n'.format(" ".join("{:.3f},{:.3f}".format(*image2(p)) for p in edge)))
        svg.write('</g>\n')

        if show_references and reference_entities:
            length = reference_scale * 0.15 * max(span_x, span_y)
            half = reference_scale * 0.12 * max(span_x, span_y)
            svg.write('<g fill="none" stroke="{}" stroke-width="1.2" stroke-dasharray="7,5">\n'.format(reference_color))
            for item in reference_entities:
                origin_data = item.get("origin") or {}
                if not all(k in origin_data for k in ("x","y","z")):
                    continue
                origin = tuple(float(origin_data[k]) for k in ("x","y","z"))
                ox, oy = image3(origin)
                item_type = item.get("type")
                if item_type == "DATUM_PLANE" and item.get("normal"):
                    normal = tuple(item["normal"][k] for k in ("x","y","z"))
                    helper = (1,0,0) if abs(normal[0]) < 0.9 else (0,1,0)
                    ux = _normalize(_cross(normal, helper)); uy = _normalize(_cross(normal, ux))
                    corners = []
                    for sx, sy in ((-1,-1),(1,-1),(1,1),(-1,1)):
                        corners.append(image3(tuple(origin[i]+half*(sx*ux[i]+sy*uy[i]) for i in range(3))))
                    svg.write('<polygon points="{}" fill="{}" fill-opacity="0.10"/>\n'.format(" ".join("{:.3f},{:.3f}".format(*p) for p in corners), reference_color))
                else:
                    vectors = []
                    if item_type == "DATUM_CSYS":
                        vectors = [item.get(k) for k in ("x_axis","y_axis","z_axis")]
                    elif item_type == "DATUM_AXIS":
                        vectors = [item.get("direction") or item.get("z_axis")]
                    for data in vectors:
                        if not data:
                            continue
                        vector = tuple(float(data[k]) for k in ("x","y","z"))
                        first = tuple(origin[i]-vector[i]*length for i in range(3)) if item_type == "DATUM_AXIS" else origin
                        second = tuple(origin[i]+vector[i]*length for i in range(3))
                        x1,y1=image3(first); x2,y2=image3(second)
                        svg.write('<line x1="{:.3f}" y1="{:.3f}" x2="{:.3f}" y2="{:.3f}"/>\n'.format(x1,y1,x2,y2))
                name = str(item.get("name", ""))
                if show_reference_labels and (not label_set or name.upper() in label_set or "ALL" in label_set):
                    svg.write('<text x="{:.3f}" y="{:.3f}" fill="{}" stroke="none" font-size="14">{}</text>\n'.format(ox+8,oy-8,reference_color,html.escape(name)))
            svg.write('</g>\n')
        svg.write('</svg>\n')
    return output_path


def render_step_section_svg(file_path, output_path, plane_origin, plane_normal, up_direction, width=1600, height=1000, line_color="#000000", fill_color="#D9D9D9", background_color="#FFFFFF", curve_samples=64):
    normal = _normalize(plane_normal); up_req = _normalize(up_direction)
    right = _normalize(_cross(normal, up_req)); up = _normalize(_cross(right, normal))
    shape = _read_shape(file_path, 1.0, 0.35)
    plane = gp_Pln(gp_Pnt(*[float(x) for x in plane_origin]), gp_Dir(*normal))
    section = BRepAlgoAPI_Section(shape, BRepBuilderAPI_MakeFace(plane).Face(), False)
    section.Approximation(True); section.Build()
    if not section.IsDone():
        raise RuntimeError("Open CASCADE section operation failed")
    polylines=[]; ex=TopExp_Explorer(section.Shape(),TopAbs_EDGE)
    while ex.More():
        adaptor=BRepAdaptor_Curve(topods.Edge(ex.Current())); first,last=adaptor.FirstParameter(),adaptor.LastParameter()
        if math.isfinite(first) and math.isfinite(last) and abs(last-first)>1e-14:
            n=2 if adaptor.GetType()==GeomAbs_Line else curve_samples
            polylines.append([(_dot((adaptor.Value(float(t)).X(),adaptor.Value(float(t)).Y(),adaptor.Value(float(t)).Z()),right),_dot((adaptor.Value(float(t)).X(),adaptor.Value(float(t)).Y(),adaptor.Value(float(t)).Z()),up)) for t in np.linspace(first,last,n)])
        ex.Next()
    if not polylines: raise RuntimeError("The section plane does not intersect the STEP model")
    pts=[p for line in polylines for p in line]; xmin=min(p[0] for p in pts); xmax=max(p[0] for p in pts); ymin=min(p[1] for p in pts); ymax=max(p[1] for p in pts)
    sx=max(xmax-xmin,1e-9); sy=max(ymax-ymin,1e-9); margin=40; scale=min((width-2*margin)/sx,(height-2*margin)/sy); xoff=(width-sx*scale)/2; yoff=(height-sy*scale)/2
    image=lambda p:(xoff+(p[0]-xmin)*scale,height-(yoff+(p[1]-ymin)*scale))
    paths=[]
    for line in polylines:
        converted=[image(p) for p in line]; command="M "+" L ".join("{:.3f} {:.3f}".format(*p) for p in converted)
        if math.hypot(converted[0][0]-converted[-1][0],converted[0][1]-converted[-1][1])<2: command+=" Z"
        paths.append(command)
    with open(output_path,"w",encoding="utf-8") as svg:
        svg.write('<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}"><rect width="{}" height="{}" fill="{}"/><path d="{}" fill="{}" fill-rule="evenodd" stroke="{}" stroke-width="1.5"/></svg>'.format(width,height,width,height,width,height,background_color," ".join(paths),fill_color,line_color))
    return output_path
