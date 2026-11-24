"""
STEP File Processor using Open CASCADE Technology (OCCT).

Provides:
- STEP B-Rep reading
- geometric and topological analysis
- face inspection
- validation
- robust XCAF assembly traversal with names, hierarchy, locations and colors
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

try:
    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
    from OCC.Core.BRepBndLib import brepbndlib
    from OCC.Core.BRepCheck import BRepCheck_Analyzer
    from OCC.Core.BRepGProp import brepgprop
    from OCC.Core.GeomAbs import (
        GeomAbs_BSplineSurface,
        GeomAbs_BezierSurface,
        GeomAbs_Cone,
        GeomAbs_Cylinder,
        GeomAbs_Plane,
        GeomAbs_Sphere,
        GeomAbs_Torus
    )
    from OCC.Core.GProp import GProp_GProps
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.Quantity import Quantity_Color
    from OCC.Core.STEPCAFControl import STEPCAFControl_Reader
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.TCollection import TCollection_ExtendedString
    from OCC.Core.TDF import TDF_Label, TDF_LabelSequence
    from OCC.Core.TDocStd import TDocStd_Document
    from OCC.Core.TopAbs import (
        TopAbs_COMPOUND,
        TopAbs_EDGE,
        TopAbs_FACE,
        TopAbs_SHELL,
        TopAbs_SOLID,
        TopAbs_VERTEX
    )
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopLoc import TopLoc_Location
    from OCC.Core.TopoDS import TopoDS_Shape, topods
    from OCC.Core.XCAFDoc import (
        XCAFDoc_ColorCurv,
        XCAFDoc_ColorGen,
        XCAFDoc_ColorSurf,
        XCAFDoc_DocumentTool_ColorTool,
        XCAFDoc_DocumentTool_LayerTool,
        XCAFDoc_DocumentTool_MaterialTool,
        XCAFDoc_DocumentTool_ShapeTool
    )
    from OCC.Extend.TopologyUtils import TopologyExplorer

    OCCT_AVAILABLE = True

except ImportError:
    OCCT_AVAILABLE = False


logger = logging.getLogger(__name__)


class STEPProcessor:
    """
    Process STEP files and extract B-Rep and XCAF information.
    """

    def __init__(self):

        if not OCCT_AVAILABLE:
            logger.warning(
                "pythonocc-core is not available"
            )

        self.occt_available = OCCT_AVAILABLE

    def is_available(self) -> bool:

        return self.occt_available

    def analyze_file(
        self,
        file_path: str
    ) -> Dict[str, Any]:

        self._require_file(
            file_path
        )

        shape, metadata = (
            self._read_step_file(
                file_path
            )
        )

        return {
            "file_info": {
                "file_size_bytes": os.path.getsize(
                    file_path
                ),
                "file_path": os.path.basename(
                    file_path
                )
            },
            "metadata": metadata,
            "geometry": self._extract_geometric_properties(
                shape
            ),
            "topology": self._extract_topology_info(
                shape
            ),
            "validation": self._validate_shape(
                shape
            ),
            "assembly": self.get_assembly_structure(
                file_path
            )
        }

    def get_geometric_properties(
        self,
        file_path: str
    ) -> Dict[str, Any]:

        self._require_file(
            file_path
        )

        shape, _ = self._read_step_file(
            file_path
        )

        return self._extract_geometric_properties(
            shape
        )

    def get_topology_info(
        self,
        file_path: str
    ) -> Dict[str, Any]:

        self._require_file(
            file_path
        )

        shape, _ = self._read_step_file(
            file_path
        )

        return self._extract_topology_info(
            shape
        )

    def validate_file(
        self,
        file_path: str
    ) -> Dict[str, Any]:

        self._require_file(
            file_path
        )

        shape, _ = self._read_step_file(
            file_path
        )

        return self._validate_shape(
            shape
        )

    def get_faces(
        self,
        file_path: str
    ) -> Dict[str, Any]:
        """
        Return sequential B-Rep face data for the complete
        transferred shape.
        """

        self._require_file(
            file_path
        )

        shape, metadata = self._read_step_file(
            file_path
        )

        surface_names = {
            GeomAbs_Plane: "PLANE",
            GeomAbs_Cylinder: "CYLINDER",
            GeomAbs_Cone: "CONE",
            GeomAbs_Sphere: "SPHERE",
            GeomAbs_Torus: "TORUS",
            GeomAbs_BezierSurface: "BEZIER_SURFACE",
            GeomAbs_BSplineSurface: "BSPLINE_SURFACE"
        }

        faces: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []

        explorer = TopExp_Explorer(
            shape,
            TopAbs_FACE
        )

        face_index = 1

        while explorer.More():

            try:

                face = topods.Face(
                    explorer.Current()
                )

                properties = GProp_GProps()

                brepgprop.SurfaceProperties(
                    face,
                    properties
                )

                centroid = (
                    properties.CentreOfMass()
                )

                adaptor = BRepAdaptor_Surface(
                    face
                )

                surface_type = (
                    surface_names.get(
                        adaptor.GetType(),
                        "OTHER"
                    )
                )

                bounding_box = Bnd_Box()

                brepbndlib.Add(
                    face,
                    bounding_box
                )

                (
                    xmin,
                    ymin,
                    zmin,
                    xmax,
                    ymax,
                    zmax
                ) = bounding_box.Get()

                face_data: Dict[str, Any] = {
                    "face_id": "FACE_{}".format(
                        face_index
                    ),
                    "index": face_index,
                    "surface_type": surface_type,
                    "area": {
                        "value": properties.Mass(),
                        "unit": "square_mm"
                    },
                    "centroid": {
                        "x": centroid.X(),
                        "y": centroid.Y(),
                        "z": centroid.Z(),
                        "unit": "mm"
                    },
                    "bounding_box": {
                        "min": {
                            "x": xmin,
                            "y": ymin,
                            "z": zmin
                        },
                        "max": {
                            "x": xmax,
                            "y": ymax,
                            "z": zmax
                        },
                        "unit": "mm"
                    }
                }

                if surface_type == "PLANE":

                    plane = adaptor.Plane()

                    normal = (
                        plane.Axis()
                        .Direction()
                    )

                    location = (
                        plane.Location()
                    )

                    face_data["plane"] = {
                        "normal": {
                            "x": normal.X(),
                            "y": normal.Y(),
                            "z": normal.Z()
                        },
                        "location": {
                            "x": location.X(),
                            "y": location.Y(),
                            "z": location.Z()
                        }
                    }

                elif surface_type == "CYLINDER":

                    cylinder = adaptor.Cylinder()

                    axis = cylinder.Axis()

                    direction = (
                        axis.Direction()
                    )

                    location = (
                        axis.Location()
                    )

                    face_data["cylinder"] = {
                        "radius": cylinder.Radius(),
                        "axis_direction": {
                            "x": direction.X(),
                            "y": direction.Y(),
                            "z": direction.Z()
                        },
                        "axis_location": {
                            "x": location.X(),
                            "y": location.Y(),
                            "z": location.Z()
                        }
                    }

                elif surface_type == "CONE":

                    cone = adaptor.Cone()

                    face_data["cone"] = {
                        "reference_radius": (
                            cone.RefRadius()
                        ),
                        "semi_angle_rad": (
                            cone.SemiAngle()
                        )
                    }

                elif surface_type == "SPHERE":

                    sphere = adaptor.Sphere()

                    location = (
                        sphere.Location()
                    )

                    face_data["sphere"] = {
                        "radius": sphere.Radius(),
                        "center": {
                            "x": location.X(),
                            "y": location.Y(),
                            "z": location.Z()
                        }
                    }

                elif surface_type == "TORUS":

                    torus = adaptor.Torus()

                    face_data["torus"] = {
                        "major_radius": (
                            torus.MajorRadius()
                        ),
                        "minor_radius": (
                            torus.MinorRadius()
                        )
                    }

                faces.append(
                    face_data
                )

            except Exception as exc:

                logger.warning(
                    "Could not analyze FACE_%s: %s",
                    face_index,
                    exc
                )

                errors.append(
                    {
                        "face_id": "FACE_{}".format(
                            face_index
                        ),
                        "error": str(exc)
                    }
                )

            face_index += 1

            explorer.Next()

        return {
            "metadata": metadata,
            "face_count": len(faces),
            "failed_face_count": len(errors),
            "faces": faces,
            "errors": errors
        }

    def get_assembly_structure(
        self,
        file_path: str
    ) -> Dict[str, Any]:
        """
        Read a STEP file into XCAF and return its complete
        assembly instance tree.
        """

        self._require_file(
            file_path
        )

        document = TDocStd_Document(
            TCollection_ExtendedString(
                "MDTV-XCAF"
            )
        )

        reader = STEPCAFControl_Reader()

        reader.SetColorMode(
            True
        )

        reader.SetLayerMode(
            True
        )

        reader.SetNameMode(
            True
        )

        reader.SetMatMode(
            True
        )

        status = reader.ReadFile(
            file_path
        )

        if status != IFSelect_RetDone:
            raise ValueError(
                "XCAF failed to read STEP file: {}".format(
                    file_path
                )
            )

        transferred = reader.Transfer(
            document
        )

        if transferred is False:
            raise RuntimeError(
                "XCAF failed to transfer the STEP document"
            )

        shape_tool = (
            XCAFDoc_DocumentTool_ShapeTool(
                document.Main()
            )
        )

        color_tool = (
            XCAFDoc_DocumentTool_ColorTool(
                document.Main()
            )
        )

        layer_tool = (
            XCAFDoc_DocumentTool_LayerTool(
                document.Main()
            )
        )

        material_tool = (
            XCAFDoc_DocumentTool_MaterialTool(
                document.Main()
            )
        )

        root_labels = TDF_LabelSequence()

        shape_tool.GetFreeShapes(
            root_labels
        )

        roots: List[Dict[str, Any]] = []
        flat_nodes: List[Dict[str, Any]] = []

        definition_ids = set()

        layer_names = (
            self._collect_layer_names(
                layer_tool
            )
        )

        identity_location = TopLoc_Location()

        for index in range(
            1,
            root_labels.Length() + 1
        ):

            root_label = (
                root_labels.Value(
                    index
                )
            )

            root_node = (
                self._walk_xcaf_node(
                    label=root_label,
                    shape_tool=shape_tool,
                    color_tool=color_tool,
                    parent_id=None,
                    depth=0,
                    instance_path=(
                        "ROOT_{}".format(
                            index
                        )
                    ),
                    parent_location=(
                        identity_location
                    ),
                    flat_nodes=flat_nodes,
                    definition_ids=(
                        definition_ids
                    )
                )
            )

            roots.append(
                root_node
            )

        instance_count = sum(
            1
            for node in flat_nodes
            if node["is_reference"]
        )

        assembly_count = sum(
            1
            for node in flat_nodes
            if node["is_assembly"]
        )

        simple_shape_count = sum(
            1
            for node in flat_nodes
            if node["is_simple_shape"]
        )

        solid_count = sum(
            node["topology"]["solids"]
            for node in flat_nodes
        )

        return {
            "is_assembly": (
                assembly_count > 0
                or instance_count > 0
            ),
            "root_count": len(
                roots
            ),
            "node_count": len(
                flat_nodes
            ),
            "assembly_count": (
                assembly_count
            ),
            "instance_count": (
                instance_count
            ),
            "definition_count": len(
                definition_ids
            ),
            "simple_shape_count": (
                simple_shape_count
            ),
            "solid_count_in_nodes": (
                solid_count
            ),
            "layers": layer_names,
            "roots": roots,
            "flat_nodes": flat_nodes,
            "capabilities": {
                "names": True,
                "hierarchy": True,
                "local_transform": True,
                "global_transform": True,
                "colors": True,
                "layers": True,
                "materials_tool_loaded": (
                    material_tool
                    is not None
                )
            }
        }

    def _walk_xcaf_node(
        self,
        label,
        shape_tool,
        color_tool,
        parent_id: Optional[str],
        depth: int,
        instance_path: str,
        parent_location,
        flat_nodes: List[Dict[str, Any]],
        definition_ids
    ) -> Dict[str, Any]:
        """
        Recursively walk one XCAF label and follow
        assembly-component references.
        """

        node_id = self._label_entry(
            label
        )

        name = self._label_name(
            label
        )

        is_reference = bool(
            shape_tool.IsReference(
                label
            )
        )

        is_assembly = bool(
            shape_tool.IsAssembly(
                label
            )
        )

        is_simple_shape = bool(
            shape_tool.IsSimpleShape(
                label
            )
        )

        is_free_shape = bool(
            shape_tool.IsFree(
                label
            )
        )

        local_location = (
            shape_tool.GetLocation(
                label
            )
        )

        global_location = (
            parent_location.Multiplied(
                local_location
            )
        )

        definition_id = node_id
        definition_name = name
        source_label = label

        if is_reference:

            referred_label = TDF_Label()

            if shape_tool.GetReferredShape(
                label,
                referred_label
            ):

                source_label = (
                    referred_label
                )

                definition_id = (
                    self._label_entry(
                        referred_label
                    )
                )

                definition_name = (
                    self._label_name(
                        referred_label
                    )
                )

        definition_ids.add(
            definition_id
        )

        shape = shape_tool.GetShape(
            source_label
        )

        topology = (
            self._shape_topology_summary(
                shape
            )
        )

        effective_color = (
            self._get_xcaf_color(
                label,
                source_label,
                color_tool
            )
        )

        node: Dict[str, Any] = {
            "component_id": node_id,
            "instance_path": instance_path,
            "parent_id": parent_id,
            "depth": depth,
            "name": name,
            "definition_id": definition_id,
            "definition_name": definition_name,
            "is_reference": is_reference,
            "is_assembly": is_assembly,
            "is_simple_shape": (
                is_simple_shape
            ),
            "is_free_shape": (
                is_free_shape
            ),
            "shape_type": (
                None
                if shape.IsNull()
                else int(
                    shape.ShapeType()
                )
            ),
            "topology": topology,
            "local_transform": (
                self._location_to_dict(
                    local_location
                )
            ),
            "global_transform": (
                self._location_to_dict(
                    global_location
                )
            ),
            "color": effective_color,
            "children": []
        }

        flat_nodes.append(
            {
                key: value
                for key, value in node.items()
                if key != "children"
            }
        )

        if is_reference:
            children_label = source_label
        else:
            children_label = label

        components = TDF_LabelSequence()

        if shape_tool.IsAssembly(
            children_label
        ):

            shape_tool.GetComponents(
                children_label,
                components,
                False
            )

            for child_index in range(
                1,
                components.Length() + 1
            ):

                child_label = (
                    components.Value(
                        child_index
                    )
                )

                child_path = (
                    "{}/{}".format(
                        instance_path,
                        child_index
                    )
                )

                child_node = (
                    self._walk_xcaf_node(
                        label=child_label,
                        shape_tool=shape_tool,
                        color_tool=color_tool,
                        parent_id=node_id,
                        depth=depth + 1,
                        instance_path=child_path,
                        parent_location=(
                            global_location
                        ),
                        flat_nodes=flat_nodes,
                        definition_ids=(
                            definition_ids
                        )
                    )
                )

                node["children"].append(
                    child_node
                )

        return node

    def _label_entry(
        self,
        label
    ) -> str:

        try:
            return (
                label.EntryDumpToString()
            )

        except Exception:
            return "UNKNOWN_LABEL"

    def _label_name(
        self,
        label
    ) -> str:

        try:

            name = (
                label.GetLabelName()
            )

            if name:
                return str(name)

        except Exception:
            pass

        return self._label_entry(
            label
        )

    def _shape_topology_summary(
        self,
        shape
    ) -> Dict[str, int]:

        summary = {
            "solids": 0,
            "shells": 0,
            "faces": 0,
            "edges": 0,
            "vertices": 0,
            "compounds": 0
        }

        if (
            shape is None
            or shape.IsNull()
        ):
            return summary

        type_map = (
            (
                "solids",
                TopAbs_SOLID
            ),
            (
                "shells",
                TopAbs_SHELL
            ),
            (
                "faces",
                TopAbs_FACE
            ),
            (
                "edges",
                TopAbs_EDGE
            ),
            (
                "vertices",
                TopAbs_VERTEX
            ),
            (
                "compounds",
                TopAbs_COMPOUND
            )
        )

        for key, shape_type in type_map:

            explorer = TopExp_Explorer(
                shape,
                shape_type
            )

            count = 0

            while explorer.More():

                count += 1

                explorer.Next()

            summary[key] = count

        return summary

    def _get_xcaf_color(
        self,
        instance_label,
        definition_label,
        color_tool
    ):

        color_types = (
            (
                "general",
                XCAFDoc_ColorGen
            ),
            (
                "surface",
                XCAFDoc_ColorSurf
            ),
            (
                "curve",
                XCAFDoc_ColorCurv
            )
        )

        label_sources = (
            (
                "instance",
                instance_label
            ),
            (
                "definition",
                definition_label
            )
        )

        for source_name, label in label_sources:

            for color_kind, color_type in color_types:

                color = Quantity_Color()

                try:

                    if color_tool.GetColor(
                        label,
                        color_type,
                        color
                    ):

                        red = color.Red()
                        green = color.Green()
                        blue = color.Blue()

                        return {
                            "source": source_name,
                            "type": color_kind,
                            "rgb": {
                                "r": red,
                                "g": green,
                                "b": blue
                            },
                            "hex": (
                                "#{:02X}{:02X}{:02X}"
                                .format(
                                    int(
                                        round(
                                            red * 255.0
                                        )
                                    ),
                                    int(
                                        round(
                                            green * 255.0
                                        )
                                    ),
                                    int(
                                        round(
                                            blue * 255.0
                                        )
                                    )
                                )
                            )
                        }

                except Exception:
                    continue

        return None

    def _collect_layer_names(
        self,
        layer_tool
    ) -> Listlayers: List[str] = []

        try:

            labels = TDF_LabelSequence()

            layer_tool.GetLayerLabels(
                labels
            )

            for index in range(
                1,
                labels.Length() + 1
            ):

                label = labels.Value(
                    index
                )

                name = self._label_name(
                    label
                )

                if name not in layers:
                    layers.append(
                        name
                    )

        except Exception as exc:

            logger.warning(
                "Could not extract XCAF layers: %s",
                exc
            )

        return layers

    def _location_to_dict(
        self,
        location
    ) -> Dict[str, Any]:

        transform = (
            location.Transformation()
        )

        matrix = [
            [
                transform.Value(
                    row,
                    column
                )
                for column in range(
                    1,
                    5
                )
            ]
            for row in range(
                1,
                4
            )
        ]

        matrix.append(
            [
                0.0,
                0.0,
                0.0,
                1.0
            ]
        )

        translation = (
            transform.TranslationPart()
        )

        return {
            "is_identity": bool(
                location.IsIdentity()
            ),
            "matrix_4x4": matrix,
            "translation_mm": {
                "x": translation.X(),
                "y": translation.Y(),
                "z": translation.Z()
            }
        }

    def _require_file(
        self,
        file_path: str
    ) -> None:

        if not self.occt_available:
            raise RuntimeError(
                "pythonocc-core is not installed"
            )

        if not os.path.exists(
            file_path
        ):
            raise FileNotFoundError(
                "File not found: {}".format(
                    file_path
                )
            )

    def _read_step_file(
        self,
        file_path: str
    ) -> Tuple[
        TopoDS_Shape,
        Dict[str, Any]
    ]:

        step_reader = (
            STEPControl_Reader()
        )

        status = step_reader.ReadFile(
            file_path
        )

        if status != IFSelect_RetDone:
            raise ValueError(
                "Failed to read STEP file: {}".format(
                    file_path
                )
            )

        step_reader.PrintCheckLoad(
            False,
            IFSelect_RetDone
        )

        root_count = (
            step_reader
            .NbRootsForTransfer()
        )

        step_reader.PrintCheckTransfer(
            False,
            IFSelect_RetDone
        )

        logger.info(
            "Found %s roots in STEP file",
            root_count
        )

        transferred = (
            step_reader.TransferRoots()
        )

        if transferred <= 0:
            raise RuntimeError(
                "Open CASCADE did not transfer any STEP roots"
            )

        shape = (
            step_reader.OneShape()
        )

        if shape.IsNull():
            raise RuntimeError(
                "Open CASCADE returned a null STEP shape"
            )

        return (
            shape,
            {
                "nb_roots": root_count,
                "transfer_status": "success"
            }
        )

    def _extract_geometric_properties(
        self,
        shape: TopoDS_Shape
    ) -> Dict[str, Any]:

        properties: Dict[
            str,
            Any
        ] = {}

        try:

            props = GProp_GProps()

            brepgprop.VolumeProperties(
                shape,
                props
            )

            center = (
                props.CentreOfMass()
            )

            properties["volume"] = {
                "value": props.Mass(),
                "unit": "cubic_mm"
            }

            properties[
                "center_of_mass"
            ] = {
                "x": center.X(),
                "y": center.Y(),
                "z": center.Z(),
                "unit": "mm"
            }

        except Exception as exc:

            logger.warning(
                "Could not calculate volume: %s",
                exc
            )

            properties["volume"] = None

            properties[
                "center_of_mass"
            ] = None

        try:

            props = GProp_GProps()

            brepgprop.SurfaceProperties(
                shape,
                props
            )

            properties[
                "surface_area"
            ] = {
                "value": props.Mass(),
                "unit": "square_mm"
            }

        except Exception as exc:

            logger.warning(
                "Could not calculate surface area: %s",
                exc
            )

            properties[
                "surface_area"
            ] = None

        try:

            bounding_box = Bnd_Box()

            brepbndlib.Add(
                shape,
                bounding_box
            )

            (
                xmin,
                ymin,
                zmin,
                xmax,
                ymax,
                zmax
            ) = bounding_box.Get()

            properties[
                "bounding_box"
            ] = {
                "min": {
                    "x": xmin,
                    "y": ymin,
                    "z": zmin
                },
                "max": {
                    "x": xmax,
                    "y": ymax,
                    "z": zmax
                },
                "dimensions": {
                    "length_x": (
                        xmax - xmin
                    ),
                    "length_y": (
                        ymax - ymin
                    ),
                    "length_z": (
                        zmax - zmin
                    )
                },
                "unit": "mm"
            }

        except Exception as exc:

            logger.warning(
                "Could not calculate bounding box: %s",
                exc
            )

            properties[
                "bounding_box"
            ] = None

        return properties

    def _extract_topology_info(
        self,
        shape: TopoDS_Shape
    ) -> Dict[str, Any]:

        explorer = TopologyExplorer(
            shape
        )

        return {
            "solids": (
                explorer.number_of_solids()
            ),
            "compounds": (
                explorer.number_of_compounds()
            ),
            "shells": (
                explorer.number_of_shells()
            ),
            "faces": (
                explorer.number_of_faces()
            ),
            "edges": (
                explorer.number_of_edges()
            ),
            "vertices": (
                explorer.number_of_vertices()
            ),
            "details": {
                "has_free_edges": (
                    len(
                        list(
                            explorer.edges()
                        )
                    )
                    != explorer.number_of_edges()
                ),
                "shape_type": int(
                    shape.ShapeType()
                )
            }
        }

    def _validate_shape(
        self,
        shape: TopoDS_Shape
    ) -> Dict[str, Any]:

        validation: Dict[
            str,
            Any
        ] = {
            "is_valid": False,
            "is_done": False,
            "issues": []
        }

        try:

            analyzer = (
                BRepCheck_Analyzer(
                    shape
                )
            )

            validation[
                "is_valid"
            ] = bool(
                analyzer.IsValid()
            )

            validation[
                "is_done"
            ] = True

            if not validation[
                "is_valid"
            ]:

                validation[
                    "issues"
                ].append(
                    "Shape contains geometric "
                    "or topological errors"
                )

        except Exception as exc:

            logger.error(
                "Validation error: %s",
                exc
            )

            validation[
                "issues"
            ].append(
                "Validation failed: {}".format(
                    exc
                )
            )

        try:

            explorer = TopologyExplorer(
                shape
            )

            degenerate_count = 0

            for edge in explorer.edges():

                if BRep_Tool.Degenerated(
                    edge
                ):
                    degenerate_count += 1

            if degenerate_count > 0:

                validation[
                    "issues"
                ].append(
                    "Found {} degenerate edges".format(
                        degenerate_count
                    )
                )

            validation[
                "quality_metrics"
            ] = {
                "degenerate_edges": (
                    degenerate_count
                ),
                "total_edges": (
                    explorer.number_of_edges()
                )
            }

        except Exception as exc:

            logger.warning(
                "Quality check error: %s",
                exc
            )

        return validation
