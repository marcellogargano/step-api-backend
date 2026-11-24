"""XCAF assembly and STEP reference processor for pythonocc-core 7.7.2."""

import logging
import math
import os
import re
from typing import Any, Dict, List, Optional, Tuple

try:
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.STEPCAFControl import STEPCAFControl_Reader
    from OCC.Core.TCollection import TCollection_ExtendedString
    from OCC.Core.TDF import TDF_Label, TDF_LabelSequence
    from OCC.Core.TDocStd import TDocStd_Document
    from OCC.Core.TopLoc import TopLoc_Location
    from OCC.Core.XCAFDoc import XCAFDoc_DocumentTool

    XCAF_AVAILABLE = True
    XCAF_IMPORT_ERROR = None
except ImportError as exc:
    XCAF_AVAILABLE = False
    XCAF_IMPORT_ERROR = str(exc)

logger = logging.getLogger(__name__)


class XCAFProcessor:
    """Read assembly hierarchy and named STEP coordinate references."""

    def __init__(self):
        self.xcaf_available = XCAF_AVAILABLE
        if not XCAF_AVAILABLE:
            logger.warning("XCAF modules are not available: %s", XCAF_IMPORT_ERROR)

    def is_available(self) -> bool:
        return self.xcaf_available

    def analyze_assembly(self, file_path: str) -> Dict[str, Any]:
        self._require_file(file_path)
        logger.info("XCAF 0: creating XCAF document")

        document = TDocStd_Document(TCollection_ExtendedString("MDTV-XCAF"))
        shape_tool = XCAFDoc_DocumentTool.ShapeTool(document.Main())

        if shape_tool is None:
            raise RuntimeError("XCAF ShapeTool initialization failed")

        logger.info("XCAF 1: reading STEP")
        reader = STEPCAFControl_Reader()
        reader.SetNameMode(True)
        reader.SetColorMode(True)
        reader.SetLayerMode(True)
        reader.SetMatMode(True)
        status = reader.ReadFile(file_path)

        if status != IFSelect_RetDone:
            raise ValueError("XCAF failed to read STEP file: {}".format(file_path))

        logger.info("XCAF 2: transferring STEP")
        if not reader.Transfer(document):
            raise RuntimeError("XCAF failed to transfer STEP into the document")

        roots_sequence = TDF_LabelSequence()
        shape_tool.GetFreeShapes(roots_sequence)
        roots = []
        flat_nodes = []

        for root_index in range(1, roots_sequence.Length() + 1):
            roots.append(
                self._walk_label(
                    roots_sequence.Value(root_index),
                    shape_tool,
                    None,
                    0,
                    "ROOT_{}".format(root_index),
                    TopLoc_Location(),
                    flat_nodes,
                )
            )

        assembly_count = sum(1 for node in flat_nodes if node["is_assembly"])
        reference_count = sum(1 for node in flat_nodes if node["is_reference"])
        logger.info("XCAF 3: extracted %s nodes", len(flat_nodes))

        return {
            "status": "success",
            "root_count": len(roots),
            "node_count": len(flat_nodes),
            "assembly_count": assembly_count,
            "reference_count": reference_count,
            "is_assembly": assembly_count > 0 or reference_count > 0,
            "roots": roots,
            "flat_nodes": flat_nodes,
        }

    def analyze_references(self, file_path: str) -> Dict[str, Any]:
        """Extract named AXIS2_PLACEMENT_3D, AXIS1_PLACEMENT and CARTESIAN_POINT entities."""
        self._require_file(file_path)
        text = open(file_path, "r", encoding="utf-8", errors="ignore").read()
        entities = self._parse_step_entities(text)
        references = []

        for entity_id, expression in entities.items():
            kind = expression.split("(", 1)[0].strip().upper()
            if kind not in ("AXIS2_PLACEMENT_3D", "AXIS1_PLACEMENT", "CARTESIAN_POINT"):
                continue
            name = self._entity_name(expression)
            if not name:
                continue

            if kind == "CARTESIAN_POINT":
                point = self._cartesian_point(entity_id, entities)
                if point is not None:
                    references.append({"id": "#{}".format(entity_id), "name": name, "type": "POINT", "origin": point})
                continue

            refs = [int(value) for value in re.findall(r"#(\d+)", expression)]
            origin = self._cartesian_point(refs[0], entities) if refs else None
            z_axis = self._direction(refs[1], entities) if len(refs) > 1 else (0.0, 0.0, 1.0)
            x_axis = self._direction(refs[2], entities) if len(refs) > 2 else self._orthogonal_x(z_axis)
            if origin is None or z_axis is None or x_axis is None:
                continue
            y_axis = self._normalize(self._cross(z_axis, x_axis))
            references.append({
                "id": "#{}".format(entity_id),
                "name": name,
                "type": "COORDINATE_SYSTEM" if kind == "AXIS2_PLACEMENT_3D" else "AXIS",
                "origin": origin,
                "x_axis": x_axis,
                "y_axis": y_axis,
                "z_axis": z_axis,
            })

        return {"reference_count": len(references), "references": references}

    def _walk_label(self, label, shape_tool, parent_id, depth, path, parent_location, flat_nodes):
        node_id = self._label_entry(label)
        is_reference = bool(shape_tool.IsReference(label))
        source_label = label
        definition_id = node_id
        definition_name = self._label_name(label)

        if is_reference:
            referred = TDF_Label()
            if shape_tool.GetReferredShape(label, referred):
                source_label = referred
                definition_id = self._label_entry(referred)
                definition_name = self._label_name(referred)

        local_location = shape_tool.GetLocation(label)
        global_location = parent_location.Multiplied(local_location)
        is_assembly = bool(shape_tool.IsAssembly(source_label))
        shape = shape_tool.GetShape(source_label)
        node = {
            "component_id": node_id,
            "instance_path": path,
            "parent_id": parent_id,
            "depth": depth,
            "name": self._label_name(label),
            "definition_id": definition_id,
            "definition_name": definition_name,
            "is_reference": is_reference,
            "is_assembly": is_assembly,
            "is_simple_shape": bool(shape_tool.IsSimpleShape(source_label)),
            "shape_type": None if shape.IsNull() else int(shape.ShapeType()),
            "local_transform": self._location_dict(local_location),
            "global_transform": self._location_dict(global_location),
            "children": [],
        }
        flat_nodes.append({key: value for key, value in node.items() if key != "children"})

        if is_assembly:
            components = TDF_LabelSequence()
            shape_tool.GetComponents(source_label, components, False)
            for index in range(1, components.Length() + 1):
                node["children"].append(
                    self._walk_label(
                        components.Value(index), shape_tool, node_id, depth + 1,
                        "{}/{}".format(path, index), global_location, flat_nodes,
                    )
                )
        return node

    def _location_dict(self, location):
        transform = location.Transformation()
        matrix = [[transform.Value(row, col) for col in range(1, 5)] for row in range(1, 4)]
        matrix.append([0.0, 0.0, 0.0, 1.0])
        translation = transform.TranslationPart()
        return {
            "is_identity": bool(location.IsIdentity()),
            "matrix_4x4": matrix,
            "translation_mm": {"x": translation.X(), "y": translation.Y(), "z": translation.Z()},
        }

    def _label_entry(self, label):
        try:
            return label.EntryDumpToString()
        except Exception:
            return "UNKNOWN_LABEL"

    def _label_name(self, label):
        try:
            name = label.GetLabelName()
            return str(name) if name else self._label_entry(label)
        except Exception:
            return self._label_entry(label)

    def _parse_step_entities(self, text):
        return {int(match.group(1)): match.group(2).strip() for match in re.finditer(r"#(\d+)\s*=\s*(.*?);", text, re.S)}

    def _entity_name(self, expression):
        match = re.match(r"[A-Z0-9_]+\s*\(\s*'([^']*)'", expression, re.I)
        return match.group(1).strip() if match and match.group(1).strip() else None

    def _cartesian_point(self, entity_id, entities):
        expression = entities.get(entity_id, "")
        if not expression.upper().startswith("CARTESIAN_POINT"):
            return None
        values = re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:E[-+]?\d+)?", expression, re.I)
        if len(values) < 3:
            return None
        xyz = tuple(float(value) for value in values[-3:])
        return {"x": xyz[0], "y": xyz[1], "z": xyz[2], "unit": "mm"}

    def _direction(self, entity_id, entities):
        expression = entities.get(entity_id, "")
        if not expression.upper().startswith("DIRECTION"):
            return None
        values = re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:E[-+]?\d+)?", expression, re.I)
        if len(values) < 3:
            return None
        return self._normalize(tuple(float(value) for value in values[-3:]))

    def _orthogonal_x(self, z_axis):
        candidate = (1.0, 0.0, 0.0) if abs(z_axis[0]) < 0.9 else (0.0, 1.0, 0.0)
        return self._normalize(self._cross(candidate, z_axis))

    @staticmethod
    def _cross(a, b):
        return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])

    @staticmethod
    def _normalize(vector):
        magnitude = math.sqrt(sum(value * value for value in vector))
        if magnitude < 1.0e-12:
            return None
        return tuple(value / magnitude for value in vector)

    def _require_file(self, file_path):
        if not self.xcaf_available:
            raise RuntimeError("XCAF is not available: {}".format(XCAF_IMPORT_ERROR))
        if not os.path.exists(file_path):
            raise FileNotFoundError("File not found: {}".format(file_path))
        if os.path.splitext(file_path)[1].lower() not in (".step", ".stp"):
            raise ValueError("Invalid STEP file extension")
