"""XCAF assembly and STEP reference processor for pythonocc-core 7.7.2."""

import json
import logging
import math
import os
import re
import subprocess
import sys
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
        """Run native XCAF in a child process; fall back without killing FastAPI."""
        self._require_file(file_path)
        command = [sys.executable, os.path.abspath(__file__), "--worker", file_path]
        try:
            completed = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=120,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            fallback = self._analyze_step_product_structure(file_path)
            fallback["native_xcaf"] = {"status": "timeout", "stderr": (exc.stderr or "")[-2000:]}
            return fallback

        if completed.returncode == 0:
            try:
                payload = json.loads(completed.stdout)
                payload["worker_isolated"] = True
                payload["native_xcaf"] = {"status": "success"}
                return payload
            except json.JSONDecodeError:
                pass

        fallback = self._analyze_step_product_structure(file_path)
        fallback["worker_isolated"] = True
        fallback["native_xcaf"] = {
            "status": "native_crash_or_error",
            "return_code": completed.returncode,
            "stderr": completed.stderr[-4000:],
            "stdout": completed.stdout[-2000:],
        }
        return fallback

    def _analyze_step_product_structure(self, file_path: str) -> Dict[str, Any]:
        """Read product/occurrence metadata when native XCAF is unavailable."""
        with open(file_path, "r", encoding="utf-8", errors="ignore") as stream:
            entities = self._parse_step_entities(stream.read())
        products = []
        definitions = []
        occurrences = []
        for entity_id, expression in entities.items():
            upper = expression.upper()
            strings = re.findall(r"'([^']*)'", expression)
            refs = [int(value) for value in re.findall(r"#(\d+)", expression)]
            if upper.startswith("PRODUCT("):
                products.append({
                    "entity_id": "#{}".format(entity_id),
                    "product_id": strings[0] if strings else "",
                    "name": strings[1] if len(strings) > 1 and strings[1] else (strings[0] if strings else "PRODUCT_{}".format(entity_id)),
                })
            elif upper.startswith("PRODUCT_DEFINITION("):
                definitions.append({"entity_id": "#{}".format(entity_id), "references": ["#{}".format(value) for value in refs]})
            elif "NEXT_ASSEMBLY_USAGE_OCCURRENCE" in upper:
                occurrences.append({
                    "entity_id": "#{}".format(entity_id),
                    "name": next((value for value in strings if value), "OCCURRENCE_{}".format(entity_id)),
                    "parent_definition": "#{}".format(refs[-2]) if len(refs) >= 2 else None,
                    "child_definition": "#{}".format(refs[-1]) if refs else None,
                })
        return {
            "status": "fallback",
            "backend_used": "step_product_structure_fallback",
            "is_assembly": bool(occurrences),
            "product_count": len(products),
            "definition_count": len(definitions),
            "occurrence_count": len(occurrences),
            "products": products,
            "definitions": definitions,
            "occurrences": occurrences,
            "limitations": [
                "Native XCAF terminated in the current runtime.",
                "Fallback does not resolve XCAF colors or instance transforms.",
            ],
        }

    def _analyze_assembly_native(self, file_path: str) -> Dict[str, Any]:
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
        """Extract Creo datum planes, axes, coordinate systems and points."""
        self._require_file(file_path)
        with open(file_path, "r", encoding="utf-8", errors="ignore") as stream:
            text = stream.read()
        entities = self._parse_step_entities(text)
        reverse_refs = self._build_reverse_references(entities)
        reference_map = {}

        # Named placements and points available independently of Creo layers.
        for entity_id, expression in entities.items():
            kind = self._entity_kind(expression)
            name = self._entity_name(expression)
            if not name:
                continue
            if kind == "AXIS2_PLACEMENT_3D":
                item = self._placement_reference(entity_id, entities, name, "DATUM_CSYS")
            elif kind == "AXIS1_PLACEMENT":
                item = self._placement_reference(entity_id, entities, name, "DATUM_AXIS")
            elif kind == "CARTESIAN_POINT":
                point = self._cartesian_point(entity_id, entities)
                item = {"id": "#{}".format(entity_id), "name": name, "type": "DATUM_POINT", "origin": point} if point else None
            else:
                item = None
            if item:
                reference_map[(item["id"], item["type"])] = item

        # Creo writes the semantic datum classification in presentation layers.
        for layer_id, expression in entities.items():
            if self._entity_kind(expression) != "PRESENTATION_LAYER_ASSIGNMENT":
                continue
            strings = re.findall(r"'([^']*)'", expression)
            layer_name = strings[0] if strings else ""
            upper_layer = layer_name.upper()
            if "DTM_PLN" in upper_layer:
                datum_type = "DATUM_PLANE"
                target_kinds = {"PLANE"}
            elif "DTM_CSYS" in upper_layer:
                datum_type = "DATUM_CSYS"
                target_kinds = {"AXIS2_PLACEMENT_3D"}
            elif "AXES" in upper_layer or "DTM_AXIS" in upper_layer:
                datum_type = "DATUM_AXIS"
                target_kinds = {"AXIS1_PLACEMENT", "LINE", "TRIMMED_CURVE"}
            elif "DTM_PNT" in upper_layer or "POINT" in upper_layer:
                datum_type = "DATUM_POINT"
                target_kinds = {"CARTESIAN_POINT"}
            else:
                continue

            owner = self._reference_owner(layer_name)
            for seed_id in self._entity_refs(expression):
                target_id = self._resolve_entity_of_kinds(seed_id, entities, target_kinds)
                if target_id is None:
                    continue
                target_expression = entities[target_id]
                datum_name, datum_aliases = self._best_reference_name(
                    seed_id,
                    target_id,
                    entities,
                    datum_type,
                    reverse_refs,
                )
                item = self._datum_from_entity(target_id, datum_name, datum_type, entities)
                if not item:
                    continue
                item["layer"] = layer_name
                item["owner"] = owner
                item["source_entity"] = "#{}".format(seed_id)
                item["aliases"] = datum_aliases
                item["display_name"] = datum_name
                reference_map[(item["id"], item["type"], layer_name)] = item

        raw_references = list(reference_map.values())
        consolidated = {}
        for item in raw_references:
            key = (item.get("id"), item.get("type"))
            current = consolidated.get(key)
            if current is None:
                current = dict(item); current["layers"] = []
                if item.get("layer"): current["layers"].append(item["layer"])
                consolidated[key] = current; continue
            if item.get("layer") and item["layer"] not in current["layers"]: current["layers"].append(item["layer"])
            for alias in item.get("aliases", []):
                current.setdefault("aliases", [])
                if alias not in current["aliases"]: current["aliases"].append(alias)
            candidate = item.get("display_name") or item.get("name") or ""; existing = current.get("display_name") or current.get("name") or ""
            if candidate and not candidate.startswith("#") and existing.startswith("#"):
                current["name"] = candidate; current["display_name"] = candidate
        references = list(consolidated.values())
        references.sort(key=lambda item: (item.get("owner", ""), item["type"], item.get("display_name", item.get("name", "")), item["id"]))
        counts = {}
        for item in references:
            counts[item["type"]] = counts.get(item["type"], 0) + 1
        return {"reference_count": len(references), "counts_by_type": counts, "references": references}

    def analyze_colors(self, file_path: str) -> Dict[str, Any]:
        """Return STEP color palette and style assignments, including occurrence overrides."""
        self._require_file(file_path)
        with open(file_path, "r", encoding="utf-8", errors="ignore") as stream:
            entities = self._parse_step_entities(stream.read())
        palette = {}
        for entity_id, expression in entities.items():
            kind = self._entity_kind(expression)
            if kind == "COLOUR_RGB":
                values = self._numeric_values(expression)
                if len(values) >= 3:
                    rgb = tuple(max(0.0, min(1.0, value)) for value in values[-3:])
                    palette[entity_id] = self._color_record(entity_id, self._entity_name(expression) or "RGB", rgb)
            elif kind == "DRAUGHTING_PRE_DEFINED_COLOUR":
                name = (self._entity_name(expression) or "").lower()
                predefined = {
                    "black": (0.0, 0.0, 0.0), "white": (1.0, 1.0, 1.0),
                    "red": (1.0, 0.0, 0.0), "green": (0.0, 1.0, 0.0),
                    "blue": (0.0, 0.0, 1.0), "yellow": (1.0, 1.0, 0.0),
                    "magenta": (1.0, 0.0, 1.0), "cyan": (0.0, 1.0, 1.0),
                }
                if name in predefined:
                    palette[entity_id] = self._color_record(entity_id, name, predefined[name])

        assignments = []
        for entity_id, expression in entities.items():
            kind = self._entity_kind(expression)
            if kind not in ("STYLED_ITEM", "OVER_RIDING_STYLED_ITEM", "CONTEXT_DEPENDENT_OVER_RIDING_STYLED_ITEM"):
                continue
            target_refs = self._entity_refs(expression)
            style_refs = target_refs[:-1] if len(target_refs) > 1 else target_refs
            reachable = set()
            for style_ref in style_refs:
                reachable.update(self._collect_reachable(style_ref, entities, 8))
            colors = [palette[item] for item in sorted(reachable) if item in palette]
            if not colors:
                continue
            assignments.append({
                "style_entity": "#{}".format(entity_id),
                "style_type": kind,
                "scope": "INSTANCE_OVERRIDE" if "OVER_RIDING" in kind else "GEOMETRY_OR_FACE",
                "target_entities": ["#{}".format(value) for value in target_refs],
                "colors": colors,
            })

        used_ids = {int(color["entity_id"][1:]) for assignment in assignments for color in assignment["colors"]}

        # Build a deterministic FACE_n color table from STEP ADVANCED_FACE order.
        # This preserves explicit face colors for the common STEP export case.
        target_colors = {}
        for assignment in assignments:
            if assignment["scope"] != "GEOMETRY_OR_FACE" or not assignment["colors"]:
                continue
            selected_color = assignment["colors"][0]
            for target in assignment["target_entities"]:
                try:
                    target_id = int(target[1:])
                except (TypeError, ValueError):
                    continue
                if self._entity_kind(entities.get(target_id, "")) == "ADVANCED_FACE":
                    target_colors[target_id] = selected_color

        advanced_faces = [
            entity_id for entity_id, expression in entities.items()
            if self._entity_kind(expression) == "ADVANCED_FACE"
        ]
        face_colors = {}
        face_color_sources = {}
        for face_index, entity_id in enumerate(advanced_faces, 1):
            color = target_colors.get(entity_id)
            if color:
                face_id = "FACE_{}".format(face_index)
                face_colors[face_id] = color["hex"]
                face_color_sources[face_id] = {
                    "step_face_entity": "#{}".format(entity_id),
                    "color_entity": color["entity_id"],
                    "color_name": color["name"],
                    "hex": color["hex"],
                }

        return {
            "palette_count": len(palette),
            "used_color_count": len(used_ids),
            "assignment_count": len(assignments),
            "palette": list(palette.values()),
            "assignments": assignments,
            "has_instance_overrides": any(item["scope"] == "INSTANCE_OVERRIDE" for item in assignments),
            "face_colors": face_colors,
            "face_color_sources": face_color_sources,
            "face_color_mapping_note": (
                "FACE_n follows ADVANCED_FACE exchange order. User overrides still have priority."
            ),
        }

    def analyze_instance_colors(self, file_path: str) -> Dict[str, Any]:
        """Extract occurrence-level and context-dependent STEP style overrides."""
        self._require_file(file_path)
        with open(file_path, "r", encoding="utf-8", errors="ignore") as stream:
            entities = self._parse_step_entities(stream.read())
        base = self.analyze_colors(file_path)
        palette = {int(item["entity_id"][1:]): item for item in base.get("palette", [])}
        occurrences = []
        for entity_id, expression in entities.items():
            kind = self._entity_kind(expression)
            if "OVER_RIDING_STYLED_ITEM" not in kind and "CONTEXT_DEPENDENT" not in kind:
                continue
            refs = self._entity_refs(expression)
            reachable = self._collect_reachable(entity_id, entities, 12)
            colors = [palette[item] for item in sorted(reachable) if item in palette]
            if not colors:
                continue
            representation_refs = []
            usage_refs = []
            for item in reachable:
                item_kind = self._entity_kind(entities.get(item, ""))
                if "REPRESENTATION_RELATIONSHIP" in item_kind or "CONTEXT_DEPENDENT_SHAPE_REPRESENTATION" in item_kind:
                    representation_refs.append("#{}".format(item))
                if "NEXT_ASSEMBLY_USAGE_OCCURRENCE" in item_kind:
                    usage_refs.append("#{}".format(item))
            occurrences.append({
                "style_entity": "#{}".format(entity_id),
                "style_type": kind,
                "target_entities": ["#{}".format(item) for item in refs],
                "context_entities": representation_refs,
                "usage_occurrences": usage_refs,
                "color": colors[0],
            })
        return {
            "occurrence_color_count": len(occurrences),
            "occurrence_colors": occurrences,
            "native_instance_mapping_available": False,
            "note": "Occurrence colors are preserved as STEP presentation metadata. Native XCAF is still required for guaranteed occurrence-to-TopoDS mapping.",
        }

    def _best_reference_name(self, seed_id, target_id, entities, datum_type, reverse_refs=None):
        """Resolve a datum name using only a narrow, typed reverse path.

        V9 deliberately refuses layer/style/property/product names. If a local
        semantic datum name is not found, the STEP entity id remains the safe
        fallback instead of inventing a wrong label.
        """
        reverse_refs = reverse_refs or self._build_reverse_references(entities)
        allowed_kinds = {
            "DATUM_AXIS": {
                "LINE", "TRIMMED_CURVE", "COMPOSITE_CURVE", "GEOMETRIC_CURVE_SET",
                "AXIS1_PLACEMENT", "REPRESENTATION_ITEM",
            },
            "DATUM_PLANE": {
                "PLANE", "AXIS2_PLACEMENT_3D", "GEOMETRIC_SET", "REPRESENTATION_ITEM",
            },
            "DATUM_CSYS": {"AXIS2_PLACEMENT_3D", "REPRESENTATION_ITEM"},
            "DATUM_POINT": {"CARTESIAN_POINT", "REPRESENTATION_ITEM"},
        }.get(datum_type, set())
        blocked_fragments = (
            "LAYER", "STYLE", "COLOUR", "COLOR", "PRESENTATION", "PROPERTY",
            "MEASURE", "PRODUCT", "SHAPE_REPRESENTATION", "ASSEMBLY_USAGE",
            "CONTEXT_DEPENDENT", "REPRESENTATION_RELATIONSHIP",
        )
        frontier=[(target_id,0),(seed_id,0)]; seen=set(); candidates=[]
        while frontier:
            eid,depth=frontier.pop(0)
            if eid in seen or eid not in entities or depth>3: continue
            seen.add(eid); expr=entities[eid]; kind=self._entity_kind(expr)
            if any(fragment in kind for fragment in blocked_fragments):
                continue
            name=self._entity_name(expr)
            if name:
                name=name.strip(); upper=name.upper()
                is_layer_like = bool(re.search(r'(ALL_|DEF_|DTM_|AXES|FEATURES|COMPONENTS)', upper))
                is_generic = upper in {"CONTINUOUS","DESIGN","AUTOMOTIVE DESIGN","SUPPLEMENTAL GEOMETRY","CENTROID","VOLUME","SURFACE AREA"}
                if name and not re.fullmatch(r"[#0-9_.+\- ]+",name) and not is_layer_like and not is_generic:
                    score=0
                    if re.fullmatch(r"A[_-]?\d+",upper): score+=1000
                    if upper.startswith("DTM"): score+=900
                    if upper.startswith(("ASM_","PRT_")): score+=850
                    if "CSYS" in upper: score+=800
                    if kind=="TRIMMED_CURVE": score+=500
                    if kind in allowed_kinds: score+=200
                    score-=depth*25
                    candidates.append((score,name,eid))
            if depth<3:
                for parent in reverse_refs.get(eid,set()):
                    pkind=self._entity_kind(entities.get(parent,""))
                    if pkind in allowed_kinds or pkind in {"TRIMMED_CURVE","COMPOSITE_CURVE","GEOMETRIC_CURVE_SET","GEOMETRIC_SET"}:
                        frontier.append((parent,depth+1))
                for child in self._entity_refs(expr):
                    ckind=self._entity_kind(entities.get(child,""))
                    if ckind in allowed_kinds:
                        frontier.append((child,depth+1))
        candidates.sort(key=lambda x:(-x[0],x[2]))
        aliases=[]
        for _,name,_ in candidates:
            if name not in aliases: aliases.append(name)
        for value in ("#{}".format(seed_id),str(seed_id),"#{}".format(target_id),str(target_id)):
            if value not in aliases: aliases.append(value)
        return (candidates[0][1],aliases) if candidates else ("#{}".format(target_id),aliases)

    def _build_reverse_references(entities):
        reverse_refs = {}
        for parent_id, expression in entities.items():
            for child_id in XCAFProcessor._entity_refs(expression):
                reverse_refs.setdefault(child_id, set()).add(parent_id)
        return reverse_refs

    def _datum_from_entity(self, entity_id, name, datum_type, entities):
        if datum_type == "DATUM_CSYS":
            return self._placement_reference(entity_id, entities, name, datum_type)
        if datum_type == "DATUM_POINT":
            point = self._cartesian_point(entity_id, entities)
            return {"id": "#{}".format(entity_id), "name": name, "type": datum_type, "origin": point} if point else None
        if datum_type == "DATUM_PLANE":
            expression = entities.get(entity_id, "")
            refs = self._entity_refs(expression)
            placement_id = self._resolve_entity_of_kinds(refs[0], entities, {"AXIS2_PLACEMENT_3D"}) if refs else None
            placement = self._placement_reference(placement_id, entities, name, datum_type) if placement_id else None
            if placement:
                placement["normal"] = placement.pop("z_axis")
                placement.pop("x_axis", None)
                placement.pop("y_axis", None)
            return placement
        if datum_type == "DATUM_AXIS":
            return self._axis_reference(entity_id, entities, name)
        return None

    def _placement_reference(self, entity_id, entities, name, datum_type):
        if entity_id is None:
            return None
        expression = entities.get(entity_id, "")
        refs = self._entity_refs(expression)
        origin = self._cartesian_point(refs[0], entities) if refs else None
        z_axis_tuple = self._direction(refs[1], entities) if len(refs) > 1 else (0.0, 0.0, 1.0)
        x_axis_tuple = self._direction(refs[2], entities) if len(refs) > 2 else self._orthogonal_x(z_axis_tuple)
        if origin is None or z_axis_tuple is None or x_axis_tuple is None:
            return None
        y_axis_tuple = self._normalize(self._cross(z_axis_tuple, x_axis_tuple))
        return {
            "id": "#{}".format(entity_id), "name": name, "type": datum_type,
            "origin": origin, "x_axis": self._vector_dict(x_axis_tuple),
            "y_axis": self._vector_dict(y_axis_tuple), "z_axis": self._vector_dict(z_axis_tuple),
        }

    def _axis_reference(self, entity_id, entities, name):
        expression = entities.get(entity_id, "")
        kind = self._entity_kind(expression)
        if kind == "AXIS1_PLACEMENT":
            placement = self._placement_reference(entity_id, entities, name, "DATUM_AXIS")
            if placement:
                placement["direction"] = placement["z_axis"]
            return placement
        line_id = entity_id if kind == "LINE" else self._resolve_entity_of_kinds(entity_id, entities, {"LINE"})
        if line_id is None:
            return None
        refs = self._entity_refs(entities[line_id])
        point_id = self._resolve_entity_of_kinds(refs[0], entities, {"CARTESIAN_POINT"}) if refs else None
        vector_id = self._resolve_entity_of_kinds(refs[1], entities, {"VECTOR", "DIRECTION"}) if len(refs) > 1 else None
        origin = self._cartesian_point(point_id, entities) if point_id else None
        direction = self._vector_or_direction(vector_id, entities) if vector_id else None
        if origin is None or direction is None:
            return None
        return {"id": "#{}".format(entity_id), "name": name, "type": "DATUM_AXIS", "origin": origin, "direction": self._vector_dict(direction)}

    def _vector_or_direction(self, entity_id, entities):
        expression = entities.get(entity_id, "")
        if self._entity_kind(expression) == "VECTOR":
            refs = self._entity_refs(expression)
            return self._direction(refs[0], entities) if refs else None
        return self._direction(entity_id, entities)

    def _resolve_entity_of_kinds(self, seed_id, entities, target_kinds, visited=None):
        visited = visited or set()
        if seed_id in visited or seed_id not in entities:
            return None
        visited.add(seed_id)
        if self._entity_kind(entities[seed_id]) in target_kinds:
            return seed_id
        for child in self._entity_refs(entities[seed_id]):
            result = self._resolve_entity_of_kinds(child, entities, target_kinds, visited)
            if result is not None:
                return result
        return None

    def _collect_reachable(self, seed_id, entities, max_depth):
        found = set()
        frontier = [(seed_id, 0)]
        while frontier:
            current, depth = frontier.pop()
            if current in found or current not in entities or depth > max_depth:
                continue
            found.add(current)
            frontier.extend((child, depth + 1) for child in self._entity_refs(entities[current]))
        return found

    @staticmethod
    def _reference_owner(layer_name):
        upper = layer_name.upper()
        if "ASM" in upper:
            return "ASSEMBLY"
        if "PRT" in upper:
            return "PART"
        return "UNSPECIFIED"

    @staticmethod
    def _entity_kind(expression):
        return expression.split("(", 1)[0].strip().upper()

    @staticmethod
    def _entity_refs(expression):
        return [int(value) for value in re.findall(r"#(\d+)", expression)]

    @staticmethod
    def _numeric_values(expression):
        return [float(value) for value in re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:E[-+]?\d+)?", expression, re.I)]

    @staticmethod
    def _vector_dict(vector):
        return {"x": vector[0], "y": vector[1], "z": vector[2]}

    @staticmethod
    def _color_record(entity_id, name, rgb):
        return {
            "entity_id": "#{}".format(entity_id), "name": name,
            "rgb": {"r": rgb[0], "g": rgb[1], "b": rgb[2]},
            "rgb_255": {"r": int(round(rgb[0] * 255)), "g": int(round(rgb[1] * 255)), "b": int(round(rgb[2] * 255))},
            "hex": "#{:02X}{:02X}{:02X}".format(int(round(rgb[0] * 255)), int(round(rgb[1] * 255)), int(round(rgb[2] * 255))),
        }

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


def _worker_main():
    if len(sys.argv) != 3 or sys.argv[1] != "--worker":
        raise SystemExit(2)
    payload = XCAFProcessor()._analyze_assembly_native(sys.argv[2])
    sys.stdout.write(json.dumps(payload, separators=(",", ":")))


if __name__ == "__main__":
    _worker_main()
