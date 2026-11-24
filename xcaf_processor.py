"""
Minimal XCAF STEP assembly processor.

This module is isolated from step_processor.py and performs the first XCAF
validation stage: create a correctly initialized XCAF document, read a STEP
file with STEPCAFControl_Reader, transfer it into the document, obtain the
ShapeTool, and enumerate the free/root shapes.
"""

import logging
import os
from typing import Any, Dict

try:
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.STEPCAFControl import STEPCAFControl_Reader
    from OCC.Core.TCollection import TCollection_ExtendedString
    from OCC.Core.TDF import TDF_Label, TDF_LabelSequence
    from OCC.Core.TDocStd import TDocStd_Document
    from OCC.Core.XCAFApp import XCAFApp_Application_GetApplication
    from OCC.Core.XCAFDoc import XCAFDoc_DocumentTool_ShapeTool

    XCAF_AVAILABLE = True
    XCAF_IMPORT_ERROR = None

except ImportError as exc:
    XCAF_AVAILABLE = False
    XCAF_IMPORT_ERROR = str(exc)


logger = logging.getLogger(__name__)


class XCAFProcessor:
    """Minimal isolated processor for STEP assembly/XCAF information."""

    def __init__(self):
        self.xcaf_available = XCAF_AVAILABLE

        if not XCAF_AVAILABLE:
            logger.warning(
                "XCAF modules are not available: %s",
                XCAF_IMPORT_ERROR,
            )

    def is_available(self) -> bool:
        return self.xcaf_available

    def analyze_assembly(
        self,
        file_path: str,
    ) -> Dict[str, Any]:
        """Read a STEP file using XCAF and return top-level information."""

        self._require_file(file_path)

        logger.info(
            "XCAF 0: creating XCAF application and document"
        )

        application = XCAFApp_Application_GetApplication()

        if application is None:
            raise RuntimeError(
                "XCAF application is not available"
            )

        document = TDocStd_Document(
            TCollection_ExtendedString(
                "MDTV-XCAF"
            )
        )

        application.NewDocument(
            TCollection_ExtendedString(
                "MDTV-XCAF"
            ),
            document,
        )

        logger.info(
            "XCAF 0A: XCAF document initialized"
        )

        reader = STEPCAFControl_Reader()

        reader.SetNameMode(
            True
        )

        reader.SetColorMode(
            True
        )

        reader.SetLayerMode(
            True
        )

        logger.info(
            "XCAF 1: reading STEP"
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

        logger.info(
            "XCAF 2: STEP read completed"
        )

        logger.info(
            "XCAF 3: transferring STEP into XCAF document"
        )

        transferred = reader.Transfer(
            document
        )

        if transferred is False:
            raise RuntimeError(
                "XCAF failed to transfer the STEP file "
                "into the document"
            )

        logger.info(
            "XCAF 4: transfer completed"
        )

        shape_tool = XCAFDoc_DocumentTool_ShapeTool(
            document.Main()
        )

        if shape_tool is None:
            raise RuntimeError(
                "XCAF ShapeTool is not available"
            )

        logger.info(
            "XCAF 5: ShapeTool obtained"
        )

        root_labels = TDF_LabelSequence()

        shape_tool.GetFreeShapes(
            root_labels
        )

        logger.info(
            "XCAF 6: found %s free/root shapes",
            root_labels.Length(),
        )

        roots = []

        for index in range(
            1,
            root_labels.Length() + 1,
        ):
            label = root_labels.Value(
                index
            )

            root_data = self._describe_label(
                label=label,
                shape_tool=shape_tool,
                index=index,
            )

            roots.append(
                root_data
            )

        assembly_root_count = sum(
            1
            for item in roots
            if item["is_assembly"]
        )

        simple_root_count = sum(
            1
            for item in roots
            if item["is_simple_shape"]
        )

        reference_root_count = sum(
            1
            for item in roots
            if item["is_reference"]
        )

        return {
            "status": "success",
            "xcaf_available": True,
            "root_count": root_labels.Length(),
            "assembly_root_count": assembly_root_count,
            "simple_root_count": simple_root_count,
            "reference_root_count": reference_root_count,
            "is_assembly": assembly_root_count > 0,
            "roots": roots,
            "stage": "xcaf_root_detection",
        }

    def _describe_label(
        self,
        label,
        shape_tool,
        index: int,
    ) -> Dict[str, Any]:
        """Return basic information for one top-level XCAF label."""

        label_entry = self._label_entry(
            label
        )

        label_name = self._label_name(
            label
        )

        is_assembly = bool(
            shape_tool.IsAssembly(
                label
            )
        )

        is_reference = bool(
            shape_tool.IsReference(
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

        component_count = 0

        if is_assembly:
            components = TDF_LabelSequence()

            shape_tool.GetComponents(
                label,
                components,
                False,
            )

            component_count = (
                components.Length()
            )

        referred_definition = None

        if is_reference:
            referred_label = TDF_Label()

            if shape_tool.GetReferredShape(
                label,
                referred_label,
            ):
                referred_definition = {
                    "label": self._label_entry(
                        referred_label
                    ),
                    "name": self._label_name(
                        referred_label
                    ),
                }

        shape = shape_tool.GetShape(
            label
        )

        shape_type = None

        if (
            shape is not None
            and not shape.IsNull()
        ):
            shape_type = int(
                shape.ShapeType()
            )

        return {
            "index": index,
            "label": label_entry,
            "name": label_name,
            "is_assembly": is_assembly,
            "is_reference": is_reference,
            "is_simple_shape": is_simple_shape,
            "is_free_shape": is_free_shape,
            "component_count": component_count,
            "shape_type": shape_type,
            "referred_definition": referred_definition,
        }

    def _label_entry(
        self,
        label,
    ) -> str:

        try:
            return label.EntryDumpToString()

        except Exception:
            return "UNKNOWN_LABEL"

    def _label_name(
        self,
        label,
    ) -> str:

        try:
            name = label.GetLabelName()

            if name:
                return str(name)

        except Exception:
            pass

        return self._label_entry(
            label
        )

    def _require_file(
        self,
        file_path: str,
    ) -> None:

        if not self.xcaf_available:
            raise RuntimeError(
                "XCAF is not available: {}".format(
                    XCAF_IMPORT_ERROR
                )
            )

        if not os.path.exists(
            file_path
        ):
            raise FileNotFoundError(
                "File not found: {}".format(
                    file_path
                )
            )

        extension = os.path.splitext(
            file_path
        )[1].lower()

        if extension not in (
            ".step",
            ".stp",
        ):
            raise ValueError(
                "Invalid STEP extension: {}".format(
                    extension
                )
            )
