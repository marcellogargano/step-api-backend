"""
STEP File Processor using Open CASCADE Technology (OCCT)
Handles reading, analysis, and extraction of data from STEP files
"""

import logging
from typing import Dict, Any, List, Tuple
import os

try:
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import (
        TopAbs_SOLID, TopAbs_SHELL, TopAbs_FACE, 
        TopAbs_EDGE, TopAbs_VERTEX, TopAbs_COMPOUND
    )
    from OCC.Core.GProp import GProp_GProps
    from OCC.Core.BRepGProp import (
        brepgprop_VolumeProperties,
        brepgprop_SurfaceProperties,
        brepgprop_LinearProperties
    )
    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.BRepBndLib import brepbndlib_Add
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
    from OCC.Core.GeomAbs import (
    GeomAbs_Plane,
    GeomAbs_Cylinder,
    GeomAbs_Cone,
    GeomAbs_Sphere,
    GeomAbs_Torus,
    GeomAbs_BezierSurface,
    GeomAbs_BSplineSurface
)
    from OCC.Core.BRepCheck import BRepCheck_Analyzer
    from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Compound, topods
    from OCC.Core.TopTools import TopTools_IndexedMapOfShape
    from OCC.Core.TopExp import topexp
    from OCC.Extend.TopologyUtils import TopologyExplorer
    from OCC.Core.STEPCAFControl import STEPCAFControl_Reader
    from OCC.Core.TDocStd import TDocStd_Document
    from OCC.Core.XCAFDoc import (
        XCAFDoc_DocumentTool_ShapeTool,
        XCAFDoc_DocumentTool_ColorTool,
        XCAFDoc_DocumentTool_LayerTool,
        XCAFDoc_DocumentTool_MaterialTool
    )
    from OCC.Core.TCollection import TCollection_ExtendedString
    from OCC.Core.TDF import TDF_LabelSequence
    
    OCCT_AVAILABLE = True
except ImportError:
    OCCT_AVAILABLE = False

logger = logging.getLogger(__name__)


class STEPProcessor:
    """Process STEP files and extract various data points"""
    
    def __init__(self):
        """Initialize the STEP processor"""
        if not OCCT_AVAILABLE:
            logger.warning("pythonocc-core not available. Install it for STEP processing.")
        self.occt_available = OCCT_AVAILABLE
    
    def is_available(self) -> bool:
        """Check if OCCT is available"""
        return self.occt_available
    
    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """
        Comprehensive analysis of STEP file
        
        Args:
            file_path: Path to STEP file
            
        Returns:
            Dictionary with all analysis results
        """
        if not self.occt_available:
            raise RuntimeError("pythonocc-core not installed")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Read the STEP file
        shape, metadata = self._read_step_file(file_path)
        
        # Gather all data
        result = {
            "file_info": {
                "file_size_bytes": os.path.getsize(file_path),
                "file_path": os.path.basename(file_path)
            },
            "metadata": metadata,
            "geometry": self._extract_geometric_properties(shape),
            "topology": self._extract_topology_info(shape),
            "validation": self._validate_shape(shape),
            "assembly": self._extract_assembly_structure(file_path)
        }
        
        return result
    
    def get_geometric_properties(self, file_path: str) -> Dict[str, Any]:
        """Extract only geometric properties"""
        shape, _ = self._read_step_file(file_path)
        return self._extract_geometric_properties(shape)
    
    def get_topology_info(self, file_path: str) -> Dict[str, Any]:
        """Extract only topology information"""
        shape, _ = self._read_step_file(file_path)
        return self._extract_topology_info(shape)
    
    def validate_file(self, file_path: str) -> Dict[str, Any]:
        """Validate STEP file"""
        shape, _ = self._read_step_file(file_path)
        return self._validate_shape(shape)

    def get_faces(self, file_path: str) -> Dict[str, Any]:
        """
        Extract all B-Rep faces from the STEP model.

        Each face receives a sequential identifier and its exact
        Open CASCADE area, centroid, surface type and bounding box.
        """

        if not self.occt_available:
            raise RuntimeError("pythonocc-core not installed")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        shape, metadata = self._read_step_file(file_path)

        surface_names = {
            GeomAbs_Plane: "PLANE",
            GeomAbs_Cylinder: "CYLINDER",
            GeomAbs_Cone: "CONE",
            GeomAbs_Sphere: "SPHERE",
            GeomAbs_Torus: "TORUS",
            GeomAbs_BezierSurface: "BEZIER_SURFACE",
            GeomAbs_BSplineSurface: "BSPLINE_SURFACE"
        }

        faces = []
        errors = []

        explorer = TopExp_Explorer(shape, TopAbs_FACE)
        face_index = 1

        while explorer.More():

            try:
                face = topods.Face(explorer.Current())

                props = GProp_GProps()
                brepgprop_SurfaceProperties(face, props)

                area = props.Mass()
                centroid = props.CentreOfMass()

                adaptor = BRepAdaptor_Surface(face)
                surface_type_code = adaptor.GetType()

                surface_type = surface_names.get(
                    surface_type_code,
                    "OTHER"
                )

                bbox = Bnd_Box()
                brepbndlib_Add(face, bbox)

                xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()

                face_data = {
                    "face_id": f"FACE_{face_index}",
                    "index": face_index,
                    "surface_type": surface_type,
                    "area": {
                        "value": area,
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
                    axis = plane.Axis()
                    normal = axis.Direction()

                    face_data["plane"] = {
                        "normal": {
                            "x": normal.X(),
                            "y": normal.Y(),
                            "z": normal.Z()
                        },
                        "location": {
                            "x": plane.Location().X(),
                            "y": plane.Location().Y(),
                            "z": plane.Location().Z()
                        }
                    }

                elif surface_type == "CYLINDER":
                    cylinder = adaptor.Cylinder()
                    axis = cylinder.Axis()
                    direction = axis.Direction()

                    face_data["cylinder"] = {
                        "radius": cylinder.Radius(),
                        "axis_direction": {
                            "x": direction.X(),
                            "y": direction.Y(),
                            "z": direction.Z()
                        },
                        "axis_location": {
                            "x": axis.Location().X(),
                            "y": axis.Location().Y(),
                            "z": axis.Location().Z()
                        }
                    }

                elif surface_type == "CONE":
                    cone = adaptor.Cone()

                    face_data["cone"] = {
                        "reference_radius": cone.RefRadius(),
                        "semi_angle_rad": cone.SemiAngle()
                    }

                elif surface_type == "SPHERE":
                    sphere = adaptor.Sphere()

                    face_data["sphere"] = {
                        "radius": sphere.Radius(),
                        "center": {
                            "x": sphere.Location().X(),
                            "y": sphere.Location().Y(),
                            "z": sphere.Location().Z()
                        }
                    }

                elif surface_type == "TORUS":
                    torus = adaptor.Torus()

                    face_data["torus"] = {
                        "major_radius": torus.MajorRadius(),
                        "minor_radius": torus.MinorRadius()
                    }

                faces.append(face_data)

            except Exception as exc:
                logger.warning(
                    f"Could not analyze FACE_{face_index}: {exc}"
                )

                errors.append({
                    "face_id": f"FACE_{face_index}",
                    "error": str(exc)
                })

            face_index += 1
            explorer.Next()

        return {
            "metadata": metadata,
            "face_count": len(faces),
            "failed_face_count": len(errors),
            "faces": faces,
            "errors": errors
        }
    
    def _read_step_file(self, file_path: str) -> Tuple[TopoDS_Shape, Dict[str, Any]]:
        """
        Read STEP file and return shape and metadata
        
        Args:
            file_path: Path to STEP file
            
        Returns:
            Tuple of (shape, metadata)
        """
        step_reader = STEPControl_Reader()
        status = step_reader.ReadFile(file_path)
        
        if status != IFSelect_RetDone:
            raise ValueError(f"Failed to read STEP file: {file_path}")
        
        # Transfer roots
        step_reader.PrintCheckLoad(False, IFSelect_RetDone)
        nb_roots = step_reader.NbRootsForTransfer()
        step_reader.PrintCheckTransfer(False, IFSelect_RetDone)
        
        logger.info(f"Found {nb_roots} roots in STEP file")
        
        # Transfer all roots
        step_reader.TransferRoots()
        shape = step_reader.OneShape()
        
        # Extract basic metadata
        metadata = {
            "nb_roots": nb_roots,
            "transfer_status": "success"
        }
        
        return shape, metadata
    
    def _extract_geometric_properties(self, shape: TopoDS_Shape) -> Dict[str, Any]:
        """
        Extract geometric properties (volume, surface area, bounding box)
        
        Args:
            shape: TopoDS_Shape to analyze
            
        Returns:
            Dictionary with geometric properties
        """
        properties = {}
        
        # Volume calculation
        try:
            props = GProp_GProps()
            brepgprop_VolumeProperties(shape, props)
            volume = props.Mass()
            center_of_mass = props.CentreOfMass()
            
            properties["volume"] = {
                "value": volume,
                "unit": "cubic_units"
            }
            properties["center_of_mass"] = {
                "x": center_of_mass.X(),
                "y": center_of_mass.Y(),
                "z": center_of_mass.Z()
            }
        except Exception as e:
            logger.warning(f"Could not calculate volume: {e}")
            properties["volume"] = None
        
        # Surface area calculation
        try:
            props = GProp_GProps()
            brepgprop_SurfaceProperties(shape, props)
            surface_area = props.Mass()
            
            properties["surface_area"] = {
                "value": surface_area,
                "unit": "square_units"
            }
        except Exception as e:
            logger.warning(f"Could not calculate surface area: {e}")
            properties["surface_area"] = None
        
        # Bounding box
        try:
            bbox = Bnd_Box()
            brepbndlib_Add(shape, bbox)
            
            xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
            
            properties["bounding_box"] = {
                "min": {"x": xmin, "y": ymin, "z": zmin},
                "max": {"x": xmax, "y": ymax, "z": zmax},
                "dimensions": {
                    "length_x": xmax - xmin,
                    "length_y": ymax - ymin,
                    "length_z": zmax - zmin
                }
            }
        except Exception as e:
            logger.warning(f"Could not calculate bounding box: {e}")
            properties["bounding_box"] = None
        
        return properties
    
    def _extract_topology_info(self, shape: TopoDS_Shape) -> Dict[str, Any]:
        """
        Extract topology information (counts of solids, faces, edges, vertices)
        
        Args:
            shape: TopoDS_Shape to analyze
            
        Returns:
            Dictionary with topology counts
        """
        topology = {
            "solids": 0,
            "compounds": 0,
            "shells": 0,
            "faces": 0,
            "edges": 0,
            "vertices": 0
        }
        
        # Count topology elements
        explorer = TopologyExplorer(shape)
        
        topology["solids"] = explorer.number_of_solids()
        topology["compounds"] = explorer.number_of_compounds()
        topology["shells"] = explorer.number_of_shells()
        topology["faces"] = explorer.number_of_faces()
        topology["edges"] = explorer.number_of_edges()
        topology["vertices"] = explorer.number_of_vertices()
        
        # Additional details
        topology["details"] = {
            "has_free_edges": len(list(explorer.edges())) != topology["edges"],
            "shape_type": shape.ShapeType()
        }
        
        return topology
    
    def _validate_shape(self, shape: TopoDS_Shape) -> Dict[str, Any]:
        """
        Validate shape quality and check for issues
        
        Args:
            shape: TopoDS_Shape to validate
            
        Returns:
            Dictionary with validation results
        """
        validation = {
            "is_valid": False,
            "is_done": False,
            "issues": []
        }
        
        try:
            analyzer = BRepCheck_Analyzer(shape)
            validation["is_valid"] = analyzer.IsValid()
            
            if not validation["is_valid"]:
                # The shape has issues
                validation["issues"].append("Shape contains geometric or topological errors")
            
            validation["is_done"] = True
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            validation["issues"].append(f"Validation failed: {str(e)}")
        
        # Additional quality checks
        try:
            explorer = TopologyExplorer(shape)
            
            # Check for degenerate edges
            degenerate_count = 0
            for edge in explorer.edges():
                if BRep_Tool.Degenerated(edge):
                    degenerate_count += 1
            
            if degenerate_count > 0:
                validation["issues"].append(f"Found {degenerate_count} degenerate edges")
            
            validation["quality_metrics"] = {
                "degenerate_edges": degenerate_count,
                "total_edges": explorer.number_of_edges()
            }
            
        except Exception as e:
            logger.warning(f"Quality check error: {e}")
        
        return validation
    
    def _extract_assembly_structure(self, file_path: str) -> Dict[str, Any]:
        """
        Extract assembly structure and metadata using XCAF
        
        Args:
            file_path: Path to STEP file
            
        Returns:
            Dictionary with assembly information
        """
        assembly_info = {
            "is_assembly": False,
            "parts": [],
            "layers": [],
            "colors": []
        }
        
        try:
            # Create XCAF document
            doc = TDocStd_Document(TCollection_ExtendedString("MDTV-XCAF"))
            
            # Read STEP file with XCAF
            reader = STEPCAFControl_Reader()
            reader.SetColorMode(True)
            reader.SetLayerMode(True)
            reader.SetNameMode(True)
            
            status = reader.ReadFile(file_path)
            
            if status == IFSelect_RetDone:
                reader.Transfer(doc)
                
                # Get shape tool
                shape_tool = XCAFDoc_DocumentTool_ShapeTool(doc.Main())
                
                # Get free shapes (top-level components)
                labels = TDF_LabelSequence()
                shape_tool.GetFreeShapes(labels)
                
                assembly_info["is_assembly"] = labels.Length() > 1
                assembly_info["part_count"] = labels.Length()
                
                # Extract part names if available
                parts = []
                for i in range(1, labels.Length() + 1):
                    label = labels.Value(i)
                    name = label.EntryDumpToString()
                    
                    # Try to get actual name
                    # Note: Name extraction can be complex, this is simplified
                    parts.append({
                        "label": name,
                        "index": i
                    })
                
                assembly_info["parts"] = parts
                
        except Exception as e:
            logger.warning(f"Could not extract assembly structure: {e}")
            assembly_info["error"] = str(e)
        
        return assembly_info
