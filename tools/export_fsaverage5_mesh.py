"""
Export fsaverage5 brain mesh to OBJ format for Unity import.

This script uses nilearn to fetch the fsaverage5 surface mesh and
converts it to .obj files with proper coordinate system transformation
for Unity's left-handed Y-up system.

Usage
-----
::

    python tools/export_fsaverage5_mesh.py --output-dir unity/TribeBrainViz/Assets/Models

Produces:
- fsaverage5_lh.obj  (left hemisphere, ~10242 vertices)
- fsaverage5_rh.obj  (right hemisphere, ~10242 vertices)
- fsaverage5_mapping.json  (vertex index mapping for shader buffers)
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def fetch_fsaverage5_mesh():
    """Fetch fsaverage5 mesh data using nilearn."""
    from nilearn import datasets

    fsaverage = datasets.fetch_surf_fsaverage("fsaverage5")
    return fsaverage


def load_surface(mesh_path: str):
    """Load a FreeSurfer/GIFTI surface mesh.

    Returns (vertices, faces) as numpy arrays.
    """
    try:
        from nilearn.surface import load_surf_mesh
        mesh = load_surf_mesh(mesh_path)
        # nilearn returns (coordinates, faces)
        if hasattr(mesh, 'coordinates'):
            return np.array(mesh.coordinates), np.array(mesh.faces)
        else:
            return np.array(mesh[0]), np.array(mesh[1])
    except Exception:
        import nibabel
        surf = nibabel.load(mesh_path)
        coords = surf.darrays[0].data
        faces = surf.darrays[1].data
        return np.array(coords), np.array(faces)


def ras_to_unity(vertices: np.ndarray) -> np.ndarray:
    """Convert RAS (right-anterior-superior) to Unity's left-handed Y-up.

    FreeSurfer RAS: X=right, Y=anterior, Z=superior
    Unity:          X=right, Y=up,       Z=forward

    Transform: Unity_X = RAS_X, Unity_Y = RAS_Z, Unity_Z = RAS_Y
    Also negate X to flip handedness.
    """
    out = np.zeros_like(vertices)
    out[:, 0] = -vertices[:, 0]   # negate X for handedness
    out[:, 1] = vertices[:, 2]     # Z → Y (up)
    out[:, 2] = vertices[:, 1]     # Y → Z (forward)

    # Scale from mm to Unity units (1 unit ≈ 1 cm → divide by 10)
    out *= 0.01

    return out


def compute_vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Compute per-vertex normals by averaging face normals."""
    normals = np.zeros_like(vertices)

    for face in faces:
        v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
        edge1 = v1 - v0
        edge2 = v2 - v0
        face_normal = np.cross(edge1, edge2)
        norm = np.linalg.norm(face_normal)
        if norm > 0:
            face_normal /= norm
        for idx in face:
            normals[idx] += face_normal

    # Normalize
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normals /= norms

    return normals


def write_obj(
    filepath: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
) -> None:
    """Write a Wavefront OBJ file."""
    with open(filepath, "w") as f:
        f.write(f"# fsaverage5 brain mesh\n")
        f.write(f"# Vertices: {len(vertices)}\n")
        f.write(f"# Faces: {len(faces)}\n\n")

        # Vertices
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

        f.write("\n")

        # Normals
        for n in normals:
            f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")

        f.write("\n")

        # Faces (OBJ is 1-indexed)
        for face in faces:
            i0, i1, i2 = face[0] + 1, face[1] + 1, face[2] + 1
            f.write(f"f {i0}//{i0} {i1}//{i1} {i2}//{i2}\n")

    logger.info("Wrote OBJ: %s (%d verts, %d faces)", filepath, len(vertices), len(faces))


def main():
    parser = argparse.ArgumentParser(description="Export fsaverage5 mesh for Unity")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="unity/TribeBrainViz/Assets/Models",
        help="Output directory for OBJ files",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Fetching fsaverage5 mesh...")
    fsaverage = fetch_fsaverage5_mesh()

    mapping = {}

    for hemi, key_pial in [("lh", "pial_left"), ("rh", "pial_right")]:
        logger.info("Processing %s hemisphere...", hemi)

        mesh_path = fsaverage[key_pial]
        vertices_ras, faces = load_surface(mesh_path)

        # Transform to Unity coordinates
        vertices_unity = ras_to_unity(vertices_ras)

        # Compute normals
        normals = compute_vertex_normals(vertices_unity, faces)

        # Write OBJ
        obj_path = output_dir / f"fsaverage5_{hemi}.obj"
        write_obj(obj_path, vertices_unity, faces, normals)

        # Store mapping info
        mapping[hemi] = {
            "obj_file": str(obj_path.name),
            "vertex_count": int(len(vertices_unity)),
            "face_count": int(len(faces)),
            "vertex_offset": 0 if hemi == "lh" else int(len(vertices_ras)),
            "bounds": {
                "min": vertices_unity.min(axis=0).tolist(),
                "max": vertices_unity.max(axis=0).tolist(),
            },
        }

    # Write mapping JSON
    mapping_path = output_dir / "fsaverage5_mapping.json"
    with open(mapping_path, "w") as f:
        json.dump(mapping, f, indent=2)
    logger.info("Wrote mapping: %s", mapping_path)

    print(f"\n✅ Export complete!")
    print(f"   Left hemisphere:  {output_dir / 'fsaverage5_lh.obj'}")
    print(f"   Right hemisphere: {output_dir / 'fsaverage5_rh.obj'}")
    print(f"   Mapping:          {mapping_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
