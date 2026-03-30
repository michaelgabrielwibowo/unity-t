using System.IO;
using UnityEngine;

/// <summary>
/// Loads fsaverage5 brain mesh OBJ files and sets up the Unity Mesh
/// with proper vertex normals and submeshes for left/right hemispheres.
/// </summary>
public class FsaverageMeshLoader : MonoBehaviour
{
    [Header("Mesh Files")]
    [SerializeField] private string leftHemispherePath = "Models/fsaverage5_lh";
    [SerializeField] private string rightHemispherePath = "Models/fsaverage5_rh";

    [Header("Transform")]
    [SerializeField] private float scaleFactor = 1.0f;
    [SerializeField] private Vector3 positionOffset = Vector3.zero;

    [Header("Status")]
    [SerializeField] private int totalVertices = 0;
    [SerializeField] private int totalTriangles = 0;
    [SerializeField] private bool isLoaded = false;

    private Mesh _combinedMesh;

    // ===================================================================
    // Public API
    // ===================================================================

    public Mesh LoadedMesh => _combinedMesh;
    public bool IsLoaded => isLoaded;
    public int TotalVertices => totalVertices;

    /// <summary>
    /// Load and combine both hemispheres into a single mesh.
    /// </summary>
    public Mesh LoadBrainMesh()
    {
        // Try loading from Resources
        var lhMesh = LoadOBJFromResources(leftHemispherePath);
        var rhMesh = LoadOBJFromResources(rightHemispherePath);

        if (lhMesh == null || rhMesh == null)
        {
            Debug.LogWarning("[MeshLoader] Could not load meshes from Resources. Generating placeholder sphere.");
            _combinedMesh = GeneratePlaceholderBrain();
            isLoaded = true;
            return _combinedMesh;
        }

        _combinedMesh = CombineHemispheres(lhMesh, rhMesh);
        isLoaded = true;
        return _combinedMesh;
    }

    void Start()
    {
        if (!isLoaded)
        {
            var mesh = LoadBrainMesh();
            var mf = GetComponent<MeshFilter>();
            if (mf != null)
            {
                mf.mesh = mesh;
            }
        }
    }

    // ===================================================================
    // Mesh Loading
    // ===================================================================

    private Mesh LoadOBJFromResources(string resourcePath)
    {
        // Unity Resources.Load expects path without extension
        var textAsset = Resources.Load<TextAsset>(resourcePath);
        if (textAsset == null)
        {
            Debug.LogWarning($"[MeshLoader] Resource not found: {resourcePath}");
            return null;
        }

        return ParseOBJ(textAsset.text, Path.GetFileNameWithoutExtension(resourcePath));
    }

    private Mesh ParseOBJ(string objText, string meshName)
    {
        var vertices = new System.Collections.Generic.List<Vector3>();
        var normals = new System.Collections.Generic.List<Vector3>();
        var triangles = new System.Collections.Generic.List<int>();

        using (var reader = new StringReader(objText))
        {
            string line;
            while ((line = reader.ReadLine()) != null)
            {
                line = line.Trim();
                if (line.Length == 0 || line[0] == '#') continue;

                string[] parts = line.Split(new[] { ' ', '\t' },
                    System.StringSplitOptions.RemoveEmptyEntries);

                if (parts[0] == "v" && parts.Length >= 4)
                {
                    float x = float.Parse(parts[1], System.Globalization.CultureInfo.InvariantCulture);
                    float y = float.Parse(parts[2], System.Globalization.CultureInfo.InvariantCulture);
                    float z = float.Parse(parts[3], System.Globalization.CultureInfo.InvariantCulture);
                    vertices.Add(new Vector3(x, y, z) * scaleFactor + positionOffset);
                }
                else if (parts[0] == "vn" && parts.Length >= 4)
                {
                    float nx = float.Parse(parts[1], System.Globalization.CultureInfo.InvariantCulture);
                    float ny = float.Parse(parts[2], System.Globalization.CultureInfo.InvariantCulture);
                    float nz = float.Parse(parts[3], System.Globalization.CultureInfo.InvariantCulture);
                    normals.Add(new Vector3(nx, ny, nz));
                }
                else if (parts[0] == "f" && parts.Length >= 4)
                {
                    // Parse face: "f v1//vn1 v2//vn2 v3//vn3" or "f v1 v2 v3"
                    int[] faceIndices = new int[parts.Length - 1];
                    for (int i = 1; i < parts.Length; i++)
                    {
                        string[] components = parts[i].Split('/');
                        faceIndices[i - 1] = int.Parse(components[0]) - 1; // OBJ is 1-indexed
                    }

                    // Triangulate (fan triangulation for n-gons)
                    for (int i = 1; i < faceIndices.Length - 1; i++)
                    {
                        triangles.Add(faceIndices[0]);
                        triangles.Add(faceIndices[i]);
                        triangles.Add(faceIndices[i + 1]);
                    }
                }
            }
        }

        var mesh = new Mesh
        {
            name = meshName,
            indexFormat = vertices.Count > 65535
                ? UnityEngine.Rendering.IndexFormat.UInt32
                : UnityEngine.Rendering.IndexFormat.UInt16
        };

        mesh.SetVertices(vertices);

        if (normals.Count == vertices.Count)
        {
            mesh.SetNormals(normals);
        }

        mesh.SetTriangles(triangles, 0);

        if (normals.Count != vertices.Count)
        {
            mesh.RecalculateNormals();
        }

        mesh.RecalculateBounds();
        mesh.RecalculateTangents();

        Debug.Log($"[MeshLoader] Loaded {meshName}: {vertices.Count} vertices, {triangles.Count / 3} triangles");
        return mesh;
    }

    // ===================================================================
    // Combine Hemispheres
    // ===================================================================

    private Mesh CombineHemispheres(Mesh lh, Mesh rh)
    {
        int lhVertCount = lh.vertexCount;
        int rhVertCount = rh.vertexCount;

        var combinedVertices = new Vector3[lhVertCount + rhVertCount];
        var combinedNormals = new Vector3[lhVertCount + rhVertCount];

        // Copy left hemisphere
        System.Array.Copy(lh.vertices, 0, combinedVertices, 0, lhVertCount);
        System.Array.Copy(lh.normals, 0, combinedNormals, 0, lhVertCount);

        // Copy right hemisphere (offset indices)
        System.Array.Copy(rh.vertices, 0, combinedVertices, lhVertCount, rhVertCount);
        System.Array.Copy(rh.normals, 0, combinedNormals, lhVertCount, rhVertCount);

        // Triangles: offset right hemisphere indices
        int[] lhTris = lh.triangles;
        int[] rhTris = rh.triangles;
        int[] combinedTris = new int[lhTris.Length + rhTris.Length];

        System.Array.Copy(lhTris, 0, combinedTris, 0, lhTris.Length);
        for (int i = 0; i < rhTris.Length; i++)
        {
            combinedTris[lhTris.Length + i] = rhTris[i] + lhVertCount;
        }

        var mesh = new Mesh
        {
            name = "fsaverage5_combined",
            indexFormat = UnityEngine.Rendering.IndexFormat.UInt32
        };

        mesh.vertices = combinedVertices;
        mesh.normals = combinedNormals;
        mesh.triangles = combinedTris;
        mesh.RecalculateBounds();
        mesh.RecalculateTangents();

        totalVertices = combinedVertices.Length;
        totalTriangles = combinedTris.Length / 3;

        Debug.Log($"[MeshLoader] Combined brain mesh: {totalVertices} vertices, {totalTriangles} triangles");
        return mesh;
    }

    // ===================================================================
    // Placeholder when real mesh is unavailable
    // ===================================================================

    private Mesh GeneratePlaceholderBrain()
    {
        // Generate a sphere as placeholder (icosphere-like)
        int resolution = 50;
        var mesh = new Mesh { name = "placeholder_brain" };

        var vertices = new System.Collections.Generic.List<Vector3>();
        var normals = new System.Collections.Generic.List<Vector3>();
        var triangles = new System.Collections.Generic.List<int>();

        // Generate sphere vertices
        for (int lat = 0; lat <= resolution; lat++)
        {
            float theta = Mathf.PI * lat / resolution;
            for (int lon = 0; lon <= resolution; lon++)
            {
                float phi = 2 * Mathf.PI * lon / resolution;

                float x = Mathf.Sin(theta) * Mathf.Cos(phi);
                float y = Mathf.Cos(theta);
                float z = Mathf.Sin(theta) * Mathf.Sin(phi);

                // Slightly elongated for brain shape
                vertices.Add(new Vector3(x * 0.6f, y * 0.45f, z * 0.5f));
                normals.Add(new Vector3(x, y, z).normalized);
            }
        }

        // Generate triangles
        for (int lat = 0; lat < resolution; lat++)
        {
            for (int lon = 0; lon < resolution; lon++)
            {
                int a = lat * (resolution + 1) + lon;
                int b = a + resolution + 1;

                triangles.Add(a);
                triangles.Add(b);
                triangles.Add(a + 1);

                triangles.Add(b);
                triangles.Add(b + 1);
                triangles.Add(a + 1);
            }
        }

        mesh.SetVertices(vertices);
        mesh.SetNormals(normals);
        mesh.SetTriangles(triangles, 0);
        mesh.RecalculateBounds();

        totalVertices = vertices.Count;
        totalTriangles = triangles.Count / 3;

        Debug.Log($"[MeshLoader] Generated placeholder brain: {totalVertices} vertices");
        return mesh;
    }
}
