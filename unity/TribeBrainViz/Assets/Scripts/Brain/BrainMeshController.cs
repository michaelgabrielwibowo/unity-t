using UnityEngine;

/// <summary>
/// Drives the fsaverage5 brain mesh visualization using double-buffered
/// ComputeBuffers for GPU-side interpolation between prediction steps.
///
/// Provides smooth 60fps visuals from 1Hz brain state updates by
/// lerping between the previous and current state in the shader.
/// </summary>
[RequireComponent(typeof(MeshFilter), typeof(MeshRenderer))]
public class BrainMeshController : MonoBehaviour
{
    [Header("References")]
    [SerializeField] private TribeOscReceiver oscReceiver;
    [SerializeField] private Material brainMaterial;
    [SerializeField] private FsaverageMeshLoader meshLoader;

    [Header("Visualization")]
    [SerializeField] private float activationMin = -3f;
    [SerializeField] private float activationMax = 3f;
    [SerializeField] private float interpolationSpeed = 2.0f;
    [SerializeField] private float emissionIntensity = 1.5f;
    [SerializeField] private bool showParticles = true;

    [Header("Color Mapping")]
    [SerializeField] private Gradient colorGradient;
    [SerializeField] private Texture2D colormapTexture;

    [Header("Debug")]
    [SerializeField] private float currentInterpolation = 0f;
    [SerializeField] private int vertexCount = 0;
    [SerializeField] private int updatesReceived = 0;

    // --- GPU buffers ---
    private ComputeBuffer _currentStateBuffer;
    private ComputeBuffer _previousStateBuffer;
    private ComputeBuffer _colormapBuffer;

    // --- CPU buffers ---
    private float[] _currentStateCPU;
    private float[] _previousStateCPU;

    // --- State ---
    private float _interpolationT = 1f;
    private MeshFilter _meshFilter;
    private MeshRenderer _meshRenderer;
    private Material _materialInstance;

    // Shader property IDs (cached for performance)
    private static readonly int PROP_PREV_STATE = Shader.PropertyToID("_PrevState");
    private static readonly int PROP_CURR_STATE = Shader.PropertyToID("_CurrState");
    private static readonly int PROP_INTERP_T = Shader.PropertyToID("_InterpolationT");
    private static readonly int PROP_ACT_MIN = Shader.PropertyToID("_ActivationMin");
    private static readonly int PROP_ACT_MAX = Shader.PropertyToID("_ActivationMax");
    private static readonly int PROP_EMISSION = Shader.PropertyToID("_EmissionIntensity");
    private static readonly int PROP_COLORMAP = Shader.PropertyToID("_ColormapTex");

    // ===================================================================
    // Unity Lifecycle
    // ===================================================================

    void Awake()
    {
        _meshFilter = GetComponent<MeshFilter>();
        _meshRenderer = GetComponent<MeshRenderer>();

        // Create default gradient if not set
        if (colorGradient == null)
        {
            colorGradient = CreateDefaultGradient();
        }
    }

    void Start()
    {
        // Initialize with mesh from loader or attached mesh
        Mesh mesh = _meshFilter.sharedMesh;
        if (mesh != null)
        {
            InitializeBuffers(mesh.vertexCount);
        }

        // Create material instance
        if (brainMaterial != null)
        {
            _materialInstance = new Material(brainMaterial);
            _meshRenderer.material = _materialInstance;
        }
        else
        {
            _materialInstance = _meshRenderer.material;
        }

        // Generate colormap texture
        GenerateColormapTexture();

        // Subscribe to OSC events
        if (oscReceiver != null)
        {
            oscReceiver.OnBrainStateReceived += OnBrainStateReceived;
        }
    }

    void Update()
    {
        // Smoothly interpolate between prediction steps
        if (_interpolationT < 1f)
        {
            _interpolationT = Mathf.Min(
                _interpolationT + Time.deltaTime * interpolationSpeed,
                1f
            );
        }

        // Update shader properties
        if (_materialInstance != null && _currentStateBuffer != null)
        {
            _materialInstance.SetBuffer(PROP_PREV_STATE, _previousStateBuffer);
            _materialInstance.SetBuffer(PROP_CURR_STATE, _currentStateBuffer);
            _materialInstance.SetFloat(PROP_INTERP_T, _interpolationT);
            _materialInstance.SetFloat(PROP_ACT_MIN, activationMin);
            _materialInstance.SetFloat(PROP_ACT_MAX, activationMax);
            _materialInstance.SetFloat(PROP_EMISSION, emissionIntensity);

            if (colormapTexture != null)
            {
                _materialInstance.SetTexture(PROP_COLORMAP, colormapTexture);
            }
        }

        currentInterpolation = _interpolationT;
    }

    void OnDestroy()
    {
        _currentStateBuffer?.Release();
        _previousStateBuffer?.Release();
        _colormapBuffer?.Release();

        if (_materialInstance != null)
            Destroy(_materialInstance);

        if (oscReceiver != null)
            oscReceiver.OnBrainStateReceived -= OnBrainStateReceived;
    }

    // ===================================================================
    // Buffer Management
    // ===================================================================

    private void InitializeBuffers(int count)
    {
        vertexCount = count;
        _currentStateCPU = new float[count];
        _previousStateCPU = new float[count];

        _currentStateBuffer?.Release();
        _previousStateBuffer?.Release();

        _currentStateBuffer = new ComputeBuffer(count, sizeof(float));
        _previousStateBuffer = new ComputeBuffer(count, sizeof(float));

        _currentStateBuffer.SetData(_currentStateCPU);
        _previousStateBuffer.SetData(_previousStateCPU);

        Debug.Log($"[BrainMesh] Initialized buffers for {count} vertices");
    }

    // ===================================================================
    // Brain State Updates
    // ===================================================================

    /// <summary>
    /// Called when a new brain state arrives from OSC (~1 Hz).
    /// Swaps buffers and resets interpolation.
    /// </summary>
    public void OnBrainStateReceived(float[] vertices)
    {
        if (vertices == null || vertices.Length == 0) return;

        // Initialize buffers on first data
        if (_currentStateBuffer == null || vertexCount != vertices.Length)
        {
            InitializeBuffers(vertices.Length);
        }

        // Swap: previous ← current
        var tempBuffer = _previousStateBuffer;
        _previousStateBuffer = _currentStateBuffer;
        _currentStateBuffer = tempBuffer;

        System.Array.Copy(_currentStateCPU, _previousStateCPU, vertexCount);

        // Update current with new data
        int copyLen = Mathf.Min(vertices.Length, vertexCount);
        System.Array.Copy(vertices, _currentStateCPU, copyLen);
        _currentStateBuffer.SetData(_currentStateCPU);

        // Reset interpolation
        _interpolationT = 0f;
        updatesReceived++;
    }

    // ===================================================================
    // Colormap
    // ===================================================================

    private void GenerateColormapTexture()
    {
        colormapTexture = new Texture2D(256, 1, TextureFormat.RGBA32, false)
        {
            wrapMode = TextureWrapMode.Clamp,
            filterMode = FilterMode.Bilinear
        };

        for (int i = 0; i < 256; i++)
        {
            float t = i / 255f;
            Color color = colorGradient.Evaluate(t);
            colormapTexture.SetPixel(i, 0, color);
        }

        colormapTexture.Apply();
    }

    private Gradient CreateDefaultGradient()
    {
        // "Hot" colormap: blue → cyan → green → yellow → red
        var gradient = new Gradient();
        gradient.SetKeys(
            new GradientColorKey[]
            {
                new GradientColorKey(new Color(0.04f, 0.00f, 0.35f), 0.0f),    // dark blue
                new GradientColorKey(new Color(0.10f, 0.30f, 0.80f), 0.15f),   // blue
                new GradientColorKey(new Color(0.00f, 0.70f, 0.80f), 0.30f),   // cyan
                new GradientColorKey(new Color(0.10f, 0.80f, 0.30f), 0.45f),   // green
                new GradientColorKey(new Color(0.95f, 0.90f, 0.10f), 0.65f),   // yellow
                new GradientColorKey(new Color(0.95f, 0.45f, 0.05f), 0.80f),   // orange
                new GradientColorKey(new Color(0.85f, 0.05f, 0.05f), 1.0f),    // red
            },
            new GradientAlphaKey[]
            {
                new GradientAlphaKey(1f, 0f),
                new GradientAlphaKey(1f, 1f),
            }
        );
        return gradient;
    }

    // ===================================================================
    // Public API
    // ===================================================================

    public void SetActivationRange(float min, float max)
    {
        activationMin = min;
        activationMax = max;
    }

    public void SetEmission(float intensity)
    {
        emissionIntensity = intensity;
    }

    public int UpdatesReceived => updatesReceived;
    public float InterpolationProgress => _interpolationT;
}
