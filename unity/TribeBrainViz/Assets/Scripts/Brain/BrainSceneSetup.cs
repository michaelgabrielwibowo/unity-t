using UnityEngine;

/// <summary>
/// Scene setup controller for the TRIBE v2 brain visualization.
/// Handles camera orbit, UI overlay, and scene configuration.
/// </summary>
public class BrainSceneSetup : MonoBehaviour
{
    [Header("References")]
    [SerializeField] private TribeOscReceiver oscReceiver;
    [SerializeField] private BrainMeshController brainController;
    [SerializeField] private Transform brainTransform;

    [Header("Camera Orbit")]
    [SerializeField] private float orbitDistance = 2.5f;
    [SerializeField] private float orbitSpeed = 15f;
    [SerializeField] private float autoRotateSpeed = 5f;
    [SerializeField] private float verticalAngle = 15f;
    [SerializeField] private bool autoRotate = true;

    [Header("Lighting")]
    [SerializeField] private Light mainLight;
    [SerializeField] private Light rimLight;
    [SerializeField] private Color ambientColor = new Color(0.05f, 0.05f, 0.12f);

    [Header("UI")]
    [SerializeField] private bool showDebugUI = true;

    // Camera orbit state
    private float _horizontalAngle = 0f;
    private float _currentVerticalAngle;
    private Vector3 _targetPosition;
    private bool _isDragging = false;
    private Vector3 _lastMousePos;

    // Performance tracking
    private float _fps;
    private float _fpsUpdateTimer;
    private int _fpsFrameCount;
    private string _statusText = "";

    // ===================================================================
    // Unity Lifecycle
    // ===================================================================

    void Start()
    {
        // Scene setup
        SetupScene();
        _currentVerticalAngle = verticalAngle;

        if (brainTransform == null)
        {
            brainTransform = transform;
        }
    }

    void Update()
    {
        UpdateCameraOrbit();
        UpdateFPS();
        UpdateStatusText();
    }

    void OnGUI()
    {
        if (!showDebugUI) return;
        DrawDebugUI();
    }

    // ===================================================================
    // Scene Setup
    // ===================================================================

    private void SetupScene()
    {
        // Background
        Camera.main.backgroundColor = new Color(0.02f, 0.02f, 0.06f);
        Camera.main.clearFlags = CameraClearFlags.SolidColor;

        // Ambient
        RenderSettings.ambientMode = UnityEngine.Rendering.AmbientMode.Flat;
        RenderSettings.ambientLight = ambientColor;

        // Main directional light
        if (mainLight == null)
        {
            var lightGO = new GameObject("MainLight");
            mainLight = lightGO.AddComponent<Light>();
            mainLight.type = LightType.Directional;
            mainLight.intensity = 0.8f;
            mainLight.color = new Color(0.95f, 0.95f, 1.0f);
            mainLight.transform.rotation = Quaternion.Euler(35f, -30f, 0f);
        }

        // Rim light for depth
        if (rimLight == null)
        {
            var rimGO = new GameObject("RimLight");
            rimLight = rimGO.AddComponent<Light>();
            rimLight.type = LightType.Directional;
            rimLight.intensity = 0.3f;
            rimLight.color = new Color(0.4f, 0.6f, 1.0f);
            rimLight.transform.rotation = Quaternion.Euler(-20f, 150f, 0f);
        }
    }

    // ===================================================================
    // Camera Orbit
    // ===================================================================

    private void UpdateCameraOrbit()
    {
        // Mouse drag for manual orbit
        if (Input.GetMouseButtonDown(0))
        {
            _isDragging = true;
            _lastMousePos = Input.mousePosition;
            autoRotate = false;
        }

        if (Input.GetMouseButtonUp(0))
        {
            _isDragging = false;
        }

        if (_isDragging)
        {
            Vector3 delta = Input.mousePosition - _lastMousePos;
            _horizontalAngle += delta.x * orbitSpeed * Time.deltaTime;
            _currentVerticalAngle -= delta.y * orbitSpeed * Time.deltaTime;
            _currentVerticalAngle = Mathf.Clamp(_currentVerticalAngle, -80f, 80f);
            _lastMousePos = Input.mousePosition;
        }

        // Auto rotation
        if (autoRotate)
        {
            _horizontalAngle += autoRotateSpeed * Time.deltaTime;
        }

        // Double-click to reset
        if (Input.GetMouseButtonDown(1))
        {
            autoRotate = true;
            _currentVerticalAngle = verticalAngle;
        }

        // Scroll wheel for zoom
        float scroll = Input.GetAxis("Mouse ScrollWheel");
        orbitDistance = Mathf.Clamp(orbitDistance - scroll * 2f, 0.5f, 10f);

        // Compute camera position
        float radH = _horizontalAngle * Mathf.Deg2Rad;
        float radV = _currentVerticalAngle * Mathf.Deg2Rad;

        Vector3 target = brainTransform != null ? brainTransform.position : Vector3.zero;
        Vector3 offset = new Vector3(
            Mathf.Cos(radV) * Mathf.Sin(radH),
            Mathf.Sin(radV),
            Mathf.Cos(radV) * Mathf.Cos(radH)
        ) * orbitDistance;

        Camera.main.transform.position = Vector3.Lerp(
            Camera.main.transform.position,
            target + offset,
            Time.deltaTime * 5f
        );
        Camera.main.transform.LookAt(target);
    }

    // ===================================================================
    // FPS Tracking
    // ===================================================================

    private void UpdateFPS()
    {
        _fpsFrameCount++;
        _fpsUpdateTimer += Time.unscaledDeltaTime;

        if (_fpsUpdateTimer >= 0.5f)
        {
            _fps = _fpsFrameCount / _fpsUpdateTimer;
            _fpsFrameCount = 0;
            _fpsUpdateTimer = 0;
        }
    }

    private void UpdateStatusText()
    {
        bool connected = oscReceiver != null && oscReceiver.IsConnected;
        int updates = brainController != null ? brainController.UpdatesReceived : 0;
        float latency = oscReceiver != null ? oscReceiver.LastLatencyMs : 0;
        int seqId = oscReceiver != null ? oscReceiver.LastSequenceId : -1;
        float interp = brainController != null ? brainController.InterpolationProgress : 0;

        _statusText = $"TRIBE v2 Streaming Brain Visualization\n" +
                      $"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n" +
                      $"Status: {(connected ? "<color=#44ff44>CONNECTED</color>" : "<color=#ff4444>DISCONNECTED</color>")}\n" +
                      $"FPS: {_fps:F0}  |  Latency: {latency:F0}ms\n" +
                      $"Brain Updates: {updates}  |  Seq: {seqId}\n" +
                      $"Interpolation: {interp:P0}\n" +
                      $"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n" +
                      $"LMB: Orbit  |  RMB: Reset  |  Scroll: Zoom";
    }

    // ===================================================================
    // Debug UI
    // ===================================================================

    private void DrawDebugUI()
    {
        // Semi-transparent panel
        GUIStyle panelStyle = new GUIStyle(GUI.skin.box);
        panelStyle.normal.background = MakeTex(2, 2, new Color(0, 0, 0, 0.7f));

        GUIStyle labelStyle = new GUIStyle(GUI.skin.label);
        labelStyle.fontSize = 14;
        labelStyle.normal.textColor = new Color(0.85f, 0.9f, 1.0f);
        labelStyle.richText = true;

        Rect panelRect = new Rect(10, 10, 360, 180);
        GUI.Box(panelRect, "", panelStyle);
        GUI.Label(new Rect(20, 15, 340, 170), _statusText, labelStyle);
    }

    private Texture2D MakeTex(int width, int height, Color col)
    {
        Color[] pix = new Color[width * height];
        for (int i = 0; i < pix.Length; i++)
            pix[i] = col;

        Texture2D result = new Texture2D(width, height);
        result.SetPixels(pix);
        result.Apply();
        return result;
    }

    // ===================================================================
    // Public Controls
    // ===================================================================

    public void ToggleAutoRotate() => autoRotate = !autoRotate;
    public void ToggleUI() => showDebugUI = !showDebugUI;
    public void SetOrbitSpeed(float speed) => autoRotateSpeed = speed;
}
