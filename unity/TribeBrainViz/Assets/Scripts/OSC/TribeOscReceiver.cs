using System;
using System.Collections.Generic;
using System.Threading;
using UnityEngine;

/// <summary>
/// Receives TRIBE v2 brain state data via OSC (Open Sound Control).
/// 
/// Uses OscCore for high-performance UDP reception. Reassembles
/// chunked vertex data into a single float[] and provides thread-safe
/// events for consumers.
///
/// OSC Addresses handled:
///   /tribe/brain/full/{i}  — Chunked vertex data (seq_id, chunk_idx, total_chunks, ...floats)
///   /tribe/brain/roi       — ROI averages (float[])
///   /tribe/meta            — Metadata (timestamp, sequence_id, latency_ms)
///   /tribe/heartbeat       — Keep-alive
///   /tribe/brain/pca       — PCA components (float[])
///   /tribe/brain/delta     — Brain state delta info
/// </summary>
public class TribeOscReceiver : MonoBehaviour
{
    [Header("OSC Settings")]
    [SerializeField] private int listenPort = 9000;
    [SerializeField] private int expectedVertexCount = 20484;
    [SerializeField] private int expectedChunks = 5;

    [Header("Status")]
    [SerializeField] private bool isConnected = false;
    [SerializeField] private int messagesReceived = 0;
    [SerializeField] private float lastHeartbeatTime = 0f;
    [SerializeField] private float lastLatencyMs = 0f;
    [SerializeField] private int lastSequenceId = -1;

    // --- Events ---
    public event Action<float[]> OnBrainStateReceived;
    public event Action<float[]> OnROIReceived;
    public event Action<float[]> OnPCAReceived;
    public event Action<float, int, float> OnMetaReceived; // timestamp, seq_id, latency_ms

    // --- Double-buffered vertex data ---
    private float[] _assemblyBuffer;
    private float[] _readyBuffer;
    private readonly object _bufferLock = new object();
    private int _chunksReceived = 0;
    private int _currentAssemblySeqId = -1;
    private bool _newStateAvailable = false;

    // --- ROI / PCA latest data ---
    private float[] _latestROI;
    private float[] _latestPCA;
    private bool _newROIAvailable = false;
    private bool _newPCAAvailable = false;

    // --- UDP receiver (minimal, compatible without OscCore) ---
    private System.Net.Sockets.UdpClient _udpClient;
    private Thread _receiveThread;
    private bool _running = false;

    // ===================================================================
    // Unity Lifecycle
    // ===================================================================

    void Awake()
    {
        _assemblyBuffer = new float[expectedVertexCount];
        _readyBuffer = new float[expectedVertexCount];
    }

    void OnEnable()
    {
        StartListening();
    }

    void OnDisable()
    {
        StopListening();
    }

    void Update()
    {
        // Check for new brain state on main thread
        lock (_bufferLock)
        {
            if (_newStateAvailable)
            {
                OnBrainStateReceived?.Invoke(_readyBuffer);
                _newStateAvailable = false;
            }

            if (_newROIAvailable && _latestROI != null)
            {
                OnROIReceived?.Invoke(_latestROI);
                _newROIAvailable = false;
            }

            if (_newPCAAvailable && _latestPCA != null)
            {
                OnPCAReceived?.Invoke(_latestPCA);
                _newPCAAvailable = false;
            }
        }

        // Check heartbeat timeout
        if (isConnected && Time.time - lastHeartbeatTime > 5f)
        {
            isConnected = false;
            Debug.LogWarning("[TribeOSC] Connection lost (no heartbeat for 5s)");
        }
    }

    // ===================================================================
    // UDP Listener
    // ===================================================================

    private void StartListening()
    {
        if (_running) return;

        try
        {
            _udpClient = new System.Net.Sockets.UdpClient(listenPort);
            _running = true;
            _receiveThread = new Thread(ReceiveLoop)
            {
                IsBackground = true,
                Name = "TribeOSC-Receiver"
            };
            _receiveThread.Start();
            Debug.Log($"[TribeOSC] Listening on port {listenPort}");
        }
        catch (Exception ex)
        {
            Debug.LogError($"[TribeOSC] Failed to start listener: {ex.Message}");
        }
    }

    private void StopListening()
    {
        _running = false;
        _udpClient?.Close();
        _receiveThread?.Join(1000);
        Debug.Log("[TribeOSC] Listener stopped");
    }

    private void ReceiveLoop()
    {
        var remoteEP = new System.Net.IPEndPoint(System.Net.IPAddress.Any, 0);

        while (_running)
        {
            try
            {
                byte[] data = _udpClient.Receive(ref remoteEP);
                if (data != null && data.Length > 0)
                {
                    ParseOscMessage(data);
                    Interlocked.Increment(ref messagesReceived);
                }
            }
            catch (System.Net.Sockets.SocketException)
            {
                // Socket closed during shutdown
                break;
            }
            catch (Exception ex)
            {
                if (_running)
                    Debug.LogWarning($"[TribeOSC] Receive error: {ex.Message}");
            }
        }
    }

    // ===================================================================
    // OSC Parsing (minimal OSC 1.0 parser)
    // ===================================================================

    private void ParseOscMessage(byte[] data)
    {
        try
        {
            int offset = 0;
            string address = ReadOscString(data, ref offset);
            string typetag = ReadOscString(data, ref offset);

            if (string.IsNullOrEmpty(address)) return;

            // Extract float arguments
            List<float> floatArgs = new List<float>();
            List<int> intArgs = new List<int>();

            if (typetag != null && typetag.Length > 0)
            {
                for (int i = 0; i < typetag.Length; i++)
                {
                    char t = typetag[i];
                    if (t == ',') continue;
                    if (t == 'f')
                    {
                        floatArgs.Add(ReadOscFloat(data, ref offset));
                    }
                    else if (t == 'i')
                    {
                        int val = ReadOscInt(data, ref offset);
                        intArgs.Add(val);
                        floatArgs.Add(val); // also store as float for convenience
                    }
                    else if (t == 'd')
                    {
                        // double — read 8 bytes
                        floatArgs.Add((float)ReadOscDouble(data, ref offset));
                    }
                }
            }

            // Route by address
            if (address.StartsWith("/tribe/brain/full/"))
            {
                HandleFullVertexChunk(address, floatArgs);
            }
            else if (address == "/tribe/brain/roi")
            {
                HandleROI(floatArgs);
            }
            else if (address == "/tribe/brain/pca")
            {
                HandlePCA(floatArgs);
            }
            else if (address == "/tribe/meta")
            {
                HandleMeta(floatArgs);
            }
            else if (address == "/tribe/heartbeat")
            {
                HandleHeartbeat();
            }
        }
        catch (Exception ex)
        {
            Debug.LogWarning($"[TribeOSC] Parse error: {ex.Message}");
        }
    }

    // ===================================================================
    // Message Handlers
    // ===================================================================

    private void HandleFullVertexChunk(string address, List<float> args)
    {
        // Args: [seq_id, chunk_idx, total_chunks, ...vertex_data]
        if (args.Count < 4) return;

        int seqId = (int)args[0];
        int chunkIdx = (int)args[1];
        int totalChunks = (int)args[2];

        lock (_bufferLock)
        {
            // Start new assembly if sequence changed
            if (seqId != _currentAssemblySeqId)
            {
                _currentAssemblySeqId = seqId;
                _chunksReceived = 0;
                Array.Clear(_assemblyBuffer, 0, _assemblyBuffer.Length);
            }

            // Copy vertex data into assembly buffer
            int chunkSize = args.Count - 3;
            int startIdx = chunkIdx * (expectedVertexCount / totalChunks);

            for (int i = 0; i < chunkSize && (startIdx + i) < expectedVertexCount; i++)
            {
                _assemblyBuffer[startIdx + i] = args[3 + i];
            }

            _chunksReceived++;

            // All chunks received — swap buffers
            if (_chunksReceived >= totalChunks)
            {
                // Swap assembly → ready
                var temp = _readyBuffer;
                _readyBuffer = _assemblyBuffer;
                _assemblyBuffer = temp;
                _newStateAvailable = true;
                lastSequenceId = seqId;
            }
        }
    }

    private void HandleROI(List<float> args)
    {
        lock (_bufferLock)
        {
            _latestROI = args.ToArray();
            _newROIAvailable = true;
        }
    }

    private void HandlePCA(List<float> args)
    {
        lock (_bufferLock)
        {
            _latestPCA = args.ToArray();
            _newPCAAvailable = true;
        }
    }

    private void HandleMeta(List<float> args)
    {
        if (args.Count >= 3)
        {
            float timestamp = args[0];
            int seqId = (int)args[1];
            float latency = args[2];

            lastLatencyMs = latency;
            isConnected = true;

            OnMetaReceived?.Invoke(timestamp, seqId, latency);
        }
    }

    private void HandleHeartbeat()
    {
        lastHeartbeatTime = Time.time;
        isConnected = true;
    }

    // ===================================================================
    // OSC Binary Helpers
    // ===================================================================

    private string ReadOscString(byte[] data, ref int offset)
    {
        int start = offset;
        while (offset < data.Length && data[offset] != 0)
            offset++;

        string result = System.Text.Encoding.ASCII.GetString(data, start, offset - start);
        offset++; // skip null terminator

        // Align to 4-byte boundary
        while (offset % 4 != 0) offset++;

        return result;
    }

    private float ReadOscFloat(byte[] data, ref int offset)
    {
        if (BitConverter.IsLittleEndian)
        {
            byte[] bytes = new byte[4];
            Array.Copy(data, offset, bytes, 0, 4);
            Array.Reverse(bytes);
            offset += 4;
            return BitConverter.ToSingle(bytes, 0);
        }
        else
        {
            float val = BitConverter.ToSingle(data, offset);
            offset += 4;
            return val;
        }
    }

    private int ReadOscInt(byte[] data, ref int offset)
    {
        if (BitConverter.IsLittleEndian)
        {
            byte[] bytes = new byte[4];
            Array.Copy(data, offset, bytes, 0, 4);
            Array.Reverse(bytes);
            offset += 4;
            return BitConverter.ToInt32(bytes, 0);
        }
        else
        {
            int val = BitConverter.ToInt32(data, offset);
            offset += 4;
            return val;
        }
    }

    private double ReadOscDouble(byte[] data, ref int offset)
    {
        if (BitConverter.IsLittleEndian)
        {
            byte[] bytes = new byte[8];
            Array.Copy(data, offset, bytes, 0, 8);
            Array.Reverse(bytes);
            offset += 8;
            return BitConverter.ToDouble(bytes, 0);
        }
        else
        {
            double val = BitConverter.ToDouble(data, offset);
            offset += 8;
            return val;
        }
    }

    // ===================================================================
    // Public Accessors
    // ===================================================================

    public bool IsConnected => isConnected;
    public int MessagesReceived => messagesReceived;
    public float LastLatencyMs => lastLatencyMs;
    public int LastSequenceId => lastSequenceId;
}
