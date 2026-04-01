using UnityEngine;
using System.Collections.Generic;
using System;
// Note: Requires an OSC library like OscCore or extOSC
// using OscCore;

public class OscBrainReceiver : MonoBehaviour
{
    [Header("Network")]
    public int port = 9000;
    public string addressPrefix = "/tribe/vtx";

    [Header("Rendering (Compute Shader via StructuredBuffer)")]
    public MeshRenderer brainRenderer;
    
    [Header("Debug")]
    [Tooltip("If true, bypasses GPU buffer and uses CPU Mesh.colors for debugging")]
    public bool useCpuDebugMode = false;

    [Header("Sequence Settings")]
    [Tooltip("Timeout in seconds before an incomplete sequence is abandoned")]
    public float sequenceTimeoutSec = 2.0f;

    // The total number of vertices expected for fsaverage5
    private const int EXPECTED_VERTICES = 20484;
    private const int CHUNK_SIZE = 256; // Standard size sent from python
    
    // Compute Buffer to push data directly to the GPU
    // This utilizes a StructuredBuffer<float> in HLSL avoiding CPU mesh.colors overhead.
    private ComputeBuffer brainActivationBuffer;
    private float[] currentBrainState;

    // Explicit Per-Chunk Assembly State
    private int currentSequenceId = -1;
    private int lastCompletedSequenceId = -1;
    private int expectedChunks = -1;
    private int receivedChunkCount = 0;
    private bool[] chunkReceivedFlags;
    
    private float lastPacketTime = 0f;

    void Start()
    {
        // 1. Initialize the Compute Buffer (20484 floats, 4 bytes each)
        // Bound as StructuredBuffer<float> _BrainActivations in the HLSL Shader
        brainActivationBuffer = new ComputeBuffer(EXPECTED_VERTICES, sizeof(float));
        currentBrainState = new float[EXPECTED_VERTICES];
        
        // Pass the buffer Reference to the Material's Shader
        if (brainRenderer != null)
        {
            brainRenderer.material.SetBuffer("_BrainActivations", brainActivationBuffer);
        }

        // 2. Setup OSC Listener
        // Replace with your preferred OSC library's initialization
        // Example for OscCore:
        // var receiver = new OscServer(port);
        // receiver.TryAddMethod(addressPrefix, ReadBrainChunk);
        
        Debug.Log($"[BrainReceiver] OSC -> GPU StructuredBuffer Pipeline ready on port {port}.");
    }

    void Update()
    {
        // Timeout Logic for Incomplete Sequences
        if (currentSequenceId != -1 && receivedChunkCount < expectedChunks)
        {
            if (Time.time - lastPacketTime > sequenceTimeoutSec)
            {
                Debug.LogWarning($"[BrainReceiver] Sequence {currentSequenceId} timed out. Missing {expectedChunks - receivedChunkCount}/{expectedChunks} chunks. Dropping incomplete frame.");
                // Reset so we don't keep waiting for it, we will render the last completed state
                currentSequenceId = -1;
            }
        }
    }

    /// <summary>
    /// Callback triggered whenever an OSC packet arrives at /tribe/vtx/<chunk_index>
    /// </summary>
    public void ReadBrainChunk(int sequenceId, int chunkIndex, int totalChunks, float[] values)
    {
        // Check if we already completed this sequence
        if (sequenceId <= lastCompletedSequenceId) return;

        // If a NEWER sequence arrives, abandon the old one completely to prevent latency pile-ups
        if (sequenceId > currentSequenceId)
        {
            currentSequenceId = sequenceId;
            expectedChunks = totalChunks;
            receivedChunkCount = 0;
            
            // Re-allocate flag array if totalChunks somehow changed (unlikely but safe)
            if (chunkReceivedFlags == null || chunkReceivedFlags.Length != totalChunks)
            {
                chunkReceivedFlags = new bool[totalChunks];
            }
            else
            {
                Array.Clear(chunkReceivedFlags, 0, chunkReceivedFlags.Length);
            }
        }
        else if (sequenceId < currentSequenceId)
        {
            // Drop late packets from older sequences
            return;
        }

        // Safety check for valid chunk index
        if (chunkIndex < 0 || chunkIndex >= expectedChunks) return;

        // Explicit Deduplication: Do not increment count twice if packet was heavily duplicated by UDP switch
        if (chunkReceivedFlags[chunkIndex]) return; 

        // Update the last packet time ONLY for valid packets of the current/newer sequence!
        lastPacketTime = Time.time;

        // Calculate where to insert this chunk into the massive 20k array
        int valuesPerChunk = values.Length; // The last payload might be naturally shorter
        int startIndex = chunkIndex * CHUNK_SIZE;

        for (int i = 0; i < valuesPerChunk; i++)
        {
            if (startIndex + i < EXPECTED_VERTICES)
            {
                currentBrainState[startIndex + i] = values[i];
            }
        }

        // Mark as natively received
        chunkReceivedFlags[chunkIndex] = true;
        receivedChunkCount++;

        // Once all chunks have definitively arrived, push exactly once to the GPU!
        if (receivedChunkCount == expectedChunks)
        {
            lastCompletedSequenceId = currentSequenceId;
            RenderBrainState();
        }
    }

    private void RenderBrainState()
    {
        if (useCpuDebugMode)
        {
            if (brainRenderer != null)
            {
                MeshFilter mf = brainRenderer.GetComponent<MeshFilter>();
                if (mf != null && mf.mesh != null)
                {
                    Color[] colors = new Color[EXPECTED_VERTICES];
                    for (int i = 0; i < EXPECTED_VERTICES; i++)
                    {
                        // Simple 1D float logic to grayscale for debug
                        float v = Mathf.Clamp((currentBrainState[i] + 1.5f) / 3.0f, 0f, 1f);
                        colors[i] = new Color(v, v, v, 1f);
                    }
                    mf.mesh.colors = colors;
                }
            }
            return;
        }

        // Near-instantaneous memory copy to the GPU VRAM
        if (brainActivationBuffer != null)
        {
            brainActivationBuffer.SetData(currentBrainState);
        }
    }

    void OnDestroy()
    {
        // Always release ComputeBuffers to prevent memory leaks in Unity
        if (brainActivationBuffer != null)
        {
            brainActivationBuffer.Release();
        }
    }
}
