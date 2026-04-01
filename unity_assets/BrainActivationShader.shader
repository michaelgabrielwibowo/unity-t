Shader "Custom/BrainActivationShader"
{
    Properties
    {
        _MainTex ("Base (RGB)", 2D) = "white" {}
        
        // A 1D gradient texture (Blue -> Black -> Red)
        // Values -2.0 to 2.0 map to UV.x 0.0 to 1.0!
        _Colormap ("Colormap (1D)", 2D) = "white" {}
        
        // Limits of activation to squash floats (e.g. min -2.0, max 2.0)
        _MinActivation ("Min Activation", Float) = -1.5
        _MaxActivation ("Max Activation", Float) = 1.5
    }
    
    SubShader
    {
        Tags { "RenderType"="Opaque" "Queue"="Geometry" }
        LOD 200

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma target 5.0 // Required for StructuredBuffers

            #include "UnityCG.cginc"

            struct appdata_t
            {
                float4 vertex : POSITION;
                uint id : SV_VertexID; // The literal index point on the mesh (must exactly match 20,484!)
            };

            struct v2f
            {
                float4 pos : SV_POSITION;
                float activationVal : TEXCOORD0; // The actual loaded fMRI activation value
            };

            // This is matched to `ComputeBuffer(20484, sizeof(float))` from C#
            // You MUST instantiate a material with this shader and use `material.SetBuffer` in C#.
            StructuredBuffer<float> _BrainActivations;

            sampler2D _Colormap;
            float _MinActivation;
            float _MaxActivation;

            v2f vert(appdata_t v)
            {
                v2f o;
                o.pos = UnityObjectToClipPos(v.vertex);
                
                // Read the exact fMRI activation using this Vertex's ID lookup in the GPU RAM
                float rawActivation = _BrainActivations[v.id];
                
                // Keep it bounded within -1.5 and 1.5, then normalize to 0.0 -> 1.0
                float normalized = clamp((rawActivation - _MinActivation) / (_MaxActivation - _MinActivation), 0.0, 1.0);
                o.activationVal = normalized;
                
                return o;
            }

            fixed4 frag(v2f i) : SV_Target
            {
                // Sample the cool-to-warm colormap gradient using our normalized float as the UV lookup!
                fixed4 col = tex2D(_Colormap, float2(i.activationVal, 0.5));
                return col;
            }
            ENDCG
        }
    }
}
