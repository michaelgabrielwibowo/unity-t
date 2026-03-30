Shader "TRIBE/BrainSurface"
{
    Properties
    {
        _ColormapTex ("Colormap", 2D) = "white" {}
        _ActivationMin ("Activation Min", Float) = -3.0
        _ActivationMax ("Activation Max", Float) = 3.0
        _InterpolationT ("Interpolation T", Range(0,1)) = 0
        _EmissionIntensity ("Emission Intensity", Range(0,5)) = 1.5
        _FresnelPower ("Fresnel Power", Range(0.1, 5)) = 2.0
        _FresnelColor ("Fresnel Color", Color) = (0.3, 0.5, 1.0, 1.0)
        _Opacity ("Opacity", Range(0,1)) = 1.0
        _Smoothness ("Smoothness", Range(0,1)) = 0.6
        _BaseColor ("Base Tint", Color) = (0.15, 0.15, 0.2, 1.0)
    }

    SubShader
    {
        Tags
        {
            "RenderType" = "Opaque"
            "Queue" = "Geometry"
        }
        LOD 200

        CGPROGRAM
        #pragma surface surf Standard fullforwardshadows vertex:vert addshadow
        #pragma target 5.0

        // --- Buffers (set from C# via ComputeBuffer) ---
        #ifdef SHADER_API_D3D11
            StructuredBuffer<float> _PrevState;
            StructuredBuffer<float> _CurrState;
        #endif

        // --- Properties ---
        sampler2D _ColormapTex;
        float _ActivationMin;
        float _ActivationMax;
        float _InterpolationT;
        float _EmissionIntensity;
        float _FresnelPower;
        float4 _FresnelColor;
        float _Opacity;
        float _Smoothness;
        float4 _BaseColor;

        struct Input
        {
            float3 worldPos;
            float3 viewDir;
            float3 worldNormal;
            float activation;
            INTERNAL_DATA
        };

        // --- Vertex Shader ---
        void vert(inout appdata_full v, out Input o)
        {
            UNITY_INITIALIZE_OUTPUT(Input, o);

            #ifdef SHADER_API_D3D11
                uint vertexID = v.vertex.w; // Using w component for vertex ID
                // Alternative: use SV_VertexID if available
                // For surface shaders, we can use the vertex index
                // Read activation from buffers and interpolate
                float prevAct = _PrevState[vertexID];
                float currAct = _CurrState[vertexID];
                float activation = lerp(prevAct, currAct, _InterpolationT);
            #else
                float activation = 0;
            #endif

            o.activation = activation;

            // Optional: displace vertices slightly based on activation
            float displacement = activation * 0.001;
            v.vertex.xyz += v.normal * displacement;
        }

        // --- Surface Shader ---
        void surf(Input IN, inout SurfaceOutputStandard o)
        {
            // Normalize activation to [0, 1] for colormap lookup
            float normalizedAct = saturate(
                (IN.activation - _ActivationMin) / (_ActivationMax - _ActivationMin)
            );

            // Sample colormap texture
            float4 colormapColor = tex2D(_ColormapTex, float2(normalizedAct, 0.5));

            // Fresnel rim effect
            float fresnel = pow(1.0 - saturate(dot(IN.viewDir, IN.worldNormal)), _FresnelPower);
            float3 fresnelColor = _FresnelColor.rgb * fresnel;

            // Base color with colormap
            float3 baseColor = lerp(_BaseColor.rgb, colormapColor.rgb, 0.9);

            // Output
            o.Albedo = baseColor;
            o.Emission = colormapColor.rgb * _EmissionIntensity * normalizedAct + fresnelColor;
            o.Smoothness = _Smoothness;
            o.Metallic = 0.1;
            o.Alpha = _Opacity;
        }
        ENDCG
    }

    // --- Fallback for older GPUs ---
    SubShader
    {
        Tags { "RenderType" = "Opaque" }
        LOD 100

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag

            #include "UnityCG.cginc"

            struct appdata
            {
                float4 vertex : POSITION;
                float3 normal : NORMAL;
            };

            struct v2f
            {
                float4 pos : SV_POSITION;
                float3 normal : TEXCOORD0;
                float3 viewDir : TEXCOORD1;
            };

            float4 _BaseColor;
            float _FresnelPower;
            float4 _FresnelColor;

            v2f vert(appdata v)
            {
                v2f o;
                o.pos = UnityObjectToClipPos(v.vertex);
                o.normal = UnityObjectToWorldNormal(v.normal);
                o.viewDir = normalize(WorldSpaceViewDir(v.vertex));
                return o;
            }

            float4 frag(v2f i) : SV_Target
            {
                float fresnel = pow(1.0 - saturate(dot(i.viewDir, i.normal)), _FresnelPower);
                float3 diffuse = _BaseColor.rgb * max(0.2, dot(i.normal, float3(0.5, 1, 0.3)));
                float3 color = diffuse + _FresnelColor.rgb * fresnel;
                return float4(color, 1.0);
            }
            ENDCG
        }
    }

    FallBack "Diffuse"
}
