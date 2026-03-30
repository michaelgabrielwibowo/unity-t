Shader "TRIBE/BrainParticles"
{
    Properties
    {
        _MainTex ("Particle Texture", 2D) = "white" {}
        _Color ("Tint", Color) = (0.4, 0.7, 1.0, 1.0)
        _EmissionColor ("Emission Color", Color) = (0.3, 0.5, 1.0, 1.0)
        _EmissionStrength ("Emission Strength", Range(0, 10)) = 3.0
        _Size ("Base Size", Range(0.001, 0.1)) = 0.01
        _ActivationThreshold ("Activation Threshold", Range(0, 5)) = 1.0
        _PulseSpeed ("Pulse Speed", Range(0, 10)) = 2.0
        _Softness ("Edge Softness", Range(0.01, 1)) = 0.5
    }

    SubShader
    {
        Tags
        {
            "Queue" = "Transparent"
            "RenderType" = "Transparent"
            "IgnoreProjector" = "True"
        }

        Blend SrcAlpha OneMinusSrcAlpha
        ZWrite Off
        Cull Off

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma multi_compile_particles

            #include "UnityCG.cginc"

            struct appdata
            {
                float4 vertex : POSITION;
                float4 color : COLOR;
                float2 texcoord : TEXCOORD0;
                float3 normal : NORMAL;
            };

            struct v2f
            {
                float4 pos : SV_POSITION;
                float4 color : COLOR;
                float2 uv : TEXCOORD0;
                float3 worldPos : TEXCOORD1;
            };

            sampler2D _MainTex;
            float4 _Color;
            float4 _EmissionColor;
            float _EmissionStrength;
            float _Size;
            float _ActivationThreshold;
            float _PulseSpeed;
            float _Softness;

            v2f vert(appdata v)
            {
                v2f o;

                // Billboard the particle to face camera
                float3 worldPos = mul(unity_ObjectToWorld, v.vertex).xyz;

                // Pulse animation based on time
                float pulse = sin(_Time.y * _PulseSpeed + worldPos.x * 3.14159) * 0.5 + 0.5;

                o.pos = UnityObjectToClipPos(v.vertex);
                o.color = v.color * _Color;
                o.color.a *= (0.3 + pulse * 0.7);
                o.uv = v.texcoord;
                o.worldPos = worldPos;
                return o;
            }

            float4 frag(v2f i) : SV_Target
            {
                // Radial gradient for soft particles
                float2 center = i.uv - 0.5;
                float dist = length(center) * 2.0;
                float alpha = smoothstep(1.0, 1.0 - _Softness, dist);

                // Sample texture
                float4 texColor = tex2D(_MainTex, i.uv);

                // Final color with emission
                float4 color = i.color * texColor;
                color.rgb += _EmissionColor.rgb * _EmissionStrength * alpha;
                color.a *= alpha;

                // Discard nearly transparent fragments
                clip(color.a - 0.01);

                return color;
            }
            ENDCG
        }
    }

    FallBack "Particles/Alpha Blended"
}
