// ===================================================================
//  TRIBE v2 — Brain Data Sonification Engine for SuperCollider
// ===================================================================
//
//  A layered generative soundscape driven by real-time brain state
//  predictions from the TRIBE v2 streaming engine.
//
//  OSC Addresses consumed:
//    /tribe/brain/pca   — 8 float PCA components (primary control)
//    /tribe/brain/roi   — ~75 float ROI averages (secondary control)
//    /tribe/meta        — timestamp, seq_id, latency_ms
//    /tribe/heartbeat   — keep-alive
//
//  Architecture:
//    Layer 1: Drone Bed        — PCA[0:2] → fundamental + harmonics
//    Layer 2: Granular Texture  — PCA[2:4] → grain density + scatter
//    Layer 3: Rhythmic Pulse    — PCA[4:6] → tempo + accent
//    Layer 4: Spectral Shimmer  — PCA[6:8] → filter + resonance
//
//  All parameters smoothed with Lag.kr to ensure continuous, glitch-free
//  transitions even at 1 Hz update rate.
//
//  Usage:
//    1. Boot SuperCollider server
//    2. Execute this file (Ctrl+Enter on each block, or Cmd+Enter on Mac)
//    3. Start the TRIBE v2 streaming engine
//
// ===================================================================

(
// =========================
// 1. SETUP & INITIALIZATION
// =========================

s.waitForBoot({

    // --- Control Buses ---
    ~pca = Array.fill(8, { Bus.control(s, 1) });
    ~roi = Array.fill(75, { Bus.control(s, 1) });
    ~meta = Dictionary.new;
    ~meta[\seq] = 0;
    ~meta[\latency] = 0;
    ~connected = false;

    // Master volume
    ~masterVol = Bus.control(s, 1);
    ~masterVol.set(0.4);

    "=== TRIBE v2 Sonification: Buses initialized ===".postln;

    // =========================
    // 2. OSC RECEIVERS
    // =========================

    // --- PCA Components (primary sonification control) ---
    OSCdef(\pcaReceiver, { |msg, time, addr, recvPort|
        msg.copyRange(1, msg.size - 1).do({ |val, i|
            if(i < 8, {
                ~pca[i].set(val);
            });
        });
    }, '/tribe/brain/pca');

    // --- ROI Averages (secondary control) ---
    OSCdef(\roiReceiver, { |msg, time, addr, recvPort|
        msg.copyRange(1, msg.size - 1).do({ |val, i|
            if(i < 75, {
                ~roi[i].set(val);
            });
        });
    }, '/tribe/brain/roi');

    // --- Metadata ---
    OSCdef(\metaReceiver, { |msg|
        if(msg.size >= 4, {
            ~meta[\seq] = msg[2];
            ~meta[\latency] = msg[3];
        });
        ~connected = true;
    }, '/tribe/meta');

    // --- Heartbeat ---
    OSCdef(\heartbeat, { |msg|
        ~connected = true;
    }, '/tribe/heartbeat');

    "=== TRIBE v2 Sonification: OSC receivers active ===".postln;

    // =========================
    // 3. SYNTH DEFINITIONS
    // =========================

    // --- Layer 1: Drone Bed ---
    // Deep, evolving drone controlled by PCA[0] (pitch) and PCA[1] (timbre)
    SynthDef(\tribeDrone, { |out=0, pcaBus0, pcaBus1, vol=0.3|
        var pca0 = In.kr(pcaBus0).lag(1.5);  // Smooth over 1.5s
        var pca1 = In.kr(pcaBus1).lag(2.0);

        // Map PCA to musical parameters
        var baseFreq = pca0.linexp(-3, 3, 40, 160);    // Low fundamental
        var brightness = pca1.linlin(-3, 3, 0.1, 0.9);

        // Harmonics with detuning
        var sig = Mix.fill(6, { |i|
            var harmonic = (i + 1);
            var detune = LFNoise1.kr(0.1 + (i * 0.05)).range(-0.5, 0.5);
            var amp = (1 / harmonic) * brightness.pow(i * 0.3);
            SinOsc.ar(baseFreq * harmonic + detune, 0, amp);
        });

        // Slow amplitude modulation
        var tremolo = SinOsc.kr(0.07).range(0.6, 1.0);
        sig = sig * tremolo;

        // Gentle low-pass filter
        var cutoff = pca1.linexp(-3, 3, 200, 4000);
        sig = LPF.ar(sig, cutoff.lag(1.0));

        // Stereo widening
        sig = Splay.ar(Array.fill(3, { sig * LFNoise1.kr(0.02).range(0.5, 1.0) }), 0.7);

        Out.ar(out, sig * vol * Lag.kr(In.kr(~masterVol), 0.5));
    }).add;

    // --- Layer 2: Granular Texture ---
    // Shimmering grains controlled by PCA[2] (density) and PCA[3] (pitch scatter)
    SynthDef(\tribeGrain, { |out=0, pcaBus2, pcaBus3, vol=0.15|
        var pca2 = In.kr(pcaBus2).lag(1.0);
        var pca3 = In.kr(pcaBus3).lag(1.2);

        // Grain parameters
        var density = pca2.linexp(-3, 3, 2, 30);      // Grains per second
        var pitchSpread = pca3.linlin(-3, 3, 0.5, 4);  // Pitch variation

        // Generate granular texture using dust triggers
        var trig = Dust.kr(density);
        var baseFreq = TRand.kr(400, 2000, trig) * pitchSpread;
        var dur = TRand.kr(0.02, 0.15, trig);
        var pan = TRand.kr(-0.8, 0.8, trig);

        // Grain envelope
        var env = EnvGen.ar(Env.perc(0.005, dur), trig);
        var grain = SinOsc.ar(baseFreq) * env;

        // Add some noise texture
        var noise = PinkNoise.ar(0.03) * env;
        grain = grain + noise;

        // Reverb-like diffusion
        grain = FreeVerb.ar(grain, 0.6, 0.8, 0.3);

        Out.ar(out, Pan2.ar(grain, pan) * vol * Lag.kr(In.kr(~masterVol), 0.5));
    }).add;

    // --- Layer 3: Rhythmic Pulse ---
    // Subtle rhythmic element driven by PCA[4] (tempo) and PCA[5] (accent)
    SynthDef(\tribeRhythm, { |out=0, pcaBus4, pcaBus5, vol=0.1|
        var pca4 = In.kr(pcaBus4).lag(2.0);
        var pca5 = In.kr(pcaBus5).lag(1.5);

        // Map to tempo and accent
        var tempo = pca4.linexp(-3, 3, 0.3, 2.0);   // Hz
        var accent = pca5.linlin(-3, 3, 0.2, 0.9);

        // Pulse with soft attack
        var pulse = SinOsc.kr(tempo);
        var env = pulse.max(0).pow(3);  // Soft onset

        // Resonant body
        var freq = pca4.linexp(-3, 3, 60, 200);
        var sig = SinOsc.ar(freq) * env;
        sig = sig + (SinOsc.ar(freq * 2.01) * env * accent * 0.3);

        // Gentle shaping
        sig = LPF.ar(sig, 500);
        sig = Pan2.ar(sig, SinOsc.kr(0.03).range(-0.3, 0.3));

        Out.ar(out, sig * vol * Lag.kr(In.kr(~masterVol), 0.5));
    }).add;

    // --- Layer 4: Spectral Shimmer ---
    // High-frequency spectral content driven by PCA[6] (filter) and PCA[7] (resonance)
    SynthDef(\tribeShimmer, { |out=0, pcaBus6, pcaBus7, vol=0.08|
        var pca6 = In.kr(pcaBus6).lag(1.0);
        var pca7 = In.kr(pcaBus7).lag(0.8);

        // Map to filter parameters
        var cutoff = pca6.linexp(-3, 3, 1000, 12000);
        var rq = pca7.linexp(-3, 3, 0.5, 0.05);  // Narrower = more resonant

        // Source: band-limited noise
        var source = WhiteNoise.ar(0.1) + PinkNoise.ar(0.05);

        // Resonant bandpass filter
        var sig = BPF.ar(source, cutoff, rq);

        // Add shimmering harmonics
        var harmonic = SinOsc.ar(cutoff * [1, 1.5, 2, 2.5]) * 0.02;
        sig = sig + Mix(harmonic);

        // Chorus effect
        sig = sig + CombL.ar(sig, 0.05, SinOsc.kr(0.1).range(0.01, 0.04), 1) * 0.3;

        sig = Splay.ar([sig, DelayL.ar(sig, 0.02, 0.015)], 0.9);

        Out.ar(out, sig * vol * Lag.kr(In.kr(~masterVol), 0.5));
    }).add;

    // --- Master Limiter ---
    SynthDef(\tribeMaster, { |out=0, in=0|
        var sig = In.ar(in, 2);
        sig = Limiter.ar(sig, 0.9, 0.01);
        sig = LeakDC.ar(sig);
        ReplaceOut.ar(out, sig);
    }).add;

    s.sync;
    "=== TRIBE v2 Sonification: SynthDefs loaded ===".postln;

    // =========================
    // 4. LAUNCH SYNTHS
    // =========================

    // Start all layers (they run continuously)
    ~synths = Dictionary.new;

    ~synths[\drone] = Synth(\tribeDrone, [
        \out, 0,
        \pcaBus0, ~pca[0].index,
        \pcaBus1, ~pca[1].index,
        \vol, 0.3
    ]);

    ~synths[\grain] = Synth(\tribeGrain, [
        \out, 0,
        \pcaBus2, ~pca[2].index,
        \pcaBus3, ~pca[3].index,
        \vol, 0.15
    ]);

    ~synths[\rhythm] = Synth(\tribeRhythm, [
        \out, 0,
        \pcaBus4, ~pca[4].index,
        \pcaBus5, ~pca[5].index,
        \vol, 0.1
    ]);

    ~synths[\shimmer] = Synth(\tribeShimmer, [
        \out, 0,
        \pcaBus6, ~pca[6].index,
        \pcaBus7, ~pca[7].index,
        \vol, 0.08
    ]);

    // Master limiter (runs last in the chain)
    ~synths[\master] = Synth.after(~synths[\shimmer], \tribeMaster, [
        \out, 0,
        \in, 0
    ]);

    "=== TRIBE v2 Sonification Engine LIVE ===".postln;
    "    Listening for OSC on port 57120".postln;
    "    Layers: Drone | Grain | Rhythm | Shimmer".postln;
    "    Send /tribe/brain/pca with 8 floats to drive sonification".postln;

    // =========================
    // 5. MONITORING / DEBUG
    // =========================

    // Print status every 5 seconds
    ~monitor = Routine({
        loop {
            var pcaVals = ~pca.collect({ |bus| bus.getSynchronous });
            (
                "TRIBE Sonification | seq: % | latency: %ms | PCA: %"
                .format(~meta[\seq], ~meta[\latency].round(0.1), pcaVals.round(0.01))
            ).postln;
            5.wait;
        }
    }).play;

});
)

// ===================================================================
// CONTROLS (evaluate individually)
// ===================================================================

// Volume control
// ~masterVol.set(0.5);

// Mute/unmute layers
// ~synths[\drone].set(\vol, 0);
// ~synths[\grain].set(\vol, 0.15);

// Stop all
// ~synths.do({ |synth| synth.free });
// ~monitor.stop;

// Debug: print all incoming OSC
// OSCFunc.trace(true);
// OSCFunc.trace(false);
