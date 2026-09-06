// ARKit-only phoneme/viseme retargeting table
// Subtly and anatomically calibrated to avoid over-articulation while preserving natural phonetic shapes.
export const VISEME_TO_ARKIT = Object.freeze({
  0: { // viseme_aa (open vowel: ah, father)
    jawOpen: 0.28,
    mouthLowerDownLeft: 0.04,
    mouthLowerDownRight: 0.04,
  },
  1: { // viseme_E (front open-mid: eh, dress)
    jawOpen: 0.16,
    mouthStretchLeft: 0.20,
    mouthStretchRight: 0.20,
    mouthDimpleLeft: 0.10,
    mouthDimpleRight: 0.10,
  },
  2: { // viseme_I (front close: ee/ih, fleece/bit)
    jawOpen: 0.08,
    mouthStretchLeft: 0.24,
    mouthStretchRight: 0.24,
    mouthDimpleLeft: 0.12,
    mouthDimpleRight: 0.12,
  },
  3: { // viseme_O (back mid-close: oh, goat)
    jawOpen: 0.20,
    mouthFunnel: 0.30,
    mouthPucker: 0.16,
  },
  4: { // viseme_U (back close: oo, goose)
    jawOpen: 0.08,
    mouthPucker: 0.32,
    mouthFunnel: 0.18,
  },
  5: { // viseme_PP (bilabials: p, b, m) - clean lip seal without jaw distortion
    mouthPressLeft: 0.38,
    mouthPressRight: 0.38,
    mouthRollLower: 0.18,
    mouthRollUpper: 0.18,
  },
  6: { // viseme_SS (alveolar sibilants: s, z) - narrow jaw, horizontal stretch
    jawOpen: 0.05,
    mouthStretchLeft: 0.18,
    mouthStretchRight: 0.18,
    mouthLowerDownLeft: 0.04,
    mouthLowerDownRight: 0.04,
  },
  7: { // viseme_TH (dental fricatives: th) - slight tongue display
    jawOpen: 0.10,
    mouthPressLeft: 0.08,
    mouthPressRight: 0.08,
    tongueOut: 0.20,
  },
  8: { // viseme_DD (alveolar stops: t, d, n)
    jawOpen: 0.10,
    mouthStretchLeft: 0.10,
    mouthStretchRight: 0.10,
    mouthLowerDownLeft: 0.04,
    mouthLowerDownRight: 0.04,
  },
  9: { // viseme_FF (labiodentals: f, v) - lower lip under upper incisors
    mouthShrugLower: 0.26,
    mouthRollLower: 0.22,
    mouthUpperUpLeft: 0.10,
    mouthUpperUpRight: 0.10,
    jawOpen: 0.04,
  },
  10: { // viseme_kk (velars: k, g)
    jawOpen: 0.14,
    mouthLowerDownLeft: 0.06,
    mouthLowerDownRight: 0.06,
  },
  11: { // viseme_nn (nasals: n, ng)
    jawOpen: 0.08,
    mouthStretchLeft: 0.08,
    mouthStretchRight: 0.08,
  },
  12: { // viseme_RR (liquids: r, er)
    jawOpen: 0.10,
    mouthPucker: 0.16,
    mouthFunnel: 0.12,
  },
  13: { // viseme_CH (affricates/fricatives: ch, j, sh)
    jawOpen: 0.10,
    mouthFunnel: 0.26,
    mouthShrugLower: 0.14,
    mouthUpperUpLeft: 0.08,
    mouthUpperUpRight: 0.08,
  },
  14: { // viseme_sil (silence / neutral rest)
  }
});

// Analytical critically damped spring angular frequencies (rad/s)
// Solves: x''(t) + 2*omega*x'(t) + omega^2*(x - target) = 0 exactly.
// Unconditionally stable for any dt >= 0 without oscillation or overshoot.
export const ARTICULATOR_OMEGA = Object.freeze({
  jawOpen: 14,             // Slower, heavy mandible bone
  mouthLowerDownLeft: 16,
  mouthLowerDownRight: 16,
  mouthStretchLeft: 18,     // Medium vowel spread
  mouthStretchRight: 18,
  mouthDimpleLeft: 18,
  mouthDimpleRight: 18,
  mouthFunnel: 20,          // Medium lip rounding
  mouthPucker: 20,
  mouthShrugLower: 22,      // Fast labiodental
  mouthUpperUpLeft: 22,
  mouthUpperUpRight: 22,
  tongueOut: 22,            // Fast tongue
  mouthPressLeft: 28,       // Fast bilabial closure (P/B/M)
  mouthPressRight: 28,
  mouthRollLower: 26,
  mouthRollUpper: 26,
});

export const REQUIRED_OCULUS_VISEMES = Object.freeze([
  "viseme_sil","viseme_PP","viseme_FF","viseme_TH","viseme_DD","viseme_kk","viseme_CH",
  "viseme_SS","viseme_nn","viseme_RR","viseme_aa","viseme_E","viseme_I","viseme_O","viseme_U"
]);

export const MANAGED_LIP_TARGETS = Object.freeze([
  'jawOpen', 'mouthClose', 'mouthFunnel', 'mouthPucker',
  'mouthSmileLeft', 'mouthSmileRight', 'mouthFrownLeft', 'mouthFrownRight',
  'mouthDimpleLeft', 'mouthDimpleRight', 'mouthStretchLeft', 'mouthStretchRight',
  'mouthRollLower', 'mouthRollUpper', 'mouthShrugLower', 'mouthShrugUpper',
  'mouthPressLeft', 'mouthPressRight', 'mouthLowerDownLeft', 'mouthLowerDownRight',
  'mouthUpperUpLeft', 'mouthUpperUpRight', 'tongueOut',
  ...REQUIRED_OCULUS_VISEMES
]);

/**
 * Exact analytical solution for critically damped spring step.
 * Guaranteed 100% stable for any dt >= 0.
 */
export function criticallyDampedStep(value, velocity, target, omega, dt) {
  const x = value - target;
  const c = velocity + omega * x;
  const e = Math.exp(-omega * dt);
  const newValue = target + (x + c * dt) * e;
  const newVelocity = (velocity - omega * c * dt) * e;
  return {
    value: Math.max(0, Math.min(1, newValue)),
    velocity: newVelocity
  };
}

export class LipSyncController {
  constructor(head = null, { log = () => {}, telemetry } = {}) {
    this.head = head;
    this.log = log;
    this.telemetry = telemetry;

    this.visemeStrength = 0.85;
    this.vowelBoost = 1.0;
    this.lipSpeed = 1.0;
    this.playbackDelayMs = 90;

    this.vadActive = false;
    this.audioDb = -100;
    this.isSpeaking = false;
    this.currViseme = 14;
    this.lastVadData = null;
    this.lastVisemeData = null;
    this.onTelemetry = null;

    // Per-channel state: { value, velocity, target }
    this.channels = {};
    for (const name of MANAGED_LIP_TARGETS) {
      this.channels[name] = { value: 0, velocity: 0, target: 0 };
    }
  }

  setHead(head) {
    this.head = head;
  }

  setStrength(strength) {
    this.visemeStrength = Math.max(0.1, Math.min(1.5, Number(strength) || 0.85));
    // Re-apply target scaling
    this.applyTargetViseme(this.currViseme);
  }

  setVowelBoost(boost) {
    this.vowelBoost = Math.max(1.0, Math.min(1.6, Number(boost) || 1.0));
    this.applyTargetViseme(this.currViseme);
  }

  setLipSpeed(speed) {
    // Scales the critically damped spring omegas: >1 snappier lips, <1 lazier.
    this.lipSpeed = Math.max(0.5, Math.min(2, Number(speed) || 1));
  }

  setPlaybackDelay(delayMs) {
    this.playbackDelayMs = Math.max(0, Math.min(500, Number(delayMs) || 0));
  }

  onVad(data) {
    if (!data) return;
    this.lastVadData = data;
    this.vadActive = Boolean(data.active > 0);
    this.audioDb = typeof data.db === 'number' ? data.db : (this.vadActive ? -20 : -100);
  }

  onStarted() {
    this.isSpeaking = true;
  }

  onEnded() {
    this.isSpeaking = false;
    this.applyTargetViseme(14);
  }

  onViseme(data) {
    if (!data) return;
    this.lastVisemeData = data;
    const rawViseme = data.viseme;

    if (rawViseme === null || rawViseme === undefined || rawViseme === 14 || rawViseme === -1) {
      this.applyTargetViseme(14);
      return;
    }

    // Breath / quiet audio gate: if signal is below -48 dB and VAD is inactive, treat as silence
    if (this.audioDb < -48 && !this.vadActive) {
      this.applyTargetViseme(14);
      return;
    }

    this.applyTargetViseme(rawViseme);
  }

  applyTargetViseme(visemeIdx) {
    // Crucial fix: Do NOT restart or recompute if the viseme hasn't changed
    if (visemeIdx === this.currViseme) {
      return;
    }

    this.currViseme = visemeIdx;
    const shapeDef = VISEME_TO_ARKIT[visemeIdx] || {};
    const isVowel = (visemeIdx >= 0 && visemeIdx <= 4);

    for (const name of MANAGED_LIP_TARGETS) {
      const ch = this.channels[name];
      if (!ch) continue;

      if (REQUIRED_OCULUS_VISEMES.includes(name)) {
        ch.target = 0; // Oculus visemes always held at 0
        continue;
      }

      const baseVal = shapeDef[name] || 0;
      if (baseVal === 0) {
        ch.target = 0;
      } else {
        const val = isVowel ? baseVal * this.vowelBoost : baseVal;
        ch.target = val * this.visemeStrength;
      }
    }
  }

  update(dt) {
    // Convert dt to seconds (clamp to prevent extreme dt jumps, though analytical solver is unconditionally stable)
    const dtSec = Math.min(Math.max(dt / 1000, 0.001), 0.1);

    for (const name of MANAGED_LIP_TARGETS) {
      const ch = this.channels[name];
      if (!ch) continue;

      const omega = (ARTICULATOR_OMEGA[name] || 20) * this.lipSpeed;
      const { value, velocity } = criticallyDampedStep(ch.value, ch.velocity, ch.target, omega, dtSec);
      ch.value = value;
      ch.velocity = velocity;

      // Clean cutoff for near-zero resting values
      if (ch.target === 0 && ch.value < 0.0005 && Math.abs(ch.velocity) < 0.005) {
        ch.value = 0;
        ch.velocity = 0;
      }
    }

    if (this.head?.mtAvatar) {
      // Apply clamped analytical spring values directly to TalkingHead morph targets
      for (const name of MANAGED_LIP_TARGETS) {
        const target = this.head.mtAvatar[name];
        if (target) {
          const val = this.channels[name]?.value || 0;
          Object.assign(target, { newvalue: val, needsUpdate: true });
        }
      }
    }

    if (this.onTelemetry && typeof this.onTelemetry === 'function') {
      try { this.onTelemetry(dt); } catch {}
    }
  }

  reset() {
    this.currViseme = 14;
    this.isSpeaking = false;
    for (const name of MANAGED_LIP_TARGETS) {
      const ch = this.channels[name];
      if (ch) {
        ch.value = 0;
        ch.velocity = 0;
        ch.target = 0;
      }
      if (this.head?.mtAvatar?.[name]) {
        Object.assign(this.head.mtAvatar[name], { newvalue: 0, needsUpdate: true });
      }
    }
  }
}
