import {DEFAULT_HEADAUDIO_MODEL} from './headaudio-config.js';

export const AVATAR_LAB_DEFAULTS = Object.freeze({
  "voice": "Orus",
  "lipSyncEngine": "headaudio",
  "headaudioModel": DEFAULT_HEADAUDIO_MODEL,
  "avatarUrl": "model3.glb",
  "playbackDelayMs": 90,
  "visemeStrength": 1,
  "vowelBoost": 1,
  "lipSpeed": 1,
  "audioSpeed": 1,
  "silenceDebounceMs": 50,
  "hardSilenceClose": true,
  "freezeIdle": true,
  "eyePitch": -0.75,
  "lidLift": 1.15,
  "gazeTargetX": 0,
  "gazeTargetY": 0,
  "headPitchBias": -0.12,
  "headYawBias": 0,
  "eyeYawBias": 0,
  "browLift": 0,
  "outputPrebufferChunks": 2,
  "localPcmChunkMs": 40
});

export const GAZE_DEFAULTS = Object.freeze({
  "gazeTargetX": 0,
  "gazeTargetY": 0,
  "headPitchBias": -0.1,
  "headYawBias": 0,
  "eyeYawBias": 0,
  "browLift": 0,
  "eyePitch": -0.7,
  "lidLift": 1.2,
  "freezeAnimation": false
});

export const AVATAR_LAB_LIMITS = Object.freeze({
  playbackDelayMs: { min: 0, max: 250, step: 10 },
  visemeStrength: { min: 0.1, max: 1, step: 0.05 },
  vowelBoost: { min: 1, max: 1.6, step: 0.05 },
  lipSpeed: { min: 0.5, max: 2, step: 0.05 },
  audioSpeed: { min: 0.75, max: 1.5, step: 0.05 },
  silenceDebounceMs: { min: 20, max: 200, step: 10 },
  eyePitch: { min: -1, max: 1, step: 0.05 },
  lidLift: { min: 0, max: 2, step: 0.05 },
  gazeTargetX: { min: -200, max: 200, step: 10 },
  gazeTargetY: { min: -200, max: 200, step: 10 },
  headPitchBias: { min: -0.6, max: 0.6, step: 0.05 },
  headYawBias: { min: -0.8, max: 0.8, step: 0.05 },
  eyeYawBias: { min: -1, max: 1, step: 0.05 },
  browLift: { min: 0, max: 1, step: 0.05 },
});

const LEGACY_QUERY_KEYS = Object.freeze({
  playbackDelayMs: 'delay', visemeStrength: 'viseme',
  outputPrebufferChunks: 'prebuffer',
});

const STORAGE_KEY = 'avatar-lab-settings';
const GAZE_KEY = 'avatar-lab-gaze';

const GAZE_LIMITS = Object.freeze({
  gazeTargetX: { min: -200, max: 200, step: 10 }, gazeTargetY: { min: -200, max: 200, step: 10 },
  headPitchBias: { min: -0.6, max: 0.6, step: 0.05 }, headYawBias: { min: -0.8, max: 0.8, step: 0.05 },
  eyeYawBias: { min: -1, max: 1, step: 0.05 }, browLift: { min: 0, max: 1, step: 0.05 },
  eyePitch: { min: -1, max: 1, step: 0.05 }, lidLift: { min: 0, max: 2, step: 0.05 },
});

export function loadGazeSettings(storage = localStorage) {
  try {
    const parsed = JSON.parse(storage.getItem(GAZE_KEY) || '{}');
    const g = {};
    for (const key of Object.keys(GAZE_LIMITS)) g[key] = boundedNumber(parsed[key], GAZE_DEFAULTS[key], GAZE_LIMITS[key]);
    g.freezeAnimation = parsed.freezeAnimation !== undefined ? Boolean(parsed.freezeAnimation) : GAZE_DEFAULTS.freezeAnimation;
    return g;
  } catch { return { ...GAZE_DEFAULTS }; }
}
export function saveGazeSettings(gaze, storage = localStorage) {
  try {
    const out = {};
    for (const k of Object.keys(GAZE_LIMITS)) out[k] = Number(gaze[k]);
    out.freezeAnimation = Boolean(gaze.freezeAnimation !== false);
    storage.setItem(GAZE_KEY, JSON.stringify(out));
    return true;
  }
  catch { return false; }
}
export function clearSavedGazeSettings(storage = localStorage) {
  try { storage.removeItem(GAZE_KEY); } catch {}
}
export function readGazeControls(root = document) {
  const g = { ...GAZE_DEFAULTS };
  for (const key of Object.keys(GAZE_LIMITS)) {
    const input = root.querySelector(`[data-gaze-setting="${key}"]`);
    if (input) g[key] = boundedNumber(input.value, GAZE_DEFAULTS[key], GAZE_LIMITS[key]);
  }
  const freezeInput = root.querySelector('[data-gaze-setting="freezeAnimation"]');
  if (freezeInput) g.freezeAnimation = Boolean(freezeInput.checked);
  return g;
}
export function populateGazeControls(gaze, root = document) {
  for (const [k, v] of Object.entries(GAZE_LIMITS)) {
    const input = root.querySelector(`[data-gaze-setting="${k}"]`);
    if (!input) continue;
    input.value = String(gaze[k]);
    const out = root.querySelector(`[data-gaze-setting-value="${k}"]`);
    if (out) out.value = String(gaze[k]);
  }
  const freezeInput = root.querySelector('[data-gaze-setting="freezeAnimation"]');
  if (freezeInput) freezeInput.checked = Boolean(gaze.freezeAnimation !== false);
}

// Persisted lab configuration. Precedence: defaults < localStorage < URL params.
export function loadSavedSettings(storage = localStorage) {
  try {
    const parsed = JSON.parse(storage.getItem(STORAGE_KEY) || '{}');
    const saved = {};
    for (const key of Object.keys(AVATAR_LAB_LIMITS)) {
      if (parsed[key] != null) saved[key] = boundedNumber(parsed[key], AVATAR_LAB_DEFAULTS[key], AVATAR_LAB_LIMITS[key]);
    }
    if (typeof parsed.voice === 'string' && parsed.voice) saved.voice = parsed.voice;
    if (['headaudio','wav2vec2'].includes(parsed.lipSyncEngine)) saved.lipSyncEngine = parsed.lipSyncEngine;
    if (typeof parsed.headaudioModel === 'string' && parsed.headaudioModel && /^[\w.-]+\.bin$/.test(parsed.headaudioModel)) saved.headaudioModel = parsed.headaudioModel;
    if (typeof parsed.avatarUrl === 'string') saved.avatarUrl = parsed.avatarUrl;
    if (parsed.hardSilenceClose != null) saved.hardSilenceClose = Boolean(parsed.hardSilenceClose);
    if (parsed.freezeIdle != null) saved.freezeIdle = Boolean(parsed.freezeIdle);
    return saved;
  } catch { return {}; }
}

export function saveSettings(settings, storage = localStorage) {
  try {
    const out = { voice: String(settings.voice || ''), lipSyncEngine: settings.lipSyncEngine === 'wav2vec2' ? 'wav2vec2' : 'headaudio', hardSilenceClose: Boolean(settings.hardSilenceClose), freezeIdle: Boolean(settings.freezeIdle), avatarUrl: String(settings.avatarUrl || '') };
    const headaudioModel = String(settings.headaudioModel || '').split(/[\\/]/).pop();
    if (/^[\w.-]+\.bin$/.test(headaudioModel)) out.headaudioModel = headaudioModel;
    for (const key of Object.keys(AVATAR_LAB_LIMITS)) out[key] = Number(settings[key]);
    storage.setItem(STORAGE_KEY, JSON.stringify(out));
    return true;
  } catch { return false; }
}

export function clearSavedSettings(storage = localStorage) {
  try { storage.removeItem(STORAGE_KEY); } catch {}
}

function boundedNumber(value, fallback, limits) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return fallback;
  return Math.max(limits.min, Math.min(limits.max, parsed));
}

export function settingsFromSearch(search = location.search, storage = localStorage) {
  const params = new URLSearchParams(search);
  const settings = { ...AVATAR_LAB_DEFAULTS, ...loadSavedSettings(storage) };
  for (const [key, limits] of Object.entries(AVATAR_LAB_LIMITS)) {
    const raw = params.get(LEGACY_QUERY_KEYS[key] || key);
    if (raw !== null) settings[key] = boundedNumber(raw, settings[key], limits);
  }
  if (params.has('voice')) settings.voice = params.get('voice') || settings.voice;
  if (params.has('lipSyncEngine')) settings.lipSyncEngine = params.get('lipSyncEngine') === 'wav2vec2' ? 'wav2vec2' : 'headaudio';
  if (params.has('model')) {
    const model = String(params.get('model') || '').split(/[\\/]/).pop();
    if (/^[\w.-]+\.bin$/.test(model)) settings.headaudioModel = model;
  }
  if (params.has('avatar')) settings.avatarUrl = params.get('avatar');
  if (params.has('hardSilenceClose')) settings.hardSilenceClose = params.get('hardSilenceClose') !== '0';
  if (params.has('freezeIdle')) settings.freezeIdle = params.get('freezeIdle') !== '0';
  return settings;
}

export function readSettingsControls(root = document) {
  const settings = { ...AVATAR_LAB_DEFAULTS };
  for (const key of Object.keys(AVATAR_LAB_LIMITS)) {
    const input = root.querySelector(`[data-avatar-setting="${key}"]`);
    if (input) settings[key] = boundedNumber(input.value, settings[key], AVATAR_LAB_LIMITS[key]);
  }
  settings.voice = root.querySelector('[data-avatar-setting="voice"]')?.value || settings.voice;
  settings.lipSyncEngine = root.querySelector('[data-avatar-setting="lipSyncEngine"]')?.value === 'wav2vec2' ? 'wav2vec2' : 'headaudio';
  const modelSelect = root.querySelector('[data-avatar-setting="headaudioModel"]');
  if (modelSelect) {
    const model = String(modelSelect.value || '').split(/[\\/]/).pop();
    settings.headaudioModel = /^[\w.-]+\.bin$/.test(model) ? model : settings.headaudioModel;
  }
  const avatarInput = root.querySelector('[data-avatar-setting="avatarUrl"]');
  if (avatarInput) settings.avatarUrl = avatarInput.value.trim();
  settings.hardSilenceClose = Boolean(root.querySelector('[data-avatar-setting="hardSilenceClose"]')?.checked);
  settings.freezeIdle = Boolean(root.querySelector('[data-avatar-setting="freezeIdle"]')?.checked);
  return settings;
}

export function populateSettingsControls(settings, root = document) {
  for (const [key, value] of Object.entries(settings)) {
    const input = root.querySelector(`[data-avatar-setting="${key}"]`);
    if (!input) continue;
    if (input.type === 'checkbox') input.checked = Boolean(value); else input.value = String(value);
    const output = root.querySelector(`[data-avatar-setting-value="${key}"]`);
    if (output) {
      output.value = String(value);
      output.textContent = String(value);
    }
  }
}

// Overlay explicit query-string parameters on top of existing (e.g. panel-read) settings.
export function overrideFromSearch(settings, search = location.search) {
  const merged = { ...settings };
  const params = new URLSearchParams(search);
  for (const [key, limits] of Object.entries(AVATAR_LAB_LIMITS)) {
    const raw = params.get(LEGACY_QUERY_KEYS[key] || key);
    if (raw !== null) merged[key] = boundedNumber(raw, merged[key], limits);
  }
  if (params.has('voice')) merged.voice = params.get('voice') || merged.voice;
  if (params.has('lipSyncEngine')) merged.lipSyncEngine = params.get('lipSyncEngine') === 'wav2vec2' ? 'wav2vec2' : 'headaudio';
  if (params.has('model')) {
    const model = String(params.get('model') || '').split(/[\\/]/).pop();
    if (/^[\w.-]+\.bin$/.test(model)) merged.headaudioModel = model;
  }
  if (params.has('hardSilenceClose')) merged.hardSilenceClose = params.get('hardSilenceClose') !== '0';
  return merged;
}

export function currentAvatarSettings(_search = location.search, root = document) {
  // Query parameters seed the controls once in settingsFromSearch(). After that the
  // visible panel is authoritative so users can switch engines without the original
  // ?lipSyncEngine=... URL silently overriding their selection.
  return readSettingsControls(root);
}
