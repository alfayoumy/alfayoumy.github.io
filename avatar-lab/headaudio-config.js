export const DEFAULT_HEADAUDIO_MODEL = 'model-en-gemini-male-orus-24k-hybrid.bin';
export const DEFAULT_HEADAUDIO_METADATA = 'model-en-gemini-male-orus-24k-hybrid.json';
export const DEFAULT_HEADAUDIO_RUNTIME = 'headaudio_runtime_tuned.json';

const MODEL_NAME = /^[\w.-]+\.bin$/;
const FINITE_RUNTIME_KEYS = Object.freeze([
  'sampleRate', 'speakerMeanHz', 'voteWindow', 'silSensitivity', 'vadMode',
  'vadGateActiveDb', 'vadGateActiveMs', 'vadGateInactiveDb', 'vadGateInactiveMs'
]);

function modelsUrl(filename, baseUrl = './models') {
  return `${String(baseUrl).replace(/\/$/, '')}/${filename}`;
}

async function fetchJson(url, label, fetchImpl) {
  const response = await fetchImpl(url);
  if (!response.ok) throw new Error(`Failed to fetch ${label}: ${response.status} ${response.statusText}`);
  return response.json();
}

export async function loadHeadAudioConfig(modelFile = DEFAULT_HEADAUDIO_MODEL, fetchImpl = fetch, baseUrl = './models') {
  const safeModel = String(modelFile || '').split(/[\\/]/).pop();
  if (!MODEL_NAME.test(safeModel)) throw new Error(`Invalid HeadAudio model filename: ${modelFile}`);
  const candidateMetadataFile = safeModel.replace(/\.bin$/i, '.json');
  const metadataRequest = (async () => {
    const response = await fetchImpl(modelsUrl(candidateMetadataFile, baseUrl));
    if (!response.ok) {
      if (safeModel === DEFAULT_HEADAUDIO_MODEL) throw new Error(`Failed to fetch HeadAudio model metadata: ${response.status} ${response.statusText}`);
      return null; // Historical lab models predate paired metadata.
    }
    return response.json();
  })();
  const [metadata, rawRuntime] = await Promise.all([
    metadataRequest,
    fetchJson(modelsUrl(DEFAULT_HEADAUDIO_RUNTIME, baseUrl), 'HeadAudio runtime tuning', fetchImpl)
  ]);
  if (metadata && metadata.model !== safeModel) {
    throw new Error(`HeadAudio metadata mismatch: expected "${safeModel}", got "${metadata.model || 'missing'}"`);
  }
  const metadataFile = metadata ? candidateMetadataFile : null;
  const runtime = {};
  for (const key of FINITE_RUNTIME_KEYS) {
    const value = Number(rawRuntime?.[key]);
    if (!Number.isFinite(value)) throw new Error(`Invalid HeadAudio runtime setting "${key}"`);
    runtime[key] = value;
  }
  if (runtime.sampleRate !== 24000) throw new Error(`HeadAudio runtime sampleRate must be 24000, got ${runtime.sampleRate}`);
  return {
    modelFile: safeModel,
    modelUrl: modelsUrl(safeModel, baseUrl),
    metadataFile,
    metadataUrl: metadataFile ? modelsUrl(metadataFile, baseUrl) : null,
    metadata,
    runtimeFile: DEFAULT_HEADAUDIO_RUNTIME,
    runtimeUrl: modelsUrl(DEFAULT_HEADAUDIO_RUNTIME, baseUrl),
    runtime
  };
}

// Runtime tuning is immutable from browser settings. Return a defensive copy so
// callers cannot mutate the validated profile object accidentally.
export function resolveHeadAudioRuntime(runtime) {
  return { ...runtime };
}
