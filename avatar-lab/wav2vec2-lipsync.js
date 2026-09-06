const MODEL_ID = 'onnx-community/wav2vec2-ljspeech-gruut-ONNX';
const WEBGPU_DTYPE = 'q4f16';
const COMPAT_DTYPE = 'q4';
const MODEL_URL = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0/+esm';
const INPUT_RATE = 24000;
const MODEL_RATE = 16000;
const WINDOW_MS = 1200;
const STRIDE_MS = 200;
const RIGHT_CONTEXT_MS = 100;
const MIN_AUDIO_MS = 400;
const BLANK_ID = 42;
const UNK_ID = 41;
const ID_TO_TOKEN = ['|','aɪ','aʊ','b','d','d͡ʒ','eɪ','f','h','i','j','k','l','m','n','oʊ','p','s','t','t͡ʃ','u','v','w','z','æ','ð','ŋ','ɑ','ɔ','ɔɪ','ə','ɚ','ɛ','ɡ','ɪ','ɹ','ʃ','ʊ','ʌ','ʒ','θ','[UNK]','[PAD]'];
const VISEME_NAMES = ['aa','E','I','O','U','PP','SS','TH','DD','FF','kk','nn','RR','CH','sil'];
const PHONEME_TO_VISEME = {
  'aɪ':'aa','aʊ':'aa',b:'PP',d:'DD','d͡ʒ':'CH','eɪ':'E',f:'FF',h:'kk',i:'I',j:'I',k:'kk',l:'DD',m:'PP',n:'nn','oʊ':'O',p:'PP',s:'SS',t:'DD','t͡ʃ':'CH',u:'U',v:'FF',w:'U',z:'SS',æ:'aa',ð:'TH',ŋ:'nn',ɑ:'aa',ɔ:'O','ɔɪ':'O',ə:'aa',ɚ:'RR',ɛ:'E',ɡ:'kk',ɪ:'I',ɹ:'RR',ʃ:'CH',ʊ:'U',ʌ:'aa',ʒ:'CH',θ:'TH'
};

function normalize(input) {
  let mean = 0;
  for (const value of input) mean += value;
  mean /= input.length;
  let variance = 0;
  for (const value of input) variance += (value - mean) ** 2;
  const scale = 1 / Math.sqrt(variance / input.length + 1e-7);
  const output = new Float32Array(input.length);
  for (let i = 0; i < input.length; i++) output[i] = (input[i] - mean) * scale;
  return output;
}

export class Wav2Vec2LipSync {
  constructor(controller, { log = () => {}, telemetry = null, playbackDelayMs = 90 } = {}) {
    this.controller = controller;
    this.log = log;
    this.telemetry = telemetry;
    this.playbackDelayMs = playbackDelayMs;
    this.pcm = [];
    this.totalSamples = 0;
    this.nextInferenceSample = INPUT_RATE * MIN_AUDIO_MS / 1000;
    this.lastCommittedSourceMs = -Infinity;
    this.lastTokenSourceMs = -Infinity;
    this.processing = false;
    this.pending = false;
    this.ended = false;
    this.speaking = false;
    this.inferenceCount = 0;
    this.streamStartWallMs = null;
    this.playbackStartWallMs = null;
    this.pendingActions = [];
    this.actionTimers = new Set();
  }
  async init() {
    const started = performance.now();
    this.worker = new Worker('./wav2vec2-worker.js', { type: 'module' });
    this.workerRequests = new Map();
    this.nextRequestId = 1;
    await new Promise((resolve, reject) => {
      const onMessage = event => {
        const data = event.data;
        if (data.type === 'ready') {
          this.device = data.device;
          this.dtype = data.dtype;
          resolve();
        } else if (data.type === 'progress') {
          const progress = data.progress || {};
          if (progress.status === 'progress' && Number.isFinite(progress.progress)) {
            this.telemetry?.set('Wav2Vec2 download', `${progress.progress.toFixed(0)}%`);
          } else if (progress.file) {
            this.telemetry?.set('Wav2Vec2 loading', progress.file);
          }
        } else if (data.type === 'fallback') {
          this.log('Wav2Vec2 WebGPU compatibility path failed; worker is retrying with WASM.', { message: data.message });
        } else if (data.type === 'error' && data.id == null) {
          reject(new Error(data.message));
        }
      };
      this.worker.addEventListener('message', onMessage);
      this.worker.addEventListener('error', event => reject(event.error || new Error(event.message)), { once: true });
      this.worker.postMessage({ type: 'init' });
    });
    this.worker.addEventListener('message', event => {
      const data = event.data;
      if ((data.type !== 'result' && data.type !== 'error') || data.id == null) return;
      const request = this.workerRequests.get(data.id);
      if (!request) return;
      this.workerRequests.delete(data.id);
      if (data.type === 'error') request.reject(new Error(data.message));
      else request.resolve({ logits: { dims: data.dims, data: new Float32Array(data.values) }, inferenceMs: data.inferenceMs });
    });
    this.telemetry?.set('Lip-sync engine', `Wav2Vec2 ${this.dtype}`);
    this.telemetry?.set('Wav2Vec2 backend', `${this.device} / ${this.dtype} / worker`);
    this.telemetry?.set('Wav2Vec2 load + warmup', `${Math.round(performance.now() - started)} ms`);
    this.log('Wav2Vec2 loaded and warmed in a dedicated worker.', { device: this.device, dtype: this.dtype });
  }
  _runInference(input) {
    return new Promise((resolve, reject) => {
      const id = this.nextRequestId++;
      this.workerRequests.set(id, { resolve, reject });
      this.worker.postMessage({ type: 'infer', id, input: input.buffer }, [input.buffer]);
    });
  }
  setPlaybackStarted(wallTimeMs = performance.now()) {
    if (this.playbackStartWallMs != null) return;
    this.playbackStartWallMs = wallTimeMs;
    const pending = this.pendingActions;
    this.pendingActions = [];
    for (const action of pending) this._scheduleAction(action);
  }
  _scheduleAction(action) {
    if (this.playbackStartWallMs == null) {
      this.pendingActions.push(action);
      return;
    }
    const targetWallMs = this.playbackStartWallMs + action.sourceTimeMs + this.playbackDelayMs;
    const run = () => {
      if (action.type === 'viseme') {
        if (!this.controllerSpeaking) {
          this.controllerSpeaking = true;
          this.controller.onStarted();
        }
        this.controller.onVad({ active: 1, db: -10, t: action.sourceTimeMs / 1000 });
        this.controller.onViseme({ viseme: action.viseme, t: action.sourceTimeMs / 1000 });
      } else {
        this.controllerSpeaking = false;
        this.controller.onVad({ active: 0, db: -100, t: action.sourceTimeMs / 1000 });
        this.controller.onEnded();
      }
    };
    const delayMs = targetWallMs - performance.now();
    if (delayMs <= 0) {
      run();
      return;
    }
    const timer = setTimeout(() => {
      this.actionTimers.delete(timer);
      run();
    }, delayMs);
    this.actionTimers.add(timer);
  }
  reset() {
    for (const timer of this.actionTimers || []) clearTimeout(timer);
    this.actionTimers?.clear?.();
    this.pcm = [];
    this.totalSamples = 0;
    this.nextInferenceSample = INPUT_RATE * MIN_AUDIO_MS / 1000;
    this.lastCommittedSourceMs = -Infinity;
    this.lastTokenSourceMs = -Infinity;
    this.pending = false;
    this.ended = false;
    this.speaking = false;
    this.controllerSpeaking = false;
    this.streamStartWallMs = null;
    this.playbackStartWallMs = null;
    this.pendingActions = [];
    this.actionTimers = new Set();
    this.controller.onEnded();
  }
  push(bytes) {
    if (!bytes?.byteLength) return;
    if (this.ended && this.totalSamples > 0) this.reset();
    this.ended = false;
    if (this.streamStartWallMs == null) this.streamStartWallMs = performance.now();
    const view = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes.buffer || bytes);
    for (let i = 0; i + 1 < view.byteLength; i += 2) {
      let value = view[i] | (view[i + 1] << 8);
      if (value & 0x8000) value -= 0x10000;
      this.pcm.push(value / 32768);
    }
    this.totalSamples += view.byteLength / 2;
    if (this.totalSamples >= this.nextInferenceSample) {
      this.pending = true;
      this._drain();
    }
  }
  async flush() {
    this.ended = true;
    this.pending = true;
    while (this.processing) await new Promise(resolve => setTimeout(resolve, 10));
    await this._drain(true);
    if (this.speaking) {
      this.speaking = false;
      this._scheduleAction({ type: 'end', sourceTimeMs: this.totalSamples / INPUT_RATE * 1000 });
    }
  }
  async _drain(force = false) {
    if (this.processing || !this.worker) return;
    this.processing = true;
    try {
      while (this.pending && (force || this.totalSamples >= this.nextInferenceSample)) {
        this.pending = false;
        const strideSamples = INPUT_RATE * STRIDE_MS / 1000;
        let endSample = this.totalSamples;
        if (!force) {
          // Latest-only scheduling: if WASM is slower than the 200 ms cadence,
          // discard obsolete intermediate windows instead of building an inference
          // backlog that monopolizes CPU and makes animation/audio stutter.
          const skipped = Math.max(0, Math.floor((this.totalSamples - this.nextInferenceSample) / strideSamples));
          endSample = this.nextInferenceSample + skipped * strideSamples;
          this.nextInferenceSample = endSample + strideSamples;
          if (skipped) this.telemetry?.set('Wav2Vec2 skipped stale windows', skipped);
        }
        await this._infer(endSample, force);
        force = false;
        if (this.totalSamples >= this.nextInferenceSample) this.pending = true;
      }
    } catch (error) {
      this.log('Wav2Vec2 rolling inference failed.', { message: error.message });
      this.telemetry?.set('Wav2Vec2 status', `ERROR: ${error.message}`);
    } finally {
      this.processing = false;
    }
  }
  _windowInput(endSample) {
    const startSample = Math.max(0, endSample - INPUT_RATE * WINDOW_MS / 1000);
    const source = this.pcm.slice(startSample, endSample);
    const outputLength = Math.floor(source.length * MODEL_RATE / INPUT_RATE);
    const output = new Float32Array(outputLength);
    for (let i = 0; i < outputLength; i++) {
      const x = i * INPUT_RATE / MODEL_RATE, j = Math.floor(x), f = x - j;
      output[i] = source[j] * (1 - f) + source[Math.min(j + 1, source.length - 1)] * f;
    }
    return { input: normalize(output), startMs: startSample / INPUT_RATE * 1000, endMs: endSample / INPUT_RATE * 1000 };
  }
  async _infer(endSample, force) {
    const { input, startMs, endMs } = this._windowInput(endSample);
    if (!input.length) return;
    const output = await this._runInference(input);
    const inferenceMs = output.inferenceMs;
    const tokens = this._decode(output.logits, startMs, endMs - startMs);
    const commitBefore = force ? endMs : endMs - RIGHT_CONTEXT_MS;
    let emitted = 0;
    for (const token of tokens) {
      if (token.sourceTimeMs > commitBefore || token.sourceTimeMs <= this.lastCommittedSourceMs + 30) continue;
      this.lastCommittedSourceMs = token.sourceTimeMs;
      this.lastTokenSourceMs = token.sourceTimeMs;
      const viseme = VISEME_NAMES.indexOf(token.viseme);
      if (viseme < 0) continue;
      if (!this.speaking) this.speaking = true;
      this._scheduleAction({ type: 'viseme', sourceTimeMs: token.sourceTimeMs, viseme });
      const decisionWallMs = performance.now();
      const playbackOrigin = this.playbackStartWallMs ?? this.streamStartWallMs;
      const estimatedAudibleWallMs = playbackOrigin + token.sourceTimeMs + this.playbackDelayMs;
      const decisionLeadMs = estimatedAudibleWallMs - decisionWallMs;
      this.telemetry?.set('Wav2Vec2 decision vs audible', `${decisionLeadMs >= 0 ? '+' : ''}${decisionLeadMs.toFixed(0)} ms`);
      emitted++;
    }
    if (this.speaking && endMs - this.lastTokenSourceMs >= 400) {
      this.speaking = false;
      this._scheduleAction({ type: 'end', sourceTimeMs: endMs });
    }
    this.inferenceCount++;
    this.telemetry?.set('Wav2Vec2 inference', `${inferenceMs.toFixed(0)} ms`);
    this.telemetry?.set('Wav2Vec2 source lead', `${Math.max(0, endMs - this.lastCommittedSourceMs).toFixed(0)} ms`);
    this.telemetry?.set('Wav2Vec2 tokens', `${emitted} new / ${tokens.length} window`);
  }
  _decode(logits, startMs, durationMs) {
    const frames = logits.dims[1], vocab = logits.dims[2], tokens = [];
    let previousId = BLANK_ID;
    for (let frame = 0; frame < frames; frame++) {
      let bestId = 0, best = -Infinity;
      const base = frame * vocab;
      for (let id = 0; id < vocab; id++) {
        const value = logits.data[base + id];
        if (value > best) { best = value; bestId = id; }
      }
      if (bestId !== BLANK_ID && bestId !== UNK_ID && bestId !== 0 && bestId !== previousId) {
        const phoneme = ID_TO_TOKEN[bestId], viseme = PHONEME_TO_VISEME[phoneme];
        if (viseme) tokens.push({ phoneme, viseme, sourceTimeMs: startMs + (frame + 0.5) / frames * durationMs });
      }
      previousId = bestId;
    }
    return tokens;
  }
  async destroy() {
    this.reset();
    this.worker?.postMessage({ type: 'destroy' });
    this.worker?.terminate();
    this.worker = null;
    for (const request of this.workerRequests?.values?.() || []) request.reject(new Error('Wav2Vec2 worker destroyed'));
    this.workerRequests?.clear?.();
  }
}
