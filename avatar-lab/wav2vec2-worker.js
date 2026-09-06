const MODEL_ID = 'onnx-community/wav2vec2-ljspeech-gruut-ONNX';
const MODEL_URL = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0/+esm';
const WEBGPU_DTYPE = 'q4f16';
let model = null;
let Tensor = null;
let device = 'webgpu';
let dtype = WEBGPU_DTYPE;

async function loadAndWarm() {
  if (!navigator.gpu) throw new Error('WEBGPU_UNAVAILABLE: navigator.gpu is not exposed in this browser worker');
  let adapter = null;
  try { adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' }); } catch {}
  if (!adapter) throw new Error('WEBGPU_UNAVAILABLE: no WebGPU adapter was returned');
  if (!adapter.features?.has('shader-f16')) throw new Error('WEBGPU_F16_UNAVAILABLE: adapter does not expose shader-f16 required by q4f16');
  const transformers = await import(MODEL_URL);
  Tensor = transformers.Tensor;
  transformers.env.allowLocalModels = false;
  transformers.env.useBrowserCache = true;
  const options = {
    dtype,
    device,
    progress_callback: progress => postMessage({ type: 'progress', progress })
  };
  model = await transformers.AutoModelForCTC.from_pretrained(MODEL_ID, options);
  const warmup = new Float32Array(16000 * 1.2);
  await model({ input_values: new Tensor('float32', warmup, [1, warmup.length]) });
  postMessage({ type: 'ready', device, dtype });
}

self.onmessage = async event => {
  const data = event.data;
  try {
    if (data.type === 'init') {
      await loadAndWarm();
    } else if (data.type === 'infer') {
      const input = new Float32Array(data.input);
      const started = performance.now();
      const output = await model({ input_values: new Tensor('float32', input, [1, input.length]) });
      const logits = output.logits;
      const values = logits.data instanceof Float32Array ? logits.data : Float32Array.from(logits.data);
      postMessage({ type: 'result', id: data.id, dims: logits.dims, values: values.buffer, inferenceMs: performance.now() - started }, [values.buffer]);
    } else if (data.type === 'destroy') {
      await model?.dispose?.();
      model = null;
      close();
    }
  } catch (error) {
    postMessage({ type: 'error', id: data.id, message: error?.message || String(error) });
  }
};
