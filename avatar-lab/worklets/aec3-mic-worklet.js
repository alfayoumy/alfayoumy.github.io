import {Aec3FramePairer} from './aec3-frame-pipeline.js';

class Aec3MicWorklet extends AudioWorkletProcessor {
  constructor() {
    super();
    this.pairer = new Aec3FramePairer({sampleRate});
    this.stopped = false;
    this.port.onmessage = event => {
      if (event.data?.type === 'stop') this.stopped = true;
    };
  }

  process(inputs, outputs) {
    outputs[0]?.[0]?.fill(0);
    if (this.stopped) return false;
    const capture = inputs[0]?.[0];
    if (!capture) return true;
    const render = inputs[1]?.[0] || new Float32Array(capture.length);
    for (const frame of this.pairer.push(capture,render)) {
      this.port.postMessage(
        {type:'frame',capture:frame.capture,render:frame.render,sampleRate},
        [frame.capture.buffer,frame.render.buffer]
      );
    }
    return true;
  }
}

registerProcessor('aec3-mic-worklet', Aec3MicWorklet);
