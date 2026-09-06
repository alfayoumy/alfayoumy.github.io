function append(left, right) {
  if (!left.length) return right.slice();
  const joined = new Float32Array(left.length + right.length);
  joined.set(left);
  joined.set(right, left.length);
  return joined;
}

/** Aligns render/capture AudioWorklet quanta into paired 10 ms AEC3 frames. */
export class Aec3FramePairer {
  constructor({sampleRate, frameMs = 10}) {
    this.frameSamples = Math.round(sampleRate * frameMs / 1000);
    this.capturePending = new Float32Array(0);
    this.renderPending = new Float32Array(0);
  }

  push(capture, render) {
    if (!(capture instanceof Float32Array) || !(render instanceof Float32Array)) {
      throw new TypeError('AEC3 capture and render inputs must be Float32Array');
    }
    if (capture.length !== render.length) throw new Error('AEC3 capture/render quantum mismatch');
    this.capturePending = append(this.capturePending, capture);
    this.renderPending = append(this.renderPending, render);
    const frames = [];
    while (this.capturePending.length >= this.frameSamples) {
      frames.push({
        capture: this.capturePending.slice(0, this.frameSamples),
        render: this.renderPending.slice(0, this.frameSamples),
      });
      this.capturePending = this.capturePending.slice(this.frameSamples);
      this.renderPending = this.renderPending.slice(this.frameSamples);
    }
    return frames;
  }

  clear() {
    this.capturePending = new Float32Array(0);
    this.renderPending = new Float32Array(0);
  }
}
