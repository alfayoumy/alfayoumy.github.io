class MicPcmProcessor extends AudioWorkletProcessor {
  process(inputs){const channel=inputs[0]?.[0];if(channel?.length){const copy=new Float32Array(channel.length);copy.set(channel);this.port.postMessage(copy,[copy.buffer]);}return true;}
}
registerProcessor('mic-pcm-worklet',MicPcmProcessor);
