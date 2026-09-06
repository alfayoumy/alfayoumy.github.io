import WebRtcAec3 from '../vendor/aec3/webrtcaec3-0.3.0.mjs';

let aec = null;
let failed = false;

try {
  const AEC3Module = await WebRtcAec3();
  aec = new AEC3Module.AEC3(48000, 1, 1);
  aec.setAudioBufferDelay(0);
  self.postMessage({type:'ready'});
} catch (error) {
  failed = true;
  self.postMessage({type:'error',message:error?.message||'AEC3 initialization failed'});
}

self.onmessage = event => {
  if (event.data?.type === 'stop') {
    aec?.free();
    aec = null;
    self.close();
    return;
  }
  if (failed || !aec || event.data?.type !== 'frame') return;
  try {
    const {capture,render,sampleRate} = event.data;
    const options = {sampleRateIn:sampleRate,sampleRateOut:sampleRate};
    aec.analyze([render],options);
    const clean = new Float32Array(aec.processSize([capture],options));
    aec.process([clean],[capture],options);
    if(clean.length)self.postMessage({type:'clean',samples:clean,sampleRate},[clean.buffer]);
  } catch (error) {
    failed = true;
    self.postMessage({type:'error',message:error?.message||'AEC3 processing failed'});
  }
};
