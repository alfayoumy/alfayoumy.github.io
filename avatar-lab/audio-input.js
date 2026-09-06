import {float32ToPcm16,resampleLinear,PcmChunker,int16ToLittleEndianBytes,bytesToBase64} from './audio-utils.js';

export class MicrophoneInput {
  constructor({onChunk,onEnded=()=>{},onError=()=>{},log=()=>{},chunkMs=100,renderReference=null}={}){
    this.onChunk=onChunk;
    this.onEnded=onEnded;
    this.onError=onError;
    this.log=log;
    this.chunkMs=chunkMs;
    this.renderReference=renderReference;
    this.ctx=null;
    this.ownsContext=false;
    this.stream=null;
    this.source=null;
    this.worklet=null;
    this.silent=null;
    this.chunker=null;
    this.renderNode=null;
    this.worker=null;
    this.aecActive=false;
    this.boundTrackEnded=this.#handleTrackEnded.bind(this);
  }

  #setAudioSessionType(type){
    try{
      if(navigator.audioSession)navigator.audioSession.type=type;
    }catch{}
  }

  #handleTrackEnded(){
    this.log('Microphone track ended or device disconnected.');
    this.onEnded?.();
    this.stop().catch(()=>{});
  }

  async start(){
    if(this.stream)return;
    const reference=this.renderReference;
    const canUseAec3=Boolean(reference?.context&&reference?.node&&reference.context.state!=='closed');
    // WebKit can otherwise leave full-duplex audio on the phone-call receiver.
    // Reset first, acquire the mic, then explicitly declare full-duplex intent.
    this.#setAudioSessionType('auto');
    try{
      this.stream=await navigator.mediaDevices.getUserMedia({
        audio:{
          channelCount:1,
          echoCancellation:canUseAec3?false:true,
          noiseSuppression:canUseAec3?false:true,
          autoGainControl:canUseAec3?false:true
        }
      });
    }catch(err){
      if(err.name==='NotAllowedError'||err.name==='PermissionDeniedError'){
        const e=new Error('Microphone permission was denied. Please allow microphone access in your browser settings.');
        e.code='MIC_PERMISSION_DENIED';
        throw e;
      }
      if(err.name==='NotFoundError'||err.name==='DevicesNotFoundError'){
        const e=new Error('No microphone device was found on this system.');
        e.code='MIC_NOT_FOUND';
        throw e;
      }
      if(err.name==='NotReadableError'||err.name==='TrackStartError'){
        const e=new Error('Microphone is in use by another application or not accessible.');
        e.code='MIC_NOT_READABLE';
        throw e;
      }
      throw err;
    }

    this.#setAudioSessionType('play-and-record');
    for(const track of this.stream.getTracks())track.addEventListener('ended',this.boundTrackEnded);

    try{
      this.ctx=canUseAec3?reference.context:new AudioContext();
      this.ownsContext=!canUseAec3;
      if(this.ctx.state==='suspended')await this.ctx.resume();
      this.chunker=new PcmChunker({
        sampleRate:16000,
        chunkMs:this.chunkMs,
        onChunk:pcm=>this.onChunk?.({pcm,base64:bytesToBase64(int16ToLittleEndianBytes(pcm)),mimeType:'audio/pcm;rate=16000'})
      });
      this.source=this.ctx.createMediaStreamSource(this.stream);

      if(canUseAec3){
        try{
          await this.ctx.audioWorklet.addModule('/avatar-lab/worklets/aec3-mic-worklet.js');
          this.worker=new Worker('/avatar-lab/worklets/aec3-worker.js',{type:'module'});
          await new Promise((resolve,reject)=>{
            const timer=setTimeout(()=>reject(new Error('AEC3 initialization timed out')),10000);
            this.worker.onmessage=event=>{
              if(event.data?.type==='ready'){clearTimeout(timer);resolve();}
              else if(event.data?.type==='error'){clearTimeout(timer);reject(new Error(event.data.message||'AEC3 initialization failed'));}
            };
            this.worker.onerror=event=>{clearTimeout(timer);reject(new Error(event.message||'AEC3 worker failed'));};
          });
          this.worklet=new AudioWorkletNode(this.ctx,'aec3-mic-worklet',{
            numberOfInputs:2,
            numberOfOutputs:1,
            outputChannelCount:[1],
            channelCount:1,
            channelCountMode:'explicit'
          });
          this.renderNode=reference.node;
          this.aecActive=true;
        }catch(error){
          try{this.worker?.terminate();}catch{}
          this.worker=null;
          const unavailable=new Error(`AEC3 initialization failed: ${error?.message||String(error)}`);
          unavailable.code='AEC_UNAVAILABLE';
          throw unavailable;
        }
      }

      if(!this.worklet){
        await this.ctx.audioWorklet.addModule('/avatar-lab/worklets/mic-pcm-worklet.js');
        this.worklet=new AudioWorkletNode(this.ctx,'mic-pcm-worklet');
      }

      this.silent=this.ctx.createGain();
      this.silent.gain.value=0;
      const consumeSamples=(samples,sourceRate=this.ctx.sampleRate)=>{
        if(!this.ctx||!(samples instanceof Float32Array))return;
        const resampled=resampleLinear(samples,sourceRate,16000);
        this.chunker?.push(float32ToPcm16(resampled));
      };
      const reportAecError=message=>{
        const error=new Error(message||'AEC3 processing failed');
        error.code='AEC_PROCESSING_FAILED';
        this.onError?.(error);
      };
      if(this.aecActive){
        this.worker.onmessage=event=>{
          if(event.data?.type==='clean')consumeSamples(event.data.samples,event.data.sampleRate);
          else if(event.data?.type==='error')reportAecError(event.data.message);
        };
        this.worker.onerror=event=>reportAecError(event.message);
        this.worklet.port.onmessage=event=>{
          if(event.data?.type!=='frame'||!this.worker)return;
          const {capture,render}=event.data;
          this.worker.postMessage(event.data,[capture.buffer,render.buffer]);
        };
      }else{
        this.worklet.port.onmessage=event=>consumeSamples(event.data);
      }
      this.source.connect(this.worklet,0,0);
      if(this.aecActive)this.renderNode.connect(this.worklet,0,1);
      this.worklet.connect(this.silent).connect(this.ctx.destination);
      if(this.ctx.state==='suspended')await this.ctx.resume();
      const settings=this.stream.getAudioTracks?.()[0]?.getSettings?.()||{};
      this.log('Microphone started',{
        captureRate:this.ctx.sampleRate,
        networkRate:16000,
        chunkMs:this.chunkMs,
        echoCancellation:this.aecActive?'webrtc-aec3-wasm':(settings.echoCancellation?'browser':'unavailable')
      });
    }catch(err){
      await this.stop();
      throw err;
    }
  }

  async stop(){
    this.chunker?.clear();
    try{this.worklet?.port.postMessage({type:'stop'});}catch{}
    try{this.worker?.postMessage({type:'stop'});}catch{}
    try{this.worker?.terminate();}catch{}
    if(this.stream){
      for(const track of this.stream.getTracks()){
        try{track.removeEventListener('ended',this.boundTrackEnded);}catch{}
        try{track.stop();}catch{}
      }
      // Force WebKit to leave its call-style route after capture ends.
      this.#setAudioSessionType('playback');
      this.#setAudioSessionType('auto');
    }
    try{this.renderNode?.disconnect(this.worklet);}catch{}
    for(const node of [this.source,this.worklet,this.silent]){
      try{node?.disconnect();}catch{}
    }
    if(this.ownsContext&&this.ctx&&this.ctx.state!=='closed'){
      try{await this.ctx.close();}catch{}
    }
    this.ctx=this.stream=this.source=this.worklet=this.silent=this.chunker=this.renderNode=this.worker=null;
    this.ownsContext=false;
    this.aecActive=false;
    this.log('Microphone resources released.');
  }
}
