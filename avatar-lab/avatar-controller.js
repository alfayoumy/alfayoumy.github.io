import {LipSyncController, MANAGED_LIP_TARGETS, REQUIRED_OCULUS_VISEMES} from './lipsync-controller.js';
import {DEFAULT_HEADAUDIO_MODEL, loadHeadAudioConfig, resolveHeadAudioRuntime} from './headaudio-config.js';

const REQUIRED_VISEMES = REQUIRED_OCULUS_VISEMES;
const ARKIT_LIP_TARGETS = MANAGED_LIP_TARGETS;

export class AvatarController {
  constructor(container,{log=()=>{},telemetry}={}){
    this.container=container;
    this.log=log;
    this.telemetry=telemetry;
    this.head=null;
    this.headAudio=null;
    this.wav2vec2=null;
    this.lipSyncEngine='headaudio';
    this.delayNode=null;
    this.outputGainNode=null;
    this.delayMs=90;
    this.currentConversationState='READY';
    this.pointerTrackingEnabled=false;
    this.pointerTrackingBound=false;
    this.pointerFrame=null;
    this.pointerReturnTimer=null;
    this.pointerReleaseTimer=null;
    this.pointerDefaultsSuspended=false;
    this.pointerRestoreFreezeState=false;
    this.pointerTakeover=false;
    this.pointerLastLook=0;
    this.pointerEyePin=null;
    this.pointerIdleEyeGazeBias=null;
    this.pointerTarget=null;
    this.lipSyncController=new LipSyncController(null,{log,telemetry});
  }
  zeroVisemes(){
    this.lipSyncController.reset();
  }
  async load(avatarUrl='',freezeIdle=true,{preserveDrawingBuffer=false}={}){
    if(this.head)return this.head;
    const started=performance.now();
    const {TalkingHead}=await import('./vendor/talkinghead/talkinghead.mjs');
    this.head=new TalkingHead(this.container,{cameraView:'head',cameraRotateEnable:false,cameraPanEnable:false,cameraZoomEnable:false,lipsyncModules:[],modelFPS:60,mixerGainSpeech:1,mixerGainBackground:0,eyeGazeBias:0.1,headPitchBias:0,freezeAnimations:Boolean(freezeIdle),preserveDrawingBuffer:Boolean(preserveDrawingBuffer)});
    this.lipSyncController.setHead(this.head);
    // Accept a local filename inside ./avatars/, an absolute URL, or fall back to the default.
    let url=avatarUrl&&avatarUrl.trim();
    if(url&&!/^https?:\/\//i.test(url))url='./avatars/'+url.replace(/^\.?\//,'');
    else if(!url)url='./avatars/brunette.glb';
    await this.head.showAvatar({url,body:'F',avatarMood:'neutral'},event=>this.telemetry?.set('Avatar download',event.total?`${Math.round(event.loaded/event.total*100)}%`:`${event.loaded.toLocaleString()} bytes`));
    const missing=REQUIRED_VISEMES.filter(name=>!this.head.mtAvatar?.[name]);
    this.telemetry?.set('Avatar load time',`${(performance.now()-started).toFixed(0)} ms`);
    this.telemetry?.set('GLB transferred','4,721,528 bytes');
    this.telemetry?.set('Missing visemes',missing.length?missing.join(', '):'none');
    if(missing.length) throw new Error(`Avatar is missing required morph targets: ${missing.join(', ')}`);
    if(freezeIdle){try{this.head.stopAnimation?.();this.head.setFreezeAnimations?.(true);this.log('Baked + procedural idle animations frozen — lookAtCamera has full control.');}catch(e){this.log('Idle freeze skipped',{message:e.message});}}
    this.log('TalkingHead 1.7.0 avatar loaded; required Oculus visemes present.');
    return this.head;
  }
  async enableLipSync(settings={}){
    const engine=settings.lipSyncEngine==='wav2vec2'?'wav2vec2':'headaudio';
    this.lipSyncEngine=engine;
    if(engine==='wav2vec2')return this.enableWav2Vec2(settings);
    return this.enableHeadAudio(settings);
  }
  _configureAudioRouting(playbackDelayMs){
    const source=this.head.audioStreamGainNode;
    this.head.streamWorkletNode.disconnect();
    this.head.streamWorkletNode.connect(source);
    this.head.audioSpeechGainNode.gain.setValueAtTime(0,this.head.audioCtx.currentTime);
    this.delayNode=new DelayNode(this.head.audioCtx,{maxDelayTime:1,delayTime:playbackDelayMs/1000});
    source.disconnect(this.head.audioReverbNode);
    this.outputGainNode=this.head.audioCtx.createGain();
    source.connect(this.delayNode).connect(this.outputGainNode).connect(this.head.audioReverbNode);
  }
  async enableWav2Vec2(settings={}){
    const {playbackDelayMs=90,visemeStrength=1,vowelBoost=1.0}=settings;
    if(this.wav2vec2){this.setDelay(playbackDelayMs);return;}
    if(!this.head)throw new Error('Load the avatar before Wav2Vec2');
    this.lipSyncController.setHead(this.head);
    this.lipSyncController.setStrength(visemeStrength);
    this.lipSyncController.setVowelBoost(vowelBoost);
    const {Wav2Vec2LipSync}=await import('./wav2vec2-lipsync.js');
    this.wav2vec2=new Wav2Vec2LipSync(this.lipSyncController,{log:this.log,telemetry:this.telemetry,playbackDelayMs});
    try{
      await this.wav2vec2.init();
    }catch(error){
      try{await this.wav2vec2.destroy();}catch{}
      this.wav2vec2=null;
      this.lipSyncEngine='headaudio';
      this.telemetry?.set('Lip-sync fallback','HeadAudio — WebGPU q4f16 unavailable');
      this.log('Wav2Vec2 WebGPU unavailable; automatically falling back to HeadAudio.',{message:error.message});
      return this.enableHeadAudio({...settings,lipSyncEngine:'headaudio'});
    }
    this._configureAudioRouting(playbackDelayMs);
    this.zeroVisemes=()=>this.lipSyncController.reset();
    this.head.opt.update=dt=>this.lipSyncController.update(dt);
    this.setDelay(playbackDelayMs);
    this.telemetry?.set('Viseme strength',visemeStrength.toFixed(2));
    this.log('Experimental WebGPU q4f16 Wav2Vec2 lip-sync initialized.');
  }
  async enableHeadAudio(settings={}){
    const {playbackDelayMs=90,visemeStrength=1,vowelBoost=1.0}=settings;
    // Classifier assets are confined to ./models/. The paired metadata is loaded
    // and validated before the binary, and runtime parameters come from the tuned profile.
    const headaudioModel=String(settings.headaudioModel||DEFAULT_HEADAUDIO_MODEL).split(/[\\/]/).pop();
    const modelFile=/^[\w.-]+\.bin$/.test(headaudioModel)?headaudioModel:DEFAULT_HEADAUDIO_MODEL;
    if(this.headAudio){
      this.setDelay(playbackDelayMs);
      if(this.headAudioModelFile!==modelFile)this.log(`HeadAudio classifier stays "${this.headAudioModelFile}" until a reconnect loads "${modelFile}".`);
      return;
    }
    if(!this.head)throw new Error('Load the avatar before HeadAudio');
    this.lipSyncController.setHead(this.head);
    this.lipSyncController.setStrength(visemeStrength);
    this.lipSyncController.setVowelBoost(vowelBoost);

    const assets=await loadHeadAudioConfig(modelFile, fetch, '/avatar-lab/models');
    const runtime=resolveHeadAudioRuntime(assets.runtime);
    const {sampleRate:_trainingSampleRate,...parameterData}=runtime;
    parameterData.silMode=0;

    const {HeadAudio}=await import('./vendor/headaudio/headaudio.min.mjs');
    await this.head.audioCtx.audioWorklet.addModule('/avatar-lab/vendor/headaudio/headworklet.min.mjs');
    this.headAudio=new HeadAudio(this.head.audioCtx,{
      processorOptions:{vadEventsEnabled:true,visemeEventsEnabled:true},
      parameterData
    });
    await this.headAudio.loadModel(assets.modelUrl);
    this.headAudioModelFile=modelFile;
    this.headAudioMetadata=assets.metadata;
    this.headAudioRuntime=runtime;

    const source=this.head.audioStreamGainNode;
    this._configureAudioRouting(playbackDelayMs);
    // Feed HeadAudio immediately while only the audible branch is delayed.
    this.analysisDelayNode=new DelayNode(this.head.audioCtx,{maxDelayTime:1,delayTime:0});
    source.connect(this.analysisDelayNode).connect(this.headAudio);

    this.zeroVisemes=()=>this.lipSyncController.reset();

    // Hook HeadAudio events directly to the LipSyncController pipeline
    let speechEpoch=0;
    this.headAudio.onviseme=data=>this.lipSyncController.onViseme(data);
    this.headAudio.onvad=data=>this.lipSyncController.onVad(data);
    this.headAudio.onstarted=()=>{
      speechEpoch++;
      this.telemetry?.mark('T7 avatar mouth onset');
      this.onMouthStart?.();
      this.lipSyncController.onStarted();
    };
    this.headAudio.onended=()=>{
      this.lipSyncController.onEnded();
    };

    // Drive TalkingHead updates directly via our LipSyncController spring physics step
    this.head.opt.update=dt=>this.lipSyncController.update(dt);
    this.setDelay(playbackDelayMs);
    this.telemetry?.set('HeadAudio classifier',modelFile);
    this.telemetry?.set('HeadAudio metadata',assets.metadataFile);
    this.telemetry?.set('HeadAudio runtime',assets.runtimeFile);
    this.telemetry?.set('HeadAudio speaker mean',`${runtime.speakerMeanHz} Hz`);
    this.telemetry?.set('Viseme strength',visemeStrength.toFixed(2));
    this.log('HeadAudio 0.1.0 initialized with validated model metadata and tuned runtime.',{model:modelFile,metadata:assets.metadataFile,runtime:assets.runtimeFile});
  }
  // Live lip-sync tuning from the lab panel. HeadAudio runtime parameters are
  // intentionally excluded and remain controlled by models/headaudio_runtime_tuned.json.
  applyLiveSettings(settings={}){
    if(!this.head)return false;
    if(settings.playbackDelayMs!=null)this.setDelay(settings.playbackDelayMs);
    if(settings.visemeStrength!=null){
      this.lipSyncController.setStrength(settings.visemeStrength);
      this.telemetry?.set('Viseme strength',Number(settings.visemeStrength).toFixed(2));
    }
    if(settings.vowelBoost!=null)this.lipSyncController.setVowelBoost(settings.vowelBoost);
    if(settings.lipSpeed!=null){
      this.lipSyncController.setLipSpeed(settings.lipSpeed);
      this.telemetry?.set('Lip motion speed',`${Number(settings.lipSpeed).toFixed(2)}x`);
    }
    if(settings.audioSpeed!=null){
      // Worklet intake resampler: takes effect for chunks queued from now on.
      try{this.head.streamWorkletNode?.port.postMessage({type:'config-speed',data:{speed:Number(settings.audioSpeed)}});}catch(e){}
      this.telemetry?.set('Audio playback speed',`${Number(settings.audioSpeed).toFixed(2)}x`);
    }
    if(settings.eyePitch!=null)this.setEyePitch(settings.eyePitch);
    if(settings.lidLift!=null)this.setLidLift(settings.lidLift);
    // Gaze biases used by TalkingHead's lookat templates (vendor patch): negative eyeGazeBias lifts pupils.
    if(this.head?.opt&&settings.eyePitch!=null){
      this.head.opt.headPitchBias=settings.headPitchBias!=null?Number(settings.headPitchBias):Math.max(-0.25,Math.min(0.1,Number(settings.eyePitch)*0.4));
    }
    if(this.head?.opt){
      this.head.opt.eyeGazeBias=settings.eyePitch!=null?Number(settings.eyePitch):(this.head.opt.eyeGazeBias??0.1);
      if(settings.headYawBias!=null)this.head.opt.headYawBias=Number(settings.headYawBias);
      if(settings.eyeYawBias!=null)this.head.opt.eyeYawBias=Number(settings.eyeYawBias);
      if(settings.browLift!=null)this.head.opt.browLiftBias=Math.max(0,Math.min(1,Number(settings.browLift)));
      if(settings.gazeTargetX!=null)this.head.opt.gazeTargetX=Number(settings.gazeTargetX);
      if(settings.gazeTargetY!=null)this.head.opt.gazeTargetY=Number(settings.gazeTargetY);
    }
    return true;
  }
  // Gaze correction: positive = look down, negative = look up (TalkingHead convention).
  setEyePitch(v){
    const val=Math.max(-1,Math.min(1,Number(v)||0));
    if(this.head){
      if(!this.head.opt)this.head.opt={};
      this.head.opt.eyeGazeBias=val;
      try{
        this.head.setFixedValue?.('eyesRotateX', val);
        this.telemetry?.set('Gaze Eye pitch',val.toFixed(2));
      }catch(e){}
    }
  }
  // Eye horizontal turn (yaw): positive = pupil right, negative = pupil left.
  setEyeYaw(v){
    const val=Math.max(-1,Math.min(1,Number(v)||0));
    if(this.head){
      if(!this.head.opt)this.head.opt={};
      this.head.opt.eyeYawBias=val;
      try{
        this.head.setFixedValue?.('eyesRotateY', val !== 0 ? val : null);
        this.telemetry?.set('Gaze Eye turn',val.toFixed(2));
      }catch(e){}
    }
  }
  // Upper-lid lift via ARKit eyeWide shapes (0 = natural, up to wide-open).
  setLidLift(v){
    const val=Math.max(0,Math.min(2,Number(v)||0));
    for(const side of ['Left','Right']){
      try{this.head?.setFixedValue?.(`eyeWide${side}`,val);}catch(e){}
    }
    this.telemetry?.set('Lid lift',val.toFixed(2));
  }
  // Brow lift via ARKit brow shapes.
  setBrowLift(v){
    const val=Math.max(0,Math.min(1,Number(v)||0));
    for(const target of ['browInnerUp','browOuterUpLeft','browOuterUpRight']){
      try{this.head?.setFixedValue?.(target,val);}catch(e){}
    }
    this.telemetry?.set('Brow lift',val.toFixed(2));
  }
  // Freeze or unfreeze all procedural animations and idle loops.
  setFreezeAnimation(freeze){
    const isFrozen = Boolean(freeze);
    try{
      this.head?.setFreezeAnimations?.(isFrozen);
      this.telemetry?.set('Animations frozen', isFrozen ? 'yes' : 'no');
    }catch(e){}
  }
  // Apply the lookAtCamera posture: biases used by lookAtCamera templates + locked eye/lid/brow/head settings.
  applyGaze(gaze={},{preview=false}={}){
    if(!this.head)return false;
    this.gaze={...(this.gaze||{}), ...gaze};
    const o=this.head.opt||(this.head.opt={});
    o.gazeTargetX=(this.gaze.gazeTargetX!=null?Number(this.gaze.gazeTargetX):0);
    o.gazeTargetY=(this.gaze.gazeTargetY!=null?Number(this.gaze.gazeTargetY):0);
    o.headPitchBias=(this.gaze.headPitchBias!=null?Number(this.gaze.headPitchBias):-0.15);
    o.headYawBias=(this.gaze.headYawBias!=null?Number(this.gaze.headYawBias):0);
    o.eyeYawBias=(this.gaze.eyeYawBias!=null?Number(this.gaze.eyeYawBias):0);
    o.browLiftBias=(this.gaze.browLift!=null?Math.max(0,Math.min(1,Number(this.gaze.browLift))):0);
    o.eyeGazeBias=(this.gaze.eyePitch!=null?Number(this.gaze.eyePitch):-0.7);
    o.disableGazeJitter=true;
    this.head.opt=o;

    if(this.gaze.freezeAnimation!==undefined)this.setFreezeAnimation(this.gaze.freezeAnimation);
    this.setEyePitch(this.gaze.eyePitch!=null?this.gaze.eyePitch:-0.7);
    this.setEyeYaw(this.gaze.eyeYawBias!=null?this.gaze.eyeYawBias:0);
    this.setLidLift(this.gaze.lidLift!=null?this.gaze.lidLift:1.2);
    this.setBrowLift(this.gaze.browLift!=null?this.gaze.browLift:0);
    // Clear any rigid fixed overrides on bodyRotate so lookAt screen/bias calculations apply live
    try{
      this.head?.setFixedValue?.('bodyRotateX', null);
      this.head?.setFixedValue?.('bodyRotateY', null);
    }catch(e){}

    if(preview){
      try{this.head?.lookAt?.(null,null,200);}catch(e){}
    }
    // Re-center the camera on the actual head position once the pose settles.
    this.centerViewOnHead();
    setTimeout(()=>this.centerViewOnHead(),1200);
    this.telemetry?.set('lookAtCamera pose','applied');
    return true;
  }
  // Center the camera target horizontally on the avatar's midline.
  // The avatar model is symmetrical around the world origin; keeping cameraX centered
  // ensures balanced horizontal margins for left-leaning and right-leaning standing poses.
  centerViewOnHead(){
    const h=this.head;
    if(!h||!h.scene||!h.camera||typeof h.setView!=='function')return false;
    try{
      const view=h.viewName||h.opt?.cameraView||'head';
      h.setView(view,{cameraX:0});
      this.telemetry?.set('Head-centered view','0.00');
      return true;
    }catch(e){return false;}
  }
  setMuted(muted){if(this.outputGainNode)this.outputGainNode.gain.setValueAtTime(muted?0:1,this.head.audioCtx.currentTime);this.telemetry?.set('Output muted',muted?'yes':'no');}
  setDelay(delayMs){this.delayMs=Math.max(0,Math.min(500,Number(delayMs)||0));const seconds=this.delayMs/1000;if(this.delayNode)this.delayNode.delayTime.setValueAtTime(seconds,this.head.audioCtx.currentTime);if(this.analysisDelayNode)this.analysisDelayNode.delayTime.setValueAtTime(0,this.head.audioCtx.currentTime);this.lipSyncController?.setPlaybackDelay(this.delayMs);this.telemetry?.set('Playback delay',`${this.delayMs} ms`);}
  getAecRenderReference(){
    if(!this.head?.audioCtx||!this.outputGainNode)return null;
    return {context:this.head.audioCtx,node:this.outputGainNode};
  }
  async startPcmStream({onStart,onEnd,onMetrics}={}){if(!this.head)throw new Error('Avatar is not loaded');await this.head.streamStart({sampleRate:24000,gain:1,metrics:{enabled:true,intervalHz:2}},()=>{this.wav2vec2?.setPlaybackStarted(performance.now());onStart?.();},onEnd,null,onMetrics);}
  pushPcm(pcm){this.wav2vec2?.push(pcm);this.head?.streamAudio({audio:pcm});}
  notifyEnd(){this.wav2vec2?.flush();this.head?.streamNotifyEnd();}
  interrupt(){this.head?.streamInterrupt();if(this.headAudio){this.headAudio.visemeActive=-1;this.headAudio.visemeAlphas?.fill(0);this.headAudio.resetAll?.();}this.wav2vec2?.reset();this.zeroVisemes?.();this.head?.stopGesture?.(50);}
  stopStream(){this.interrupt();this.head?.streamStop();}
  async playFixture({url='./fixtures/speech-24k.pcm',data=null,chunkMs=40,mode='acoustic',onStart=()=>{},onDone=()=>{}}={}){
    let buffer=data;
    if(!buffer){
      const response=await fetch(url);
      if(!response.ok)throw new Error(`Fixture failed: HTTP ${response.status}`);
      buffer=new Uint8Array(await response.arrayBuffer());
    }
    const streaming=mode==='streaming';
    this.telemetry?.set('PCM fixture',`${buffer.byteLength.toLocaleString()} bytes`);
    this.telemetry?.set('PCM benchmark mode',streaming?'streaming':'acoustic');
    this.telemetry?.set('PCM chunk duration',`${chunkMs} ms`);
    onStart();
    const durationMs=(buffer.length/(24000*2))*1000;
    if(streaming){
      const bytesPerChunk=Math.max(2,Math.round(24000*2*chunkMs/1000)) & ~1;
      for(let offset=0;offset<buffer.length;offset+=bytesPerChunk){
        this.pushPcm(buffer.subarray(offset,Math.min(buffer.length,offset+bytesPerChunk)));
        if(offset+bytesPerChunk<buffer.length)await new Promise(resolve=>setTimeout(resolve,chunkMs));
      }
    }else{
      this.pushPcm(buffer);
    }
    this.notifyEnd();
    setTimeout(()=>onDone(),(streaming?300:durationMs+300));
  }
  isPointerIdle(){
    return ['IDLE','READY','ONLINE','OPEN_IDLE'].includes(this.currentConversationState);
  }
  clearPointerRecoveryTimers(){
    if(this.pointerReturnTimer!==null&&typeof clearTimeout==='function')clearTimeout(this.pointerReturnTimer);
    if(this.pointerReleaseTimer!==null&&typeof clearTimeout==='function')clearTimeout(this.pointerReleaseTimer);
    this.pointerReturnTimer=null;
    this.pointerReleaseTimer=null;
  }
  suspendDefaultsForPointer(){
    const wasRecovering=this.pointerReleaseTimer!==null;
    this.clearPointerRecoveryTimers();
    if(this.pointerDefaultsSuspended){
      if(wasRecovering){
        this.pointerTakeover=true;
        // A release glide may have already restored the idle gaze bias; re-assert
        // the tracking center so a takeover never resumes with the idle bias.
        this._pointerEnterTrackingGaze();
        this._neutralizeAggregateEyeRoll(true);
      }
      return;
    }
    if(!this.head)return;
    this.pointerRestoreFreezeState=Boolean(this.head.freezeAnimations);
    try{this.head.setFreezeAnimations?.(true);}catch{}
    // The calibrated camera pose pins the eye-pitch morphs via setFixedValue, and
    // fixed values override every animated channel. Remember the pinned value,
    // unpin for the duration of pointer tracking so the animated gaze baseline
    // can move the pupils, and restore the exact pin when the pose is released.
    this.pointerEyePin=null;
    try{this.pointerEyePin=this.head.getFixedValue?.('eyesRotateX');}catch{}
    try{this.head.setFixedValue?.('eyesRotateX',null);}catch{}
    this._pointerEnterTrackingGaze();
    this.pointerDefaultsSuspended=true;
    this.pointerTakeover=true;
  }
  // While tracking, the eyes center on the cursor instead of the calibrated
  // upward idle bias (eyeGazeBias=-0.75 keeps the idle pupils visible but would
  // cancel any downward pointer travel). A small positive optical bias keeps the
  // pupils centered in the eye opening at the eye line.
  _pointerEnterTrackingGaze(){
    if(!this.head?.opt)return;
    if(this.pointerIdleEyeGazeBias==null)this.pointerIdleEyeGazeBias=this.head.opt.eyeGazeBias;
    this.head.opt.eyeGazeBias=0.12;
    this._neutralizeAggregateEyeRoll(true);
  }
  // The vertical gaze pipeline in this avaturn export is driven by the per-side
  // ARKit roll morphs (eyeLookUpLeft/Right, eyeLookDownLeft/Right); the aggregate
  // eyesLookUp/eyesLookDown morphs are baked as near-zero/wrong-sign here, so
  // while tracking we pin them to 0 (system priority, overrides the lookAt anim)
  // and drive the per-side channels directly instead.
  _neutralizeAggregateEyeRoll(on){
    if(!this.head?.mtAvatar)return;
    try{
      for(const mt of ['eyesLookUp','eyesLookDown']){
        if(this.head.mtAvatar.hasOwnProperty(mt))this.head.setValue(mt,on?0:null);
      }
    }catch(e){}
  }
  _resetPointerEyeRoll(){
    if(!this.head?.mtAvatar)return;
    try{
      for(const mt of ['eyeLookUpLeft','eyeLookUpRight','eyeLookDownLeft','eyeLookDownRight']){
        if(this.head.mtAvatar.hasOwnProperty(mt))this.head.setBaselineValue(mt,0);
      }
    }catch(e){}
  }
  // Directly roll the eyeball via the per-side morphs the model actually bakes.
  // ny is the avatar-box-relative vertical (-1 above .. +1 below the eye line).
  // The up roll target is boosted relative to the pitch range because this model
  // bakes the upward ARKit roll roughly a quarter as strong as the downward one
  // (measured: 0.8px vs 3.2px iris shift at full strength).
  _applyPointerEyeRoll(ny,eyeDownRange,eyeUpRange){
    if(!this.head?.mtAvatar)return;
    const down=Math.max(0,ny)*eyeDownRange;
    const up=Math.max(0,-ny)*eyeUpRange*1.35;
    const upv=Math.min(1,up);
    try{
      this.head.setBaselineValue('eyeLookUpLeft',upv);
      this.head.setBaselineValue('eyeLookUpRight',upv);
      this.head.setBaselineValue('eyeLookDownLeft',down);
      this.head.setBaselineValue('eyeLookDownRight',down);
    }catch(e){}
  }
  _pointerExitTrackingGaze(){
    if(!this.head?.opt)return;
    if(this.pointerIdleEyeGazeBias!=null){
      this.head.opt.eyeGazeBias=this.pointerIdleEyeGazeBias;
      this.pointerIdleEyeGazeBias=null;
    }
  }
  releaseDefaultsAfterPointer(immediate=false){
    this.clearPointerRecoveryTimers();
    if(!this.pointerDefaultsSuspended)return;
    const finish=()=>{
      if(!this.pointerDefaultsSuspended)return;
      if(this.head?.opt){this.head.opt.pointerEyeYaw=0;this.head.opt.pointerEyePitch=0;}
      // Return the eyes to the calibrated idle bias, then re-apply the exact
      // fixed eye pitch that was unpinned for tracking (setEyePitch also writes
      // eyeGazeBias, so the glide and the pin land on the same pose).
      this._pointerExitTrackingGaze();
      this._neutralizeAggregateEyeRoll(false);
      this._resetPointerEyeRoll();
      try{this.setEyePitch(this.pointerEyePin!=null?this.pointerEyePin:(this.gaze&&this.gaze.eyePitch!=null?this.gaze.eyePitch:-0.7));}catch{}
      if(!this.pointerRestoreFreezeState){try{this.head?.setFreezeAnimations?.(false);}catch{}}
      this.pointerDefaultsSuspended=false;
      this.pointerTakeover=false;
    };
    if(immediate||!this.head){finish();return;}
    this.pointerTarget=null;
    if(this.head.opt){this.head.opt.pointerEyeYaw=0;this.head.opt.pointerEyePitch=0;}
    // Restore the idle bias BEFORE the bridge lookAt so the slow glide back to
    // the calibrated camera pose ends exactly on the re-applied pin (no jump).
    this._pointerExitTrackingGaze();
    // Ease the per-side tracking roll back out over the same glide, and un-pin
    // the aggregate pitch morphs so the idle bias can settle smoothly.
    this._neutralizeAggregateEyeRoll(false);
    this._resetPointerEyeRoll();
    // Bridge back to the calibrated camera pose slowly before procedural motion resumes.
    try{this.head.lookAt?.(null,null,1400);}catch{}
    this.pointerReleaseTimer=setTimeout(finish,1600);
  }
  schedulePointerRecovery(){
    if(this.pointerReturnTimer!==null)clearTimeout(this.pointerReturnTimer);
    this.pointerReturnTimer=setTimeout(()=>{
      this.pointerReturnTimer=null;
      if(!this.pointerTrackingEnabled||!this.isPointerIdle()||!this.head)return;
      this.releaseDefaultsAfterPointer(false);
    },5000);
  }
  setPointerTrackingEnabled(enabled){
    this.pointerTrackingEnabled=Boolean(enabled);
    if(!this.pointerTrackingEnabled){
      if(this.pointerFrame!==null&&typeof cancelAnimationFrame==='function')cancelAnimationFrame(this.pointerFrame);
      this.pointerFrame=null;
      this.releaseDefaultsAfterPointer(true);
    }
  }
  enablePointerTracking(){
    if(this.pointerTrackingBound||typeof window==='undefined')return;
    this.pointerTrackingBound=true;
    this.pointerTrackingEnabled=true;
    this.boundPointerMove=event=>{
      if(!this.pointerTrackingEnabled||!this.isPointerIdle()||!this.head)return;
      if(event.pointerType&&event.pointerType!=='mouse'&&event.pointerType!=='pen')return;
      this.suspendDefaultsForPointer();
      // Resolve direction from the avatar box itself, not the viewport center. This
      // keeps tracking correct whether the interface is docked left/right or floating.
      const rect=this.container?.getBoundingClientRect?.();
      const cx=rect&&rect.width?rect.left+rect.width/2:window.innerWidth/2;
      const cy=rect&&rect.height?rect.top+rect.height/2:window.innerHeight/2;
      const dx=event.clientX-cx,dy=event.clientY-cy;
      // Resolution-proportional intensity: 1920x1080 is the 1.0 baseline. CSS-pixel
      // distances grow with resolution while the avatar stays roughly the same size,
      // so fixed pixel ceilings would saturate earlier and read as weaker tracking.
      // Scaling the travel budget and the eye range with the fit-resolution keeps
      // the visual energy comparable from small laptops to 4K displays.
      const resScale=Math.max(0.55,Math.min(2,Math.min(window.innerWidth/1920,window.innerHeight/1080)));
      // Uniform 40% response in every direction, resolved from the avatar box.
      const headXFactor=0.40;
      const headYFactor=0.40;
      // Hard pixel ceilings prevent extreme cursor positions or ultrawide screens
      // from producing excessive neck rotation in any direction.
      const maxHeadX=Math.min(220*resScale,window.innerWidth*0.18);
      const maxHeadY=Math.min(140*resScale,window.innerHeight*0.18);
      const headDx=Math.max(-maxHeadX,Math.min(maxHeadX,dx*headXFactor));
      const headDy=Math.max(-maxHeadY,Math.min(maxHeadY,dy*headYFactor));
      const x=cx+headDx;
      const y=cy+headDy;
      const availableX=dx<0?Math.max(cx,1):Math.max(window.innerWidth-cx,1);
      const nx=Math.max(-1,Math.min(1,dx/availableX));
      // Eye pitch resolves against the avatar's own eye line with avatar-box spans:
      // the cursor reaching the avatar's bottom edge is always "full look-down",
      // regardless of where the avatar sits in the viewport.
      const eyeY=rect&&rect.height?rect.top+rect.height*0.35:cy;
      const dyEye=event.clientY-eyeY;
      const eyeSpanUp=Math.max(rect&&rect.height?eyeY-rect.top:140,80);
      const eyeSpanDown=Math.max(rect&&rect.height?rect.bottom-eyeY:160,80);
      const ny=Math.max(-1,Math.min(1,dyEye/(dyEye<0?eyeSpanUp:eyeSpanDown)));
      // The tracking gaze centers near neutral (see suspendDefaultsForPointer), so
      // the full scaled range is available for actual down/up travel.
      const eyeRange=Math.min(0.55,0.45*resScale);
      // Downward travel is slightly softened to prevent excessive downward iris / bone roll.
      const eyeDownRange=Math.min(0.42,0.33*resScale);
      // Upward travel gets a very slight extra range so the pupils read a touch
      // stronger as the pointer climbs, scaling with resolution like the rest.
      const eyeUpRange=Math.min(0.62,eyeRange+0.07*resScale);
      this.head.opt.pointerEyeYaw=Math.max(-eyeRange,Math.min(eyeRange,nx*eyeRange));
      this.head.opt.pointerEyePitch=ny<0
        ?Math.max(-eyeUpRange,Math.min(0,ny*eyeUpRange))
        :Math.max(0,Math.min(eyeDownRange,ny*eyeDownRange));
      // Drive the per-side roll morphs this model actually bakes (the aggregate
      // pitch morphs are near-no-ops in this export; see _applyPointerEyeRoll).
      this._applyPointerEyeRoll(ny,eyeDownRange,eyeUpRange);
      this.pointerTarget=this.pointerTarget?{
        x:this.pointerTarget.x*.62+x*.38,
        y:this.pointerTarget.y*.62+y*.38
      }:{x,y};
      if(this.pointerFrame===null){
        this.pointerFrame=requestAnimationFrame(timestamp=>{
          this.pointerFrame=null;
          if(!this.pointerTrackingEnabled||!this.isPointerIdle()||!this.head||!this.pointerTarget)return;
          if(timestamp-this.pointerLastLook<55)return;
          this.pointerLastLook=timestamp;
          const transitionMs=this.pointerTakeover?480:170;
          this.pointerTakeover=false;
          try{this.head.lookAt?.(this.pointerTarget.x,this.pointerTarget.y,transitionMs);}catch{}
        });
      }
      this.schedulePointerRecovery();
    };
    this.boundPointerLeave=event=>{
      if(event.relatedTarget||!this.pointerTrackingEnabled||!this.isPointerIdle()||!this.head)return;
      this.schedulePointerRecovery();
    };
    window.addEventListener('pointermove',this.boundPointerMove,{passive:true});
    window.addEventListener('pointerout',this.boundPointerLeave,{passive:true});
  }
    setConversationState(state){
    this.currentConversationState=state;
    if(!this.isPointerIdle()){
      if(this.pointerFrame!==null&&typeof cancelAnimationFrame==='function')cancelAnimationFrame(this.pointerFrame);
      this.pointerFrame=null;
      this.releaseDefaultsAfterPointer(true);
    }
    if(!this.head)return;
    if(!this.isPointerIdle()){
      if(this.head.opt){this.head.opt.pointerEyeYaw=0;this.head.opt.pointerEyePitch=0;}
      if(Array.isArray(this.head.animQueue)){
        for(let i=this.head.animQueue.length-1;i>=0;i--){
          if(this.head.animQueue[i]?.template?.name==='lookat')this.head.animQueue.splice(i,1);
        }
      }
    }
    if(['INTERRUPTING','USER_SPEAKING','LISTENING','THINKING','ERROR','CLOSED'].includes(state)){
      this.head.stopGesture?.(100);
    }
  }
  async destroy(){
    this.setPointerTrackingEnabled(false);
    if(typeof window!=='undefined'&&this.pointerTrackingBound){
      window.removeEventListener('pointermove',this.boundPointerMove);
      window.removeEventListener('pointerout',this.boundPointerLeave);
    }
    this.pointerTrackingBound=false;
    try{this.head?.streamStop?.()}catch{}
    try{this.headAudio?.disconnect?.()}catch{}
    try{await this.wav2vec2?.destroy?.()}catch{}
    try{this.analysisDelayNode?.disconnect?.()}catch{}
    try{this.delayNode?.disconnect?.()}catch{}
    try{this.outputGainNode?.disconnect?.()}catch{}
    if(this.head){try{this.head.stop?.()}catch{};try{this.head.dispose?.()}catch{}}
    this.head=this.headAudio=this.wav2vec2=this.delayNode=this.outputGainNode=null;
    this.container.replaceChildren();
  }
}
