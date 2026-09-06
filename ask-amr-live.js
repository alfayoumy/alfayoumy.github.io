import {GeminiLiveClient, GEMINI_LIVE_MODEL, GEMINI_LIVE_FALLBACK_MODEL} from './avatar-lab/live-client.js';
import {MicrophoneInput} from './avatar-lab/audio-input.js';
import {LiveTokenProvider} from './avatar-lab/token-provider.js';
import {base64ToPcm16} from './avatar-lab/audio-utils.js';
import {AVATAR_LAB_DEFAULTS} from './avatar-lab/avatar-config.js';

export function createAskAmrVoice({getAvatar, onState, onInput, onOutput, onInterrupted, onNotice, turnstileContainer}={}){
  let client=null,mic=null,provider=null,output='',pending=[],streamReady=false,starting=false,playbackStarted=false;
  let activeModel=GEMINI_LIVE_MODEL,activeKeyIndex=0,handingOver=false,closed=false;
  let conversationHistory=[];
  const state=value=>onState?.(value);
  const notice=value=>onNotice?.(value);
  const wait=ms=>new Promise(resolve=>setTimeout(resolve,ms));

  function getFailoverSequence(model, keyIndex){
    if(model === GEMINI_LIVE_MODEL && keyIndex === 0){
      return [
        { model: GEMINI_LIVE_MODEL, keyIndex: 1 },
        { model: GEMINI_LIVE_FALLBACK_MODEL, keyIndex: 0 },
        { model: GEMINI_LIVE_FALLBACK_MODEL, keyIndex: 1 }
      ];
    }
    if(model === GEMINI_LIVE_FALLBACK_MODEL && keyIndex === 1){
      return [
        { model: GEMINI_LIVE_FALLBACK_MODEL, keyIndex: 0 },
        { model: GEMINI_LIVE_MODEL, keyIndex: 0 },
        { model: GEMINI_LIVE_MODEL, keyIndex: 1 }
      ];
    }
    if(model === GEMINI_LIVE_MODEL && keyIndex === 1){
      return [
        { model: GEMINI_LIVE_MODEL, keyIndex: 0 },
        { model: GEMINI_LIVE_FALLBACK_MODEL, keyIndex: 1 },
        { model: GEMINI_LIVE_FALLBACK_MODEL, keyIndex: 0 }
      ];
    }
    return [
      { model: GEMINI_LIVE_FALLBACK_MODEL, keyIndex: 1 },
      { model: GEMINI_LIVE_MODEL, keyIndex: 0 },
      { model: GEMINI_LIVE_MODEL, keyIndex: 1 }
    ];
  }

  async function connectSession(avatar, targetModel, targetKeyIndex, history=null){
    const c=new GeminiLiveClient({voice:'Orus', model: targetModel});
    let turnUser='';

    c.addEventListener('inputtranscript',e=>{
      const text=e.detail.text||'';
      turnUser+=text;
      onInput?.(text);
      state('listening');
    });
    c.addEventListener('outputtranscript',e=>{
      output+=e.detail.text;
      onOutput?.(output,false);
    });
    c.addEventListener('audio',e=>{
      state('speaking');
      const pcm=base64ToPcm16(e.detail.data);
      if(!playbackStarted){
        pending.push(pcm);
        if(pending.length>=2){
          pending.forEach(x=>avatar.pushPcm(x));
          pending=[];
          playbackStarted=true;
        }
      }else{
        avatar.pushPcm(pcm);
      }
    });
    c.addEventListener('interrupted',()=>{
      pending=[];
      playbackStarted=false;
      avatar.interrupt();
      const interrupted=output.trim();
      if(turnUser.trim()){
        conversationHistory.push({role:'user',content:turnUser.trim()});
        turnUser='';
      }
      if(interrupted){
        conversationHistory.push({role:'assistant',content:interrupted+' [INTERRUPTED]'});
      }
      conversationHistory=conversationHistory.slice(-8);
      onInterrupted?.(interrupted);
      output='';
      state('listening');
    });
    c.addEventListener('turncomplete',()=>{
      pending.forEach(x=>avatar.pushPcm(x));
      pending=[];
      playbackStarted=false;
      avatar.notifyEnd();
      if(turnUser.trim()){
        conversationHistory.push({role:'user',content:turnUser.trim()});
        turnUser='';
      }
      if(output.trim()){
        conversationHistory.push({role:'assistant',content:output.trim()});
        onOutput?.(output.trim(),true);
      }
      conversationHistory=conversationHistory.slice(-8);
      output='';
      state('listening');
    });
    
    c.addEventListener('close',e=>{
      if(!e.detail.intentional && client===c && !handingOver && !closed){
        performHandover('socket close');
      }
    });
    c.addEventListener('servererror',e=>{
      if(client===c && !handingOver && !closed){
        performHandover(e.detail.message||`server error ${e.detail.code||''}`);
      }
    });
    c.addEventListener('goaway',()=>{
      if(client===c && !handingOver && !closed){
        performHandover('server goaway');
      }
    });

    const token=await provider.issue({
      model: targetModel,
      keyIndex: targetKeyIndex,
      history: history!==null ? history : conversationHistory.slice(-8)
    });
    await c.connectWithToken(token);
    return c;
  }

  async function performHandover(reason='mid-session error'){
    if(handingOver || !client || closed) return;
    handingOver = true;
    const wasListening = Boolean(mic);

    state('reconnecting');
    notice('Connection interrupted. Switching voice engine...');

    pending = [];
    playbackStarted = false;
    try{
      const avatar = await getAvatar();
      avatar?.notifyEnd();
    }catch{}

    const oldClient = client;
    client = null;
    try{ oldClient?.close(); }catch{}

    let connected = false;

    // Step 1: Before switching to a different model or API key, attempt the same request
    // while preserving only 4 conversation turns instead of 8 to reduce TPM/rate pressure.
    try {
      await wait(500);
      const avatar = await getAvatar();
      const trimmedHistory = conversationHistory.slice(-4);
      const newClient = await connectSession(avatar, activeModel, activeKeyIndex, trimmedHistory);
      client = newClient;
      conversationHistory = trimmedHistory;
      connected = true;
      state(wasListening ? 'listening' : 'online');
      notice('');
    } catch (sameReqError) {
      try { client?.close(); } catch {}
      client = null;
    }

    // Step 2: If same-request attempt failed, proceed to ordered failover sequence across keys & models
    if (!connected && !closed) {
      const sequence = getFailoverSequence(activeModel, activeKeyIndex);
      const maxCycles = 2;

      for (let cycle = 0; cycle < maxCycles && !connected && !closed; cycle += 1) {
        for (let i = 0; i < sequence.length && !connected && !closed; i += 1) {
          const target = sequence[i];
          try {
            await wait(cycle === 0 ? 500 : 1000 + cycle * 500);
            const avatar = await getAvatar();
            const newClient = await connectSession(avatar, target.model, target.keyIndex, conversationHistory.slice(-4));
            client = newClient;
            activeModel = target.model;
            activeKeyIndex = target.keyIndex;
            conversationHistory = conversationHistory.slice(-4);
            connected = true;
            state(wasListening ? 'listening' : 'online');
            notice('');
            break;
          } catch (trialError) {
            try { client?.close(); } catch {}
            client = null;
          }
        }
      }
    }

    if (!connected && !closed) {
      state('voice_unavailable');
      notice('Voice connection lost. Tap TALK to retry, or continue with text.');
    }
    handingOver = false;
  }

  async function ensureConnected(){
    if(client?.setupComplete)return;
    if(starting)throw new Error('Voice is already connecting.');
    starting=true;closed=false;state('connecting');
    try{
      const avatar=await getAvatar();
      if(!avatar)throw new Error('The avatar is still initializing. Please try again.');
      if(!streamReady){await avatar.startPcmStream();streamReady=true;}
      await avatar.enableLipSync(AVATAR_LAB_DEFAULTS);
      provider = provider || new LiveTokenProvider({container: turnstileContainer});

      let initialSuccess = false;
      try{
        client=await connectSession(avatar, GEMINI_LIVE_MODEL, 0);
        activeModel=GEMINI_LIVE_MODEL;
        activeKeyIndex=0;
        initialSuccess=true;
      }catch(initialError){
        try{client?.close();}catch{}
        client=null;
      }

      if(!initialSuccess){
        const sequence = getFailoverSequence(GEMINI_LIVE_MODEL, 0);
        const maxCycles = 2;
        for (let cycle = 0; cycle < maxCycles && !initialSuccess && !closed; cycle += 1) {
          for (let i = 0; i < sequence.length && !initialSuccess && !closed; i += 1) {
            const target = sequence[i];
            await wait(cycle === 0 ? 500 : 1000 + cycle * 500);
            try{
              client=await connectSession(avatar, target.model, target.keyIndex);
              activeModel=target.model;
              activeKeyIndex=target.keyIndex;
              initialSuccess=true;
              break;
            }catch(err){
              try{client?.close();}catch{}
              client=null;
            }
          }
        }
      }

      if(!initialSuccess){
        throw new Error('Unable to establish voice connection. Tap RETRY to try again.');
      }
      state('online');
    }catch(error){
      try{client?.close();}catch{}
      client=null;
      throw error;
    }finally{starting=false;}
  }
  async function start(){
    try{
      await ensureConnected();
      if(mic){await stop();return false;}
      const avatar=await getAvatar();
      mic=new MicrophoneInput({
        chunkMs:100,
        renderReference:avatar?.getAecRenderReference?.()||null,
        onChunk:chunk=>{
          if(client?.setupComplete){
            try{client.sendAudio(chunk.base64,chunk.mimeType);}catch{}
          }
        },
        onEnded:()=>stop().catch(()=>{}),
        onError:error=>{
          const failedMic=mic;
          mic=null;
          failedMic?.stop().catch(()=>{});
          try{client?.sendAudioStreamEnd();}catch{}
          state('voice_unavailable');
          notice('Echo cancellation failed. Tap TALK to retry, or continue with text.');
        }
      });
      await mic.start();state('listening');return true;
    }catch(error){state('voice_unavailable');notice(error?.code==='MIC_PERMISSION_DENIED'?'Microphone blocked. Text chat is still available.':error?.code==='AEC_UNAVAILABLE'?'Echo cancellation is unavailable in this browser. Text chat is still available.':(error.message||'Voice is temporarily unavailable. Continue with text.'));throw error;}
  }
  async function stop(){if(mic){await mic.stop();mic=null;try{client?.sendAudioStreamEnd();}catch{}}state('online');return false;}
  async function close(){closed=true;pending=[];playbackStarted=false;await stop();client?.close();client=null;provider?.destroy();provider=null;const avatar=await getAvatar();avatar?.stopStream();streamReady=false;}
  return {start,stop,close,get listening(){return Boolean(mic)}};
}
