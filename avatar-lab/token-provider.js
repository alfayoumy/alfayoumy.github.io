function endpointUrl(endpoint,path){const base=endpoint.replace(/\/$/,'').replace(/\/chat$/,'');return `${base}${path}`;}
let fallbackSessionCounter=0;
function randomSessionId(){
  try{
    const bytes=globalThis.crypto.getRandomValues(new Uint8Array(18));
    return Array.from(bytes,b=>b.toString(16).padStart(2,'0')).join('');
  }catch{
    // This is a public rate-limit/session label, not an authorization token.
    // Browser RNG failures must not prevent verification or voice startup.
    // Time and a counter keep IDs distinct even if Math.random repeats.
    fallbackSessionCounter+=1;
    return `voice_${Date.now().toString(36)}_${fallbackSessionCounter.toString(36)}_${Math.random().toString(36).slice(2).padEnd(12,'0')}`;
  }
}
function loadTurnstile(){if(typeof window!=='undefined'&&window.turnstile)return Promise.resolve();return new Promise((resolve,reject)=>{if(typeof document==='undefined')return resolve();const existing=document.querySelector('script[data-avatar-turnstile]');if(existing){existing.addEventListener('load',resolve,{once:true});existing.addEventListener('error',reject,{once:true});return;}const script=document.createElement('script');script.src='https://challenges.cloudflare.com/turnstile/v0/api.js?render=explicit';script.async=true;script.defer=true;script.dataset.avatarTurnstile='';script.onload=resolve;script.onerror=()=>reject(new Error('Turnstile failed to load'));document.head.append(script);});}
export class LiveTokenProvider {
  constructor({log=()=>{},container=null}={}){this.log=log;this.configPromise=null;this.widgetId=null;this.pending=null;this.sessionId=randomSessionId();this.container=null;this.mountTarget=container||null;}
  async #config(){if(!this.configPromise)this.configPromise=fetch('/chat-config.json',{cache:'no-store'}).then(r=>{if(!r.ok)throw new Error(`Chat configuration failed: HTTP ${r.status}`);return r.json();});return this.configPromise;}
  async #verificationToken(sitekey){
    await loadTurnstile();if(this.pending)throw new Error('Verification is already active.');
    if(!this.container&&typeof document!=='undefined'){this.container=document.createElement('div');this.container.id='avatar-turnstile';this.container.setAttribute('aria-label','Human verification');if(this.mountTarget){this.mountTarget.append(this.container);}else{(document.querySelector('.ask-amr-verify')||document.querySelector('.controls')||document.body).append(this.container);}}
    return new Promise((resolve,reject)=>{this.pending={resolve,reject};const finish=(error,value)=>{const pending=this.pending;this.pending=null;if(error)pending?.reject(error);else pending?.resolve(value);};
      if(typeof window==='undefined'||!window.turnstile)return finish(null,'test-token');
      const options={sitekey,theme:document.documentElement.getAttribute('data-theme')==='dark'?'dark':'light',size:'flexible',appearance:'interaction-only',execution:'execute',retry:'auto','retry-interval':3000,'refresh-expired':'auto','refresh-timeout':'auto',callback:value=>finish(null,value),'expired-callback':()=>finish(new Error('Verification expired.')),'timeout-callback':()=>finish(new Error('Verification timed out.')),'error-callback':()=>{finish(new Error('Verification failed.'));return true;}};
      if(this.widgetId===null)this.widgetId=window.turnstile.render(this.container,options);else window.turnstile.reset(this.widgetId);window.turnstile.execute(this.widgetId);
    });
  }
  async issue({fallback=false,model=null,keyIndex=undefined,history=undefined}={}){const config=(await this.#config())||{};const endpoint=config.endpoint||config.apiUrl;const turnstileSiteKey=config.turnstileSiteKey||config.siteKey;if(!endpoint||!turnstileSiteKey)throw new Error('Live-token configuration is unavailable.');const loc=typeof location!=='undefined'?location:{href:'http://localhost/',hostname:'localhost'};const endpointHost=new URL(endpoint,loc.href).hostname;const loopback=value=>['localhost','127.0.0.1','::1'].includes(value);const localDev=loopback(loc.hostname)&&loopback(endpointHost);const turnstileToken=localDev?'local-dev-bypass':await this.#verificationToken(turnstileSiteKey);const requestPayload={turnstileToken,sessionId:this.sessionId};if(fallback)requestPayload.fallback=true;if(model)requestPayload.model=model;if(keyIndex!==undefined)requestPayload.keyIndex=keyIndex;if(history!==undefined)requestPayload.history=history;try{const response=await fetch(endpointUrl(endpoint,'/live-token'),{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(requestPayload)});const body=await response.json().catch(()=>({}));if(!response.ok)throw new Error(body.error||`Live token failed: HTTP ${response.status}`);if(typeof body.token!=='string'||!body.token)throw new Error('Worker returned no ephemeral token.');this.lastModel=body.model||null;this.lastKeyIndex=body.keyIndex??null;this.log('One-use ephemeral authorization issued by Worker.',{model:this.lastModel,keyIndex:this.lastKeyIndex});return body.token;}finally{if(this.widgetId!==null&&typeof window!=='undefined'&&window.turnstile)window.turnstile.reset(this.widgetId);}}
  destroy(){if(this.widgetId!==null&&typeof window!=='undefined'&&window.turnstile)try{window.turnstile.remove(this.widgetId)}catch{}this.container?.remove();this.widgetId=null;this.container=null;this.pending=null;this.sessionId=randomSessionId();}
}
