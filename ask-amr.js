(function () {
  'use strict';

  var STORAGE_KEY = 'amr-portfolio-chat-v1';
  var VOICE_STORAGE_KEY = 'amr-portfolio-voice-v1';
  var MODE_STORAGE_KEY = 'amr-portfolio-interface-mode-v1';
  var HISTORY_CONTENT_LIMIT = 4000;
  var suggestions = [
    'What has Amr shipped?',
    'Tell me about his GenAI work.',
    'Why should I hire Amr?',
    'Tell me about his fraud and AML work.',
    'Which projects demonstrate his research experience?',
    'What is Amr like outside of work?'
  ];
  var pendingPhrases = [
    'Harvesting spice',
    'Riding the sandworm',
    'Training with the Bene Gesserit',
    'Sweating in a stillsuit',
    'Spotting wormsign',
    'Entering the spice trance',
    'Summoning Shai-Hulud',
    'Using the Voice',
    'Flying the ornithopter',
    'Awaiting Lisan al-Gaib',
    'Plotting against House Harkonnen',
    'Surviving Arrakis',
    'Riding Shai-Hulud',
    'Walking without rhythm',
    'Folding space',
    'Seeing all possible futures',
    'Waking the sleeper',
    'Becoming desert power',
    'Accidentally starting a holy war',
    'Threatening spice production',
    'Bothering the Spacing Guild',
    'Unlocking ancestral memories',
    'Asking the Reverend Mother',
    'Choosing the least terrible future',
    'Escaping the Coriolis storm',
    'Becoming a Fremen'
  ];

  function loadHistory() {
    try {
      var value = JSON.parse(sessionStorage.getItem(STORAGE_KEY) || '[]');
      return Array.isArray(value) ? value.slice(-6) : [];
    } catch (error) {
      return [];
    }
  }

  function saveHistory(history) {
    try { sessionStorage.setItem(STORAGE_KEY, JSON.stringify(history.slice(-6))); } catch (error) {}
  }

  function loadVoiceHistory() {
    try {
      var value = JSON.parse(sessionStorage.getItem(VOICE_STORAGE_KEY) || '[]');
      return Array.isArray(value) ? value.slice(-12) : [];
    } catch (error) { return []; }
  }

  function saveVoiceHistory(history) {
    try { sessionStorage.setItem(VOICE_STORAGE_KEY, JSON.stringify(history.slice(-12))); } catch (error) {}
  }

  function loadMode() {
    try { return sessionStorage.getItem(MODE_STORAGE_KEY) === 'chat' ? 'chat' : 'avatar'; }
    catch (error) { return 'avatar'; }
  }

  function saveMode(mode) {
    try { sessionStorage.setItem(MODE_STORAGE_KEY, mode); } catch (error) {}
  }

  function sessionId() {
    var key = STORAGE_KEY + '-session';
    try {
      var current = sessionStorage.getItem(key);
      if (current) return current;
      var created = (crypto.randomUUID ? crypto.randomUUID() : Math.random().toString(36) + Date.now().toString(36)).replace(/-/g, '');
      sessionStorage.setItem(key, created);
      return created;
    } catch (error) {
      return (Math.random().toString(36) + Date.now().toString(36)).replace(/[^a-z0-9]/g, '').padEnd(16, '0');
    }
  }

  function create(tag, className, text) {
    var element = document.createElement(tag);
    if (className) element.className = className;
    if (typeof text === 'string') element.textContent = text;
    return element;
  }

  function appendInlineMarkdown(parent, text) {
    var pattern = /(\*\*([^*]+)\*\*|`([^`]+)`|\*([^*]+)\*)/g;
    var cursor = 0;
    var match;
    while ((match = pattern.exec(text)) !== null) {
      if (match.index > cursor) parent.appendChild(document.createTextNode(text.slice(cursor, match.index)));
      var element;
      if (match[2]) element = create('strong', '', match[2]);
      else if (match[3]) element = create('code', '', match[3]);
      else element = create('em', '', match[4]);
      parent.appendChild(element);
      cursor = pattern.lastIndex;
    }
    if (cursor < text.length) parent.appendChild(document.createTextNode(text.slice(cursor)));
  }

  function renderMarkdown(element, text) {
    var lines = String(text || '').replace(/\r\n?/g, '\n').split('\n');
    var list = null;
    var listType = '';

    function closeList() { list = null; listType = ''; }

    lines.forEach(function (line) {
      var unordered = line.match(/^\s*[-*]\s+(.+)$/);
      var ordered = line.match(/^\s*\d+[.)]\s+(.+)$/);
      if (unordered || ordered) {
        var type = ordered ? 'ol' : 'ul';
        if (!list || listType !== type) {
          closeList();
          list = create(type);
          listType = type;
          element.appendChild(list);
        }
        var item = create('li');
        appendInlineMarkdown(item, (ordered || unordered)[1]);
        list.appendChild(item);
        return;
      }
      closeList();
      if (!line.trim()) return;
      var paragraph = create('p');
      appendInlineMarkdown(paragraph, line.trim());
      element.appendChild(paragraph);
    });
  }

  function loadTurnstile() {
    if (window.turnstile) return Promise.resolve();
    return new Promise(function (resolve, reject) {
      var existing = document.querySelector('script[data-turnstile-script]');
      if (existing) {
        existing.addEventListener('load', resolve, { once: true });
        existing.addEventListener('error', reject, { once: true });
        return;
      }
      var script = document.createElement('script');
      script.src = 'https://challenges.cloudflare.com/turnstile/v0/api.js?render=explicit';
      script.async = true;
      script.defer = true;
      script.dataset.turnstileScript = '';
      script.onload = resolve;
      script.onerror = reject;
      document.head.appendChild(script);
    });
  }

  function endpointUrl(endpoint) {
    return endpoint.replace(/\/$/, '') + (/\/chat$/.test(endpoint) ? '' : '/chat');
  }

  function isLocalDevelopmentEndpoint(endpoint) {
    try {
      var endpointHost = new URL(endpoint, window.location.href).hostname;
      var loopback = function (value) { return ['localhost', '127.0.0.1', '::1'].indexOf(value) !== -1; };
      return loopback(window.location.hostname) && loopback(endpointHost);
    } catch (error) { return false; }
  }

  function focusableElements(container) {
    return Array.prototype.slice.call(container.querySelectorAll('a[href], button:not([disabled]), input:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])'));
  }

  function setBackgroundInert(exception, inert) {
    Array.prototype.slice.call(document.body.children).forEach(function (element) {
      if (element === exception) return;
      if (inert) element.setAttribute('inert', '');
      else element.removeAttribute('inert');
    });
  }

  function initialize(config) {
    config = config || {};
    var endpoint = config.endpoint || config.apiUrl;
    var turnstileSiteKey = config.turnstileSiteKey || config.siteKey;
    if (!endpoint || !turnstileSiteKey) return;
    config.endpoint = endpoint;
    config.turnstileSiteKey = turnstileSiteKey;

    var history = loadHistory();
    var voiceHistory = loadVoiceHistory();
    var initialMode = loadMode();
    var token = '';
    var widgetId = null;
    var busy = false;
    var previousFocus = null;
    var pendingRequest = null;

    var launcher = create('button', 'ask-amr-launcher');
    launcher.type = 'button';
    launcher.setAttribute('aria-haspopup', 'dialog');
    launcher.setAttribute('aria-expanded', 'false');
    launcher.setAttribute('aria-label', 'Open Ask Amr.AI portfolio interface');
    launcher.innerHTML = '<span class="ask-amr-status" aria-hidden="true"></span><span>ASK AMR.AI</span><span aria-hidden="true">↗</span>';

    var backdrop = create('div', 'ask-amr-backdrop');
    var panel = create('section', 'ask-amr-panel');
    panel.setAttribute('role', 'dialog');
    panel.setAttribute('aria-modal', 'true');
    panel.setAttribute('aria-labelledby', 'ask-amr-title');
    panel.dataset.state = 'online';
    panel.dataset.mode = initialMode;
    panel.innerHTML = [
      '<header class="ask-amr-head">',
      '<div><div class="ask-amr-kicker">// ASK AMR.AI</div><h2 id="ask-amr-title">Live portfolio interface</h2></div>',
      '<div class="ask-amr-head-actions"><span class="ask-amr-live"><i></i><b>ONLINE</b></span><button type="button" class="ask-amr-clear-mode" aria-label="Clear current conversation">CLEAR</button><button type="button" class="ask-amr-minimize" aria-label="Minimize Ask Amr.AI">−</button><button type="button" class="ask-amr-close" aria-label="Close Ask Amr.AI">×</button></div>',
      '</header>',
      '<section class="ask-amr-avatar" aria-label="Amr.AI avatar">',
      '<div class="ask-amr-avatar-grid" aria-hidden="true"></div>',
      '<div class="ask-amr-avatar-canvas" data-avatar-target="panel"></div>',
      '<div class="ask-amr-avatar-meta"><span>// AI INTERFACE</span><span class="ask-amr-broadcast-time" aria-label="Local date and time"></span></div>',
      '<div class="ask-amr-waveform" aria-hidden="true"><i></i><i></i><i></i><i></i><i></i><i></i><i></i></div>',
      '<div class="ask-amr-turnstile-host" role="presentation"><div class="ask-amr-dialing"><span class="cli-spinner" aria-hidden="true"></span><span>Dialing...</span></div></div>',
      '</section>',
      '<div class="ask-amr-log" aria-live="polite"></div>',
      '<section class="ask-amr-topics"><div class="ask-amr-topics-head"><div class="ask-amr-suggestions-title">SUGGESTED TOPICS</div><button type="button" class="ask-amr-topics-toggle" aria-expanded="true" aria-label="Minimize suggested topics">−</button></div><div class="ask-amr-suggestions" aria-label="Suggested topics"></div></section>',
      '<form class="ask-amr-form">',
      '<label for="ask-amr-input">Ask about Amr\'s work, projects, research, or skills.</label>',
      '<div class="ask-amr-mode-row"><button type="button" class="ask-amr-talk"><span aria-hidden="true">🎙</span> TALK TO AMR.AI</button><span>OR</span><button type="button" class="ask-amr-type">CHAT INSTEAD</button></div>',
      '<div class="ask-amr-input-row"><span aria-hidden="true">&gt;</span><textarea id="ask-amr-input" rows="1" maxlength="1500" placeholder="Ask about Amr\'s work..."></textarea><button type="submit">SEND <span aria-hidden="true">↵</span></button></div>',
      '<div class="ask-amr-voice-note" role="status" hidden></div>',
      '<div class="ask-amr-verify"></div>',
      '<div class="ask-amr-foot"><span>GROUNDED IN PORTFOLIO SOURCES</span><button type="button" class="ask-amr-clear">CLEAR</button></div>',
      '</form>'
    ].join('');
    backdrop.appendChild(panel);
    document.body.appendChild(launcher);
    document.body.appendChild(backdrop);

    var closeButton = panel.querySelector('.ask-amr-close');
    var minimizeButton = panel.querySelector('.ask-amr-minimize');
    var clearButton = panel.querySelector('.ask-amr-clear');
    var clearModeButton = panel.querySelector('.ask-amr-clear-mode');
    var talkButton = panel.querySelector('.ask-amr-talk');
    var typeButton = panel.querySelector('.ask-amr-type');
    var voiceNote = panel.querySelector('.ask-amr-voice-note');
        var avatarCanvas = panel.querySelector('.ask-amr-avatar-canvas');
    var turnstileHost = panel.querySelector('.ask-amr-turnstile-host');
    var avatarPortal = create('div', 'ask-amr-avatar-portal');
    avatarPortal.innerHTML = '<div class="ask-amr-avatar-loading"><span class="cli-spinner"></span><small>INITIALIZING AVATAR</small></div>';
    var avatarParking = create('div', 'ask-amr-avatar-parking');
    avatarCanvas.appendChild(avatarPortal);
    document.body.appendChild(avatarParking);
    var homeTerminal = null;
    var liveLabel = panel.querySelector('.ask-amr-live b');
    var avatarStarted = false;
    var voiceController = null;
    var liveInputTranscript = '';
    var liveInputRendered = false;
    var liveAssistantMessage = null;
    var windowAnimationActive = false;
    var log = panel.querySelector('.ask-amr-log');
    var topicsSection = panel.querySelector('.ask-amr-topics');
    var topicsToggle = panel.querySelector('.ask-amr-topics-toggle');
    var suggestionBox = panel.querySelector('.ask-amr-suggestions');
    var form = panel.querySelector('.ask-amr-form');
    var input = panel.querySelector('textarea');
    var submit = form.querySelector('button[type="submit"]');
    var verify = panel.querySelector('.ask-amr-verify');
    var broadcastTime = panel.querySelector('.ask-amr-broadcast-time');
    var homeMount = document.getElementById('ask-amr-home-mount');
    var homeInput = null;

    if (homeMount) {
      homeTerminal = create('section', 'ask-amr-home-terminal ask-amr-home-away');
      homeTerminal.setAttribute('aria-labelledby', 'ask-amr-home-title');
      homeTerminal.innerHTML = [
        '<div class="ask-amr-home-head"><span>// ASK AMR.AI</span><span><i aria-hidden="true"></i> UNDOCKED</span></div>',
        '<div class="ask-amr-away-screen" role="status">',
        '<span class="cli-spinner ask-amr-away-spinner" aria-hidden="true"></span>',
        '<div class="ask-amr-away-kicker">// INTERFACE IN TRANSIT</div>',
        '<h2 id="ask-amr-home-title">Continue in the floating window.</h2>',
        '<p>The live session is active at the edge of your screen.</p>',
        '<span class="ask-amr-away-signal" aria-hidden="true">SIGNAL →</span>',
        '</div>'
      ].join('');
      homeTerminal.hidden = true;
      homeMount.appendChild(homeTerminal);
      homeMount.appendChild(panel);
      panel.classList.add('is-home');
      panel.setAttribute('aria-modal', 'false');
      avatarPortal.dataset.owner = 'home';
      document.documentElement.dataset.askAmrAvatarOwner = 'home';
      loadAvatar();
    } else {
      avatarPortal.dataset.owner = panel.dataset.mode === 'avatar' ? 'panel' : 'parked';
      document.documentElement.dataset.askAmrAvatarOwner = avatarPortal.dataset.owner;
    }

    function updateBroadcastTime() {
      var now = new Date();
      var date = [now.getFullYear(), String(now.getMonth() + 1).padStart(2, '0'), String(now.getDate()).padStart(2, '0')].join('.');
      var time = [String(now.getHours()).padStart(2, '0'), String(now.getMinutes()).padStart(2, '0'), String(now.getSeconds()).padStart(2, '0')].join(':');
      broadcastTime.textContent = date + ' · ' + time;
    }
    updateBroadcastTime();
    window.setInterval(updateBroadcastTime, 1000);

    function refreshAvatarViewport() {
      window.requestAnimationFrame(function () {
        window.requestAnimationFrame(function () {
          try {
            if (window.askAmrAvatar && window.askAmrAvatar.head) {
              window.askAmrAvatar.head.resize?.();
              window.dispatchEvent(new Event('resize'));
            }
          } catch (error) {}
        });
      });
    }

    function setMode(mode, options) {
      mode = mode === 'chat' ? 'chat' : 'avatar';
      panel.dataset.mode = mode;
      saveMode(mode);
      if (mode === 'chat') {
        if (voiceController && voiceController.listening) voiceController.stop().catch(function () {});
        if (avatarPortal.parentElement !== avatarParking) avatarParking.appendChild(avatarPortal);
        avatarPortal.dataset.owner = 'parked';
        document.documentElement.dataset.askAmrAvatarOwner = 'parked';
        refreshAvatarViewport();
        talkButton.innerHTML = '<span aria-hidden="true">🎙</span> TALK INSTEAD';
        typeButton.hidden = true;
        voiceNote.hidden = true;
        if (!options || options.focus !== false) window.setTimeout(function () { input.focus(); }, 80);
      } else {
        if (avatarPortal.parentElement !== avatarCanvas) avatarCanvas.appendChild(avatarPortal);
        avatarPortal.dataset.owner = panel.classList.contains('is-home') ? 'home' : 'panel';
        document.documentElement.dataset.askAmrAvatarOwner = avatarPortal.dataset.owner;
        talkButton.innerHTML = '<span aria-hidden="true">🎙</span> TALK TO AMR.AI';
        typeButton.hidden = false;
        typeButton.textContent = 'CHAT INSTEAD';
        input.value = '';
        loadAvatar();
        refreshAvatarViewport();
      }
      if (window.askAmrAvatar) window.askAmrAvatar.setPointerTrackingEnabled(mode === 'avatar');
      renderCurrentMode();
    }

    function setInterfaceState(state) {
      state = state || 'online';
      panel.dataset.state = state;
      liveLabel.textContent = state.toUpperCase().replace('_', ' ');
      launcher.dataset.state = state;
      if (window.askAmrAvatar) {
        var avatarStates = {
          online: 'READY', connecting: 'CONNECTING', reconnecting: 'RECONNECTING',
          listening: 'LISTENING', thinking: 'THINKING', speaking: 'MODEL_SPEAKING',
          error: 'ERROR', voice_unavailable: 'ERROR'
        };
        window.askAmrAvatar.setConversationState(avatarStates[state] || 'READY');
      }
    }

    function transferAvatar(target, destination) {
      if (!target || avatarPortal.parentElement === target) return;
      var from = avatarPortal.getBoundingClientRect();
      var ghost = create('div', 'ask-amr-avatar-transfer-ghost');
      ghost.style.cssText = 'left:' + from.left + 'px;top:' + from.top + 'px;width:' + from.width + 'px;height:' + from.height + 'px';
      document.body.appendChild(ghost);
      target.appendChild(avatarPortal);
      avatarPortal.dataset.owner = destination;
      document.documentElement.dataset.askAmrAvatarOwner = destination;
      if (homeTerminal) homeTerminal.classList.toggle('avatar-away', destination === 'panel');
      var portals = document.querySelectorAll('.ask-amr-avatar-portal');
      if (portals.length !== 1 && window.console && console.error) console.error('Ask Amr.AI invariant violated: expected one shared avatar portal, found ' + portals.length);
      var to = target.getBoundingClientRect();
      avatarPortal.classList.add('is-transferring');
      avatarPortal.style.transformOrigin = 'top left';
      avatarPortal.style.transform = 'translate(' + (from.left - to.left) + 'px,' + (from.top - to.top) + 'px) scale(' + (from.width / Math.max(to.width, 1)) + ',' + (from.height / Math.max(to.height, 1)) + ')';
      window.requestAnimationFrame(function () {
        window.requestAnimationFrame(function () {
          avatarPortal.style.transition = 'transform 520ms cubic-bezier(.16,.8,.2,1), opacity 260ms ease';
          avatarPortal.style.transform = 'none';
          ghost.style.transition = 'opacity 180ms ease';
          ghost.style.opacity = '0';
        });
      });
      window.setTimeout(function () {
        avatarPortal.classList.remove('is-transferring');
        avatarPortal.style.transition = '';
        avatarPortal.style.transform = '';
        ghost.remove();
        if (window.askAmrAvatar && window.askAmrAvatar.head) {
          try { window.askAmrAvatar.head.resize?.(); } catch (error) {}
        }
      }, 560);
    }

    function loadAvatar() {
      if (avatarStarted) return;
      avatarStarted = true;
      Promise.all([import('/avatar-lab/avatar-controller.js'), import('/avatar-lab/avatar-config.js')]).then(function (modules) {
        var AvatarController = modules[0].AvatarController;
        var settings = modules[1].settingsFromSearch(window.location.search);
        var gaze = modules[1].loadGazeSettings();
        var controller = new AvatarController(avatarPortal, { log: function () {} });
        return controller.load(window.location.origin + '/avatar-lab/avatars/' + settings.avatarUrl, settings.freezeIdle).then(function () {
          var loading = avatarPortal.querySelector('.ask-amr-avatar-loading');
          if (loading) loading.remove();
          avatarPortal.classList.add('is-ready');
          controller.applyGaze(gaze, { preview: false });
          controller.applyLiveSettings(settings);
          window.askAmrAvatar = controller;
          controller.enablePointerTracking();
          controller.setPointerTrackingEnabled(panel.dataset.mode === 'avatar');
          setInterfaceState(panel.dataset.state || 'online');
        });
      }).catch(function (error) {
        if (window.console && console.warn) console.warn('Ask Amr.AI avatar fallback:', error);
        avatarPortal.classList.add('is-fallback');
        var loading = avatarPortal.querySelector('.ask-amr-avatar-loading');
        if (loading) loading.innerHTML = '<span class="ask-amr-fallback-mark">AMR.AI</span><small>INTERFACE ONLINE</small>';
      });
    }

    function recordVoiceUser() {
      var value = liveInputTranscript.trim();
      if (liveInputRendered || !value) return;
      voiceHistory.push({ role: 'user', content: value });
      voiceHistory = voiceHistory.slice(-12);
      saveVoiceHistory(voiceHistory);
      if (panel.dataset.mode === 'avatar') {
        var intro = log.querySelector('.ask-amr-intro');
        if (intro) intro.remove();
        panel.classList.add('has-conversation');
        addMessage('user', value);
      }
      liveInputRendered = true;
    }

    function recordVoiceAssistant(value) {
      value = String(value || '').trim();
      if (!value) return;
      voiceHistory.push({ role: 'assistant', content: value });
      voiceHistory = voiceHistory.slice(-12);
      saveVoiceHistory(voiceHistory);
    }

    function ensureVoiceController() {
      if (voiceController) return Promise.resolve(voiceController);
      return import('/ask-amr-live.js').then(function (module) {
        voiceController = module.createAskAmrVoice({
          turnstileContainer: turnstileHost,
          getAvatar: function () {
            return new Promise(function (resolve) {
              if (window.askAmrAvatar) { resolve(window.askAmrAvatar); return; }
              var attempts = 0;
              var timer = window.setInterval(function () {
                attempts += 1;
                if (window.askAmrAvatar || attempts > 80) {
                  window.clearInterval(timer);
                  resolve(window.askAmrAvatar || null);
                }
              }, 100);
            });
          },
          onState: setInterfaceState,
          onNotice: function (message) { voiceNote.hidden = false; voiceNote.textContent = message; },
          onInput: function (text) {
            if (liveInputRendered) {
              liveInputTranscript = '';
              liveInputRendered = false;
              liveAssistantMessage = null;
            }
            liveInputTranscript += String(text || '');
          },
          onOutput: function (text, complete) {
            recordVoiceUser();
            if (panel.dataset.mode === 'avatar') {
              if (!liveAssistantMessage || !liveAssistantMessage.isConnected) liveAssistantMessage = addMessage('assistant', '');
              updateAssistantMessage(liveAssistantMessage, text, []);
            }
            if (complete) {
              recordVoiceAssistant(text);
              liveAssistantMessage = null;
              liveInputTranscript = '';
              liveInputRendered = false;
            }
          },
          onInterrupted: function (text) {
            recordVoiceUser();
            var interrupted = String(text || '').trim();
            var content = (interrupted ? interrupted + '\n\n' : '') + '[INTERRUPTED]';
            if (panel.dataset.mode === 'avatar') {
              if (!liveAssistantMessage || !liveAssistantMessage.isConnected) liveAssistantMessage = addMessage('assistant', '');
              updateAssistantMessage(liveAssistantMessage, content, []);
            }
            recordVoiceAssistant(content);
            liveAssistantMessage = null;
            liveInputTranscript = '';
            liveInputRendered = false;
          }
        });
        return voiceController;
      });
    }

    function startVoiceInput() {
      voiceNote.hidden = true;
      liveInputTranscript = '';
      liveInputRendered = false;
      liveAssistantMessage = null;
      ensureVoiceController().then(function (voice) {
        return voice.start().then(function (listening) {
          talkButton.innerHTML = listening ? '<span>■</span> STOP LISTENING' : '<span aria-hidden="true">🎙</span> TALK TO AMR.AI';
        });
      }).catch(function () {
        talkButton.innerHTML = '<span aria-hidden="true">🎙</span> RETRY VOICE';
      });
    }

    function addMessage(role, content, sources, scrollMode) {
      var item = create('article', 'ask-amr-message ask-amr-message-' + role);
      var label = create('div', 'ask-amr-message-label', role === 'user' ? 'YOU / QUERY' : 'AMR.AI / RESPONSE');
      var copy = create('div', 'ask-amr-message-copy');
      renderMarkdown(copy, content);
      item.appendChild(label);
      item.appendChild(copy);
      if (sources && sources.length) {
        var sourceWrap = create('div', 'ask-amr-sources');
        sourceWrap.appendChild(create('div', 'ask-amr-source-label', 'SOURCES'));
        sources.forEach(function (source) {
          var href = safeSourceUrl(source && source.url);
          if (!href) return;
          var link = create('a', 'ask-amr-source');
          link.href = href;
          link.addEventListener('click', function () { minimize(); });
          link.appendChild(create('span', '', source.title + (source.page ? ' · p. ' + source.page : '')));
          link.appendChild(create('span', '', source.type === 'project' ? 'VIEW CASE STUDY →' : 'OPEN SOURCE ↗'));
          sourceWrap.appendChild(link);
        });
        if (sourceWrap.children.length > 1) item.appendChild(sourceWrap);
      }
      log.appendChild(item);
      if (scrollMode === 'start') {
        window.requestAnimationFrame(function () {
          window.requestAnimationFrame(function () {
            var reducedMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
            log.scrollTo({
              top: Math.max(0, item.offsetTop - log.offsetTop),
              behavior: reducedMotion ? 'auto' : 'smooth'
            });
          });
        });
      } else {
        log.scrollTop = log.scrollHeight;
      }
      return item;
    }

    function safeSourceUrl(value) {
      try {
        var url = new URL(String(value || ''), window.location.origin);
        return url.protocol === 'https:' && url.hostname === 'alfayoumy.github.io' ? url.href : '';
      } catch (error) {
        return '';
      }
    }

    function updateAssistantMessage(item, content, sources, scrollMode) {
      var copy = item.querySelector('.ask-amr-message-copy');
      copy.textContent = '';
      renderMarkdown(copy, content);
      var existingSources = item.querySelector('.ask-amr-sources');
      if (existingSources) existingSources.remove();
      if (sources && sources.length) {
        var sourceWrap = create('div', 'ask-amr-sources');
        sourceWrap.appendChild(create('div', 'ask-amr-source-label', 'SOURCES'));
        sources.forEach(function (source) {
          var href = safeSourceUrl(source && source.url);
          if (!href) return;
          var link = create('a', 'ask-amr-source');
          link.href = href;
          link.addEventListener('click', function () { minimize(); });
          link.appendChild(create('span', '', source.title + (source.page ? ' · p. ' + source.page : '')));
          link.appendChild(create('span', '', source.type === 'project' ? 'VIEW CASE STUDY →' : 'OPEN SOURCE ↗'));
          sourceWrap.appendChild(link);
        });
        if (sourceWrap.children.length > 1) item.appendChild(sourceWrap);
      }
      if (scrollMode === 'start') {
        window.requestAnimationFrame(function () {
          var reducedMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
          log.scrollTo({ top: Math.max(0, item.offsetTop - log.offsetTop), behavior: reducedMotion ? 'auto' : 'smooth' });
        });
      }
    }

    function startPendingSpinner(item) {
      var copy = item.querySelector('.ask-amr-message-copy');
      var spinner = create('span', 'cli-spinner ask-amr-pending-spinner');
      var phrase = create('span', 'ask-amr-pending-text');
      var accessibleStatus = create('span', 'ask-amr-visually-hidden', 'Searching the portfolio dossier.');
      var remaining = [];
      var last = '';
      var intervalId = null;
      var swapId = null;
      var stopped = false;

      copy.textContent = '';
      spinner.setAttribute('aria-hidden', 'true');
      phrase.setAttribute('aria-hidden', 'true');
      copy.appendChild(spinner);
      copy.appendChild(phrase);
      copy.appendChild(accessibleStatus);

      function nextPhrase() {
        if (!remaining.length) remaining = pendingPhrases.filter(function (value) { return value !== last; });
        var index = Math.floor(Math.random() * remaining.length);
        last = remaining.splice(index, 1)[0];
        return last + '...';
      }

      phrase.textContent = nextPhrase();
      var reducedMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
      if (!reducedMotion) {
        intervalId = window.setInterval(function () {
          phrase.classList.add('is-changing');
          swapId = window.setTimeout(function () {
            if (stopped) return;
            phrase.textContent = nextPhrase();
            phrase.classList.remove('is-changing');
          }, 180);
        }, 1900);
      }

      return function () {
        stopped = true;
        if (intervalId !== null) window.clearInterval(intervalId);
        if (swapId !== null) window.clearTimeout(swapId);
      };
    }

    function renderCurrentMode() {
      log.innerHTML = '';
      var activeHistory = panel.dataset.mode === 'chat' ? history : voiceHistory;
      panel.classList.toggle('has-conversation', activeHistory.length > 0);
      if (!activeHistory.length) {
        var intro = create('section', 'ask-amr-intro');
        intro.innerHTML = panel.dataset.mode === 'chat'
          ? '<div class="ask-amr-intro-kicker">GROUNDED PORTFOLIO CHAT</div><h3>Ask about Amr’s work.</h3><p>Ask by text about Amr’s shipped systems, GenAI projects, research, experience, or technical decisions. Answers stream with portfolio sources.</p>'
          : '<div class="ask-amr-intro-kicker">LIVE PORTFOLIO INTERFACE</div><h3>Ask about Amr’s work.</h3><p>Ask by voice about Amr’s shipped systems, GenAI projects, research, experience, or technical decisions.</p>';
        log.appendChild(intro);
      }
      activeHistory.forEach(function (message) { addMessage(message.role, message.content, message.sources); });
    }

    function showWelcome() { renderCurrentMode(); }

    function askQuestion(question) {
      var value = String(question || '').trim();
      if (!value || busy) return;
      setMode('chat', { focus: false });
      if (!panel.classList.contains('is-home')) open(false);
      beginRequest(value);
    }

    suggestions.forEach(function (question) {
      var button = create('button', '', question);
      button.type = 'button';
      button.addEventListener('click', function () { askQuestion(question); });
      suggestionBox.appendChild(button);
    });

    function setTopicsMinimized(minimized) {
      var isMin = Boolean(minimized);
      if (topicsSection) topicsSection.classList.toggle('is-minimized', isMin);
      if (topicsToggle) {
        topicsToggle.setAttribute('aria-expanded', isMin ? 'false' : 'true');
        topicsToggle.setAttribute('aria-label', isMin ? 'Expand suggested topics' : 'Minimize suggested topics');
        topicsToggle.textContent = isMin ? '+' : '−';
      }
    }

    if (topicsToggle) {
      topicsToggle.addEventListener('click', function () {
        var willMinimize = !topicsSection.classList.contains('is-minimized');
        setTopicsMinimized(willMinimize);
      });
    }



    function beginRequest(message) {
      var context = history.slice(-6).map(function (item) {
        return { role: item.role, content: String(item.content || '').slice(0, HISTORY_CONTENT_LIMIT) };
      });
      input.value = '';
      input.style.height = 'auto';
      if (homeInput) homeInput.value = '';
      var intro = log.querySelector('.ask-amr-intro');
      if (intro) intro.remove();
      addMessage('user', message);
      history.push({ role: 'user', content: message });
      busy = true;
      panel.classList.add('has-conversation');
      setInterfaceState('thinking');
      submit.disabled = true;
      var pending = addMessage('assistant', '');
      pending.classList.add('is-pending');
      pendingRequest = {
        message: message,
        context: context,
        pending: pending,
        stopSpinner: startPendingSpinner(pending),
        sent: false
      };
      if (token) dispatchPendingRequest();
      else setupTurnstile(true);
    }

    function dispatchPendingRequest() {
      if (!pendingRequest || pendingRequest.sent || !token) return;
      var request = pendingRequest;
      var requestToken = token;
      request.sent = true;
      token = '';

      fetch(endpointUrl(config.endpoint), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Accept': 'application/x-ndjson, application/json' },
        body: JSON.stringify({
          message: request.message,
          history: request.context,
          turnstileToken: requestToken,
          sessionId: sessionId(),
          corpusVersion: config.corpusVersion || ''
        })
      }).then(function (response) {
        if (!response.ok) {
          return response.json().catch(function () { return {}; }).then(function (body) {
            throw new Error(body.error || 'A Coriolis storm disrupted the spice trance. Please try your question again shortly.');
          });
        }
        if ((response.headers.get('Content-Type') || '').indexOf('application/json') !== -1) {
          return response.json().then(function (body) {
            if (!body.answer) throw new Error(body.error || 'A Coriolis storm disrupted the spice trance. Please try your question again shortly.');
            return body;
          });
        }
        if (!response.body) throw new Error('A Coriolis storm disrupted the spice trance. Please try your question again shortly.');
        var reader = response.body.getReader();
        var decoder = new TextDecoder();
        var buffer = '';
        var answer = '';
        var completed = null;
        var showingAnswer = false;
        var previewFrame = null;

        function schedulePreview() {
          if (previewFrame !== null) return;
          previewFrame = window.requestAnimationFrame(function () {
            previewFrame = null;
            updateAssistantMessage(request.pending, answer, []);
          });
        }

        function resetPreview() {
          if (previewFrame !== null) {
            window.cancelAnimationFrame(previewFrame);
            previewFrame = null;
          }
          request.stopSpinner();
          request.pending.classList.add('is-pending');
          answer = '';
          showingAnswer = false;
          request.stopSpinner = startPendingSpinner(request.pending);
        }

        function handleEvent(event) {
          if (event.type === 'retry') {
            resetPreview();
            return;
          }
          if (event.type === 'delta') {
            if (!showingAnswer) {
              request.stopSpinner();
              request.pending.classList.remove('is-pending');
              showingAnswer = true;
            }
            answer += String(event.text || '');
            schedulePreview();
            return;
          }
          if (event.type === 'complete') {
            if (previewFrame !== null) {
              window.cancelAnimationFrame(previewFrame);
              previewFrame = null;
            }
            completed = { answer: String(event.answer || ''), sources: Array.isArray(event.sources) ? event.sources : [] };
            return;
          }
          if (event.type === 'error') throw new Error(event.error || 'A Coriolis storm disrupted the spice trance. Please try your question again shortly.');
        }

        function consumeLines(flush) {
          var newline;
          while ((newline = buffer.indexOf('\n')) !== -1) {
            var line = buffer.slice(0, newline).trim();
            buffer = buffer.slice(newline + 1);
            if (line) handleEvent(JSON.parse(line));
          }
          if (flush && buffer.trim()) {
            handleEvent(JSON.parse(buffer));
            buffer = '';
          }
        }

        function read() {
          return reader.read().then(function (chunk) {
            if (chunk.done) {
              buffer += decoder.decode();
              consumeLines(true);
              if (!completed || !completed.answer) throw new Error('A Coriolis storm disrupted the spice trance. Please try your question again shortly.');
              return completed;
            }
            buffer += decoder.decode(chunk.value, { stream: true });
            consumeLines(false);
            return read();
          });
        }
        return read();
      }).then(function (result) {
        request.stopSpinner();
        request.pending.classList.remove('is-pending');
        updateAssistantMessage(request.pending, result.answer, result.sources || [], 'start');
        history.push({ role: 'assistant', content: result.answer, sources: result.sources || [] });
        history = history.slice(-6);
        saveHistory(history);
      }).catch(function (error) {
        request.stopSpinner();
        request.pending.remove();
        if (history.length && history[history.length - 1].role === 'user' && history[history.length - 1].content === request.message) history.pop();
        addMessage('assistant', error.message || 'A Coriolis storm disrupted the spice trance. Please try your question again shortly.');
      }).finally(function () {
        if (pendingRequest === request) pendingRequest = null;
        busy = false;
        setInterfaceState('online');
        token = '';
        if (widgetId !== null && window.turnstile) window.turnstile.reset(widgetId);
        submit.disabled = false;
      });
    }

    function failQueuedRequest(message) {
      if (!pendingRequest || pendingRequest.sent) return;
      var request = pendingRequest;
      pendingRequest = null;
      request.stopSpinner();
      request.pending.remove();
      if (history.length && history[history.length - 1].role === 'user' && history[history.length - 1].content === request.message) history.pop();
      addMessage('assistant', message);
      busy = false;
      submit.disabled = false;
    }

    function setupTurnstile(refresh) {
      if (isLocalDevelopmentEndpoint(config.endpoint)) {
        token = 'local-dev-bypass';
        submit.disabled = busy;
        dispatchPendingRequest();
        return Promise.resolve();
      }
      return loadTurnstile().then(function () {
        if (widgetId !== null) {
          if (refresh && !token) {
            window.turnstile.reset(widgetId);
            window.turnstile.execute(widgetId);
          }
          return;
        }
        widgetId = window.turnstile.render(verify, {
          sitekey: config.turnstileSiteKey,
          theme: document.documentElement.getAttribute('data-theme') === 'dark' ? 'dark' : 'light',
          size: 'flexible',
          appearance: 'interaction-only',
          execution: 'execute',
          retry: 'auto',
          'retry-interval': 3000,
          'refresh-expired': 'auto',
          'refresh-timeout': 'auto',
          callback: function (value) {
            token = value;
            submit.disabled = busy;
            dispatchPendingRequest();
          },
          'expired-callback': function () {
            token = '';
            submit.disabled = busy;
          },
          'timeout-callback': function () {
            token = '';
            submit.disabled = busy;
          },
          'error-callback': function () {
            token = '';
            submit.disabled = busy;
            return true;
          }
        });
        if (refresh) window.turnstile.execute(widgetId);
      }).catch(function () {
        failQueuedRequest('The Bene Gesserit lost the verification signal. Check your connection, then try the question again.');
      });
    }

    function dockPanelInFloatingWindow() {
      if (!panel.classList.contains('is-home')) return;
      homeTerminal.hidden = false;
      backdrop.appendChild(panel);
      panel.classList.remove('is-home');
      panel.setAttribute('aria-modal', 'true');
      avatarPortal.dataset.owner = panel.dataset.mode === 'avatar' ? 'panel' : 'parked';
      document.documentElement.dataset.askAmrAvatarOwner = avatarPortal.dataset.owner;
    }

    function dockPanelOnHomepage() {
      if (!homeMount || panel.classList.contains('is-home')) return;
      homeMount.appendChild(panel);
      panel.classList.add('is-home');
      panel.setAttribute('aria-modal', 'false');
      homeTerminal.hidden = true;
      avatarPortal.dataset.owner = panel.dataset.mode === 'avatar' ? 'home' : 'parked';
      document.documentElement.dataset.askAmrAvatarOwner = avatarPortal.dataset.owner;
      panel.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 220, easing: 'ease-out' });
      refreshAvatarViewport();
    }

    function animateWindowFromLauncher(opening) {
      var reduced = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
      if (reduced) return Promise.resolve();
      var panelRect = panel.getBoundingClientRect();
      var launcherRect = launcher.getBoundingClientRect();
      var scaleX = Math.max(.08, Math.min(1, launcherRect.width / Math.max(panelRect.width, 1)));
      var scaleY = Math.max(.05, Math.min(1, launcherRect.height / Math.max(panelRect.height, 1)));
      var translateX = launcherRect.left - panelRect.left;
      var translateY = launcherRect.top - panelRect.top;
      var collapsed = {
        transform: 'translate(' + translateX + 'px,' + translateY + 'px) scale(' + scaleX + ',' + scaleY + ')',
        transformOrigin: 'top left',
        opacity: .18,
        clipPath: 'inset(0 round 3px)'
      };
      var expanded = { transform: 'none', transformOrigin: 'top left', opacity: 1, clipPath: 'inset(0 round 0)' };
      panel.classList.add('is-window-animating');
      var animation = panel.animate(opening ? [collapsed, expanded] : [expanded, collapsed], {
        duration: opening ? 430 : 340,
        easing: opening ? 'cubic-bezier(.16,.8,.2,1)' : 'cubic-bezier(.4,0,.7,.2)',
        fill: 'both'
      });
      return animation.finished.catch(function () {}).then(function () {
        animation.cancel();
        panel.classList.remove('is-window-animating');
      });
    }

    function open(prepareVerification) {
      if (windowAnimationActive || backdrop.classList.contains('is-open')) return;
      windowAnimationActive = true;
      previousFocus = document.activeElement;
      dockPanelInFloatingWindow();
      panel.classList.add('is-window-animating');
      backdrop.classList.add('is-open');
      setBackgroundInert(backdrop, true);
      launcher.setAttribute('aria-expanded', 'true');
      document.body.classList.add('ask-amr-open');
      window.requestAnimationFrame(function () {
        animateWindowFromLauncher(true).finally(function () { windowAnimationActive = false; });
      });
      if (prepareVerification !== false) setupTurnstile(false);
      loadAvatar();
      if (panel.dataset.mode === 'chat') window.setTimeout(function () { input.focus(); }, 440);
    }

    function minimize() {
      if (windowAnimationActive || !backdrop.classList.contains('is-open')) return;
      windowAnimationActive = true;
      animateWindowFromLauncher(false).finally(function () {
        backdrop.classList.remove('is-open');
        launcher.setAttribute('aria-expanded', 'false');
        document.body.classList.remove('ask-amr-open');
        setBackgroundInert(backdrop, false);
        dockPanelOnHomepage();
        windowAnimationActive = false;
        if (previousFocus && previousFocus.focus) previousFocus.focus();
      });
    }

    function closeSession() {
      if (voiceController) voiceController.close().catch(function () {});
      talkButton.innerHTML = '<span aria-hidden="true">🎙</span> TALK TO AMR.AI';
      setInterfaceState('online');
      minimize();
    }

    launcher.addEventListener('click', function () { open(); });
    closeButton.addEventListener('click', closeSession);
    minimizeButton.addEventListener('click', minimize);
    typeButton.addEventListener('click', function () { setMode('chat'); });
    talkButton.addEventListener('click', function () {
      if (panel.dataset.mode === 'chat') { setMode('avatar', { focus: false }); return; }
      if (voiceController && voiceController.listening) {
        voiceController.stop().then(function () { talkButton.innerHTML = '<span aria-hidden="true">🎙</span> TALK TO AMR.AI'; });
        return;
      }
      startVoiceInput();
    });
    backdrop.addEventListener('mousedown', function (event) { if (event.target === backdrop) minimize(); });
    document.addEventListener('keydown', function (event) {
      if (!backdrop.classList.contains('is-open')) return;
      if (event.key === 'Escape') {
        minimize();
        return;
      }
      if (event.key !== 'Tab') return;
      var focusable = focusableElements(panel);
      if (!focusable.length) return;
      var first = focusable[0];
      var last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    });
    input.addEventListener('keydown', function (event) {
      if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); form.requestSubmit(); }
    });
    input.addEventListener('input', function () {
      input.style.height = 'auto';
      input.style.height = Math.min(input.scrollHeight, 112) + 'px';
    });
    function clearCurrentMode() {
      if (panel.dataset.mode === 'chat') {
        history = [];
        saveHistory(history);
      } else {
        voiceHistory = [];
        saveVoiceHistory(voiceHistory);
        liveInputTranscript = '';
        liveInputRendered = false;
        liveAssistantMessage = null;
        if (voiceController) voiceController.close().catch(function () {});
        setInterfaceState('online');
      }
      renderCurrentMode();
    }
    clearButton.addEventListener('click', clearCurrentMode);
    clearModeButton.addEventListener('click', clearCurrentMode);

    form.addEventListener('submit', function (event) {
      event.preventDefault();
      var message = input.value.trim();
      askQuestion(message);
    });

    submit.disabled = false;
    setMode(initialMode, { focus: false });
  }

  var configPromise = window.portfolioConfigPromise || (window.portfolioConfigPromise = fetch('/chat-config.json', { cache: 'no-store' })
    .then(function (response) { return response.ok ? response.json() : {}; })
    .catch(function () { return {}; }));
  configPromise
    .then(initialize)
    .catch(function () {});
})();
