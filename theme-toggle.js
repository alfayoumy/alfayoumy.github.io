(function () {
  var riseSelector = [
    'main > section',
    'main > article > .shell > header',
    'main > article > .shell > section',
    '.project-panel',
    '.home-proof-card',
    '.impact-card',
    '.career-row',
    '.faq-row',
    '.home-hire-block'
  ].join(',');

  function initRiseScroll() {
    var targets = Array.prototype.slice.call(document.querySelectorAll(riseSelector));
    if (!targets.length) return;

    var styleId = 'rise-scroll-styles';
    if (!document.getElementById(styleId)) {
      var style = document.createElement('style');
      style.id = styleId;
      style.textContent = [
        '.rise-on-scroll{opacity:0;transform:translate3d(0,32px,0);transition:opacity 720ms cubic-bezier(.2,.7,.2,1),transform 720ms cubic-bezier(.2,.7,.2,1);transition-delay:var(--rise-delay,0ms);will-change:opacity,transform}',
        '.rise-on-scroll.is-visible{opacity:1;transform:translate3d(0,0,0)}',
        '@media (prefers-reduced-motion:reduce){.rise-on-scroll{opacity:1;transform:none;transition:none;will-change:auto}}'
      ].join('');
      document.head.appendChild(style);
    }

    if (window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      targets.forEach(function (target) {
        target.classList.add('rise-on-scroll', 'is-visible');
      });
      return;
    }

    targets.forEach(function (target, index) {
      target.classList.add('rise-on-scroll');
      target.style.setProperty('--rise-delay', Math.min(index % 4, 3) * 70 + 'ms');
    });

    if (!('IntersectionObserver' in window)) {
      targets.forEach(function (target) {
        target.classList.add('is-visible');
      });
      return;
    }

    var observer = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (!entry.isIntersecting) return;
        entry.target.classList.add('is-visible');
        observer.unobserve(entry.target);
      });
    }, {
      rootMargin: '0px 0px -12% 0px',
      threshold: 0.14
    });

    targets.forEach(function (target) {
      observer.observe(target);
    });
  }

  function initThinkingTitle() {
    var rotators = Array.prototype.slice.call(document.querySelectorAll('.thinking-title-rotator'));
    if (!rotators.length) return;

    rotators.forEach(function (rotator) {
      var titles = Array.prototype.slice.call(rotator.querySelectorAll('span'));
      if (titles.length < 2) {
        if (titles[0]) titles[0].classList.add('is-active');
        return;
      }

      var reduceMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
      if (reduceMotion) {
        titles[0].classList.add('is-active');
        return;
      }

      var activeIndex = 0;
      titles[activeIndex].classList.add('is-active');
      rotator.classList.add('is-cycling');

      window.setInterval(function () {
        titles[activeIndex].classList.remove('is-active');
        activeIndex = (activeIndex + 1) % titles.length;
        titles[activeIndex].classList.add('is-active');
      }, 2400);
    });
  }

  function prefersReducedMotion() {
    return window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  }

  function focusableElements(container) {
    return Array.prototype.slice.call(container.querySelectorAll([
      'a[href]',
      'button:not([disabled])',
      'input:not([disabled])',
      'textarea:not([disabled])',
      'select:not([disabled])',
      '[tabindex]:not([tabindex="-1"])'
    ].join(','))).filter(function (element) {
      return !element.hidden && element.getAttribute('aria-hidden') !== 'true';
    });
  }

  function setBackgroundInert(exception, inert) {
    Array.prototype.slice.call(document.body.children).forEach(function (element) {
      if (element === exception) return;
      if (inert) element.setAttribute('inert', '');
      else element.removeAttribute('inert');
    });
  }

  function initMobileNavigation() {
    var headerBar = document.querySelector('body > div.fixed.top-0');
    var headerShell = headerBar && headerBar.querySelector('.shell');
    if (!headerShell || document.getElementById('mobileNavToggle')) return;

    var desktopSectionNav = headerShell.querySelector('nav[aria-label="Section navigation"]');
    var mobileSectionNav = Array.prototype.slice.call(document.querySelectorAll('body > nav[aria-label="Section navigation"]')).find(function (nav) {
      return nav !== desktopSectionNav;
    });
    if (mobileSectionNav) mobileSectionNav.classList.add('mobile-nav-legacy');

    var toggle = document.createElement('button');
    toggle.type = 'button';
    toggle.id = 'mobileNavToggle';
    toggle.className = 'mobile-nav-toggle';
    toggle.setAttribute('aria-label', 'Open navigation menu');
    toggle.setAttribute('aria-controls', 'mobileNavDrawer');
    toggle.setAttribute('aria-expanded', 'false');
    toggle.innerHTML = '<span></span><span></span><span></span>';
    headerShell.insertBefore(toggle, headerShell.firstChild);

    var siteLinks = Array.prototype.slice.call(headerShell.querySelectorAll(':scope > a'));
    var sectionLinks = desktopSectionNav ? Array.prototype.slice.call(desktopSectionNav.querySelectorAll('a')) : [];
    var backdrop = document.createElement('div');
    backdrop.className = 'mobile-nav-backdrop';
    backdrop.id = 'mobileNavBackdrop';
    backdrop.hidden = true;

    var siteMarkup = siteLinks.map(function (link, index) {
      var label = link.textContent.replace(/^\s*\/\s*/, '').trim();
      return '<a href="' + link.getAttribute('href') + '"' + (link.target === '_blank' ? ' target="_blank" rel="noopener noreferrer"' : '') + '><span>0' + (index + 1) + '</span>' + label + '</a>';
    }).join('');
    var sectionMarkup = sectionLinks.map(function (link) {
      var target = document.querySelector(link.getAttribute('href'));
      var label = link.getAttribute('data-nav-label') || (target ? sectionLabel(target) : 'Section');
      return '<a href="' + link.getAttribute('href') + '"><span>' + link.textContent.trim() + '</span>' + label + '</a>';
    }).join('');

    backdrop.innerHTML = [
      '<aside class="mobile-nav-drawer" id="mobileNavDrawer" role="dialog" aria-modal="true" aria-label="Mobile navigation" aria-hidden="true">',
      '<div class="mobile-nav-head"><span>// NAVIGATION</span><button type="button" class="mobile-nav-close" aria-label="Close navigation menu">&times;</button></div>',
      '<nav aria-label="Primary navigation"><p>Site</p>' + siteMarkup + '</nav>',
      sectionMarkup ? '<nav aria-label="On this page"><p>On this page</p>' + sectionMarkup + '</nav>' : '',
      '</aside>'
    ].join('');
    document.body.appendChild(backdrop);

    var drawer = backdrop.querySelector('.mobile-nav-drawer');
    var closeButton = backdrop.querySelector('.mobile-nav-close');
    var lastFocused = null;

    function openMenu() {
      lastFocused = document.activeElement;
      backdrop.hidden = false;
      setBackgroundInert(backdrop, true);
      window.requestAnimationFrame(function () {
        backdrop.classList.add('is-open');
        drawer.setAttribute('aria-hidden', 'false');
        toggle.setAttribute('aria-expanded', 'true');
        toggle.setAttribute('aria-label', 'Close navigation menu');
        document.body.classList.add('mobile-nav-open');
        closeButton.focus();
      });
    }

    function closeMenu() {
      backdrop.classList.remove('is-open');
      drawer.setAttribute('aria-hidden', 'true');
      toggle.setAttribute('aria-expanded', 'false');
      toggle.setAttribute('aria-label', 'Open navigation menu');
      document.body.classList.remove('mobile-nav-open');
      setBackgroundInert(backdrop, false);
      window.setTimeout(function () {
        backdrop.hidden = true;
      }, prefersReducedMotion() ? 0 : 240);
      if (lastFocused && lastFocused.focus) lastFocused.focus();
    }

    toggle.addEventListener('click', function () {
      if (toggle.getAttribute('aria-expanded') === 'true') closeMenu();
      else openMenu();
    });
    closeButton.addEventListener('click', closeMenu);
    backdrop.addEventListener('click', function (event) {
      if (event.target === backdrop || event.target.closest('a')) closeMenu();
    });
    document.addEventListener('keydown', function (event) {
      if (toggle.getAttribute('aria-expanded') !== 'true') return;
      if (event.key === 'Escape') {
        closeMenu();
        return;
      }
      if (event.key !== 'Tab') return;
      var focusable = focusableElements(drawer);
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
    window.addEventListener('resize', function () {
      if (window.innerWidth >= 768 && toggle.getAttribute('aria-expanded') === 'true') closeMenu();
    });
  }

  function initCursorTrace() {
    if (prefersReducedMotion()) return;
    if (!window.matchMedia || !window.matchMedia('(hover: hover) and (pointer: fine)').matches) return;

    var trace = document.createElement('div');
    trace.className = 'cursor-trace';
    trace.setAttribute('aria-hidden', 'true');
    trace.innerHTML = '<div class="cursor-invert"></div><div class="cursor-dot"></div>';
    document.body.appendChild(trace);

    var cardSelector = [
      '.project-panel',
      '#work a[href^="/projects/"]',
      '.home-proof-card',
      '.impact-card',
      '.home-hero-aside',
      '.home-hire-block',
      '.project-domain-panel',
      'main [style*="border:1px solid var(--rule)"]:not(#licenses-box)',
      '#licenses-box a'
    ].join(',');
    var tokens = ['fn()', '{}', '=>', '</>', 'git', 'run', 'sql', 'ai'];
    var lastX = -100;
    var lastY = -100;
    var targetX = -100;
    var targetY = -100;
    var lastTokenAt = 0;
    var tokenIndex = 0;
    var frameId = null;

    function setTracePosition(x, y) {
      trace.style.setProperty('--cursor-x', x + 'px');
      trace.style.setProperty('--cursor-y', y + 'px');
    }

    function drawToken(x, y) {
      var now = performance.now();
      var distance = Math.hypot(x - lastX, y - lastY);
      if (now - lastTokenAt < 72 || distance < 18) return;

      lastTokenAt = now;
      var token = document.createElement('span');
      token.className = 'cursor-code-token';
      token.textContent = tokens[tokenIndex % tokens.length];
      tokenIndex += 1;
      token.style.setProperty('--token-x', x + 10 + 'px');
      token.style.setProperty('--token-y', y - 12 + 'px');
      token.style.setProperty('--token-drift-x', (Math.random() * 18 - 9).toFixed(2) + 'px');
      token.style.setProperty('--token-drift-y', (-16 - Math.random() * 16).toFixed(2) + 'px');
      document.body.appendChild(token);
      window.setTimeout(function () {
        token.remove();
      }, 820);
    }

    function animate() {
      lastX += (targetX - lastX) * 0.36;
      lastY += (targetY - lastY) * 0.36;
      setTracePosition(lastX, lastY);
      frameId = window.requestAnimationFrame(animate);
    }

    function stopTrace() {
      trace.classList.remove('is-active', 'is-over-card');
      if (frameId) {
        window.cancelAnimationFrame(frameId);
        frameId = null;
      }
    }

    document.addEventListener('pointermove', function (event) {
      var hoveredCard = event.target && event.target.closest ? event.target.closest(cardSelector) : null;
      targetX = event.clientX;
      targetY = event.clientY;
      trace.classList.add('is-active');
      trace.classList.toggle('is-over-card', Boolean(hoveredCard));
      drawToken(targetX, targetY);

      if (!frameId) {
        lastX = targetX;
        lastY = targetY;
        setTracePosition(lastX, lastY);
        frameId = window.requestAnimationFrame(animate);
      }
    }, { passive: true });

    document.addEventListener('pointerleave', stopTrace);

    window.addEventListener('blur', stopTrace);
  }

  function initBootSequence() {
    if (!document.getElementById('field') || prefersReducedMotion()) return;

    var boot = document.createElement('div');
    boot.className = 'boot-sequence';
    boot.setAttribute('aria-hidden', 'true');
    boot.innerHTML = [
      '<div class="boot-terminal">',
      '<div class="boot-topline"><span>amr@portfolio ~ %</span><span>BOOT</span></div>',
      '<div class="boot-line"><span>$</span> loading profile dossier</div>',
      '<div class="boot-line"><span>$</span> indexing production case studies</div>',
      '<div class="boot-line"><span>$</span> mapping AI / data / automation signals</div>',
      '<div class="boot-line boot-ready"><span>status:</span> open to build</div>',
      '</div>'
    ].join('');

    document.body.appendChild(boot);
    window.setTimeout(function () {
      boot.classList.add('is-complete');
    }, 2300);
    window.setTimeout(function () {
      boot.remove();
    }, 3100);
  }

  var projectPreviewData = {
    'ead-mynd': {
      kind: 'avatar',
      label: 'AVATAR PIPELINE',
      metric: '3,700+ interactions',
      detail: 'voice -> rag -> ssml -> viseme',
      command: 'run avatar.agent --lang ar,en',
      steps: ['stt', 'rag', 'qa', 'ssml', 'viseme'],
      domainCommand: 'trace avatar.turn',
      domainLines: ['speech captured in Arabic / English', 'memory-aware retrieval selects source pack', 'speech agent emits SSML + lip-sync events'],
      domainNodes: ['voice', 'memory', 'rag', 'tts', 'avatar']
    },
    'ead-soer': {
      kind: 'report',
      label: 'REPORT ENGINE',
      metric: 'multi-agent drafting',
      detail: 'evidence -> agents -> reviewed chapters',
      command: 'run report.agents --evidence locked',
      steps: ['ingest', 'extract', 'draft', 'review', 'publish'],
      domainCommand: 'assemble environment.report',
      domainLines: ['structured indicators join document evidence', 'specialized agents draft report sections', 'review workflow keeps institutional control'],
      domainNodes: ['data', 'docs', 'agents', 'portal', 'report']
    },
    'mobily-cvm': {
      kind: 'migration',
      label: 'MIGRATION TRACE',
      metric: 'SAS -> Dataiku',
      detail: 'inventory -> refactor -> validate',
      command: 'migrate model.flow --target dataiku',
      steps: ['audit', 'map', 'port', 'test', 'release'],
      domainCommand: 'translate sas.campaigns',
      domainLines: ['legacy flows are inventoried and grouped', 'logic is rebuilt into governed DSS pipelines', 'outputs are validated before release'],
      domainNodes: ['sas', 'logic', 'dss', 'qa', 'ops']
    },
    'bm-ad': {
      kind: 'fraud',
      label: 'AML SIGNAL',
      metric: '12 sub-models',
      detail: 'score -> explain -> alert',
      command: 'score aml.behavior --cohort routed',
      steps: ['cohort', 'model', 'score', 'explain', 'alert'],
      domainCommand: 'triage anomaly.batch',
      domainLines: ['customers route into behavior cohorts', 'three anomaly lenses score each segment', 'top drivers become investigator narratives'],
      domainNodes: ['kyc', 'txn', 'cohort', 'risk', 'aml']
    },
    'crdb-fraud': {
      kind: 'fraud',
      label: 'FRAUD STREAM',
      metric: '80M+ monthly txns',
      detail: 'events -> features -> risk band',
      command: 'score fraud.txn --channels all',
      steps: ['txn', 'feature', 'anomaly', 'rank', 'case'],
      domainCommand: 'flatten transaction.ledger',
      domainLines: ['multi-channel activity becomes model-ready features', 'lineage and labels stay separated', 'risk bands feed governed review'],
      domainNodes: ['cards', 'mobile', 'atm', 'score', 'case']
    },
    'payments-risk': {
      kind: 'fraud',
      label: 'RISK GATE',
      metric: 'real-time scoring',
      detail: 'onboard -> monitor -> suppress',
      command: 'gate merchant.risk --realtime',
      steps: ['onboard', 'identity', 'behavior', 'risk', 'suppress'],
      domainCommand: 'watch payment.onboarding',
      domainLines: ['new entities receive instant risk checks', 'behavior changes update fraud pressure', 'suppression logic keeps alerts usable'],
      domainNodes: ['id', 'device', 'txn', 'risk', 'ops']
    },
    'swat': {
      kind: 'telemetry',
      label: 'TELEMETRY',
      metric: 'industrial anomaly',
      detail: 'sensors -> sequence -> anomaly',
      command: 'detect process.anomaly --sequence',
      steps: ['sensor', 'window', 'model', 'drift', 'alarm'],
      domainCommand: 'monitor plant.telemetry',
      domainLines: ['industrial signals are windowed over time', 'sequence model watches abnormal transitions', 'alerts map back to process context'],
      domainNodes: ['flow', 'level', 'valve', 'model', 'alarm']
    },
    'har': {
      kind: 'sensor',
      label: 'SENSOR WINDOW',
      metric: '98.95% F1',
      detail: 'accelerometer -> cnn-lstm -> activity',
      command: 'classify motion.window --live',
      steps: ['imu', 'window', 'cnn', 'lstm', 'activity'],
      domainCommand: 'stream wearable.motion',
      domainLines: ['accelerometer windows enter the model loop', 'cnn-lstm extracts local and temporal patterns', 'dashboard receives live activity state'],
      domainNodes: ['x', 'y', 'z', 'model', 'state']
    },
    'ips': {
      kind: 'position',
      label: 'BLE TRACE',
      metric: 'real-time location',
      detail: 'beacons -> model -> map coordinate',
      command: 'locate user.ble --floor live',
      steps: ['rssi', 'filter', 'model', 'x/y', 'map'],
      domainCommand: 'resolve indoor.coordinate',
      domainLines: ['beacon RSSI readings are cleaned and aligned', 'model estimates position from signal patterns', 'location state updates the monitoring view'],
      domainNodes: ['b1', 'b2', 'b3', 'x/y', 'map']
    },
    'covid': {
      kind: 'imaging',
      label: 'IMAGING PIPELINE',
      metric: 'CT + X-ray scans',
      detail: 'scans -> extraction -> ensemble',
      command: 'classify.imaging --data chest',
      steps: ['scans', 'extract', 'ensemble', 'vote', 'diagnose'],
      domainCommand: 'diagnose chest.scan',
      domainLines: ['chest CT & X-ray inputs are loaded', 'feature extraction pulls spatial patterns', 'ensemble model resolves final classification'],
      domainNodes: ['ct', 'xray', 'cnn', 'vote', 'diag']
    },
    'harips': {
      kind: 'integration',
      label: 'INTEGRATED SIGNAL',
      metric: 'activity + position',
      detail: 'sensors + beacons -> cnn-lstm + RF -> alert',
      command: 'run harips.monitor --live',
      steps: ['imu', 'rssi', 'har', 'ips', 'alert'],
      domainCommand: 'track user.state',
      domainLines: ['motion and beacon signals are stream-captured', 'dual models evaluate activity and position', 'integrated logic triggers caregiver alerts'],
      domainNodes: ['imu', 'ble', 'har', 'ips', 'alert']
    }
  };

  function slugFromHref(href) {
    var match = (href || '').match(/\/projects\/([^/]+)\//);
    return match ? match[1] : null;
  }

  function buildCasePreview(data) {
    var steps = data.steps.map(function (step, index) {
      return '<span style="--i:' + index + '">' + step + '</span>';
    }).join('');

    return [
      '<div class="case-preview case-preview-' + data.kind + '">',
      '<div class="case-preview-head"><span>// ' + data.label + '</span><span>AGENT TRACE</span></div>',
      '<div class="case-command"><span>$</span> ' + data.command + '</div>',
      '<div class="case-agent-flow" aria-hidden="true">' + steps + '</div>',
      '<div class="case-preview-detail">' + data.detail + '</div>',
      '</div>'
    ].join('');
  }

  function initProjectPreviews() {
    var cards = Array.prototype.slice.call(document.querySelectorAll('.project-panel, #work a[href^="/projects/"]'));
    if (!cards.length) return;

    cards.forEach(function (card) {
      if (card.querySelector('.case-preview')) return;
      var slug = slugFromHref(card.getAttribute('href'));
      var data = projectPreviewData[slug];
      if (!data) return;

      card.classList.add('has-case-preview', 'case-kind-' + data.kind);
      card.insertAdjacentHTML('beforeend', buildCasePreview(data));
    });
  }

  function sectionLabel(section) {
    if (!section) return 'SECTION';
    if (section.id === 'summary') return 'SUMMARY';
    var signal = section.querySelector ? section.querySelector('.caption-signal') : null;
    var heading = section.querySelector ? section.querySelector('h1, h2, h3') : null;
    var text = '';

    if (signal) text = signal.textContent;
    if (!text && heading) text = heading.textContent;
    if (!text && section.tagName && /^H[1-6]$/i.test(section.tagName)) text = section.textContent;
    if (!text) text = section.id || 'SECTION';

    return text
      .replace(/\s+/g, ' ')
      .replace(/\/\/\s*/g, '')
      .replace(/—/g, '-')
      .trim()
      .slice(0, 34);
  }

  function initFieldRecorder() {
    var sections = Array.prototype.slice.call(document.querySelectorAll('main section[id], main header[id]'));
    if (sections.length < 2) return;

    var rail = document.createElement('aside');
    rail.className = 'field-recorder';
    rail.setAttribute('aria-hidden', 'true');
    rail.innerHTML = [
      '<div class="field-recorder-label">FIELD RECORDER</div>',
      '<div class="field-recorder-track"><span></span></div>',
      '<div class="field-recorder-current">§01</div>',
      '<div class="field-recorder-title">' + sectionLabel(sections[0]) + '</div>'
    ].join('');
    document.body.appendChild(rail);

    var current = rail.querySelector('.field-recorder-current');
    var title = rail.querySelector('.field-recorder-title');
    var progress = rail.querySelector('.field-recorder-track span');

    function setActive(section) {
      var index = sections.indexOf(section);
      if (index < 0) return;
      current.textContent = '§' + String(index + 1).padStart(2, '0');
      title.textContent = sectionLabel(section);
    }

    function updateProgress() {
      var doc = document.documentElement;
      var max = Math.max(1, doc.scrollHeight - window.innerHeight);
      progress.style.transform = 'scaleY(' + Math.min(1, Math.max(0, window.scrollY / max)) + ')';
    }

    if ('IntersectionObserver' in window) {
      var observer = new IntersectionObserver(function (entries) {
        entries.forEach(function (entry) {
          if (entry.isIntersecting) setActive(entry.target);
        });
      }, {
        rootMargin: '-35% 0px -55% 0px',
        threshold: 0
      });
      sections.forEach(function (section) {
        observer.observe(section);
      });
    }

    updateProgress();
    window.addEventListener('scroll', updateProgress, { passive: true });
  }

  function initCareerTrace() {
    var stack = document.querySelector('.career-stack');
    if (!stack) return;

    var rows = Array.prototype.slice.call(stack.querySelectorAll('.career-row'));
    if (!rows.length) return;

    stack.classList.add('career-trace-ready');

    var travelText = 'traveling back in time';
    var lastScrollY = window.scrollY;

    function updateTravelText(nextText) {
      travelText = nextText || travelText;
      rows.forEach(function (row) {
        var target = row.querySelector('.career-travel-text');
        if (target) target.textContent = travelText;
      });
    }

    window.addEventListener('scroll', function () {
      var nextY = window.scrollY;
      updateTravelText(nextY >= lastScrollY ? 'traveling back in time' : 'back to the future');
      lastScrollY = nextY;
    }, { passive: true });

    rows.forEach(function (row) {
      if (row.querySelector('.career-packet')) return;
      var packet = document.createElement('div');
      packet.className = 'career-packet';
      packet.innerHTML = [
        '<span class="career-travel"><span class="career-braille" aria-hidden="true"></span><span class="career-travel-text">' + travelText + '</span></span>',
        '<em>timeline cursor active</em>'
      ].join('');
      row.appendChild(packet);
    });

    if (!('IntersectionObserver' in window)) {
      rows[0].classList.add('is-current');
      return;
    }

    var observer = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (!entry.isIntersecting) return;
        rows.forEach(function (row) {
          row.classList.toggle('is-current', row === entry.target);
        });
      });
    }, {
      rootMargin: '-35% 0px -45% 0px',
      threshold: 0.15
    });

    rows.forEach(function (row) {
      observer.observe(row);
    });
  }

  function initArchitectureMotion() {
    var svgs = Array.prototype.slice.call(document.querySelectorAll('svg[aria-labelledby*="ArchTitle"]'));
    if (!svgs.length) return;

    svgs.forEach(function (svg) {
      svg.classList.add('architecture-svg');
      var panel = svg.closest('.overflow-x-auto') || svg.parentElement;
      if (panel) panel.classList.add('architecture-panel');

      var nodeShapes = Array.prototype.slice.call(svg.querySelectorAll('rect[class]'));
      nodeShapes.forEach(function (shape, index) {
        if (shape.closest('.architecture-node')) return;

        var members = [shape];
        var sibling = shape.nextElementSibling;
        while (sibling && sibling.tagName.toLowerCase() === 'text') {
          members.push(sibling);
          sibling = sibling.nextElementSibling;
        }

        var group = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        group.classList.add('architecture-node');
        group.setAttribute('tabindex', '0');
        group.setAttribute('focusable', 'true');
        group.setAttribute('role', 'group');

        var label = members.slice(1).map(function (member) {
          return member.textContent.trim();
        }).filter(Boolean).join(' — ');
        group.setAttribute('aria-label', label || 'Architecture component ' + (index + 1));

        shape.parentNode.insertBefore(group, shape);
        members.forEach(function (member) {
          group.appendChild(member);
        });

        function syncNodeFocus() {
          var hasActiveNode = !!svg.querySelector('.architecture-node.is-pointer-active, .architecture-node:focus');
          svg.classList.toggle('has-active-node', hasActiveNode);
        }

        group.addEventListener('pointerenter', function () {
          group.classList.add('is-pointer-active');
          syncNodeFocus();
        });
        group.addEventListener('pointerleave', function () {
          group.classList.remove('is-pointer-active');
          syncNodeFocus();
        });
        group.addEventListener('focus', syncNodeFocus);
        group.addEventListener('blur', syncNodeFocus);
      });
    });
  }

  function motifForSlug(slug) {
    var data = projectPreviewData[slug];
    if (!data) return null;
    var motifLabel = {
      avatar: 'VOICE / RAG / VISEME',
      report: 'EVIDENCE / AGENTS / REPORT',
      migration: 'SAS / DATAIKU / VALIDATION',
      fraud: 'TRANSACTIONS / FEATURES / RISK',
      telemetry: 'SENSORS / SEQUENCES / ALERTS',
      sensor: 'WINDOWS / CNN-LSTM / ACTIVITY',
      position: 'BEACONS / MODEL / COORDINATE',
      imaging: 'SCANS / FEATURES / DIAGNOSIS',
      integration: 'IMU / BLE / ALERT'
    }[data.kind];

    return {
      kind: data.kind,
      label: motifLabel || data.label,
      metric: data.metric,
      command: data.domainCommand,
      lines: data.domainLines,
      nodes: data.domainNodes
    };
  }

  function buildDomainPanel(motif) {
    var lines = motif.lines.map(function (line) {
      return '<li><span>></span>' + line + '</li>';
    }).join('');
    var nodes = motif.nodes.map(function (node, index) {
      return '<span style="--i:' + index + '">' + node + '</span>';
    }).join('');

    return [
      '<div class="domain-head"><span>// DOMAIN CONSOLE</span><span>' + motif.metric + '</span></div>',
      '<div class="domain-body">',
      '<div class="domain-terminal">',
      '<div class="domain-command"><span>$</span> ' + motif.command + '</div>',
      '<ul>' + lines + '</ul>',
      '</div>',
      '<div class="domain-map" aria-hidden="true">' + nodes + '</div>',
      '</div>',
      '<div class="domain-label">' + motif.label + '</div>'
    ].join('');
  }

  function initProjectMotifs() {
    var slug = slugFromHref(window.location.pathname + '/');
    var motif = motifForSlug(slug);
    var header = document.querySelector('main article header#summary');
    if (!motif || !header || header.querySelector('.project-domain-panel')) return;

    var motifEl = document.createElement('div');
    motifEl.className = 'project-domain-panel domain-' + motif.kind;
    motifEl.innerHTML = buildDomainPanel(motif);
    header.appendChild(motifEl);
  }

  function syncThemeToggle() {
    var toggle = document.getElementById('themeToggle');
    if (!toggle) return;
    var isLight = document.documentElement.getAttribute('data-theme') !== 'dark';
    toggle.textContent = isLight ? '◐ DARK' : '◑ LIGHT';
    toggle.setAttribute('aria-label', isLight ? 'Switch to dark mode' : 'Switch to light mode');
    toggle.setAttribute('title', isLight ? 'Switch to dark mode' : 'Switch to light mode');
  }

  function initPortfolioAssistant() {
    if (document.querySelector('script[data-portfolio-assistant]')) return;
    var script = document.createElement('script');
    script.src = '/ask-amr.js';
    script.defer = true;
    script.dataset.portfolioAssistant = '';
    document.head.appendChild(script);
  }

  function initExternalLinks() {
    Array.prototype.slice.call(document.querySelectorAll('a[href^="http://"], a[href^="https://"]')).forEach(function (link) {
      var url;
      try { url = new URL(link.href, window.location.href); } catch (error) { return; }
      if (url.origin === window.location.origin) return;
      link.target = '_blank';
      link.rel = 'noopener noreferrer';
      if (!link.getAttribute('aria-label')) {
        var label = link.textContent.replace(/\s+/g, ' ').trim();
        if (label) link.setAttribute('aria-label', label + ' (opens in a new tab)');
      }
    });
  }

  function initConversionTracking() {
    if (navigator.doNotTrack === '1' || navigator.globalPrivacyControl === true) return;
    var eventEndpoint = '';
    var allowedEvents = ['resume_download', 'case_study_open', 'contact_email', 'linkedin_exit', 'assistant_open', 'assistant_question'];

    var configPromise = window.portfolioConfigPromise || (window.portfolioConfigPromise = fetch('/chat-config.json', { cache: 'no-store' })
      .then(function (response) { return response.ok ? response.json() : {}; })
      .catch(function () { return {}; }));
    configPromise
      .then(function (config) {
        config = config || {};
        var endpoint = config.endpoint || config.apiUrl;
        if (!endpoint) return;
        eventEndpoint = String(endpoint).replace(/\/chat\/?$/, '').replace(/\/$/, '') + '/event';
      })
      .catch(function () {});

    function send(eventName) {
      if (!eventEndpoint || allowedEvents.indexOf(eventName) === -1) return;
      var payload = JSON.stringify({ event: eventName, path: window.location.pathname });
      var body = new Blob([payload], { type: 'text/plain;charset=UTF-8' });
      if (navigator.sendBeacon && navigator.sendBeacon(eventEndpoint, body)) return;
      fetch(eventEndpoint, { method: 'POST', body: payload, headers: { 'Content-Type': 'text/plain;charset=UTF-8' }, keepalive: true }).catch(function () {});
    }

    document.addEventListener('click', function (event) {
      var assistantControl = event.target.closest('.ask-amr-launcher, .ask-amr-home-question');
      if (assistantControl) {
        send('assistant_open');
        return;
      }
      var link = event.target.closest('a[href]');
      if (!link) return;
      var href = link.getAttribute('href') || '';
      if (href.indexOf('mailto:') === 0) send('contact_email');
      else if (/\/docs\/Amr_Alfayoumy_[^/]+\.pdf(?:$|[?#])/.test(href)) send('resume_download');
      else if (/^\/projects\/[^/]+\//.test(href)) send('case_study_open');
      else if (/linkedin\.com\//i.test(href)) send('linkedin_exit');
    });

    document.addEventListener('submit', function (event) {
      if (event.target.matches('.ask-amr-form, .ask-amr-home-form')) send('assistant_question');
    });
  }

  document.addEventListener('DOMContentLoaded', function () {
    initMobileNavigation();
    var toggle = document.getElementById('themeToggle');
    if (toggle) {
      syncThemeToggle();
      toggle.addEventListener('click', function () {
        var nextTheme = document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
        document.documentElement.setAttribute('data-theme', nextTheme);
        try {
          localStorage.setItem('theme', nextTheme);
        } catch (e) {}
        syncThemeToggle();
      });
    }
    initBootSequence();
    initCursorTrace();
    initThinkingTitle();
    initRiseScroll();
    initProjectPreviews();
    initFieldRecorder();
    initCareerTrace();
    initArchitectureMotion();
    initProjectMotifs();
    initExternalLinks();
    initConversionTracking();
    initPortfolioAssistant();
  });
})();
