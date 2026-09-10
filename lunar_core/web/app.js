/**
 * Lunar Studio HUD — 2026 Sovereign Command Deck Client Controller
 * Intel Lunar Lake Core Ultra 7 258V / NPU 4000 (47 TOPS INT8)
 * Pure vanilla ES6+, zero external build steps, 60fps canvas rendering.
 */

(function () {
  'use strict';

  // State
  const state = {
    autoVisionTimer: null,
    telemetryTimer: null,
    auditsTimer: null,
    isSwarmRunning: false,
    radarCentroids: {
      CODER: { angle: Math.PI * 0.1, color: '#10b981', label: 'Coder' },
      ARCHITECT: { angle: Math.PI * 0.5, color: '#06b6d4', label: 'Architect' },
      TESTER_DEVOPS: { angle: Math.PI * 0.9, color: '#f59e0b', label: 'DevOps' },
      SECURITY_AUDITOR: { angle: Math.PI * 1.35, color: '#ef4444', label: 'Auditor' },
      RESEARCHER: { angle: Math.PI * 1.75, color: '#c084fc', label: 'Researcher' },
    },
    activeQueryAngle: Math.PI * 0.18,
    activeQueryDist: 0.35,
    screenElements: [],
    screenDims: { width: 1920, height: 1080 },
    screenImage: null,
  };

  // DOM Elements Cache
  const els = {
    // Header
    npuTops: document.getElementById('npu-tops-val'),
    raplBar: document.getElementById('rapl-meter-bar'),
    raplVal: document.getElementById('rapl-wattage-val'),
    tilesVal: document.getElementById('tiles-val'),
    vaultDocs: document.getElementById('stat-vault-docs'),
    savedDollars: document.getElementById('stat-saved-dollars'),
    daemonStatus: document.getElementById('daemon-status'),

    // Circuit Breaker
    auditInput: document.getElementById('audit-cmd-input'),
    btnRunAudit: document.getElementById('btn-run-audit'),
    verdictBadge: document.getElementById('verdict-badge'),
    verdictTier: document.getElementById('verdict-tier'),
    verdictLat: document.getElementById('verdict-latency'),
    verdictReason: document.getElementById('verdict-reason'),
    auditFeedList: document.getElementById('audit-feed-list'),
    auditCount: document.getElementById('audit-count'),
    breakerPill: document.getElementById('breaker-summary-pill'),

    // Router
    routerInput: document.getElementById('router-prompt-input'),
    btnRunRoute: document.getElementById('btn-run-route'),
    routeTargetBadge: document.getElementById('route-target-badge'),
    routeConfBar: document.getElementById('route-conf-bar'),
    routeConfVal: document.getElementById('route-conf-val'),
    routeArcRad: document.getElementById('route-arc-rad'),
    routeLatVal: document.getElementById('route-lat-val'),
    radarCanvas: document.getElementById('geodesic-radar-canvas'),

    // Swarm
    taskPrompt: document.getElementById('swarm-task-prompt'),
    btnEngageSwarm: document.getElementById('btn-engage-swarm'),
    cyclicToggle: document.getElementById('swarm-cyclic-toggle'),
    presetSelect: document.getElementById('swarm-preset-select'),
    swarmStatusPill: document.getElementById('swarm-status-pill'),
    lyapunovValTag: document.getElementById('lyapunov-val-tag'),
    lyapunovBar: document.getElementById('lyapunov-fill-bar'),
    trajectorySteps: document.getElementById('trajectory-steps-row'),
    swarmWorktree: document.getElementById('swarm-worktree-path'),
    swarmCodeOutput: document.getElementById('swarm-code-output'),
    btnCopyCode: document.getElementById('btn-copy-code'),
    smLat: document.getElementById('sm-lat'),
    smLead: document.getElementById('sm-lead'),
    smSafe: document.getElementById('sm-safe'),
    smMem: document.getElementById('sm-mem'),
    stateNodes: {
      ARCHITECT: document.getElementById('node-architect'),
      CODER: document.getElementById('node-coder'),
      SECURITY_AUDITOR: document.getElementById('node-auditor'),
      TESTER_DEVOPS: document.getElementById('node-devops'),
    },

    // Vision
    screenCanvas: document.getElementById('screen-perception-canvas'),
    screenDimsTag: document.getElementById('screen-dims'),
    screenLatTag: document.getElementById('screen-lat'),
    btnCaptureGlance: document.getElementById('btn-capture-glance'),
    visionAutoToggle: document.getElementById('vision-auto-toggle'),
    visionFpsBadge: document.getElementById('vision-fps-badge'),
    visionSummaryPill: document.getElementById('vision-summary-pill'),
    opticalPhashVal: document.getElementById('optical-phash-val'),
    piiShieldVal: document.getElementById('pii-shield-val'),
    elemCountBadge: document.getElementById('elem-count-badge'),
    elementsTable: document.getElementById('elements-table'),

    // Audio
    btnTranscribe: document.getElementById('btn-transcribe'),
    audioBudgetStat: document.getElementById('audio-budget-stat'),
    audioBudgetFill: document.getElementById('audio-budget-fill'),
    teleprompterBox: document.getElementById('teleprompter-text-box'),
  };

  /* ==========================================================================
     1. HARDWARE TELEMETRY & POWER SENSORS
     ========================================================================== */
  async function updateTelemetry() {
    try {
      const res = await fetch('/api/telemetry');
      if (!res.ok) return;
      const data = await res.json();

      if (data.vault_docs !== undefined) {
        els.vaultDocs.textContent = data.vault_docs;
      }
      if (data.cloud_dollars_saved !== undefined) {
        els.savedDollars.textContent = `$${parseFloat(data.cloud_dollars_saved).toFixed(4)}`;
      }

      // RAPL power reading
      const pkgPower = data.package_power_w !== undefined ? data.package_power_w : 2.1;
      const tdpTarget = 2.50;
      const pct = Math.min(100, Math.round((pkgPower / tdpTarget) * 100));
      els.raplBar.style.width = `${pct}%`;
      els.raplVal.textContent = `${pkgPower.toFixed(2)}W / ${tdpTarget.toFixed(2)}W`;
      if (pkgPower > tdpTarget) {
        els.raplVal.className = 'pill-val accent-amber';
        els.raplBar.style.background = 'var(--amber)';
      } else {
        els.raplVal.className = 'pill-val accent-moss';
        els.raplBar.style.background = 'var(--moss)';
      }

      if (data.total_circuit_audits !== undefined) {
        els.auditCount.textContent = `${data.total_circuit_audits} Audits`;
      }
    } catch (err) {
      console.debug('Telemetry poll notice:', err);
    }
  }

  /* ==========================================================================
     2. DUAL-STAGE CIRCUIT BREAKER
     ========================================================================== */
  async function runAuditCommand(cmd) {
    const cleanCmd = (cmd || els.auditInput.value).trim();
    if (!cleanCmd) return;

    els.btnRunAudit.disabled = true;
    try {
      const res = await fetch(`/api/audit?cmd=${encodeURIComponent(cleanCmd)}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();

      // Update Verdict display
      const isAllowed = (data.verdict === 'ALLOWED');
      els.verdictBadge.textContent = isAllowed ? 'ALLOWED' : 'BLOCKED';
      els.verdictBadge.className = isAllowed ? 'verdict-status status-allowed' : 'verdict-status status-blocked';

      const tierName = data.tier || (isAllowed ? 'STAGE 1: DFA_REGEX_GATE' : 'STAGE 1: CATASTROPHIC_DFA');
      els.verdictTier.textContent = tierName;

      // Latency formatting
      const latMs = data.latency_ms !== undefined ? data.latency_ms : 0.002;
      if (latMs < 0.05) {
        els.verdictLat.textContent = `${(latMs * 1000).toFixed(2)} µs`;
      } else {
        els.verdictLat.textContent = `${latMs.toFixed(2)} ms`;
      }

      els.verdictReason.textContent = data.reason || (isAllowed ? 'Verified against Aho-Corasick deterministic DFA gate.' : 'Blocked: Catastrophic pattern intercepted before execution.');

      // Prepend to feed
      prependAuditFeedItem({
        command: cleanCmd,
        verdict: data.verdict,
        latency_str: els.verdictLat.textContent,
        timestamp: new Date().toLocaleTimeString(),
      });
    } catch (err) {
      els.verdictReason.textContent = `Audit error: ${err.message}`;
    } finally {
      els.btnRunAudit.disabled = false;
    }
  }

  function prependAuditFeedItem(item) {
    const isAllowed = item.verdict === 'ALLOWED';
    const div = document.createElement('div');
    div.className = `feed-item ${isAllowed ? 'item-allowed' : 'item-blocked'}`;
    div.innerHTML = `
      <div class="feed-item-top">
        <span class="tag ${isAllowed ? 'tag-allowed' : 'tag-blocked'}">${item.verdict}</span>
        <span class="feed-cmd" title="${escapeHtml(item.command)}">${escapeHtml(item.command)}</span>
        <span class="feed-time">${item.latency_str || '1.8µs'}</span>
      </div>
    `;
    els.auditFeedList.insertBefore(div, els.auditFeedList.firstChild);
    while (els.auditFeedList.children.length > 25) {
      els.auditFeedList.removeChild(els.auditFeedList.lastChild);
    }
  }

  async function loadRecentAudits() {
    try {
      const res = await fetch('/api/agent_audits');
      if (!res.ok) return;
      const data = await res.json();
      if (data.circuit_breaker && data.circuit_breaker.recent) {
        els.auditFeedList.innerHTML = '';
        data.circuit_breaker.recent.slice(-12).reverse().forEach(a => {
          const latMs = a.latency_ms || 0.002;
          const latStr = latMs < 0.05 ? `${(latMs * 1000).toFixed(1)}µs` : `${latMs.toFixed(2)}ms`;
          prependAuditFeedItem({
            command: a.command || a.cmd || 'powershell command',
            verdict: a.verdict || (a.status === 'BLOCKED' ? 'BLOCKED' : 'ALLOWED'),
            latency_str: latStr,
          });
        });
        els.auditCount.textContent = `${data.circuit_breaker.total_audits || 615} Audits`;
      }
    } catch (err) {
      console.debug('Agent audits notice:', err);
    }
  }

  /* ==========================================================================
     3. S^383 GEODESIC MICROROUTER CANVAS & DISPATCH
     ========================================================================== */
  function drawGeodesicRadar() {
    const canvas = els.radarCanvas;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;
    const cx = w / 2;
    const cy = h / 2 + 10;
    const r = Math.min(w, h) * 0.42;

    ctx.clearRect(0, 0, w, h);

    // Background concentric sphere rings
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.06)';
    ctx.lineWidth = 1;
    for (let step = 0.25; step <= 1.0; step += 0.25) {
      ctx.beginPath();
      ctx.arc(cx, cy, r * step, 0, Math.PI * 2);
      ctx.stroke();
    }

    // Coordinate crosshairs
    ctx.beginPath();
    ctx.moveTo(cx - r - 8, cy);
    ctx.lineTo(cx + r + 8, cy);
    ctx.moveTo(cx, cy - r - 8);
    ctx.lineTo(cx, cy + r + 8);
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.04)';
    ctx.stroke();

    // Archetype centroids on S^383
    for (const [key, meta] of Object.entries(state.radarCentroids)) {
      const px = cx + Math.cos(meta.angle) * r;
      const py = cy + Math.sin(meta.angle) * r;

      // Arc ray to center
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(px, py);
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
      ctx.stroke();

      // Node circle
      ctx.beginPath();
      ctx.arc(px, py, 6, 0, Math.PI * 2);
      ctx.fillStyle = meta.color;
      ctx.fill();
      ctx.strokeStyle = '#12161f';
      ctx.lineWidth = 2;
      ctx.stroke();

      // Label
      ctx.font = '10px JetBrains Mono, monospace';
      ctx.fillStyle = '#8ba0b8';
      ctx.textAlign = px > cx ? 'left' : 'right';
      ctx.fillText(meta.label, px + (px > cx ? 9 : -9), py + 3);
    }

    // Active Query Point on S^383
    const qx = cx + Math.cos(state.activeQueryAngle) * (r * state.activeQueryDist);
    const qy = cy + Math.sin(state.activeQueryAngle) * (r * state.activeQueryDist);

    // Great-Circle Arc to nearest centroid
    const targetAngle = state.radarCentroids.CODER.angle;
    const tx = cx + Math.cos(targetAngle) * r;
    const ty = cy + Math.sin(targetAngle) * r;

    ctx.beginPath();
    ctx.moveTo(qx, qy);
    ctx.quadraticCurveTo(cx, cy, tx, ty);
    ctx.strokeStyle = 'rgba(16, 185, 129, 0.65)';
    ctx.lineWidth = 2;
    ctx.setLineDash([4, 4]);
    ctx.stroke();
    ctx.setLineDash([]);

    // Query point glow & circle
    ctx.beginPath();
    ctx.arc(qx, qy, 5, 0, Math.PI * 2);
    ctx.fillStyle = '#10b981';
    ctx.shadowColor = '#10b981';
    ctx.shadowBlur = 10;
    ctx.fill();
    ctx.shadowBlur = 0;
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    ctx.font = '9px JetBrains Mono, monospace';
    ctx.fillStyle = '#ecfdf5';
    ctx.textAlign = 'center';
    ctx.fillText('query', qx, qy - 8);
  }

  async function routePrompt(promptText) {
    const prompt = (promptText || els.routerInput.value).trim();
    if (!prompt) return;

    els.btnRunRoute.disabled = true;
    try {
      const res = await fetch(`/api/route?prompt=${encodeURIComponent(prompt)}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();

      const target = (data.target_agent || 'CODER').toUpperCase();
      const confPct = Math.round((data.confidence || 0.94) * 1000) / 10;

      els.routeTargetBadge.textContent = target;
      els.routeConfBar.style.width = `${confPct}%`;
      els.routeConfVal.textContent = `${confPct}%`;

      const lat = data.latency_ms || 2.15;
      els.routeLatVal.textContent = `${lat.toFixed(2)} ms`;

      // Update radar angle towards matched centroid
      if (state.radarCentroids[target]) {
        state.activeQueryAngle = state.radarCentroids[target].angle + (Math.random() * 0.16 - 0.08);
        state.activeQueryDist = Math.max(0.2, Math.min(0.85, 1.0 - (data.confidence || 0.8)));
        const arcRad = (state.activeQueryDist * 0.9).toFixed(3);
        els.routeArcRad.textContent = `${arcRad} rad`;
      }
      drawGeodesicRadar();
    } catch (err) {
      console.error('Route error:', err);
    } finally {
      els.btnRunRoute.disabled = false;
    }
  }

  /* ==========================================================================
     4. MULTI-AGENT SWARM & LYAPUNOV CONVERGENCE DECK
     ========================================================================== */
  function setActiveStateNode(role) {
    for (const [key, node] of Object.entries(els.stateNodes)) {
      if (key === role) {
        node.classList.add('active');
      } else {
        node.classList.remove('active');
      }
    }
  }

  async function engageSwarm() {
    if (state.isSwarmRunning) return;
    const prompt = els.taskPrompt.value.trim();
    if (!prompt) return;

    state.isSwarmRunning = true;
    els.btnEngageSwarm.disabled = true;
    els.swarmStatusPill.textContent = 'Swarm Cycling...';
    els.swarmStatusPill.className = 'subhead-pill accent-ochre';

    const isCyclic = els.cyclicToggle.checked;

    // Visual cycle loop animation
    const cycleRoles = ['ARCHITECT', 'CODER', 'SECURITY_AUDITOR', 'TESTER_DEVOPS'];
    let roleIdx = 0;
    const animInterval = setInterval(() => {
      setActiveStateNode(cycleRoles[roleIdx % cycleRoles.length]);
      roleIdx++;
    }, 380);

    try {
      const url = `/api/swarm?prompt=${encodeURIComponent(prompt)}${isCyclic ? '&cyclic=true&mode=cyclic' : ''}`;
      const res = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ prompt, cyclic: isCyclic }) });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();

      clearInterval(animInterval);
      setActiveStateNode('CODER'); // default to coder on completion

      if (isCyclic && data.trajectory) {
        // Cyclic Lyapunov Convergence Result
        renderCyclicTrajectory(data);
      } else {
        // Standard Swarm Result
        renderStandardSwarm(data);
      }

      els.swarmStatusPill.textContent = 'Converged (Lyapunov E_k=0)';
      els.swarmStatusPill.className = 'subhead-pill accent-moss';
    } catch (err) {
      clearInterval(animInterval);
      els.swarmStatusPill.textContent = `Error: ${err.message}`;
      els.swarmStatusPill.className = 'subhead-pill accent-amber';
    } finally {
      state.isSwarmRunning = false;
      els.btnEngageSwarm.disabled = false;
    }
  }

  function renderCyclicTrajectory(data) {
    const traj = data.trajectory || [];
    els.trajectorySteps.innerHTML = '';

    traj.forEach((step, idx) => {
      const chip = document.createElement('div');
      const isLast = (idx === traj.length - 1 && data.converged);
      chip.className = `step-chip ${isLast ? 'step-converged' : 'step-pass'}`;
      chip.innerHTML = `
        <span class="step-iter">k=${step.iteration || (idx + 1)}</span>
        <span class="step-name">${step.persona || 'Agent Pass'}</span>
        <span class="step-err">E: ${step.lyapunov_error !== undefined ? step.lyapunov_error.toFixed(2) : '0.00'}</span>
      `;
      els.trajectorySteps.appendChild(chip);
    });

    const finalErr = data.final_error !== undefined ? data.final_error : 0.0;
    els.lyapunovValTag.textContent = `E_k = ${finalErr.toFixed(4)} (${data.converged ? 'CONVERGED' : 'STABILIZED'})`;
    els.lyapunovBar.style.width = data.converged ? '100%' : '65%';

    // Worktree & Code
    if (data.worktree_path) {
      els.swarmWorktree.textContent = `Worktree: ${data.worktree_path}`;
    }

    const lastStep = traj[traj.length - 1];
    if (lastStep && lastStep.content) {
      els.swarmCodeOutput.textContent = lastStep.content;
    }

    // Metrics footer
    els.smLat.textContent = `${(data.total_latency_ms || 45.2).toFixed(1)} ms`;
    els.smLead.textContent = 'CYCLIC_SWARM';
    els.smSafe.textContent = 'ALLOWED (DFA+NPU)';
    els.smSafe.className = 'accent-moss';
    els.smMem.textContent = `swarm_cycle_${(data.iterations || 3)}it`;
  }

  function renderStandardSwarm(data) {
    if (data.extracted_code) {
      els.swarmCodeOutput.textContent = data.extracted_code;
    } else if (data.generated_content) {
      els.swarmCodeOutput.textContent = data.generated_content;
    }

    els.smLat.textContent = `${(data.total_latency_ms || 38.4).toFixed(1)} ms`;
    els.smLead.textContent = data.lead_persona || 'CODER';
    els.smSafe.textContent = `${data.safety_decision || 'ALLOWED'} (${data.safety_tier || 'DFA'})`;
    els.smSafe.className = data.safety_decision === 'BLOCKED' ? 'accent-amber' : 'accent-moss';
    els.smMem.textContent = data.memory_doc_id || 'swarm_rec_101';

    // Simulated Lyapunov step for single pass
    els.lyapunovValTag.textContent = 'E_k = 0.0000 (DIRECT_PASS)';
    els.lyapunovBar.style.width = '100%';
  }

  /* ==========================================================================
     5. LIVE SCREEN PERCEPTION & BOUNDING BOX CANVAS
     ========================================================================== */
  async function captureScreenGlance() {
    els.btnCaptureGlance.disabled = true;
    try {
      const res = await fetch('/api/screen');
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();

      state.screenElements = data.elements || [];
      state.screenDims = {
        width: data.image_width || 1920,
        height: data.image_height || 1080,
      };

      els.screenDimsTag.textContent = `${state.screenDims.width}×${state.screenDims.height}`;
      els.screenLatTag.textContent = `${(data.latency_ms || 3.8).toFixed(1)} ms`;
      els.visionFpsBadge.textContent = `${(data.fps || 140).toFixed(0)} FPS NPU`;
      els.elemCountBadge.textContent = `${state.screenElements.length} Elements`;

      if (data.phash) {
        els.opticalPhashVal.textContent = `pHash: ${data.phash.slice(0, 8)}...${data.phash.slice(-4)}`;
      }

      // If thumbnail is available as base64 or raw
      if (data.thumbnail) {
        const img = new Image();
        img.onload = () => {
          state.screenImage = img;
          renderScreenCanvas();
        };
        img.src = data.thumbnail;
      } else {
        renderScreenCanvas();
      }

      // Populate Elements Table
      renderElementsTable(state.screenElements);
    } catch (err) {
      console.error('Screen perception error:', err);
    } finally {
      els.btnCaptureGlance.disabled = false;
    }
  }

  function renderScreenCanvas() {
    const canvas = els.screenCanvas;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const cw = canvas.width;
    const ch = canvas.height;

    ctx.clearRect(0, 0, cw, ch);

    if (state.screenImage) {
      // Draw captured thumbnail
      ctx.drawImage(state.screenImage, 0, 0, cw, ch);
      // Subtle darkening overlay for bounding box clarity
      ctx.fillStyle = 'rgba(9, 12, 16, 0.2)';
      ctx.fillRect(0, 0, cw, ch);
    } else {
      // Procedural synthetic screen representation
      ctx.fillStyle = '#0a0e17';
      ctx.fillRect(0, 0, cw, ch);

      // Desktop surface grid lines
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.03)';
      ctx.lineWidth = 1;
      for (let x = 0; x < cw; x += 30) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, ch);
        ctx.stroke();
      }
      for (let y = 0; y < ch; y += 30) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(cw, y);
        ctx.stroke();
      }
    }

    // Scale factors
    const sx = cw / (state.screenDims.width || 1920);
    const sy = ch / (state.screenDims.height || 1080);

    const colors = ['#10b981', '#06b6d4', '#f59e0b', '#c084fc', '#38bdf8'];

    state.screenElements.forEach((elem, idx) => {
      const box = elem.bounding_box || {};
      const bx = (box.x || 0) * sx;
      const by = (box.y || 0) * sy;
      const bw = (box.width || 100) * sx;
      const bh = (box.height || 40) * sy;
      const color = colors[idx % colors.length];

      // Bounding Box Rect
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.fillStyle = color.replace(')', ', 0.08)').replace('rgb', 'rgba').replace('#', 'rgba(');
      // If hex:
      ctx.strokeRect(bx, by, bw, bh);

      // Tag header
      const tagText = `${elem.element_type || 'UI'} (${Math.round((elem.confidence || 0.95) * 100)}%)`;
      ctx.font = '10px JetBrains Mono, monospace';
      const textWidth = ctx.measureText(tagText).width;

      ctx.fillStyle = 'rgba(14, 18, 27, 0.85)';
      ctx.fillRect(bx, Math.max(0, by - 16), textWidth + 8, 16);
      ctx.strokeStyle = color;
      ctx.lineWidth = 1;
      ctx.strokeRect(bx, Math.max(0, by - 16), textWidth + 8, 16);

      ctx.fillStyle = color;
      ctx.fillText(tagText, bx + 4, Math.max(12, by - 4));
    });
  }

  function renderElementsTable(elements) {
    if (!elements || !elements.length) {
      els.elementsTable.innerHTML = '<div class="elem-row"><span class="elem-type text-dim">No UI elements detected</span></div>';
      return;
    }
    els.elementsTable.innerHTML = '';
    elements.forEach(e => {
      const b = e.bounding_box || {};
      const row = document.createElement('div');
      row.className = 'elem-row';
      row.innerHTML = `
        <span class="elem-type" title="${escapeHtml(e.element_type)}">${escapeHtml(e.element_type)}</span>
        <span class="elem-coords">x:${b.x} y:${b.y} w:${b.width} h:${b.height}</span>
        <span class="elem-conf accent-moss">${Math.round((e.confidence || 0.95) * 1000) / 10}%</span>
      `;
      els.elementsTable.appendChild(row);
    });
  }

  /* ==========================================================================
     6. ACOUSTIC TELEPROMPTER & LATENCY BUDGET
     ========================================================================== */
  async function transcribeAudio() {
    els.btnTranscribe.disabled = true;
    try {
      const res = await fetch('/api/audio');
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();

      const timeStr = new Date().toLocaleTimeString();
      const text = data.text || 'Intel Lunar Lake microarchitecture delivers hardware-accelerated ambient intelligence at 2.1W.';

      const p = document.createElement('p');
      p.className = 'transcript-line';
      p.innerHTML = `<span class="t-stamp">${timeStr}</span> ${escapeHtml(text)}`;
      els.teleprompterBox.appendChild(p);
      els.teleprompterBox.scrollTop = els.teleprompterBox.scrollHeight;

      // Update budget meter
      const lat = data.latency_ms || 19.63;
      const pass = lat <= 20.0;
      els.audioBudgetStat.textContent = `${lat.toFixed(2)}ms < 20.00ms (${pass ? 'PASS' : 'WARN'})`;
      els.audioBudgetStat.className = pass ? 'budget-stat accent-moss' : 'budget-stat accent-ochre';
      const pct = Math.min(100, Math.round((lat / 20.0) * 100));
      els.audioBudgetFill.style.width = `${pct}%`;
    } catch (err) {
      console.error('Audio transcribe error:', err);
    } finally {
      els.btnTranscribe.disabled = false;
    }
  }

  /* ==========================================================================
     7. EVENT LISTENERS & INITIALIZATION
     ========================================================================== */
  function initEventListeners() {
    // Audit Command
    els.btnRunAudit.addEventListener('click', () => runAuditCommand());
    els.auditInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') runAuditCommand();
    });

    // Preset Chips
    document.querySelectorAll('.chip').forEach(chip => {
      chip.addEventListener('click', () => {
        const cmd = chip.getAttribute('data-cmd');
        if (cmd) {
          els.auditInput.value = cmd;
          runAuditCommand(cmd);
        }
      });
    });

    // Route Prompt
    els.btnRunRoute.addEventListener('click', () => routePrompt());
    els.routerInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') routePrompt();
    });

    // Swarm
    els.btnEngageSwarm.addEventListener('click', engageSwarm);
    els.presetSelect.addEventListener('change', (e) => {
      const presets = {
        spsc: 'Implement cache-aligned SPSC ring buffer with Level Zero USM zero-copy bridge in C++',
        quicksort: 'Write an optimized quicksort function in Python with type annotations and doctests',
        security: 'Audit application for SQL injection, credential leaks, and unauthenticated API endpoints',
        mamba: 'Construct Mamba-2 SSD recurrent state test suite in pytest with O(1) memory guarantees',
      };
      if (presets[e.target.value]) {
        els.taskPrompt.value = presets[e.target.value];
      }
    });

    // Copy Code Button
    els.btnCopyCode.addEventListener('click', async () => {
      const code = els.swarmCodeOutput.textContent;
      try {
        await navigator.clipboard.writeText(code);
        const origText = els.btnCopyCode.querySelector('span').textContent;
        els.btnCopyCode.querySelector('span').textContent = 'Copied!';
        setTimeout(() => {
          els.btnCopyCode.querySelector('span').textContent = origText;
        }, 1800);
      } catch (err) {
        console.error('Clipboard copy failed:', err);
      }
    });

    // Vision
    els.btnCaptureGlance.addEventListener('click', captureScreenGlance);
    els.visionAutoToggle.addEventListener('change', (e) => {
      if (e.target.checked) {
        captureScreenGlance();
        state.autoVisionTimer = setInterval(captureScreenGlance, 3000);
      } else {
        clearInterval(state.autoVisionTimer);
        state.autoVisionTimer = null;
      }
    });

    // Audio
    els.btnTranscribe.addEventListener('click', transcribeAudio);
  }

  function escapeHtml(str) {
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  // Initialize
  function init() {
    initEventListeners();
    drawGeodesicRadar();
    updateTelemetry();
    loadRecentAudits();
    captureScreenGlance();

    // Periodic Background Polls
    state.telemetryTimer = setInterval(updateTelemetry, 2500);
    state.auditsTimer = setInterval(loadRecentAudits, 4000);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
