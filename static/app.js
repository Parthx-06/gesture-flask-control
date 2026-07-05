// ===== API Helpers =====
async function postJSON(url, data = {}) {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });
  return res.json();
}
async function getJSON(url) {
  const res = await fetch(url);
  return res.json();
}

// ===== Toast Notifications =====
function showToast(message, type = 'success') {
  const container = document.getElementById('toastContainer');
  if (!container) return;
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.textContent = message;
  container.appendChild(toast);
  setTimeout(() => {
    toast.classList.add('removing');
    setTimeout(() => toast.remove(), 300);
  }, 2500);
}

// ===== Lightbox =====
function openLightbox(src) {
  const lb = document.getElementById('lightbox');
  const img = document.getElementById('lightboxImg');
  if (!lb || !img) return;
  img.src = src;
  lb.classList.add('active');
}
function closeLightbox() {
  const lb = document.getElementById('lightbox');
  if (lb) lb.classList.remove('active');
}

// ===== Scroll Reveal =====
function initReveal() {
  const els = document.querySelectorAll('.reveal');
  if (!els.length) return;
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((e) => {
        if (e.isIntersecting) {
          e.target.classList.add('visible');
          observer.unobserve(e.target);
        }
      });
    },
    { threshold: 0.1 }
  );
  els.forEach((el) => observer.observe(el));
}

// ===== Range slider live value =====
function bindSliderValue(sliderId, displayId) {
  const slider = document.getElementById(sliderId);
  const display = document.getElementById(displayId);
  if (slider && display) {
    slider.addEventListener('input', () => {
      display.textContent = slider.value;
    });
  }
}

// ===== Button glow follow cursor =====
function initButtonGlow() {
  document.querySelectorAll('.btn').forEach((btn) => {
    btn.addEventListener('mousemove', (e) => {
      const rect = btn.getBoundingClientRect();
      btn.style.setProperty('--x', ((e.clientX - rect.left) / rect.width) * 100 + '%');
      btn.style.setProperty('--y', ((e.clientY - rect.top) / rect.height) * 100 + '%');
    });
  });
}

// ===== Status UI Update =====
function updateStatusUI(running, gesture, volume) {
  const runState = document.getElementById('runState');
  const gestureEl = document.getElementById('gesture');
  const volumeEl = document.getElementById('volume');
  const dot = document.getElementById('statusDot');
  const volBar = document.getElementById('volBar');
  if (runState) runState.textContent = running ? 'running' : 'stopped';
  if (gestureEl) gestureEl.textContent = gesture || 'None';
  if (volumeEl) volumeEl.textContent = volume ?? 0;
  if (dot) dot.classList.toggle('active', running);
  if (volBar) volBar.style.width = (volume ?? 0) + '%';
}

// ===== Server Status Refresh (non-demo mode) =====
async function refreshStatus() {
  try {
    const s = await getJSON('/api/status');
    updateStatusUI(s.running, s.gesture, s.volume);
  } catch (e) { /* ignore */ }
}

// ============================================================
// ===== BROWSER-SIDE GESTURE DETECTION (Demo / Render) =====
// ============================================================
let browserRunning = false;
let mpCamera = null;
let mpHands = null;
let lastActionTime = 0;
let lastClickTime = 0;
let browserVolume = 50;

function getFingerState(landmarks) {
  // landmarks: array of {x, y, z} normalised
  const f = [];
  // Thumb: tip(4) vs ip(3) — x axis
  f.push(landmarks[4].x < landmarks[3].x ? 1 : 0);
  // Fingers: tip vs pip — y axis (smaller y = higher on screen)
  [[8, 6], [12, 10], [16, 14], [20, 18]].forEach(([tip, pip]) => {
    f.push(landmarks[tip].y < landmarks[pip].y ? 1 : 0);
  });
  return f;
}

function recognizeGesture(f) {
  const key = f.join(',');
  const map = {
    '0,1,0,0,0': 'Cursor',
    '0,1,1,0,0': 'Click',
    '0,1,1,1,0': 'Scroll Up',
    '0,1,1,1,1': 'Scroll Down',
    '1,1,1,1,1': 'Screenshot',
    '0,0,0,0,1': 'Alt+Tab',
    '1,0,0,0,1': 'Show Desktop',
  };
  return map[key] || null;
}

function dist2D(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function onHandResults(results) {
  const canvas = document.getElementById('gestureCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  // Match canvas size to video
  const vid = document.getElementById('hiddenVideo');
  if (vid && vid.videoWidth) {
    canvas.width = vid.videoWidth;
    canvas.height = vid.videoHeight;
  }

  // Draw mirrored camera frame
  ctx.save();
  ctx.scale(-1, 1);
  ctx.drawImage(results.image, -canvas.width, 0, canvas.width, canvas.height);
  ctx.restore();

  if (!browserRunning) {
    // Dim overlay when stopped
    ctx.fillStyle = 'rgba(10,14,26,0.55)';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = '#6c7aff';
    ctx.font = 'bold 22px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Press ▶ Start to begin', canvas.width / 2, canvas.height / 2);
    updateStatusUI(false, 'None', browserVolume);
    return;
  }

  let gesture = 'None';
  let vol = browserVolume;

  if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
    const lm = results.multiHandLandmarks[0];

    // Mirror landmarks for mirrored canvas
    const mlm = lm.map(p => ({ x: 1 - p.x, y: p.y, z: p.z }));

    // Draw landmarks
    if (window.drawConnectors && window.HAND_CONNECTIONS) {
      drawConnectors(ctx, mlm, HAND_CONNECTIONS, { color: '#6c7aff', lineWidth: 2 });
    }
    if (window.drawLandmarks) {
      drawLandmarks(ctx, mlm, { color: '#a78bfa', lineWidth: 1, radius: 4 });
    }

    // Finger state & gesture
    const f = getFingerState(mlm);
    const detected = recognizeGesture(f);
    const now = Date.now() / 1000;
    const cooldown = parseFloat(document.getElementById('cooldown')?.value || 0.2);

    // Volume via pinch (thumb tip 4 ↔ index tip 8)
    const pinchDist = dist2D(mlm[4], mlm[8]);
    vol = Math.round(Math.min(100, Math.max(0, (pinchDist - 0.02) / (0.20 - 0.02) * 100)));
    browserVolume = vol;

    if (detected) {
      gesture = detected;

      // Visual feedback on canvas per gesture
      ctx.fillStyle = 'rgba(108,122,255,0.18)';
      ctx.fillRect(0, 0, canvas.width, 50);
      ctx.fillStyle = '#fff';
      ctx.font = 'bold 18px Inter, sans-serif';
      ctx.textAlign = 'left';
      ctx.fillText('✋ ' + gesture, 12, 32);

      // Screenshot: capture canvas
      if (detected === 'Screenshot' && now - lastActionTime > cooldown) {
        lastActionTime = now;
        const dataURL = canvas.toDataURL('image/png');
        const link = document.createElement('a');
        link.href = dataURL;
        link.download = 'gesture_screenshot_' + Date.now() + '.png';
        link.click();
        showToast('📸 Screenshot downloaded!');
      } else if (detected === 'Click' && now - lastActionTime > cooldown) {
        lastActionTime = now;
        const dt = now - lastClickTime;
        showToast(dt < 0.35 ? '🖱️ Double Click!' : '🖱️ Click!', 'success');
        lastClickTime = now;
      } else if (detected === 'Scroll Up' && now - lastActionTime > cooldown) {
        lastActionTime = now;
        window.scrollBy({ top: -120, behavior: 'smooth' });
      } else if (detected === 'Scroll Down' && now - lastActionTime > cooldown) {
        lastActionTime = now;
        window.scrollBy({ top: 120, behavior: 'smooth' });
      } else if (detected === 'Alt+Tab' && now - lastActionTime > cooldown) {
        lastActionTime = now;
        showToast('🔀 Alt+Tab (demo)', 'success');
      } else if (detected === 'Show Desktop' && now - lastActionTime > cooldown) {
        lastActionTime = now;
        showToast('🖥️ Show Desktop (demo)', 'success');
      }
    } else {
      gesture = 'None';
    }

    // Volume bar on canvas bottom
    const barW = (vol / 100) * canvas.width;
    ctx.fillStyle = 'rgba(108,122,255,0.25)';
    ctx.fillRect(0, canvas.height - 8, canvas.width, 8);
    ctx.fillStyle = '#6c7aff';
    ctx.fillRect(0, canvas.height - 8, barW, 8);
  } else {
    // No hand detected
    ctx.fillStyle = 'rgba(10,14,26,0.3)';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = 'rgba(255,255,255,0.35)';
    ctx.font = '16px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Show your hand to the camera', canvas.width / 2, canvas.height - 20);
  }

  updateStatusUI(true, gesture, vol);
}

function initBrowserGesture() {
  const videoEl = document.getElementById('hiddenVideo');
  const canvas = document.getElementById('gestureCanvas');
  if (!videoEl || !canvas) return;

  // Set canvas size
  canvas.width = 640;
  canvas.height = 480;

  // Draw loading state
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#0a0e1a';
  ctx.fillRect(0, 0, 640, 480);
  ctx.fillStyle = '#6c7aff';
  ctx.font = 'bold 18px Inter, sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText('Loading MediaPipe...', 320, 240);

  // Init MediaPipe Hands
  mpHands = new Hands({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
  });
  mpHands.setOptions({
    maxNumHands: 1,
    modelComplexity: 1,
    minDetectionConfidence: 0.7,
    minTrackingConfidence: 0.5,
  });
  mpHands.onResults(onHandResults);

  // Init camera
  mpCamera = new Camera(videoEl, {
    onFrame: async () => {
      await mpHands.send({ image: videoEl });
    },
    width: 640,
    height: 480,
  });
  mpCamera.start().then(() => {
    ctx.clearRect(0, 0, 640, 480);
    ctx.fillStyle = '#0a0e1a';
    ctx.fillRect(0, 0, 640, 480);
    ctx.fillStyle = '#a78bfa';
    ctx.font = 'bold 20px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Camera ready — Press ▶ Start', 320, 240);
  }).catch((err) => {
    ctx.clearRect(0, 0, 640, 480);
    ctx.fillStyle = '#0a0e1a';
    ctx.fillRect(0, 0, 640, 480);
    ctx.fillStyle = '#f87171';
    ctx.font = 'bold 16px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Camera access denied.', 320, 220);
    ctx.fillStyle = '#94a3b8';
    ctx.font = '14px Inter, sans-serif';
    ctx.fillText('Please allow camera access and refresh.', 320, 255);
    showToast('Camera permission denied — please allow access', 'error');
  });
}

// ===== Home Page Bindings =====
function bindHome() {
  const start = document.getElementById('startBtn');
  const stop = document.getElementById('stopBtn');
  const save = document.getElementById('saveCfg');

  if (start) {
    start.onclick = async () => {
      if (window.DEMO_MODE) {
        browserRunning = true;
        updateStatusUI(true, 'None', browserVolume);
        showToast('👋 Gesture detection started! Show your hand.');
      } else {
        const res = await postJSON('/api/start');
        showToast(res.demo ? 'Demo mode active' : 'Gesture control started!');
        refreshStatus();
      }
    };
  }
  if (stop) {
    stop.onclick = async () => {
      if (window.DEMO_MODE) {
        browserRunning = false;
        updateStatusUI(false, 'None', browserVolume);
        showToast('Gesture detection stopped', 'error');
      } else {
        await postJSON('/api/stop');
        showToast('Gesture control stopped', 'error');
        refreshStatus();
      }
    };
  }
  if (save) {
    save.onclick = async () => {
      const data = {
        scroll_sensitivity: Number(document.getElementById('scroll')?.value),
        mouse_sensitivity: Number(document.getElementById('mouse')?.value),
        cooldown: Number(document.getElementById('cooldown')?.value),
      };
      if (!window.DEMO_MODE) await postJSON('/api/config', data);
      showToast('Settings saved!');
    };
  }

  // Bind slider displays
  bindSliderValue('scroll', 'scrollVal');
  bindSliderValue('mouse', 'mouseVal');
  bindSliderValue('cooldown', 'cooldownVal');

  if (!window.DEMO_MODE) {
    refreshStatus();
    setInterval(refreshStatus, 800);
  }
}

// ===== Init =====
document.addEventListener('DOMContentLoaded', () => {
  bindHome();
  initReveal();
  initButtonGlow();

  // Boot browser-side gesture engine in demo mode
  if (window.DEMO_MODE) {
    initBrowserGesture();
  }

  // Lightbox close
  const lbClose = document.getElementById('lightboxClose');
  const lb = document.getElementById('lightbox');
  if (lbClose) lbClose.onclick = closeLightbox;
  if (lb) lb.addEventListener('click', (e) => {
    if (e.target === lb) closeLightbox();
  });

  // ESC to close lightbox
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') closeLightbox();
  });
});
