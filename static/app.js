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

// ===== Status Refresh =====
async function refreshStatus() {
  try {
    const s = await getJSON('/api/status');
    const runState = document.getElementById('runState');
    const gesture = document.getElementById('gesture');
    const volume = document.getElementById('volume');
    const dot = document.getElementById('statusDot');
    const volBar = document.getElementById('volBar');

    if (runState) runState.textContent = s.running ? 'running' : 'stopped';
    if (gesture) gesture.textContent = s.gesture || 'None';
    if (volume) volume.textContent = s.volume ?? 0;
    if (dot) {
      dot.classList.toggle('active', s.running);
    }
    if (volBar) {
      volBar.style.width = (s.volume ?? 0) + '%';
    }
  } catch (e) {
    /* ignore fetch errors */
  }
}

// ===== Home Page Bindings =====
function bindHome() {
  const start = document.getElementById('startBtn');
  const stop = document.getElementById('stopBtn');
  const save = document.getElementById('saveCfg');

  if (start) {
    start.onclick = async () => {
      const res = await postJSON('/api/start');
      showToast(res.demo ? 'Demo mode — run locally for control' : 'Gesture control started!');
      refreshStatus();
    };
  }
  if (stop) {
    stop.onclick = async () => {
      await postJSON('/api/stop');
      showToast('Gesture control stopped', 'error');
      refreshStatus();
    };
  }
  if (save) {
    save.onclick = async () => {
      const data = {
        scroll_sensitivity: Number(document.getElementById('scroll')?.value),
        mouse_sensitivity: Number(document.getElementById('mouse')?.value),
        cooldown: Number(document.getElementById('cooldown')?.value),
      };
      await postJSON('/api/config', data);
      showToast('Settings saved!');
    };
  }

  // Bind slider displays
  bindSliderValue('scroll', 'scrollVal');
  bindSliderValue('mouse', 'mouseVal');
  bindSliderValue('cooldown', 'cooldownVal');

  refreshStatus();
  setInterval(refreshStatus, 800);
}

// ===== Init =====
document.addEventListener('DOMContentLoaded', () => {
  bindHome();
  initReveal();
  initButtonGlow();

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
