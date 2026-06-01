export default function ({ setTriggerValue, parentElement, data }) {
  // Guard: camera initialises once; subsequent fragment reruns only update display/settings.
  if (parentElement._cameraReady) {
    if (parentElement._onUpdate) parentElement._onUpdate(data);
    return;
  }
  parentElement._cameraReady = true;

  const video    = parentElement.querySelector('#video');
  const overlay  = parentElement.querySelector('#overlay');
  const ctx      = overlay.getContext('2d');
  const guideEl  = parentElement.querySelector('#guidance');
  const confFill = parentElement.querySelector('#conf-fill');
  const errEl    = parentElement.querySelector('#error');

  let stream         = null;
  let paused         = false;
  let facingMode     = 'environment';
  let isCapturing    = false;
  let lastSpokenText = '';
  let autoSpeak      = false;
  let speakGuidance  = false;

  // ── Camera ──────────────────────────────────────────────────────────────────
  async function startCamera(facing) {
    if (stream) stream.getTracks().forEach(t => t.stop());
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: facing, width: { ideal: 1280 }, height: { ideal: 720 } }
      });
      video.srcObject = stream;
      await video.play();
      overlay.width  = video.videoWidth  || 640;
      overlay.height = video.videoHeight || 480;
      errEl.style.display = 'none';
    } catch (err) {
      errEl.textContent = 'Camera error: ' + err.message;
      errEl.style.display = 'block';
      setTriggerValue('frame', null);
    }
  }

  // ── Frame capture ────────────────────────────────────────────────────────────
  function captureFrame() {
    if (paused || isCapturing || !video.videoWidth) return;
    isCapturing = true;
    const tmp = document.createElement('canvas');
    tmp.width  = video.videoWidth;
    tmp.height = video.videoHeight;
    tmp.getContext('2d').drawImage(video, 0, 0);
    tmp.toBlob(blob => {
      const reader = new FileReader();
      reader.onloadend = () => {
        setTriggerValue('frame', { frame: reader.result.split(',')[1], ts: Date.now() });
        isCapturing = false;
      };
      reader.readAsDataURL(blob);
    }, 'image/jpeg', 0.85);
  }

  // ── Result display ───────────────────────────────────────────────────────────
  function updateDisplay(result) {
    if (!result) return;
    if (result.annotated_frame) {
      const img = new Image();
      img.onload = () => {
        ctx.clearRect(0, 0, overlay.width, overlay.height);
        ctx.globalAlpha = 0.55;
        ctx.drawImage(img, 0, 0, overlay.width, overlay.height);
        ctx.globalAlpha = 1.0;
      };
      img.src = 'data:image/jpeg;base64,' + result.annotated_frame;
    }
    if (result.guidance) {
      guideEl.textContent = result.guidance.message || '';
      guideEl.style.color = result.guidance.status === 'ok' ? '#66ff66' : '#ffcc00';
      if (speakGuidance && result.guidance.status === 'warn') speakText(result.guidance.message);
    }
    if (result.confidence !== undefined) {
      confFill.style.width = Math.round(result.confidence * 100) + '%';
    }
    if (autoSpeak && result.text && result.text !== lastSpokenText) {
      lastSpokenText = result.text;
      speakText(result.text);
    }
  }

  // ── Speech ───────────────────────────────────────────────────────────────────
  function speakText(text) {
    if (!text || !('speechSynthesis' in window)) return;
    window.speechSynthesis.cancel();
    const utt  = new SpeechSynthesisUtterance(text);
    utt.lang   = 'en-US';
    utt.rate   = 0.9;
    window.speechSynthesis.speak(utt);
  }

  // ── Data update (called on every fragment rerun via the guard at top) ─────────
  parentElement._onUpdate = function (d) {
    if (!d) return;
    autoSpeak     = !!d.auto_speak;
    speakGuidance = !!d.speak_guidance;
    updateDisplay(d.result || null);
  };
  parentElement._onUpdate(data);

  // ── Controls ─────────────────────────────────────────────────────────────────
  parentElement.querySelector('#btn-pause').addEventListener('click', () => {
    paused = !paused;
    parentElement.querySelector('#btn-pause').textContent = paused ? '▶ Resume' : '⏸ Pause';
  });
  parentElement.querySelector('#btn-flip').addEventListener('click', () => {
    facingMode = facingMode === 'environment' ? 'user' : 'environment';
    startCamera(facingMode);
  });

  // ── Boot ─────────────────────────────────────────────────────────────────────
  startCamera(facingMode).then(() => {
    setInterval(captureFrame, 300);
  });
}
