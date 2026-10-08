/* One standing calibration shared by uploads and history reanalysis. */
window.Calibration = (() => {
  const q = (id) => document.getElementById(id);
  let mainUrl = '', savedReference = '', objectUrls = [], historyId = null;
  let head = null, ground = null, markerSeconds = null, mark = 'head', snapshot = null;
  const video = q('calibrationVideo'), canvas = q('calibrationCanvas');
  function clear() {
    head = ground = markerSeconds = snapshot = null;
    canvas.removeAttribute('width'); canvas.removeAttribute('height');
    q('calibrationStatus').textContent = '尚未标记';
  }
  function loadPreview() {
    const reference = q('referenceSource').value === 'reference';
    q('referenceFileLabel').classList.toggle('hidden', !reference);
    const file = q('referenceInput').files[0];
    let url = mainUrl;
    if (reference) {
      url = savedReference;
      if (file) { url = URL.createObjectURL(file); objectUrls.push(url); }
    }
    video.src = url || ''; clear();
  }
  function status() {
    q('calibrationStatus').textContent = `标记画面 ${markerSeconds == null ? '—' : markerSeconds.toFixed(2) + ' 秒'} · 头顶 ${head ? '已标记' : '未标记'} · 脚底 ${ground ? '已标记' : '未标记'}`;
  }
  function render() {
    if (!snapshot) return;
    const ctx = canvas.getContext('2d'); ctx.drawImage(snapshot, 0, 0);
    [[head, '头顶'], [ground, '脚底支撑']].forEach(([p, label]) => {
      if (!p) return;
      const x = p[0] * canvas.width, y = p[1] * canvas.height;
      ctx.fillStyle = '#c9ff42'; ctx.beginPath(); ctx.arc(x, y, 6, 0, Math.PI * 2); ctx.fill();
      ctx.font = '16px sans-serif'; ctx.fillText(label, Math.min(x + 10, canvas.width-90), Math.max(20, y-10));
    });
    if (head && ground) {
      ctx.strokeStyle = '#c9ff42'; ctx.beginPath(); ctx.moveTo(head[0]*canvas.width, head[1]*canvas.height);
      ctx.lineTo(ground[0]*canvas.width, ground[1]*canvas.height); ctx.stroke();
    }
    status();
  }
  function enabledState() {
    ['heightCm', 'runningFacing', 'referenceSource', 'referenceInput', 'referenceStart', 'referenceEnd'].forEach((id) => { q(id).disabled = !q('correctionEnabled').checked; });
  }
  q('correctionEnabled').addEventListener('change', enabledState);
  enabledState();
  q('captureReference').addEventListener('click', () => {
    if (!video.videoWidth || video.readyState < 2) { q('calibrationStatus').textContent = '请先载入可播放的视频'; return; }
    video.pause(); clear();
    canvas.width = Math.min(video.videoWidth, 960); canvas.height = Math.round(canvas.width*video.videoHeight/video.videoWidth);
    snapshot = document.createElement('canvas'); snapshot.width = canvas.width; snapshot.height = canvas.height;
    snapshot.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
    markerSeconds = video.currentTime; mark = 'head'; render();
  });
  canvas.addEventListener('click', (event) => {
    if (!snapshot) return;
    const rect = canvas.getBoundingClientRect();
    const p = [(event.clientX-rect.left)/rect.width, (event.clientY-rect.top)/rect.height];
    if (mark === 'head') { head = p; mark = 'ground'; } else ground = p;
    render();
  });
  q('markHead').addEventListener('click', () => { mark = 'head'; q('calibrationInstructions').textContent = '在标定图像中点击头顶'; });
  q('markGround').addEventListener('click', () => { mark = 'ground'; q('calibrationInstructions').textContent = '点击双脚支撑位置中点，位于跑台表面'; });
  q('clearMarkers').addEventListener('click', () => { head = ground = null; mark = 'head'; render(); });
  q('setReferenceStart').addEventListener('click', () => { q('referenceStart').value = video.currentTime.toFixed(2); });
  q('setReferenceEnd').addEventListener('click', () => { q('referenceEnd').value = video.currentTime.toFixed(2); });
  q('referenceSource').addEventListener('change', loadPreview);
  q('referenceInput').addEventListener('change', () => { savedReference = ''; loadPreview(); });
  video.addEventListener('error', () => { q('calibrationStatus').textContent = '浏览器无法播放此参考视频，请转为 H.264 MP4 后标定'; });
  window.addEventListener('beforeunload', () => objectUrls.forEach((url) => URL.revokeObjectURL(url)));
  return {
    get historyId() { return historyId; },
    setMainFile(file) {
      objectUrls.forEach((url) => URL.revokeObjectURL(url)); objectUrls = [];
      mainUrl = URL.createObjectURL(file); objectUrls.push(mainUrl);
      savedReference = ''; historyId = null; q('referenceInput').value = '';
      q('referenceSource').value = 'video'; loadPreview();
      q('analyzeButton').innerHTML = '开始分析 <span>→</span>';
    },
    editHistory(job) {
      historyId = job.id; mainUrl = job.artifacts['player.mp4'] || job.artifacts.source;
      savedReference = job.artifacts.reference || ''; q('videoInput').value = ''; q('referenceInput').value = '';
      q('correctionEnabled').checked = true; enabledState(); q('calibrationPanel').open = true;
      const c = job.calibration;
      q('referenceSource').value = c?.source || 'video';
      q('heightCm').value = c?.height_cm || 170; q('runningFacing').value = c?.facing || 'right';
      q('referenceStart').value = c?.start_seconds ?? 0; q('referenceEnd').value = c?.end_seconds ?? 3;
      loadPreview();
      q('fileName').textContent = `重新分析：${job.filename}`;
      q('analyzeButton').disabled = false; q('analyzeButton').innerHTML = '保存标定并重新分析 <span>→</span>';
      q('calibrationStatus').textContent = '请重新截取站立画面并标记头顶和脚底支撑点';
      if (c) video.addEventListener('loadedmetadata', () => { video.currentTime = c.marker_seconds; }, { once: true });
    },
    addTo(body) {
      if (!q('correctionEnabled').checked) { body.append('calibration', 'null'); return; }
      if (!head || !ground || markerSeconds == null) throw new Error('请截取站立画面并标记头顶、脚底支撑点');
      const start = Number(q('referenceStart').value), end = Number(q('referenceEnd').value);
      const height = Number(q('heightCm').value);
      if (!(height >= 100 && height <= 230 && end-start >= 1 && end-start <= 10 && markerSeconds >= start && markerSeconds <= end)) {
        throw new Error('请输入 100–230 cm 的身高、1–10 秒的站立片段，并在片段内截取标定画面');
      }
      const source = q('referenceSource').value;
      const file = q('referenceInput').files[0];
      if (source === 'reference' && !file && !savedReference) throw new Error('请上传同机位的站立参考视频');
      body.append('calibration', JSON.stringify({ height_cm: height, start_seconds: start, end_seconds: end,
        marker_seconds: markerSeconds, head, ground, source, facing: q('runningFacing').value }));
      if (source === 'reference' && file) body.append('reference_video', file);
    },
  };
})();
