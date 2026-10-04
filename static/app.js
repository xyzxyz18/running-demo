const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

const CONNECTIONS = [
  [11,12],[11,23],[12,24],[23,24],[11,13],[13,15],[12,14],[14,16],
  [23,25],[25,27],[27,29],[29,31],[27,31],[24,26],[26,28],[28,30],[30,32],[28,32],
];
const COLORS = { left: '#c9ff42', right: '#32d5ff', grid: 'rgba(255,255,255,.09)', muted: '#77817d' };
let timeline = null;
let analysisJob = null;
let cameraStream = null;
let cameraRunning = false;
let cameraEpoch = 0;
let liveSeries = { left: [], right: [] };
const footHighlight = { left: null, right: null };

function switchMode(mode) {
  $$('.mode-button').forEach((button) => button.classList.toggle('active', button.dataset.mode === mode));
  $('#uploadMode').classList.toggle('hidden', mode !== 'upload');
  $('#cameraMode').classList.toggle('hidden', mode !== 'camera');
  $('#historyMode').classList.toggle('hidden', mode !== 'history');
  if (mode !== 'camera') stopCamera();
  if (mode === 'history') loadHistory();
}
$$('.mode-button').forEach((button) => button.addEventListener('click', () => switchMode(button.dataset.mode)));

function showSelectedFile(file) {
  if (!file) return;
  $('#fileName').textContent = file.name;
  $('#analyzeButton').disabled = false;
}
$('#videoInput').addEventListener('change', (event) => showSelectedFile(event.target.files[0]));
['dragenter', 'dragover'].forEach((name) => $('#dropZone').addEventListener(name, (event) => {
  event.preventDefault(); $('#dropZone').classList.add('dragging');
}));
['dragleave', 'drop'].forEach((name) => $('#dropZone').addEventListener(name, (event) => {
  event.preventDefault(); $('#dropZone').classList.remove('dragging');
}));
$('#dropZone').addEventListener('drop', (event) => {
  const file = event.dataTransfer.files[0];
  if (!file) return;
  const transfer = new DataTransfer(); transfer.items.add(file);
  $('#videoInput').files = transfer.files; showSelectedFile(file);
});

$('#uploadForm').addEventListener('submit', async (event) => {
  event.preventDefault();
  const file = $('#videoInput').files[0];
  if (!file) return;
  $('#analyzeButton').disabled = true;
  $('#statusCard').classList.remove('hidden', 'error');
  $('#statusTitle').textContent = '正在上传';
  $('#statusMessage').textContent = '视频较大时需要稍等片刻…';
  try {
    const body = new FormData(); body.append('video', file);
    const response = await fetch('/api/jobs', { method: 'POST', body });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || '上传失败');
    $('#uploadHero').classList.add('hidden');
    pollJob(payload.status_url);
  } catch (error) { showAnalysisError(error.message); }
});

async function pollJob(url) {
  try {
    const response = await fetch(url, { cache: 'no-store' });
    const job = await response.json();
    if (!response.ok) throw new Error(job.error || '无法读取分析状态');
    $('#statusTitle').textContent = job.state === 'queued' ? '等待分析' : `RTMPose 正在逐帧分析`;
    $('#statusMessage').textContent = job.message;
    if (job.state === 'completed') return prepareWorkspace(job);
    if (job.state === 'failed') return showAnalysisError(job.message);
    setTimeout(() => pollJob(url), 1200);
  } catch (error) { showAnalysisError(error.message); }
}

function showAnalysisError(message) {
  $('#statusCard').classList.add('error');
  $('#statusTitle').textContent = '分析失败';
  $('#statusMessage').textContent = message;
  $('#uploadHero').classList.remove('hidden');
  $('#analyzeButton').disabled = false;
}

async function prepareWorkspace(job) {
  analysisJob = job;
  $('#resultModel').textContent = `${job.model === 'rtmpose' ? 'RTMPose' : '历史分析'} · POSE OVERLAY`;
  if (!job.artifacts['timeline.json'] || !job.artifacts.source) return showLegacyResult(job);
  const response = await fetch(job.artifacts['timeline.json']);
  if (!response.ok) return showAnalysisError('无法加载逐帧姿态数据');
  timeline = await response.json();
  $('#statusCard').classList.add('hidden');
  $('#analysisWorkspace').classList.remove('hidden');
  $('#reviewGrid').classList.remove('hidden');
  $('#reviewCharts').classList.remove('hidden');
  $('#legacyReportPanel').classList.add('hidden');
  const video = $('#analysisVideo');
  video.src = job.artifacts['player.mp4'] || job.artifacts.source;
  video.load();
  renderMetrics(job.result.metrics);
  renderFeedback(job);
  renderFootMotion(job.result.foot_motion);
  renderReport(job.result);
  drawAllReviewCharts(0);
  $('#analysisWorkspace').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function showLegacyResult(job) {
  analysisJob = job; timeline = null;
  $('#statusCard').classList.add('hidden');
  $('#analysisWorkspace').classList.remove('hidden');
  $('#reviewGrid').classList.add('hidden');
  $('#reviewCharts').classList.add('hidden');
  $('#legacyReportPanel').classList.toggle('hidden', !job.artifacts['report.png']);
  if (job.artifacts['report.png']) $('#legacyReportImage').src = job.artifacts['report.png'];
  renderMetrics(job.result.metrics); renderFeedback(job);
  renderFootMotion(job.result.foot_motion);
  renderReport(job.result);
  $('#analysisWorkspace').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function renderMetrics(metrics) {
  const definitions = [
    ['cadence_steps_per_min', '步频', 'SPM'], ['detected_steps', '检测步数', 'STEPS'],
    ['step_time_seconds', '平均步时', 'SEC'], ['knee_rom_degrees', '膝关节 ROM', 'DEG'],
    ['hip_rom_degrees', '髋关节 ROM', 'DEG'], ['stride_time_asymmetry_percent', '左右跨步差', '%'],
  ];
  const root = $('#metricsPanel'); root.replaceChildren();
  definitions.forEach(([key, label, unit]) => {
    const card = document.createElement('div'); card.className = 'metric-card';
    const value = metrics[key];
    card.innerHTML = `<small>${label}</small><div><strong>${value == null ? '—' : value}</strong><em>${unit}</em></div>`;
    root.append(card);
  });
}

function renderFeedback(job) {
  const pdfButton = $('#exportPdfButton');
  pdfButton.classList.toggle('hidden', !job.artifacts?.['report.pdf']);
  if (job.artifacts?.['report.pdf']) pdfButton.href = job.artifacts['report.pdf'];
  const list = $('#feedbackList'); list.replaceChildren();
  job.result.feedback.forEach((value, index) => {
    const item = document.createElement('li');
    item.innerHTML = `<span>${String(index + 1).padStart(2, '0')}</span><p>${value}</p>`; list.append(item);
  });
  $('#disclaimer').textContent = job.result.disclaimer;
  const downloads = $('#downloadLinks'); downloads.replaceChildren();
  const files = [['report.pdf', 'PDF 报告'], ['player.mp4', '可播放视频'], ['annotated.mp4', '标注视频'], ['metrics.json', '指标 JSON'], ['landmarks.csv', '关键点 CSV'], ['report.png', '分析图'], ['report.html', '简要报告']];
  files.forEach(([name, label]) => {
    if (!job.artifacts[name]) return;
    const link = document.createElement('a'); link.href = job.artifacts[name]; link.download = name;
    link.innerHTML = `<span>${label}</span><b>↓</b>`; downloads.append(link);
  });
}

function renderFootMotion(motion) {
  const available = motion?.version === 6 && timeline?.foot_motion?.version === 6;
  $('#footMotionPanel').classList.toggle('hidden', !available);
  if (!available) return;
  $('#stabilityStatus').textContent = `${motion.assessment} · 总偏差 ${motion.overall_dispersion_body_ratio ?? '—'}`;
  ['left', 'right'].forEach((side) => {
    const dispersion = motion[side].dispersion_body_ratio;
    $(`#${side}CycleCount`).textContent = `${motion[side].cycle_count} 周期 / ${motion[side].included_count} 纳入平均 · 侧内偏差 ${dispersion ?? '—'}`;
    const list = $(`#${side}CycleList`); list.replaceChildren();
    timeline.foot_motion[side].cycles.forEach((cycle) => {
      const item = document.createElement('button'); item.type = 'button'; item.className = 'cycle-item';
      item.dataset.cycle = cycle.number;
      if (!cycle.path) item.classList.add('unavailable');
      if (!cycle.included_in_mean) item.classList.add('excluded');
      const number = document.createElement('span'); number.textContent = `#${cycle.number}`;
      const detail = document.createElement('span'); detail.textContent = `${cycle.start_frame}–${cycle.end_frame} 帧 · 可见 ${cycle.visibility_percent}%`;
      const deviation = document.createElement('b');
      deviation.textContent = cycle.deviation_body_ratio == null ? cycle.status : `${cycle.status} · ${cycle.deviation_body_ratio}`;
      item.append(number, detail, deviation);
      item.addEventListener('pointerenter', () => { footHighlight[side] = cycle.number; drawFootMotionCharts(); });
      item.addEventListener('pointerleave', () => { footHighlight[side] = null; drawFootMotionCharts(); });
      item.addEventListener('focus', () => { footHighlight[side] = cycle.number; drawFootMotionCharts(); });
      item.addEventListener('blur', () => { footHighlight[side] = null; drawFootMotionCharts(); });
      item.addEventListener('click', () => {
        const video = $('#analysisVideo');
        const stamps = timeline.timestamps;
        if (video.duration && stamps?.length > 1 && stamps.at(-1) > 0) {
          video.currentTime = video.duration * stamps[cycle.start_frame] / stamps.at(-1);
        } else if (video.duration && timeline.frame_count > 1) {
          video.currentTime = video.duration * cycle.start_frame / (timeline.frame_count - 1);
        }
      });
      list.append(item);
    });
  });
  drawFootMotionCharts(frameForTime($('#analysisVideo').currentTime, $('#analysisVideo').duration));
}

function drawFootMotionCharts(frame = frameForTime($('#analysisVideo').currentTime, $('#analysisVideo').duration)) {
  const motion = timeline?.foot_motion;
  if (motion?.version !== 6 || $('#footMotionPanel').classList.contains('hidden')) return;
  const referencePaths = ['left', 'right'].flatMap((side) => [
    ...motion[side].cycles.filter((cycle) => cycle.included_in_mean).map((cycle) => cycle.path?.filter((_, index) => {
      const phase = index / (cycle.path.length - 1);
      return phase >= (cycle.phase_start ?? 0) && phase <= (cycle.phase_end ?? 1);
    })).filter(Boolean), motion[side].mean_path,
  ]).filter((path) => path?.length);
  const allPaths = referencePaths.length ? referencePaths : ['left', 'right'].flatMap((side) =>
    motion[side].cycles.map((cycle) => cycle.path).filter(Boolean));
  const bounds = { minX: Infinity, maxX: -Infinity, minY: Infinity, maxY: -Infinity };
  allPaths.forEach((path) => path.forEach(([x, y]) => {
    bounds.minX = Math.min(bounds.minX, x); bounds.maxX = Math.max(bounds.maxX, x);
    bounds.minY = Math.min(bounds.minY, y); bounds.maxY = Math.max(bounds.maxY, y);
  }));
  for (const side of ['left', 'right']) {
    const { context: ctx, width, height } = fitCanvas($(`#${side}FootChart`));
    ctx.clearRect(0, 0, width, height);
    const cycles = motion[side].cycles || [];
    if (!Number.isFinite(bounds.minX) || !cycles.some((cycle) => cycle.path)) {
      ctx.fillStyle = COLORS.muted; ctx.font = '13px system-ui'; ctx.textAlign = 'center';
      ctx.fillText('没有可绘制的周期', width / 2, height / 2); continue;
    }
    const rangeX = Math.max(bounds.maxX - bounds.minX, .2) * 1.14;
    const rangeY = Math.max(bounds.maxY - bounds.minY, .2) * 1.14;
    const centerX = (bounds.minX + bounds.maxX) / 2;
    const centerY = (bounds.minY + bounds.maxY) / 2;
    const pixelsPerUnit = Math.min((width - 60) / rangeX, (height - 46) / rangeY);
    const project = ([x, y]) => [width / 2 + (x - centerX) * pixelsPerUnit,
      height / 2 - (y - centerY) * pixelsPerUnit];
    ctx.strokeStyle = COLORS.grid; ctx.lineWidth = 1;
    const origin = project([0, 0]);
    ctx.beginPath(); ctx.moveTo(origin[0], 0); ctx.lineTo(origin[0], height); ctx.moveTo(0, origin[1]); ctx.lineTo(width, origin[1]); ctx.stroke();
    const color = COLORS[side];
    const trace = (path, alpha, lineWidth) => {
      if (!path?.length) return;
      ctx.strokeStyle = color; ctx.globalAlpha = alpha; ctx.lineWidth = lineWidth; ctx.beginPath();
      path.forEach((point, index) => { const [x, y] = project(point); if (index) ctx.lineTo(x, y); else ctx.moveTo(x, y); });
      ctx.stroke(); ctx.globalAlpha = 1;
    };
    if (motion[side].mean_path?.length) trace(motion[side].mean_path, .9, 4);
    cycles.forEach((cycle) => {
      const active = frame >= cycle.start_frame && frame <= cycle.end_frame;
      const completed = frame > cycle.end_frame;
      const item = $(`#${side}CycleList .cycle-item[data-cycle="${cycle.number}"]`);
      if (item) item.classList.toggle('active', active);
      if (!cycle.path || (!active && !completed)) return;
      const first = Math.max(0, Math.ceil((cycle.phase_start ?? 0) * (cycle.path.length - 1)));
      const last = Math.min(cycle.path.length - 1, Math.floor((cycle.phase_end ?? 1) * (cycle.path.length - 1)));
      const stamps = timeline.timestamps;
      const validTime = stamps?.length === timeline.frame_count &&
        stamps[cycle.end_frame] > stamps[cycle.start_frame];
      const progress = active ? (validTime
        ? (stamps[frame] - stamps[cycle.start_frame]) / (stamps[cycle.end_frame] - stamps[cycle.start_frame])
        : (frame - cycle.start_frame) / (cycle.end_frame - cycle.start_frame)) : 1;
      const visibleLast = Math.min(last, Math.floor(progress * (cycle.path.length - 1)));
      const visiblePath = cycle.path.slice(first, visibleLast + 1);
      const highlighted = active || cycle.number === footHighlight[side];
      trace(visiblePath, highlighted ? 1 : cycle.included_in_mean ? .3 : .12,
        highlighted ? 3 : 1.4);
      if (active && visiblePath.length) {
        const [x, y] = project(visiblePath.at(-1));
        ctx.beginPath(); ctx.arc(x, y, 5, 0, 2 * Math.PI); ctx.fillStyle = color; ctx.fill();
      }
    });
    ctx.fillStyle = COLORS.muted; ctx.font = '10px ui-monospace, monospace'; ctx.textAlign = 'left';
    ctx.fillText('前后位移 →', 12, height - 10); ctx.fillText('↑ 向上', 12, 17);
    ctx.fillText(`${bounds.minX.toFixed(1)} … ${bounds.maxX.toFixed(1)}`, width - 92, height - 10);
  }
}

function renderReport(result) {
  const report = result.report || {
    title: '跑姿分析简报', summary: '这份历史结果没有足部稳定性分析，可重新分析原视频以生成新版报告。',
    observations: result.feedback || [], method: '基于单目侧面视频的二维姿态观察。',
  };
  $('#expandableReport').classList.remove('hidden');
  $('#expandableReport').open = false;
  const root = $('#reportContent'); root.replaceChildren();
  const heading = document.createElement('h3'); heading.textContent = report.title;
  const summary = document.createElement('p'); summary.className = 'report-summary'; summary.textContent = report.summary;
  const list = document.createElement('ul');
  (report.observations || []).forEach((observation) => { const item = document.createElement('li'); item.textContent = observation; list.append(item); });
  const method = document.createElement('p'); method.className = 'report-method'; method.textContent = report.method;
  root.append(heading, summary, list, method);
}

function frameForTime(time, duration) {
  if (!timeline) return 0;
  let rawFrame;
  const timestamps = timeline.timestamps;
  if (timestamps && timestamps.length === timeline.frame_count && timestamps.length > 1) {
    const lastTimestamp = timestamps[timestamps.length - 1];
    const target = duration > 0 && lastTimestamp > 0 ? time * lastTimestamp / duration : time;
    let low = 0, high = timestamps.length - 1;
    while (low < high) {
      const middle = Math.floor((low + high) / 2);
      if (timestamps[middle] < target) low = middle + 1; else high = middle;
    }
    rawFrame = low > 0 && Math.abs(timestamps[low - 1] - target) < Math.abs(timestamps[low] - target) ? low - 1 : low;
  } else {
    rawFrame = duration > 0
      ? Math.round((time / duration) * (timeline.frame_count - 1))
      : Math.round(time * timeline.fps);
  }
  return Math.max(0, Math.min(timeline.frame_count - 1, rawFrame));
}

function updateReview() {
  if (!timeline) return;
  const video = $('#analysisVideo');
  const frame = frameForTime(video.currentTime, video.duration);
  drawSkeleton($('#poseCanvas'), timeline.landmarks[frame], video, false);
  const angles = timeline.angles;
  $('#leftKneeValue').textContent = valueText(angles.left_knee[frame]);
  $('#rightKneeValue').textContent = valueText(angles.right_knee[frame]);
  $('#frameLabel').textContent = `FRAME ${frame + 1} / ${timeline.frame_count}`;
  $('#eventValue').textContent = eventText(timeline.events[String(frame)]);
  drawAllReviewCharts(frame);
  drawFootMotionCharts(frame);
}

function eventText(value) {
  if (!value) return '—';
  return value.replace('Left', '左脚').replace('Right', '右脚').replace('foot strike', '着地').replace('toe-off', '离地');
}
function valueText(value) { return value == null ? '—' : Number(value).toFixed(1); }

$('#analysisVideo').addEventListener('timeupdate', updateReview);
$('#analysisVideo').addEventListener('seeked', updateReview);
$('#analysisVideo').addEventListener('loadedmetadata', updateReview);
$('#analysisVideo').addEventListener('play', function animate() {
  if ($('#analysisVideo').paused || $('#analysisVideo').ended) return;
  updateReview(); requestAnimationFrame(animate);
});

function fitCanvas(canvas) {
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  const width = Math.max(1, Math.round(rect.width * ratio));
  const height = Math.max(1, Math.round(rect.height * ratio));
  if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
  const context = canvas.getContext('2d'); context.setTransform(ratio, 0, 0, ratio, 0, 0);
  return { context, width: rect.width, height: rect.height };
}

function videoRect(video, width, height) {
  if (!video.videoWidth || !video.videoHeight) return { x: 0, y: 0, width, height };
  const scale = Math.min(width / video.videoWidth, height / video.videoHeight);
  const drawWidth = video.videoWidth * scale, drawHeight = video.videoHeight * scale;
  return { x: (width - drawWidth) / 2, y: (height - drawHeight) / 2, width: drawWidth, height: drawHeight };
}

function drawSkeleton(canvas, landmarks, video, mirrored) {
  const { context: ctx, width, height } = fitCanvas(canvas); ctx.clearRect(0, 0, width, height);
  if (!landmarks) return;
  const rect = videoRect(video, width, height);
  const point = (index) => {
    const value = landmarks[index];
    if (!value || value[0] == null || value[1] == null || value[3] < .45) return null;
    return [rect.x + value[0] * rect.width, rect.y + value[1] * rect.height];
  };
  ctx.lineCap = 'round'; ctx.lineJoin = 'round'; ctx.lineWidth = Math.max(2, width / 260);
  CONNECTIONS.forEach(([a, b]) => {
    const start = point(a), end = point(b); if (!start || !end) return;
    const side = a % 2 === 1 ? 'left' : 'right'; ctx.strokeStyle = COLORS[side];
    ctx.shadowColor = COLORS[side]; ctx.shadowBlur = 8;
    ctx.beginPath(); ctx.moveTo(...start); ctx.lineTo(...end); ctx.stroke();
  });
  ctx.shadowBlur = 5;
  [...new Set(CONNECTIONS.flat())].forEach((index) => {
    const p = point(index); if (!p) return;
    ctx.fillStyle = index % 2 === 1 ? COLORS.left : COLORS.right;
    ctx.beginPath(); ctx.arc(p[0], p[1], Math.max(3, width / 180), 0, Math.PI * 2); ctx.fill();
  });
  ctx.shadowBlur = 0;
}

function drawChart(canvas, left, right, cursor = null) {
  const { context: ctx, width, height } = fitCanvas(canvas);
  ctx.clearRect(0, 0, width, height);
  const pad = { left: 38, right: 14, top: 14, bottom: 22 };
  const chartW = width - pad.left - pad.right, chartH = height - pad.top - pad.bottom;
  ctx.font = '10px ui-monospace, monospace'; ctx.textAlign = 'right'; ctx.fillStyle = COLORS.muted;
  [0, 45, 90, 135, 180].forEach((value) => {
    const y = pad.top + chartH * (1 - value / 180);
    ctx.strokeStyle = COLORS.grid; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(width - pad.right, y); ctx.stroke();
    ctx.fillText(`${value}°`, pad.left - 7, y + 3);
  });
  const count = Math.max(left.length, right.length, 2);
  const plot = (values, color) => {
    ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.shadowColor = color; ctx.shadowBlur = 4; ctx.beginPath();
    let drawing = false;
    values.forEach((value, index) => {
      if (value == null) { drawing = false; return; }
      const x = pad.left + chartW * index / (count - 1);
      const y = pad.top + chartH * (1 - Math.max(0, Math.min(180, value)) / 180);
      if (!drawing) { ctx.moveTo(x, y); drawing = true; } else ctx.lineTo(x, y);
    });
    ctx.stroke(); ctx.shadowBlur = 0;
  };
  plot(left, COLORS.left); plot(right, COLORS.right);
  if (cursor != null && count > 1) {
    const x = pad.left + chartW * cursor / (count - 1);
    ctx.strokeStyle = '#fff'; ctx.lineWidth = 1; ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(x, pad.top); ctx.lineTo(x, pad.top + chartH); ctx.stroke(); ctx.setLineDash([]);
    [[left[cursor], COLORS.left, -1], [right[cursor], COLORS.right, 1]].forEach(([value, color, side]) => {
      if (value == null) return;
      const y = pad.top + chartH * (1 - Math.max(0, Math.min(180, value)) / 180);
      ctx.fillStyle = color; ctx.beginPath(); ctx.arc(x, y, 4, 0, Math.PI * 2); ctx.fill();
      const label = `${Number(value).toFixed(1)}°`;
      ctx.font = '600 10px ui-monospace, monospace';
      const labelWidth = ctx.measureText(label).width + 10;
      const labelX = side < 0 ? Math.max(pad.left, x - labelWidth - 7) : Math.min(width - pad.right - labelWidth, x + 7);
      ctx.fillStyle = '#f3f7f5'; ctx.fillRect(labelX, y - 9, labelWidth, 17);
      ctx.fillStyle = '#101513'; ctx.textAlign = 'center'; ctx.fillText(label, labelX + labelWidth / 2, y + 3);
    });
  }
}

function drawAllReviewCharts(frame) {
  if (!timeline) return;
  drawChart($('#kneeChart'), timeline.angles.left_knee, timeline.angles.right_knee, frame);
  drawChart($('#hipChart'), timeline.angles.left_hip, timeline.angles.right_hip, frame);
}

function seekFromChart(event) {
  if (!timeline || !$('#analysisVideo').duration) return;
  const rect = event.currentTarget.getBoundingClientRect();
  const ratio = Math.max(0, Math.min(1, (event.clientX - rect.left - 38) / (rect.width - 52)));
  $('#analysisVideo').currentTime = ratio * $('#analysisVideo').duration;
}
['kneeChart', 'hipChart'].forEach((id) => $(`#${id}`).addEventListener('pointerdown', seekFromChart));
$('#newVideoButton').addEventListener('click', () => window.location.reload());

async function loadHistory() {
  $('#historyLoading').classList.remove('hidden');
  $('#historyEmpty').classList.add('hidden');
  const root = $('#historyGrid'); root.replaceChildren();
  try {
    const response = await fetch('/api/jobs', { cache: 'no-store' });
    const payload = await response.json();
    if (!response.ok) throw new Error('无法读取历史记录');
    $('#historyLoading').classList.add('hidden');
    $('#historyEmpty').classList.toggle('hidden', payload.jobs.length > 0);
    payload.jobs.forEach((job) => root.append(createHistoryCard(job)));
  } catch (error) {
    $('#historyLoading').textContent = error.message;
  }
}

function createHistoryCard(job) {
  const card = document.createElement('article'); card.className = 'history-card';
  const preview = document.createElement('div'); preview.className = 'history-preview';
  if (job.artifacts && job.artifacts['report.png']) {
    const image = document.createElement('img'); image.src = job.artifacts['report.png']; image.alt = '';
    preview.append(image);
  } else { const mark = document.createElement('span'); mark.textContent = 'PACE'; preview.append(mark); }
  const state = document.createElement('b'); state.className = `history-state ${job.state}`;
  state.textContent = job.state === 'completed' ? '已完成' : '未完成'; preview.append(state);

  const body = document.createElement('div'); body.className = 'history-body';
  const date = new Date(job.created_at);
  const time = document.createElement('small');
  time.textContent = Number.isNaN(date.getTime()) ? '时间未知' : date.toLocaleString('zh-CN', { hour12: false });
  const title = document.createElement('h3'); title.textContent = job.filename || '历史视频';
  body.append(time, title);
  const modelLabel = document.createElement('small'); modelLabel.textContent = job.model === 'rtmpose' ? 'RTMPose' : '历史分析'; body.append(modelLabel);
  if (job.result) {
    const stats = document.createElement('div'); stats.className = 'history-stats';
    const metrics = job.result.metrics;
    stats.innerHTML = `<span><b>${metrics.cadence_steps_per_min ?? '—'}</b> 步频</span><span><b>${metrics.duration_seconds ?? '—'}s</b> 时长</span><span><b>${metrics.pose_detection_rate == null ? '—' : Math.round(metrics.pose_detection_rate * 100) + '%'}</b> 识别率</span>`;
    body.append(stats);
  }
  const actions = document.createElement('div'); actions.className = 'history-actions';
  const view = document.createElement('button'); view.textContent = '查看分析'; view.disabled = !job.result;
  view.addEventListener('click', () => openHistoryResult(job.id)); actions.append(view);
  const rerun = document.createElement('button'); rerun.textContent = '重新分析'; rerun.className = 'rerun'; rerun.disabled = !job.can_reanalyze;
  rerun.addEventListener('click', () => reanalyzeHistory(job.id)); actions.append(rerun);
  const remove = document.createElement('button'); remove.textContent = '删除记录'; remove.className = 'delete';
  remove.addEventListener('click', () => deleteHistory(job.id, job.filename || '历史记录', card)); actions.append(remove);
  body.append(actions); card.append(preview, body); return card;
}

async function openHistoryResult(jobId) {
  const response = await fetch(`/api/jobs/${jobId}`, { cache: 'no-store' });
  const job = await response.json();
  if (!response.ok || !job.result) return;
  switchMode('upload'); $('#uploadHero').classList.add('hidden');
  prepareWorkspace(job);
}

async function reanalyzeHistory(jobId) {
  const response = await fetch(`/api/jobs/${jobId}/reanalyze`, { method: 'POST' });
  const payload = await response.json();
  if (!response.ok) { alert(payload.error || '无法重新分析'); return; }
  switchMode('upload');
  $('#uploadHero').classList.add('hidden'); $('#analysisWorkspace').classList.add('hidden');
  $('#statusCard').classList.remove('hidden', 'error');
  $('#statusTitle').textContent = '正在重新分析'; $('#statusMessage').textContent = '已使用保留的原视频创建新任务…';
  pollJob(payload.status_url);
}

async function deleteHistory(jobId, filename, card) {
  if (!window.confirm(`确定删除“${filename}”及其所有分析文件吗？此操作无法撤销。`)) return;
  const response = await fetch(`/api/jobs/${jobId}`, { method: 'DELETE' });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) { alert(payload.error || '无法删除历史记录'); return; }
  card.remove();
  const remaining = $('#historyGrid').children.length;
  $('#historyEmpty').classList.toggle('hidden', remaining > 0);
}

$('#historyUploadButton').addEventListener('click', () => { switchMode('upload'); window.location.reload(); });

async function startCamera() {
  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' }, audio: false });
    const video = $('#cameraVideo'); video.srcObject = cameraStream; await video.play();
    cameraRunning = true; liveSeries = { left: [], right: [] };
    $('#cameraPlaceholder').classList.add('hidden');
    $('#cameraToggle').innerHTML = '停止检测 <span>■</span>';
    $('#cameraToggle').classList.add('stop');
    $('#liveStatus').classList.add('on'); $('#liveStatus').innerHTML = '<i></i>实时分析中';
    const epoch = ++cameraEpoch;
    liveLoop(epoch);
  } catch (error) {
    $('#detectionValue').textContent = error.name === 'NotAllowedError' ? '未授权摄像头' : '无法启动';
  }
}

function stopCamera() {
  cameraRunning = false;
  cameraEpoch += 1;
  if (cameraStream) cameraStream.getTracks().forEach((track) => track.stop());
  cameraStream = null; $('#cameraVideo').srcObject = null;
  $('#cameraPoseCanvas').getContext('2d').clearRect(0, 0, $('#cameraPoseCanvas').width, $('#cameraPoseCanvas').height);
  $('#cameraPlaceholder').classList.remove('hidden');
  $('#cameraToggle').innerHTML = '开启摄像头 <span>→</span>'; $('#cameraToggle').classList.remove('stop');
  $('#liveStatus').classList.remove('on'); $('#liveStatus').innerHTML = '<i></i>等待开始';
}

$('#cameraToggle').addEventListener('click', () => cameraRunning ? stopCamera() : startCamera());

async function liveLoop(epoch) {
  if (!cameraRunning || epoch !== cameraEpoch) return;
  const video = $('#cameraVideo');
  const capture = $('#captureCanvas');
  capture.width = 640; capture.height = Math.round(640 * video.videoHeight / video.videoWidth) || 360;
  capture.getContext('2d').drawImage(video, 0, 0, capture.width, capture.height);
  try {
    const blob = await new Promise((resolve) => capture.toBlob(resolve, 'image/jpeg', .72));
    if (!cameraRunning || epoch !== cameraEpoch) return;
    const response = await fetch('/api/realtime/pose', { method: 'POST', headers: { 'Content-Type': 'image/jpeg' }, body: blob });
    const pose = await response.json();
    if (!cameraRunning || epoch !== cameraEpoch) return;
    if (!response.ok) throw new Error(pose.error || '姿态分析失败');
    if (pose.detected) {
      drawSkeleton($('#cameraPoseCanvas'), pose.landmarks, video, true);
      pushLiveValue(pose.angles.left_knee, pose.angles.right_knee);
      $('#liveLeftKnee').textContent = valueText(pose.angles.left_knee);
      $('#liveRightKnee').textContent = valueText(pose.angles.right_knee);
      $('#detectionValue').textContent = '已识别全身';
    } else {
      $('#cameraPoseCanvas').getContext('2d').clearRect(0, 0, $('#cameraPoseCanvas').width, $('#cameraPoseCanvas').height);
      pushLiveValue(null, null); $('#detectionValue').textContent = '请保持全身入镜';
    }
  } catch (error) {
    if (!cameraRunning || epoch !== cameraEpoch) return;
    stopCamera(); $('#detectionValue').textContent = error.message;
  }
  if (cameraRunning && epoch === cameraEpoch) setTimeout(() => liveLoop(epoch), 35);
}

function pushLiveValue(left, right) {
  liveSeries.left.push(left); liveSeries.right.push(right);
  if (liveSeries.left.length > 180) { liveSeries.left.shift(); liveSeries.right.shift(); }
  drawChart($('#liveChart'), liveSeries.left, liveSeries.right);
}

new ResizeObserver(() => {
  if (timeline) updateReview();
  drawFootMotionCharts();
  drawChart($('#liveChart'), liveSeries.left, liveSeries.right);
}).observe(document.body);
