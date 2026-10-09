/* Interactive 3D camera over temporal model xyz data. */
(() => {
  const panel = document.getElementById('skeleton3dPanel');
  const canvas = document.getElementById('skeleton3dCanvas');
  const status = document.getElementById('skeleton3dStatus');
  const label = document.getElementById('skeleton3dFrame');
  const sagittalPanel = document.getElementById('sagittal3dPanel');
  const sagittalStatus = document.getElementById('sagittal3dStatus');
  const charts = ['leftSagittal3d', 'rightSagittal3d'].map(id => document.getElementById(id));
  let sagittal = null, direction = 1, extent = null;
  let data = null, sample = 0, epoch = 0, yaw = -.55, pitch = .16, zoom = 1;
  let showOriginal = false;
  const originalButton=document.getElementById('toggleOriginal3d');
  const planeControl=document.getElementById('legPlaneEnabled');
  const trailsControl=document.getElementById('ankleTrailsEnabled');
  planeControl.addEventListener('change',()=>{showOriginal=false;originalButton.textContent='查看原始相机朝向';setupTrajectories();draw();drawTrajectories();});
  trailsControl.addEventListener('change',draw);
  originalButton.addEventListener('click',()=>{
    showOriginal=!showOriginal;
    originalButton.textContent=showOriginal?'返回显示校正朝向':'查看原始相机朝向';
    draw();
  });
  document.getElementById('groundExample3d').addEventListener('click',()=>{
    showOriginal=false;originalButton.textContent='查看原始相机朝向';
    yaw=-.45;pitch=.18;zoom=1;draw();
  });
  document.getElementById('exportSkeleton3d').addEventListener('click',()=>{
    if(!data)return;
    draw();const link=document.createElement('a');link.download=`3d-ground-${sample+1}.png`;
    link.href=canvas.toDataURL('image/png');link.click();
  });
  const pointers = new Map();
  const finitePoint = (p) => Array.isArray(p) && p.length === 3 && p.every(Number.isFinite);
  const colors = { left: '#c9ff42', right: '#32d5ff', center: '#edf5ef' };
  function color(index) {
    const name = data.joint_names[index];
    return name.startsWith('left_') ? colors.left : name.startsWith('right_') ? colors.right : colors.center;
  }
  function setTime(time) {
    if (!data) return;
    const stamps = data.timestamps;
    let low = 0, high = stamps.length - 1;
    while (low < high) {
      const mid = Math.floor((low + high) / 2);
      if (stamps[mid] < time) low = mid + 1; else high = mid;
    }
    sample = low > 0 && Math.abs(stamps[low - 1] - time) < Math.abs(stamps[low] - time) ? low - 1 : low;
    draw(); drawTrajectories();
  }
  function setupTrajectories() {
    sagittal = (planeControl.checked?data.plane_sagittal:data.sagittal) || {status:'unavailable',reason:'旧记录需重新分析生成侧面轨迹'}; direction = 1;
    document.getElementById('flipSagittal3d').disabled = true;
    sagittalPanel.classList.remove('hidden');
    charts.forEach(c => c.classList.toggle('hidden', sagittal.status !== 'available'));
    if (sagittal.status !== 'available') { sagittalStatus.textContent = `侧面投影不可用：${sagittal.reason}`; return; }
    const ids = ['left_hip','right_hip','left_knee','right_knee','left_ankle','right_ankle'].map(n => data.joint_names.indexOf(n));
    const points = sagittal.keypoints.flatMap(frame => ids.map(i => frame[i]).filter(Boolean));
    extent = points.reduce((e,p)=>({x:Math.max(e.x,Math.abs(p[0])),lower:Math.min(e.lower,p[1]),upper:Math.max(e.upper,p[1])}),{x:.6,lower:-.1,upper:.25});
    extent={x:extent.x+.12,lower:extent.lower-.12,upper:extent.upper+.12};
    sagittalStatus.textContent = `${sagittal.origin==='pelvis'?'双髋活动平面 · 髋中心原点（非地面高度）':'站立标定 · 髋中心正下方地面原点'} · 有效采样 ${Math.round(sagittal.valid_ratio*100)}%`;
  }
  function drawTrajectories() {
    if (!sagittal || sagittal.status !== 'available' || sagittalPanel.classList.contains('hidden')) return;
    charts.forEach((chart, sideIndex) => {
      const side = sideIndex ? 'right' : 'left', stroke = colors[side];
      const names = ['hip','knee','ankle'].map(n => `${side}_${n}`), ids = names.map(n => data.joint_names.indexOf(n));
      const [hip, knee, ankle] = ids;
      const rect = chart.getBoundingClientRect(), w = Math.max(1,rect.width), h = Math.max(1,rect.height), ratio = window.devicePixelRatio || 1;
      if (chart.width !== Math.round(w*ratio) || chart.height !== Math.round(h*ratio)) { chart.width = Math.round(w*ratio); chart.height = Math.round(h*ratio); }
      const ctx = chart.getContext('2d'); ctx.setTransform(ratio,0,0,ratio,0,0); ctx.clearRect(0,0,w,h);
      const margin = {left:48,right:24,top:45,bottom:43}, pw=w-margin.left-margin.right, ph=h-margin.top-margin.bottom;
      const scale = Math.min(pw/(extent.x*2),ph/(extent.upper-extent.lower));
      const cx = margin.left+pw/2, cy=margin.top+ph/2;
      const midY=(extent.upper+extent.lower)/2;
      const point = p => [cx+p[0]*direction*scale,cy-(p[1]-midY)*scale];
      const origin=point([0,0]);
      function path(a,b) { ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.stroke(); }
      ctx.font='11px system-ui';ctx.lineWidth=1;
      const tick=.25*Math.max(1,Math.ceil(42/(scale*.25)));
      for(let v=Math.ceil(-extent.x/tick)*tick;v<=extent.x;v+=tick) {
        const x=point([v,0])[0]; ctx.strokeStyle='#25352c';path([x,margin.top],[x,h-margin.bottom]);
        ctx.fillStyle='#83978a';ctx.textAlign='center';ctx.fillText((v*direction).toFixed(2),x,h-margin.bottom+17);
      }
      for(let v=Math.ceil(extent.lower/tick)*tick;v<=extent.upper;v+=tick) {
        const y=point([0,v])[1];ctx.strokeStyle='#25352c';path([margin.left,y],[w-margin.right,y]);
        ctx.fillStyle='#83978a';ctx.textAlign='right';ctx.fillText(v.toFixed(2),margin.left-7,y+4);
      }
      ctx.strokeStyle='#91a697';ctx.setLineDash([4,4]);path([origin[0],margin.top],[origin[0],h-margin.bottom]);path([margin.left,origin[1]],[w-margin.right,origin[1]]);ctx.setLineDash([]);
      ctx.strokeStyle=stroke;ctx.globalAlpha=.55;ctx.lineWidth=2;
      // Only adjacent valid samples are connected; never bridge an occlusion.
      for(let t=1;t<sagittal.keypoints.length;t++) {
        const a=sagittal.keypoints[t-1][ankle],b=sagittal.keypoints[t][ankle];if(a&&b)path(point(a),point(b));
      }
      ctx.globalAlpha=1;
      const current=sagittal.keypoints[sample];
      ctx.strokeStyle='#edf5ef';ctx.lineWidth=1.5;
      path([origin[0]-6,origin[1]],[origin[0]+6,origin[1]]);path([origin[0],origin[1]-6],[origin[0],origin[1]+6]);
      ctx.fillStyle='#edf5ef';ctx.textAlign='center';ctx.fillText(sagittal.origin==='pelvis'?'髋中心 (0, 0)':'地面原点 (0, 0)',origin[0],origin[1]+19);
      const pelvis=current[data.joint_names.indexOf('pelvis')];
      if(pelvis&&sagittal.origin!=='pelvis') {
        const p=point(pelvis);path([p[0]-5,p[1]],[p[0]+5,p[1]]);path([p[0],p[1]-5],[p[0],p[1]+5]);
        ctx.fillText('髋中心',p[0],p[1]-12);
      }
      if(current[hip]) {
        const p=point(current[hip]);ctx.strokeStyle=stroke;ctx.setLineDash([5,4]);
        if(current[knee])path(p,point(current[knee]));if(current[knee]&&current[ankle])path(point(current[knee]),point(current[ankle]));ctx.setLineDash([]);
        ctx.beginPath();ctx.moveTo(p[0],p[1]-6);ctx.lineTo(p[0]+6,p[1]);ctx.lineTo(p[0],p[1]+6);ctx.lineTo(p[0]-6,p[1]);ctx.closePath();ctx.fillStyle=stroke;ctx.fill();
        const offset=sideIndex?70:-70;path([p[0],p[1]],[p[0]+offset,p[1]+24]);
        ctx.textAlign='center';ctx.fillText(sideIndex?'右髋关节':'左髋关节',p[0]+offset,p[1]+39);
      }
      if(current[ankle]) {
        const p=point(current[ankle]);ctx.beginPath();ctx.arc(p[0],p[1],5,0,Math.PI*2);ctx.fillStyle=stroke;ctx.fill();
        ctx.textAlign='left';ctx.fillText('当前脚踝',p[0]+9,p[1]-9);
      }
      ctx.fillStyle='#9eb0a4';ctx.textAlign='left';ctx.fillText(sagittal.origin==='pelvis'?'↑ 相对髋部高度 / 腿长':'↑ 估计高度 / 腿长',12,18);
      ctx.textAlign='right';ctx.fillText('前后 / 腿长 →',w-16,h-9);
      ctx.textAlign='left';ctx.fillText(`${data.timestamps[sample].toFixed(2)}s${current[ankle]?'':' · 当前脚踝不可用'}`,margin.left,h-9);
    });
  }
  document.getElementById('flipSagittal3d').addEventListener('click',()=>{direction*=-1;drawTrajectories();});
  function draw() {
    if (!data || panel.classList.contains('hidden')) return;
    const bounds = canvas.getBoundingClientRect();
    const w = Math.max(1, bounds.width), h = Math.max(1, bounds.height), ratio = window.devicePixelRatio || 1;
    if (canvas.width !== Math.round(w * ratio) || canvas.height !== Math.round(h * ratio)) {
      canvas.width = Math.round(w * ratio); canvas.height = Math.round(h * ratio);
    }
    const ctx = canvas.getContext('2d'); ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, w, h);
    const gradient = ctx.createRadialGradient(w / 2, h / 2, 20, w / 2, h / 2, w * .7);
    gradient.addColorStop(0, '#18271f'); gradient.addColorStop(1, '#0a110e');
    ctx.fillStyle = gradient; ctx.fillRect(0, 0, w, h);
    const scale = Math.min(w, h) * .29 * zoom;
    function project(p) {
      const x = p[0] * Math.cos(yaw) + p[2] * Math.sin(yaw);
      const z = -p[0] * Math.sin(yaw) + p[2] * Math.cos(yaw);
      const y = p[1] * Math.cos(pitch) - z * Math.sin(pitch);
      const depth = p[1] * Math.sin(pitch) + z * Math.cos(pitch);
      const factor = 4.5 / Math.max(1.5, 4.5 - depth);
      return [w / 2 + x * scale * factor, h / 2 - y * scale * factor, depth, factor];
    }
    function line(a, b, stroke, width = 1) {
      const pa = project(a), pb = project(b);
      ctx.beginPath(); ctx.moveTo(pa[0], pa[1]); ctx.lineTo(pb[0], pb[1]);
      ctx.strokeStyle = stroke; ctx.lineWidth = width; ctx.stroke();
    }
    const estimated=data.ground_reference?.method==='standing_estimate'&&!showOriginal;
    const ground=estimated?data.ground_reference.heights[sample]:-1.1;
    // The surface and its normal are in the same 3D coordinates as the skeleton.
    const corners=[[-1.4,ground,-1.2],[1.4,ground,-1.2],[1.4,ground,1.2],[-1.4,ground,1.2]].map(project);
    ctx.beginPath();corners.forEach((p,i)=>i?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1]));
    ctx.closePath();ctx.fillStyle='rgba(70,137,120,.14)';ctx.fill();
    ctx.strokeStyle='#517568';ctx.lineWidth=1;ctx.stroke();
    for (let i = -4; i <= 4; i++) {
      line([i / 2, ground, -2], [i / 2, ground, 2], '#26372e');
      line([-2, ground, i / 2], [2, ground, i / 2], '#26372e');
    }
    const base=[-.85,ground,.55],tip=[-.85,ground+1.8,.55];
    line(base,tip,'#ffcf70',2.5);
    line(tip,[-.94,ground+1.61,.55],'#ffcf70',2.5);
    line(tip,[-.76,ground+1.61,.55],'#ffcf70',2.5);
    // A square in the X/Y plane: ground tangent is perpendicular to +Y normal.
    line([-.65,ground,.55],[-.65,ground+.2,.55],'#ffcf70',1.5);
    line([-.65,ground+.2,.55],[-.85,ground+.2,.55],'#ffcf70',1.5);
    line(base,[-.25,ground,.55],'#ffcf70',1.5);
    const normalLabel=project(tip),planeLabel=project([.8,ground,.9]);
    ctx.font='12px system-ui';ctx.fillStyle='#ffcf70';
    ctx.fillText(estimated?'地面法向 +Y（估计）':'参考平面法向 +Y',normalLabel[0]+10,normalLabel[1]);
    const squareLabel=project([-.65,ground+.2,.55]);
    ctx.fillText('90°',squareLabel[0]+5,squareLabel[1]-5);
    ctx.fillStyle='#a8cbbc';ctx.fillText(estimated?'估计地面 XZ':'参考平面 XZ（未标定）',planeLabel[0]-50,planeLabel[1]+20);
    const axes = [[[.45, 0, 0], '#ff867e', 'X'], [[0, .45, 0], '#c9ff42', 'Y'], [[0, 0, .45], '#32d5ff', 'Z']];
    ctx.font = '12px system-ui';
    axes.forEach(([end, stroke, text]) => {
      line([0, 0, 0], end, stroke, 1.3);
      const p = project(end); ctx.fillStyle = stroke; ctx.fillText(text, p[0] + 7, p[1]);
    });
    const usePlane=planeControl.checked&&!showOriginal&&data.leg_plane_constraint?.status==='available'&&data.leg_plane_constraint.version===3;
    const track=showOriginal&&data.original_keypoints?data.original_keypoints:usePlane?data.constrained_keypoints:data.keypoints;
    const points = track[sample];
    if(trailsControl.checked) {
      ['left_ankle','right_ankle'].forEach(name=>{
        const index=data.joint_names.indexOf(name);ctx.globalAlpha=.45;
        for(let t=1;t<track.length;t++) {
          if(finitePoint(track[t-1][index])&&finitePoint(track[t][index]))line(track[t-1][index],track[t][index],color(index),1.2);
        }
      });ctx.globalAlpha=1;
    }
    if(usePlane&&data.leg_plane_constraint.normals[sample]) {
      const normal=data.leg_plane_constraint.normals[sample];
      const unit=a=>{const n=Math.hypot(...a);return n>1e-8?a.map(v=>v/n):null;};
      let up=unit([0,1,0].map((v,i)=>v-normal[1]*normal[i]));
      if(!up)up=unit([1,0,0].map((v,i)=>v-normal[0]*normal[i]));
      const tangent=[normal[1]*up[2]-normal[2]*up[1],normal[2]*up[0]-normal[0]*up[2],normal[0]*up[1]-normal[1]*up[0]];
      ['left_hip','right_hip'].forEach(name=>{
        const index=data.joint_names.indexOf(name),hip=points[index];if(!finitePoint(hip))return;
        const corners=[[-.6,-1.2],[.6,-1.2],[.6,.25],[-.6,.25]].map(([a,b])=>hip.map((v,i)=>v+a*tangent[i]+b*up[i]));
        ctx.beginPath();corners.map(project).forEach((p,i)=>i?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1]));ctx.closePath();
        ctx.globalAlpha=.08;ctx.fillStyle=color(index);ctx.fill();ctx.globalAlpha=.4;ctx.strokeStyle=color(index);ctx.stroke();ctx.globalAlpha=1;
      });
    }
    const visible = points.some(finitePoint);
    label.textContent = `3D ${sample + 1} / ${data.timestamps.length} · ${data.timestamps[sample].toFixed(2)}s`;
    if (!visible) {
      ctx.textAlign = 'center'; ctx.fillStyle = '#d5dfd8';
      ctx.fillText('此采样未得到可信骨架', w / 2, h / 2 - 30); ctx.textAlign = 'left';
      return;
    }
    const segments = data.edges.filter(([a, b]) => finitePoint(points[a]) && finitePoint(points[b]));
    segments.sort(([a, b], [c, d]) => (project(points[a])[2] + project(points[b])[2]) - (project(points[c])[2] + project(points[d])[2]));
    ctx.lineCap = 'round';
    segments.forEach(([a, b]) => {
      const stroke = color(b) === colors.center ? color(a) : color(b);
      line(points[a], points[b], '#07100b', 9);
      line(points[a], points[b], stroke, 4.5);
    });
    points.map((p, index) => finitePoint(p) ? { p: project(p), index } : null).filter(Boolean)
      .sort((a, b) => a.p[2] - b.p[2]).forEach(({ p, index }) => {
        ctx.beginPath(); ctx.arc(p[0], p[1], (index === 0 ? 7 : 4.5) * p[3], 0, 2 * Math.PI);
        ctx.fillStyle = color(index); ctx.fill(); ctx.strokeStyle = '#08110c'; ctx.lineWidth = 1.5; ctx.stroke();
      });
    [['left_hip','左髋关节',70],['right_hip','右髋关节',-70],['pelvis','髋中心',0]].forEach(([name,text,dx]) => {
      const index=data.joint_names.indexOf(name);
      if(!finitePoint(points[index])) return;
      const p=project(points[index]);ctx.fillStyle=color(index);ctx.strokeStyle=color(index);ctx.textAlign='center';
      if(name!=='pelvis') {
        ctx.beginPath();ctx.arc(p[0],p[1],7,0,Math.PI*2);ctx.lineWidth=1.5;ctx.stroke();
        ctx.beginPath();ctx.moveTo(p[0],p[1]);ctx.lineTo(p[0]+dx,p[1]+12);ctx.stroke();
      }
      ctx.fillText(text,p[0]+dx,p[1]+(name==='pelvis'?-18:25));
    });
    ctx.textAlign='left';
    ctx.fillStyle = '#899d90'; ctx.fillText(estimated?'站立标定估计地面 · 非真实测量':'参考平面及法向示意 · 未标定真实地面', 16, h - 18);
  }
  async function load(job) {
    const current = ++epoch; data = null; sagittal = null; sample = 0; pointers.clear();
    sagittalPanel.classList.add('hidden');
    yaw = -.55; pitch = .16; zoom = 1;
    panel.classList.remove('hidden'); canvas.classList.add('hidden'); label.textContent = '3D';
    const url = job.artifacts?.['skeleton3d.json'];
    if (!url) {
      status.textContent = job.result?.skeleton3d?.reason
        ? `三维骨架不可用：${job.result.skeleton3d.reason}` : '此历史记录没有三维骨架，请点击“重新分析”生成。';
      return;
    }
    status.textContent = '正在加载三维骨架…';
    try {
      const response = await fetch(url);
      if (!response.ok) throw new Error('无法读取三维骨架数据');
      const payload = await response.json();
      if (current !== epoch) return;
      if (!Array.isArray(payload.timestamps) || !payload.timestamps.length || !Array.isArray(payload.joint_names)
          || !Array.isArray(payload.edges) || !Array.isArray(payload.keypoints)
          || payload.keypoints.length !== payload.timestamps.length
          || !payload.timestamps.every(Number.isFinite)) throw new Error('三维骨架数据格式不正确');
      if(payload.version!==5||payload.model!=='VideoPose3D') throw new Error('旧记录需重新分析以生成统一坐标及质量检查结果');
      data = payload; showOriginal=false;
      const plane=payload.leg_plane_constraint;
      planeControl.disabled=plane?.status!=='available'||plane?.version!==3;planeControl.checked=!planeControl.disabled;
      document.getElementById('legPlaneStatus').textContent=!planeControl.disabled
        ?`逐帧双髋连线为法向，两个平面分别经过原始左右髋；髋节点不移动。最大平面残差 ${plane.max_plane_residual_leg_ratio.toExponential(1)} 腿长；平均骨长改变 ${(plane.mean_bone_length_change_ratio*100).toFixed(1)}%。平面随髋转动，不等于重力/地面标定。`
        :`平面约束不可用：${plane?.reason||'此历史记录需重新分析'}`;
      yaw=0;pitch=0;zoom=1;
      originalButton.textContent='查看原始相机朝向';
      originalButton.classList.toggle('hidden',payload.display_alignment?.method!=='fixed_2d_trunk_reference');
      document.getElementById('displayAlignmentStatus').textContent=payload.display_alignment?.reason||'';
      canvas.classList.remove('hidden');
      setupTrajectories();
      const corrected=payload.coordinate_system==='world_pelvis_relative';
      const q=payload.quality;
      const cv=Object.values(q.bone_length_cv).reduce((a,b)=>a+b,0)/Object.keys(q.bone_length_cv).length;
      status.textContent = `${payload.model} · ${corrected?'站立标定世界坐标':'原始相机坐标，未校正'} · 有效采样 ${Math.round(payload.valid_sample_ratio * 100)}% · 时序支持 ${Math.round(q.supported_frame_ratio*100)}% · 重投影误差 ${q.weak_perspective_reprojection_rmse_image_height==null?'不可评估':(q.weak_perspective_reprojection_rmse_image_height*100).toFixed(1)+'%画面高度'} · 骨长CV ${(cv*100).toFixed(1)}% · 左右腿长差 ${(q.leg_length_asymmetry_ratio*100).toFixed(1)}% · 疑似交换 ${q.suspected_swap_frames.length} 帧`;
      setTime(document.getElementById('analysisVideo').currentTime || 0);
    } catch (error) {
      if (current === epoch) status.textContent = `三维骨架加载失败：${error.message}`;
    }
  }
  function view(name) {
    const world=data?.coordinate_system==='world_pelvis_relative';
    yaw = name === 'front' ? (world?-Math.PI/2:0) : name === 'side' ? (world?0:-Math.PI/2) : -.55;
    pitch = name === 'orbit' ? .16 : 0; zoom = 1; draw();
  }
  function changeZoom(factor) { zoom = Math.max(.45, Math.min(2.5, zoom * factor)); draw(); }
  document.querySelectorAll('[data-view3d]').forEach(button => button.addEventListener('click', () => view(button.dataset.view3d)));
  document.getElementById('reset3d').addEventListener('click', () => view('orbit'));
  canvas.addEventListener('pointerdown', event => {
    canvas.setPointerCapture(event.pointerId); pointers.set(event.pointerId, [event.clientX, event.clientY]);
  });
  canvas.addEventListener('pointermove', event => {
    const previous = pointers.get(event.pointerId); if (!previous) return;
    if (pointers.size === 2) {
      const other = [...pointers.entries()].find(([id]) => id !== event.pointerId)[1];
      const oldDistance = Math.hypot(previous[0] - other[0], previous[1] - other[1]);
      const distance = Math.hypot(event.clientX - other[0], event.clientY - other[1]);
      if (oldDistance > 1) changeZoom(distance / oldDistance);
    } else {
      yaw += (event.clientX - previous[0]) * .008;
      pitch = Math.max(-1.2, Math.min(1.2, pitch + (event.clientY - previous[1]) * .008));
    }
    pointers.set(event.pointerId, [event.clientX, event.clientY]); draw();
  });
  ['pointerup', 'pointercancel', 'lostpointercapture'].forEach(name => canvas.addEventListener(name, event => pointers.delete(event.pointerId)));
  canvas.addEventListener('wheel', event => { event.preventDefault(); changeZoom(Math.exp(-event.deltaY * .001)); }, { passive: false });
  canvas.addEventListener('keydown', event => {
    if (!['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', '+', '=', '-'].includes(event.key)) return;
    event.preventDefault();
    if (event.key === 'ArrowLeft') yaw -= .1;
    if (event.key === 'ArrowRight') yaw += .1;
    if (event.key === 'ArrowUp') pitch = Math.max(-1.2, pitch - .1);
    if (event.key === 'ArrowDown') pitch = Math.min(1.2, pitch + .1);
    if (event.key === '+' || event.key === '=') changeZoom(1.1);
    if (event.key === '-') changeZoom(1 / 1.1);
    draw();
  });
  new ResizeObserver(draw).observe(canvas);
  charts.forEach(chart=>new ResizeObserver(drawTrajectories).observe(chart));
  window.Skeleton3D = { load, setTime };
})();
