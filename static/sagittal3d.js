/* Fixed body sagittal coordinates, computed from 3D samples, not viewer yaw. */
((root) => {
  const finite = p => Array.isArray(p) && p.length === 3 && p.every(Number.isFinite);
  const sub = (a, b) => a.map((v, i) => v - b[i]);
  const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);
  const unit = a => { const n = Math.hypot(...a); return n > 1e-6 ? a.map(v => v / n) : null; };
  const cross = (a, b) => [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
  const median = values => { const s = [...values].sort((a,b) => a-b), m = Math.floor(s.length/2); return s.length%2 ? s[m] : (s[m-1]+s[m])/2; };
  const averageDirection = vectors => unit([0,1,2].map(i => median(vectors.map(v => v[i]))));
  function build(data) {
    const id = name => data.joint_names.indexOf(name);
    const ids = ['left_hip','right_hip','left_shoulder','right_shoulder','pelvis','neck',data.joint_names.includes('nose')?'nose':'head'].map(id);
    if (ids.some(i => i < 0)) return {status:'unavailable',reason:'缺少肩、髋或躯干节点'};
    const [lh,rh,ls,rs,pelvis,neck,nose] = ids;
    const samples = data.keypoints.filter(p => [lh,rh,ls,rs,pelvis,neck].every(i => finite(p[i])));
    const lateral = p => unit(sub(p[rh],p[lh]).map((v,i) => v + p[rs][i]-p[ls][i]));
    const rights = samples.map(lateral).filter(Boolean);
    const ups = samples.map(p => unit(sub(p[neck],p[pelvis]))).filter(Boolean);
    if (!rights.length || !ups.length) return {status:'unavailable',reason:'有效身体方向采样不足'};
    const up = averageDirection(ups), right0 = averageDirection(rights);
    if (!up || !right0) return {status:'unavailable',reason:'三维身体方向退化'};
    let right = unit(right0.map((v,i) => v-dot(right0,up)*up[i]));
    if (!right) return {status:'unavailable',reason:'三维身体方向退化'};
    let forward = unit(cross(right,up)), method='upper_body', motionRatio=null;
    // Side-view occlusion can corrupt the left/right hip depth ordering. Infer
    // the fixed sagittal direction from dominant leg motion when identifiable.
    let horizontal=unit([1,0,0].map((v,i)=>v-up[0]*up[i]));
    if(!horizontal) horizontal=unit([0,0,1].map((v,i)=>v-up[2]*up[i]));
    const transverse=unit(cross(up,horizontal));
    let xx=0,xy=0,yy=0,count=0;
    ['left_ankle','right_ankle','left_knee','right_knee'].map(id).filter(i=>i>=0).forEach(j=>{
      const track=data.keypoints.filter(p=>finite(p[j])&&finite(p[pelvis])).map(p=>sub(p[j],p[pelvis]));
      if(track.length<12) return;
      const mean=[0,1,2].map(i=>track.reduce((s,p)=>s+p[i],0)/track.length);
      track.forEach(p=>{const d=sub(p,mean),a=dot(d,horizontal),b=dot(d,transverse);xx+=a*a;xy+=a*b;yy+=b*b;count++;});
    });
    if(count) {
      const delta=Math.hypot(xx-yy,2*xy),large=(xx+yy+delta)/2,small=(xx+yy-delta)/2;
      motionRatio=large/Math.max(small,1e-8);
      if(large/count>.001&&motionRatio>=2) {
        const angle=.5*Math.atan2(2*xy,xx-yy);
        forward=unit(horizontal.map((v,i)=>v*Math.cos(angle)+transverse[i]*Math.sin(angle)));
        right=unit(cross(up,forward));method='leg_motion_pca';
      } else if(large/count>.001) {
        return {status:'unavailable',reason:'三维下肢运动方向不明确，不能可靠确定正侧面'};
      }
    }
    const facing = samples.filter(p => finite(p[nose])).map(p => dot(sub(p[nose],p[neck]),forward));
    const noseDirection = facing.length ? median(facing) : 0;
    if ((Math.abs(noseDirection) > .025 && noseDirection < 0) || (Math.abs(noseDirection) <= .025 && forward[0] < 0)) forward = forward.map(v => -v);
    const valid = data.keypoints.map(p => {
      if (![lh,rh,ls,rs,pelvis,neck].every(i => finite(p[i]))) return false;
      const r = lateral(p), u = unit(sub(p[neck],p[pelvis]));
      return r && u && (method==='leg_motion_pca'||dot(r,right)>Math.cos(Math.PI/4)) && dot(u,up)>Math.cos(Math.PI/4);
    });
    if (!valid.some(Boolean)) return {status:'unavailable',reason:'身体转向或三维朝向不稳定'};
    const projected = data.keypoints.map((p,t) => p.map(j => valid[t] && finite(j)
      ? [dot(sub(j,p[pelvis]),forward),dot(sub(j,p[pelvis]),up)] : null));
    return {status:'available', method, motion_ratio:motionRatio, axes:{forward,up,right}, keypoints:projected, validity:valid,
      units:'estimated_leg_length', origin:'pelvis', valid_ratio:valid.filter(Boolean).length/valid.length};
  }
  const api = {build};
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  root.Sagittal3D = api;
})(typeof window !== 'undefined' ? window : globalThis);
