const assert = require('node:assert/strict');
const {build} = require('../static/sagittal3d.js');
const names=['left_hip','right_hip','left_shoulder','right_shoulder','pelvis','neck','nose','left_ankle','right_ankle'];
const base=[[-.1,0,0],[.1,0,0],[-.2,.7,0],[.2,.7,0],[0,0,0],[0,.7,0],[0,.9,.12],[-.1,-.9,.3],[.1,-.7,-.25]];
const close=(a,b)=>assert.ok(Math.abs(a-b)<1e-8,`${a} != ${b}`);
const data=keypoints=>({joint_names:names,keypoints});
const first=build(data([base,base,base]));
assert.equal(first.status,'available');
close(first.keypoints[0][7][0],.3);close(first.keypoints[0][7][1],-.9);
assert.deepEqual(first.keypoints[0][4],[0,0]);
// Body sagittal coordinates must be invariant to a rotated camera/model.
for (const yaw of [0,.6,1.3]) {
 const rotate=p=>{
   const x=p[0]*Math.cos(yaw)+p[2]*Math.sin(yaw),z=-p[0]*Math.sin(yaw)+p[2]*Math.cos(yaw),pitch=.23;
   return [x,p[1]*Math.cos(pitch)-z*Math.sin(pitch),p[1]*Math.sin(pitch)+z*Math.cos(pitch)];
 };
 const rotated=build(data([base.map(rotate),base.map(rotate),base.map(rotate)]));
 rotated.keypoints[0].forEach((p,i)=>p.forEach((v,j)=>close(v,first.keypoints[0][i][j])));
}
const missing=base.map(p=>[...p]);missing[0]=null;
const turned=base.map(p=>[-p[0],p[1],-p[2]]);
const masked=build(data([base,base,base,missing,turned]));
assert.deepEqual(masked.validity,[true,true,true,false,false]);
assert.ok(masked.keypoints[3].every(p=>p===null));
assert.ok(masked.keypoints[4].every(p=>p===null));
assert.equal(build(data([missing])).status,'unavailable');
// Side-view upper-body depth can misidentify the sagittal axis. The motion
// plane must preserve the dominant fore/aft ankle excursion rather than z.
const moving=Array.from({length:40},(_,t)=>{
 const p=base.map(j=>[...j]),phase=t/40*Math.PI*4;
 p[7]=[.45*Math.cos(phase),-.9+.15*Math.sin(phase),.03*Math.sin(phase)];
 p[8]=[-.45*Math.cos(phase),-.9-.15*Math.sin(phase),-.03*Math.sin(phase)];
 return p;
});
const motion=build(data(moving));
assert.equal(motion.method,'leg_motion_pca');
assert.ok(Math.abs(motion.axes.forward[0])>.98);
const excursion=Math.max(...motion.keypoints.map(p=>p[7][0]))-Math.min(...motion.keypoints.map(p=>p[7][0]));
assert.ok(excursion>.85,'Fore/aft ankle motion was compressed');
console.log('PASS: fixed body projection, camera rotation invariance, hip origin, missing/turning masks');
