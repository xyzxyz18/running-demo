"""Run the reproducible RTMPose -> VideoPose3D experiment matrix.

Outputs A raw, B smoothed, C rigid-world, and D world+smoothed poses plus
quality reports. Use ``python -m tools.run_pipeline --input video.mp4 --output outputs``.
"""
from __future__ import annotations
import argparse, json
import shutil
from pathlib import Path
import cv2
import numpy as np

from pace.pipeline import extract_pose
from pace.pose.lifting import lift_pose, H36M_NAMES
from tools.pose_experiment import metrics, quality_2d, smooth_pose, align_world
from pace.visualization.video_overlay import draw_pose
from pace.pose.skeleton3d import EDGES


def save_json_frames(raw, directory):
    directory.mkdir(parents=True,exist_ok=True)
    coco=np.stack([raw[:,i,:2] for i in [0,2,5,7,8,11,12,13,14,15,16,23,24,25,26,27,28]],axis=1)
    score=raw[:,[0,2,5,7,8,11,12,13,14,15,16,23,24,25,26,27,28],3]
    # COCO 17 is the requested experiment schema. A bbox is derived from visible points.
    for frame,(xy,sc) in enumerate(zip(coco,score)):
        valid=np.isfinite(xy).all(axis=1)&(sc>=.3)
        if valid.any():
            lo,hi=xy[valid].min(axis=0),xy[valid].max(axis=0);bbox=[*lo.tolist(),*hi.tolist()]
        else:bbox=None
        (directory/f'frame_{frame+1:06d}.json').write_text(json.dumps({'keypoints':[[float(x),float(y),float(s)] if np.isfinite([x,y,s]).all() else None for (x,y),s in zip(xy,sc)],'bbox':bbox},separators=(',',':')))
    return coco,score


def render_panel(pose,extent,title):
    image=np.full((480,480,3),18,np.uint8)
    # Fixed oblique orthographic viewing transform, independent of correction.
    projection=np.array([[.85,0,.52],[-.18,.94,.29]])
    xy=pose@projection.T
    lo,hi=extent
    scale=min(380/(hi[0]-lo[0]),350/(hi[1]-lo[1]))
    center=(lo+hi)/2
    q=np.column_stack([240+(xy[:,0]-center[0])*scale,270-(xy[:,1]-center[1])*scale]).astype(int)
    cv2.putText(image,title,(16,30),0,.65,(240,240,240),1,cv2.LINE_AA)
    for a,b in EDGES:
        color=(80,230,160) if b in [4,5,6,11,12,13] else (240,200,60)
        cv2.line(image,tuple(q[a]),tuple(q[b]),color,3,cv2.LINE_AA)
    for p in q:cv2.circle(image,tuple(p),4,(235,235,235),-1)
    return image


def save_videos(source,output,raw,poses,fps):
    display_a=poses['A_raw']*[1,-1,-1]
    display_d=poses['D_rotation_plus_smoothing']
    projection=np.array([[.85,0,.52],[-.18,.94,.29]])
    combined=np.concatenate([display_a.reshape(-1,3),display_d.reshape(-1,3)])@projection.T
    extent=(combined.min(axis=0)-.1,combined.max(axis=0)+.1)
    specifications={'rtmpose_2d.mp4':(480,480),'pose3d_raw.mp4':(480,480),
                    'pose3d_corrected.mp4':(480,480),'comparison.mp4':(1440,480)}
    writers={name:cv2.VideoWriter(str(output/name),cv2.VideoWriter_fourcc(*'mp4v'),fps,size)
             for name,size in specifications.items()}
    cap=cv2.VideoCapture(str(source))
    try:
        if any(not w.isOpened() for w in writers.values()):raise RuntimeError('Cannot create output videos')
        for i in range(len(raw)):
            ok,frame=cap.read()
            if not ok:raise RuntimeError('Source video ended early')
            draw_pose(frame,raw[i],.45)
            h,w=frame.shape[:2];factor=min(480/w,440/h)
            resized=cv2.resize(frame,(int(w*factor),int(h*factor)))
            panel=np.full((480,480,3),18,np.uint8);rh,rw=resized.shape[:2]
            panel[40:40+rh,(480-rw)//2:(480+rw)//2]=resized
            cv2.putText(panel,'RTMPose 2D',(16,28),0,.65,(240,240,240),1)
            a=render_panel(display_a[i],extent,'A: raw VideoPose3D')
            d=render_panel(display_d[i],extent,'D: world + temporal smoothing')
            for name,image in [('rtmpose_2d.mp4',panel),('pose3d_raw.mp4',a),('pose3d_corrected.mp4',d),('comparison.mp4',np.concatenate([panel,a,d],axis=1))]:writers[name].write(image)
    finally:
        cap.release()
        for w in writers.values():w.release()


def save_plots(output,poses,times):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,2,figsize=(12,9),sharex=True)
    for column,(side,joint) in enumerate([('left',6),('right',3)]):
        for row,axis in enumerate(['X','Y','Z']):
            for name in ['C_coordinate_rotation','D_rotation_plus_smoothing']:
                axes[row,column].plot(times,poses[name][:,joint,row],label=name)
            axes[row,column].set_title(f'{side} ankle {axis} (model units)');axes[row,column].grid(alpha=.2)
    axes[0,0].legend(fontsize=7);fig.tight_layout();fig.savefig(output/'ankle_xyz.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    from tools.pose_experiment import knee_angles
    for ax,side in zip(axes,['left','right']):
        for name,pose in poses.items():ax.plot(times,knee_angles(pose,side),label=name)
        ax.set_title(f'{side} knee angle');ax.legend(fontsize=7);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(output/'knee_angles.png');plt.close(fig)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--camera-config',type=Path);parser.add_argument('--smooth-window',type=int,default=11);parser.add_argument('--allow-low-quality',action='store_true');args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    raw,fps,width,height,times=extract_pose(args.input)
    pixel_raw=raw.copy();pixel_raw[:,:,:2]*=[width,height]
    coco,score=save_json_frames(pixel_raw,args.output/'2d_frames')
    np.save(args.output/'keypoints_2d.npy',np.concatenate([coco,score[...,None]],axis=2))
    np.save(args.output/'timestamps.npy',times)
    quality=quality_2d(np.concatenate([coco,score[...,None]],axis=2))
    (args.output/'quality_2d.json').write_text(json.dumps(quality,indent=2))
    pelvis=coco[:,[11,12]].mean(axis=1);shoulder=coco[:,[5,6]].mean(axis=1)
    torso=np.linalg.norm(shoulder-pelvis,axis=1)
    normalized=(coco-pelvis[:,None])/np.maximum(torso[:,None,None],1e-8)
    np.save(args.output/'keypoints_2d_body_normalized.npy',normalized)
    print('Step 1: 2D exports saved. Quality:',quality['passed'],flush=True)
    if not quality['passed'] and not args.allow_low_quality:
        # Save the overlay even if lifting is blocked.
        cap=cv2.VideoCapture(str(args.input));writer=cv2.VideoWriter(str(args.output/'rtmpose_2d.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(width,height))
        try:
            for frame_points in raw:
                ok,image=cap.read()
                if not ok:break
                draw_pose(image,frame_points,.45);writer.write(image)
        finally:cap.release();writer.release()
        raise RuntimeError('2D quality gate failed. Inspect quality_2d.json and rtmpose_2d.mp4; --allow-low-quality explicitly overrides.')
    pose=lift_pose(raw,times,width/height)
    np.save(args.output/'pose3d_raw.npy',pose)
    print('Step 2: raw 3D saved',flush=True)
    smoothed=smooth_pose(pose,times,args.smooth_window)
    config_path=args.camera_config or args.input.parent/'camera_config.json'
    camera=json.loads(config_path.read_text()) if config_path.exists() else {}
    if args.camera_config and not config_path.exists():raise ValueError('Camera config not found')
    if camera.get('image_size') and camera['image_size']!=[width,height]:
        raise ValueError('Calibration image size does not match the input video')
    world,world_meta=align_world(pose,camera)
    corrected=smooth_pose(world,times,args.smooth_window)
    np.save(args.output/'pose3d_smoothed.npy',smoothed);np.save(args.output/'pose3d_world.npy',world);np.save(args.output/'pose3d_corrected.npy',corrected)
    poses={'A_raw':pose,'B_temporal_smoothing':smoothed,'C_coordinate_rotation':world,'D_rotation_plus_smoothing':corrected}
    print('Steps 3–4: world coordinates and temporal smoothing saved',flush=True)
    save_videos(args.input,args.output,raw,poses,fps)
    save_plots(args.output,poses,times)
    destination=args.output/('input'+args.input.suffix.lower())
    if destination.resolve()!=args.input.resolve():shutil.copy2(args.input,destination)
    report=dict(input=str(args.input.resolve()),fps=fps,frame_count=len(raw),schema='COCO17',
                axes=world_meta,quality_2d=quality,
                experiments={
                  'A_raw':metrics(pose,times),'B_temporal_smoothing':metrics(smoothed,times),
                  'C_coordinate_rotation':metrics(world,times,(0,1,0)),'D_rotation_plus_smoothing':metrics(corrected,times,(0,1,0))},
                interpretation=['A is the official lifting baseline.',
                                'B changes time continuity only.',
                                'C applies one rigid rotation and translation; internal bone lengths are unchanged.',
                                'D combines C and B.',
                                'Bone and left/right corrections are diagnostics only and are not automatically applied.'])
    (args.output/'camera_config.json').write_text(json.dumps(dict(camera,estimated_world=world_meta),ensure_ascii=False,indent=2))
    report['checks']=dict(rotation_orthogonality_error=float(np.max(np.abs(np.asarray(world_meta['rotation_camera_to_world'])@np.asarray(world_meta['rotation_camera_to_world']).T-np.eye(3)))),
                         rotation_knee_angle_max_difference=float(max(np.max(np.abs(np.asarray(report['experiments']['A_raw']['knees'][s])-np.asarray(report['experiments']['C_coordinate_rotation']['knees'][s]))) for s in ['left','right'])))
    if report['checks']['rotation_orthogonality_error']>1e-8 or report['checks']['rotation_knee_angle_max_difference']>1e-6:
        raise RuntimeError('Rigid coordinate conversion failed the invariance checks')
    (args.output/'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False))
    print(json.dumps({'output':str(args.output.resolve()),'frames':len(raw),'fps':fps,'experiments':['A_raw','B_temporal_smoothing','C_coordinate_rotation','D_rotation_plus_smoothing']},ensure_ascii=False))


if __name__=='__main__':main()
