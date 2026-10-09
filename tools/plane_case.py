"""Generate a reproducible plane-constraint case from saved video predictions."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pace.pose.leg_plane import constrain_leg_planes, side_projection
from pace.pose.skeleton3d import EDGES


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',default='output/experiments/coordinate-quality/real/skeleton3d.npz')
    parser.add_argument('--output',default='output/leg-plane-case')
    parser.add_argument('--landmarks',help='Saved RTMPose CSV for matching 2D cycle boundaries')
    parser.add_argument('--aspect',type=float,default=594/1174)
    args=parser.parse_args()
    source=np.load(args.input)
    pose=source['keypoints'];confidence=source['confidence'];times=source['timestamps']
    result,meta=constrain_leg_planes(pose,confidence)
    if meta['status']!='available':raise RuntimeError(meta['reason'])
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    side=side_projection(result,meta)
    motion=None
    if args.landmarks:
        import csv
        from pace.pose.landmarks import LANDMARK_NAMES
        from pace.pose.smoothing import preprocess_landmarks
        from pace.biomechanics.foot_tracking import leg_length
        from pace.biomechanics.gait_events import detect_ankle_events
        from pace.analysis.stability import foot_cycle_analysis
        points=np.full((len(times),33,4),np.nan)
        for row in csv.DictReader(open(args.landmarks)):
            index=int(row['frame_id'])
            points[index,LANDMARK_NAMES.index(row['landmark'])]=[float(row[k]) for k in ['x','y','z','visibility']]
        fps=1/np.median(np.diff(times))
        points=preprocess_landmarks(points,fps,.45,.1)
        strikes={}
        for name,ankle in [('left',27),('right',28)]:
            events,_=detect_ankle_events(points[:,ankle],leg_length(points,name,args.aspect,.45),fps,.45)
            strikes[name]=events.strikes
        motion=foot_cycle_analysis(points,strikes,1,.45,times,args.aspect,
            trajectories={'left':side[:,6],'right':side[:,3]})
        motion['method']='Hip-plane side projection; 2D ankle event boundaries; pelvis origin'
        (out/'side_cycles.json').write_text(json.dumps(motion,ensure_ascii=False,allow_nan=False))
    fig,axes=plt.subplots(1,2,figsize=(11,5))
    for ax,name,joint in zip(axes,['left','right'],[6,3]):
        if motion:
            for cycle in motion[name]['cycles']:
                if cycle.get('path'):
                    path=np.asarray(cycle['path']);ax.plot(path[:,0],path[:,1],alpha=.35,lw=1)
            mean=np.asarray(motion[name]['mean_path'])
            if len(mean):ax.plot(mean[:,0],mean[:,1],color='black',lw=3,label='Mean cycle')
        else:ax.plot(side[:,joint,0],side[:,joint,1],alpha=.6)
        ax.axhline(0,color='gray',ls='--');ax.axvline(0,color='gray',ls='--')
        ax.set_title(name+' ankle side-plane cycles');ax.set_xlabel('Forward / leg length')
        ax.set_ylabel('Height relative to pelvis / leg length');ax.set_aspect('equal',adjustable='datalim')
        if motion and len(motion[name]['mean_path']):ax.legend()
    fig.tight_layout();fig.savefig(out/'side_cycles.png',dpi=160);plt.close(fig)
    np.savez_compressed(out/'leg_plane_case.npz',original=pose,constrained=result,
                        timestamps=times,normal=meta['normal'],confidence=confidence)
    (out/'metrics.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False))
    def serial(array):
        return [[joint.tolist() if np.isfinite(joint).all() else None for joint in frame] for frame in array]
    (out/'leg_plane_case.json').write_text(json.dumps(dict(metadata=meta,
        timestamps=times.tolist(),original=serial(pose),constrained=serial(result)),allow_nan=False))
    frame=int(np.argmin(abs(times-2)))
    fig=plt.figure(figsize=(12,6))
    for panel,(array,title) in enumerate([(pose,'Original prediction'),(result,'Per-frame hip-line plane constraint')],1):
        ax=fig.add_subplot(1,2,panel,projection='3d')
        for a,b in EDGES:
            p=array[frame,[a,b]]
            ax.plot(p[:,0],p[:,2],p[:,1],color='#278ca5' if b in [1,2,3,14,15,16] else '#6f9520',lw=3)
        for ankle in [3,6]:
            ax.plot(array[:,ankle,0],array[:,ankle,2],array[:,ankle,1],alpha=.4,lw=1)
        if panel==2:
            normal=np.asarray(meta['normals'][frame])
            up=np.array([0.,1.,0.]);up-=np.dot(up,normal)*normal
            if np.linalg.norm(up)<1e-8:up=np.cross(normal,[1.,0.,0.])
            up/=np.linalg.norm(up);forward=np.cross(normal,up)
            u,v=np.meshgrid(np.linspace(-.6,.6,2),np.linspace(-1.2,.3,2))
            for hip in [1,4]:
                plane=array[frame,hip]+u[:,:,None]*forward+v[:,:,None]*up
                ax.plot_surface(plane[:,:,0],plane[:,:,2],plane[:,:,1],alpha=.12,color='#278ca5')
        ax.set_title(title+f' ({times[frame]:.2f}s)')
        ax.set_xlabel('Display X');ax.set_ylabel('Display Z');ax.set_zlabel('Display Y')
        ax.set_xlim(-1,1);ax.set_ylim(-1,1);ax.set_zlim(-1.3,.9)
        ax.set_box_aspect((2,2,2.2));ax.view_init(elev=15,azim=-65)
    fig.suptitle('Hard projection experiment: two hip-anchored parallel planes; not ground calibration')
    fig.tight_layout();fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
    import cv2
    writer=cv2.VideoWriter(str(out/'comparison.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),30,(1280,640))
    if not writer.isOpened():raise RuntimeError('Cannot create comparison video')
    try:
        for time in np.arange(times[0],times[-1]+1e-9,1/30):
            index=int(np.argmin(abs(times-time)))
            canvas=np.full((640,1280,3),(24,31,24),np.uint8)
            for panel,(array,title) in enumerate([(pose,'Original 3D'),(result,'Hard leg-plane constraint')]):
                def project(p):
                    x=p[0]*.87-p[2]*.5;z=p[0]*.5+p[2]*.87
                    return (int(panel*640+320+x*200),int(270-(p[1]*.98-z*.2)*200))
                cv2.putText(canvas,title,(panel*640+30,40),cv2.FONT_HERSHEY_SIMPLEX,.7,(230,230,230),1,cv2.LINE_AA)
                cv2.putText(canvas,f'{time:.2f}s - pelvis relative',(panel*640+30,70),cv2.FONT_HERSHEY_SIMPLEX,.5,(170,190,170),1,cv2.LINE_AA)
                for ankle,color in [(3,(255,205,50)),(6,(65,230,180))]:
                    for t in range(1,len(array)):
                        if np.isfinite(array[t-1:t+1,ankle]).all():
                            cv2.line(canvas,project(array[t-1,ankle]),project(array[t,ankle]),tuple(int(v*.5) for v in color),1,cv2.LINE_AA)
                for a,b in EDGES:
                    p=array[index,[a,b]]
                    if np.isfinite(p).all():
                        color=(255,205,50) if b in [1,2,3,14,15,16] else (65,230,180)
                        cv2.line(canvas,project(p[0]),project(p[1]),color,4,cv2.LINE_AA)
                        cv2.circle(canvas,project(p[1]),5,color,-1,cv2.LINE_AA)
            cv2.putText(canvas,'Two parallel hip-anchored planes. Projection changes bone lengths. Not ground calibration.',
                        (30,610),cv2.FONT_HERSHEY_SIMPLEX,.5,(200,200,200),1,cv2.LINE_AA)
            writer.write(canvas)
    finally:writer.release()
    print('Plane residual',meta['max_plane_residual_leg_ratio']);print('Saved',out)


if __name__=='__main__':main()
