"""Rigid coordinate alignment and separate temporal experiments."""
import numpy as np
from scipy.signal import savgol_filter

from pace.pose.skeleton3d import EDGES


def smooth_pose(pose, times, window=9):
    if window < 3 or window % 2 == 0:
        raise ValueError('Smoothing window must be odd and >= 3')
    # Resample irregular video timestamps before filtering.
    grid = np.linspace(times[0], times[-1], len(times))
    result = pose.copy()
    size = min(window, len(times) if len(times)%2 else len(times)-1)
    if size < 3:
        return result
    for joint in range(17):
        for axis in range(3):
            series = np.interp(grid,times,pose[:,joint,axis])
            filtered = savgol_filter(series,size,min(2,size-1),mode='interp')
            result[:,joint,axis] = np.interp(times,grid,filtered)
    return result


def unit(vector):
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm < 1e-8:
        raise ValueError('Body coordinate axes are degenerate')
    return vector/norm


def align_world(pose, config=None):
    """One rotation for the entire clip, without altering internal bone lengths."""
    config = config or {}
    pelvis = pose[:,[1,4]].mean(axis=1)
    up = unit(np.median(pose[:,8]-pelvis,axis=0))
    right = np.median(pose[:,1]-pose[:,4]+pose[:,14]-pose[:,11],axis=0)
    right = unit(right-up*np.dot(right,up))
    forward = unit(np.cross(up,right))
    manual = config.get('running_direction')
    if manual is not None:
        manual = np.asarray(manual,dtype=float)
        if manual.shape!=(2,) or not np.isfinite(manual).all() or np.linalg.norm(manual)<1e-8:
            raise ValueError('running_direction must be a finite nonzero 2D vector')
        # Original image has y down, as does the model camera convention.
        if np.dot(forward[:2],manual)<0:
            forward *= -1
    elif forward[0]<0:
        forward *= -1
    lateral = unit(np.cross(forward,up))
    rotation = np.stack([forward,up,lateral])
    world = pose @ rotation.T
    # Constant offset: estimated ankle low reference, not measured floor.
    ground = float(np.percentile(world[:,[3,6],1],5))
    world[:,:,1] -= ground
    return world,dict(rotation_camera_to_world=rotation.tolist(),
                      determinant=float(np.linalg.det(rotation)),
                      ankle_low_reference=ground,
                      world_axes=['running','up','lateral'],
                      orientation_method='clip_median_shoulders_hips_and_trunk',
                      manual_direction_role='sign_reference_only' if manual is not None else 'none',
                      limitations=['Torso up is an approximation, not measured gravity.',
                                   'Two image points do not uniquely calibrate 3D camera orientation.',
                                   'Ground offset is an ankle low reference, not the physical belt surface.',
                                   'Rigid rotation cannot fix internal pose or lifting errors.'])


def knee_angles(pose,side):
    h,k,a = (4,5,6) if side=='left' else (1,2,3)
    u,v = pose[:,h]-pose[:,k],pose[:,a]-pose[:,k]
    denominator = np.linalg.norm(u,axis=1)*np.linalg.norm(v,axis=1)
    return np.degrees(np.arccos(np.clip(np.sum(u*v,axis=1)/np.maximum(denominator,1e-12),-1,1)))


def metrics(pose,times,vertical=(0,-1,0)):
    bones={}
    flags=[]
    for a,b in EDGES:
        length=np.linalg.norm(pose[:,a]-pose[:,b],axis=1)
        mean=float(length.mean());std=float(length.std())
        bones[f'{a}-{b}']=dict(mean=mean,std=std,cv=std/max(mean,1e-8))
        indices=np.flatnonzero(np.abs(length-mean)>3*std) if std>1e-8 else []
        flags.extend(dict(frame=int(i),bone=[a,b],reason='length_outlier_3std') for i in indices)
    left=np.linalg.norm(pose[:,4]-pose[:,5],axis=1)+np.linalg.norm(pose[:,5]-pose[:,6],axis=1)
    right=np.linalg.norm(pose[:,1]-pose[:,2],axis=1)+np.linalg.norm(pose[:,2]-pose[:,3],axis=1)
    trunk=pose[:,8]-pose[:,[1,4]].mean(axis=1)
    tilt=np.degrees(np.arccos(np.clip(trunk@np.asarray(vertical)/np.maximum(np.linalg.norm(trunk,axis=1),1e-8),-1,1)))
    swaps=[]
    # Diagnostic only. Crossed legs alone must not trigger automatic relabeling.
    for t in range(1,len(pose)):
        previous=pose[t-1,[4,5,6,1,2,3]];current=pose[t,[4,5,6,1,2,3]]
        direct=np.linalg.norm(current-previous,axis=1).sum()
        exchanged=np.linalg.norm(current[[3,4,5,0,1,2]]-previous,axis=1).sum()
        if direct>.05 and exchanged<direct*.6:
            swaps.append(dict(frame=t,direct_distance=float(direct),swapped_distance=float(exchanged)))
    return dict(bones=bones,mean_bone_cv=float(np.mean([v['cv'] for v in bones.values()])),
                leg_length_asymmetry_ratio=float(np.mean(np.abs(left-right)/np.maximum((left+right)/2,1e-8))),
                trunk_tilt_degrees=tilt.tolist(),
                knees={s:knee_angles(pose,s).tolist() for s in ['left','right']},
                ankles={s:dict(xyz=pose[:,j].tolist(),range_xyz=np.ptp(pose[:,j],axis=0).tolist(),
                               peak_speed=float(np.linalg.norm(np.diff(pose[:,j],axis=0)/np.diff(times)[:,None],axis=1).max()))
                        for s,j in [('left',6),('right',3)]},
                bone_outliers=flags,suspected_leg_swaps=swaps,
                automatic_bone_or_swap_correction=False)


def quality_2d(coco):
    valid=np.isfinite(coco[:,:,:2]).all(axis=2)&(coco[:,:,2]>=.45)
    groups={name:float(valid[:,ids].all(axis=1).mean())
            for name,ids in [('hips',[11,12]),('shoulders',[5,6]),('knees',[13,14]),('ankles',[15,16])]}
    jumps={}
    for j in [5,6,11,12,13,14,15,16]:
        good=valid[:-1,j]&valid[1:,j]
        delta=np.linalg.norm(np.diff(coco[:,j,:2],axis=0),axis=1)
        series=delta[good]
        if len(series):
            center=np.median(series);mad=np.median(np.abs(series-center))
            jumps[str(j)]=np.flatnonzero(good&(delta>center+max(8*mad,5))).tolist()
    return dict(valid_ratios=groups,jump_candidates=jumps,
                passed=min(groups.values())>=.7,
                criterion='Both joints in each group visible in >=70% of frames',
                note='Candidate jumps are diagnostics; manually inspect the 2D overlay before attributing errors to 3D.')
