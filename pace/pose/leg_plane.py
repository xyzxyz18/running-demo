"""Per-frame leg planes perpendicular to the unmodified hip line."""
import numpy as np


def side_projection(pose, metadata):
    """Per-frame plane coordinates, origin at pelvis; not ground height."""
    projected=np.full((len(pose),17,2),np.nan)
    if metadata.get('status')!='available':
        return projected
    for frame,normal in enumerate(metadata['normals']):
        if normal is None:
            continue
        normal=np.asarray(normal)
        up=np.array([0.,1.,0.])-normal[1]*normal
        if np.linalg.norm(up)<.1:
            continue
        up/=np.linalg.norm(up)
        forward=np.cross(up,normal)
        pelvis=pose[frame,[1,4]].mean(axis=0)
        projected[frame,:,0]=(pose[frame]-pelvis)@forward
        projected[frame,:,1]=(pose[frame]-pelvis)@up
    return projected


def constrain_leg_planes(pose, confidence):
    axis=pose[:,1]-pose[:,4]
    width=np.linalg.norm(axis,axis=1)
    valid=np.isfinite(axis).all(axis=1)&(width>1e-6)
    valid&=(confidence[:,1]>=.45)&(confidence[:,4]>=.45)
    if not valid.any():
        return pose.copy(),dict(status='unavailable',reason='双髋缺失或重合，不能建立平面')
    normals=np.full_like(axis,np.nan)
    normals[valid]=axis[valid]/width[valid,None]
    result=pose.copy();displacement=[];length_change=[];residual=[]
    for hip,knee,ankle in [(4,5,6),(1,2,3)]:
        for joint in [knee,ankle]:
            good=valid&np.isfinite(pose[:,joint]).all(axis=1)
            distance=np.sum((pose[good,joint]-pose[good,hip])*normals[good],axis=1)
            result[good,joint]-=distance[:,None]*normals[good]
            result[~valid,joint]=np.nan
            displacement.extend(np.abs(distance).tolist())
            residual.extend(np.abs(np.sum((result[good,joint]-pose[good,hip])*normals[good],axis=1)).tolist())
        for a,b in [(hip,knee),(knee,ankle)]:
            before=np.linalg.norm(pose[:,a]-pose[:,b],axis=1)
            after=np.linalg.norm(result[:,a]-result[:,b],axis=1)
            good=np.isfinite(before)&np.isfinite(after)&(before>1e-8)
            length_change.extend((np.abs(after[good]-before[good])/before[good]).tolist())
    return result,dict(status='available',version=3,method='per_frame_original_hip_line_projection',
        normals=[n.tolist() if good else None for n,good in zip(normals,valid)],
        valid_frames=valid.tolist(),normal=np.median(normals[valid],axis=0).tolist(),
        hip_alignment='none; both original hips unchanged',
        max_plane_residual_leg_ratio=max(residual,default=0.),
        mean_displacement_leg_ratio=float(np.mean(displacement)) if displacement else 0.,
        mean_bone_length_change_ratio=float(np.mean(length_change)) if length_change else 0.,
        limitation='Planes follow the original hip line each frame, not gravity. Hard projection changes bone lengths and knee angles.')
