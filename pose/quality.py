"""Visibility support and diagnostics; these do not repair model depth."""
import numpy as np
from scipy.signal import savgol_filter

from pose.backends import COCO_TO_LANDMARKS


def display_alignment(pose, points, aspect):
    """Fixed approximate orientation for viewing; not a camera calibration."""
    from_xyz = pose[:,8]-pose[:,[1,4]].mean(axis=1)
    shoulders = points[:,[11,12],:2].mean(axis=1)
    hips = points[:,[23,24],:2].mean(axis=1)
    to_xy = (shoulders-hips)*[aspect,-1]
    good = np.isfinite(to_xy).all(axis=1)
    good &= (points[:,[11,12,23,24],3]>=.45).all(axis=1)
    if good.sum()<10:
        return None, '肩髋观测不足，保留原始相机显示'
    source = np.median(from_xyz[good]*[1,-1,-1],axis=0)
    target = np.r_[np.median(to_xy[good],axis=0),0.]
    if min(np.linalg.norm(source),np.linalg.norm(target))<1e-6:
        return None, '躯干方向退化，保留原始相机显示'
    source /= np.linalg.norm(source); target /= np.linalg.norm(target)
    cosine = float(np.clip(source@target,-1,1))
    if cosine < -.95:
        return None, '三维方向异常，保留原始相机显示'
    cross = np.cross(source,target)
    skew = np.array([[0,-cross[2],cross[1]],[cross[2],0,-cross[0]],[-cross[1],cross[0],0]])
    rotation = np.eye(3)+skew+skew@skew/(1+cosine)
    return rotation, '全片固定显示旋转；二维躯干方向参考，不是相机标定'


def fill_for_model(values, times):
    """Interpolate short internal gaps; hold nearest observation for model padding."""
    valid = np.flatnonzero(np.isfinite(values))
    if not len(valid):
        raise ValueError('关键点遮挡过多，无法进行三维姿态估计')
    filled = np.interp(times,times[valid],values[valid])
    for index in np.flatnonzero(~np.isfinite(values)):
        pos = np.searchsorted(valid,index)
        if pos == 0 or pos == len(valid):
            continue
        left,right = valid[pos-1],valid[pos]
        if times[right]-times[left] > .1 + 1e-9:
            nearest = left if times[index]-times[left] <= times[right]-times[index] else right
            filled[index] = values[nearest]
    return filled


def temporal_support(points, times, threshold=.45, radius=121/50):
    times = np.asarray(times)
    visible = np.isfinite(points[:, COCO_TO_LANDMARKS, :2]).all(axis=2)
    visible &= points[:, COCO_TO_LANDMARKS, 3] >= threshold
    # Faces are often hidden in profile; gate the skeleton on body observations.
    supported = np.ones(len(times), dtype=bool)
    gaps = []
    for joint in range(5, 17):
        missing = ~visible[:, joint]
        starts = np.flatnonzero(missing & ~np.r_[False, missing[:-1]])
        ends = np.flatnonzero(missing & ~np.r_[missing[1:], False])
        for start, end in zip(starts, ends):
            internal = start > 0 and end < len(times)-1
            duration = times[end+1]-times[start-1] if internal else float('inf')
            if internal and duration <= .1 + 1e-9:
                continue
            supported &= ~((times >= times[start]-radius) & (times <= times[end]+radius))
            gaps.append(dict(coco_joint=joint,start_seconds=float(times[start]),
                             end_seconds=float(times[end]),boundary=not internal))
    return supported, gaps


def smooth_supported(pose, times, supported):
    result = pose.copy()
    starts = np.flatnonzero(supported & ~np.r_[False, supported[:-1]])
    ends = np.flatnonzero(supported & ~np.r_[supported[1:], False])
    for start, end in zip(starts, ends):
        t = times[start:end+1]
        if len(t) < 3:
            continue
        grid = np.arange(t[0],t[-1]+1e-9,1/50)
        window = min(9,len(grid) if len(grid)%2 else len(grid)-1)
        if window < 3:
            continue
        for joint in range(pose.shape[1]):
            for axis in range(3):
                values = np.interp(grid,t,pose[start:end+1,joint,axis])
                values = savgol_filter(values,window,2,mode='interp')
                result[start:end+1,joint,axis] = np.interp(t,grid,values)
    return result


def diagnostics(pose, points, aspect, supported):
    edges = [(1,2),(2,3),(4,5),(5,6),(11,12),(12,13),(14,15),(15,16)]
    bones = {}
    for a,b in edges:
        lengths = np.linalg.norm(pose[:,a]-pose[:,b],axis=1)
        bones[f'{a}-{b}'] = float(np.std(lengths)/max(np.mean(lengths),1e-8))
    left = np.linalg.norm(pose[:,4]-pose[:,5],axis=1)+np.linalg.norm(pose[:,5]-pose[:,6],axis=1)
    right = np.linalg.norm(pose[:,1]-pose[:,2],axis=1)+np.linalg.norm(pose[:,2]-pose[:,3],axis=1)
    joints, common = [11,14,4,1,5,2,6,3], [11,12,23,24,25,26,27,28]
    xy = points[:,common,:2]*[aspect,1]
    center = points[:,[23,24],:2].mean(axis=1)*[aspect,1]
    target = xy-center[:,None]
    predicted = pose[:,joints,:2]
    valid = np.isfinite(target).all(axis=2) & (points[:,common,3]>=.45)
    error = None
    scale = None
    if valid.any():
        x,y = predicted[valid],target[valid]
        scale = float(np.sum(x*y)/max(np.sum(x*x),1e-12))
        error = float(np.sqrt(np.mean(np.sum((scale*x-y)**2,axis=1))))
    swapped = []
    for frame in range(1,len(pose)):
        if not supported[frame-1:frame+1].all():
            continue
        prev = pose[frame-1,[4,5,6,1,2,3]]
        curr = pose[frame,[4,5,6,1,2,3]]
        direct = np.linalg.norm(curr-prev,axis=1).sum()
        exchange = np.linalg.norm(curr[[3,4,5,0,1,2]]-prev,axis=1).sum()
        if direct > .05 and exchange < .6*direct:
            swapped.append(frame)
    return dict(bone_length_cv=bones,
                leg_length_asymmetry_ratio=float(np.mean(np.abs(left-right)/np.maximum((left+right)/2,1e-8))),
                weak_perspective_reprojection_rmse_image_height=error,
                reprojection_observation_count=int(valid.sum()),
                screen_scale=scale,suspected_swap_frames=swapped,
                supported_frame_ratio=float(supported.mean()),
                limitation='Low reprojection error does not validate depth; no label or bone correction applied.')
