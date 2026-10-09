"""Export VideoPose3D estimates for synchronized browser playback."""
import json
from pathlib import Path

import numpy as np

from pace.pose.lifting import H36M_NAMES, lift_pose
from pace.pose.quality import temporal_support, smooth_supported, diagnostics, display_alignment
from pace.pose.leg_plane import constrain_leg_planes, side_projection

EDGES = [[0,1],[1,2],[2,3],[0,4],[4,5],[5,6],[0,7],[7,8],
         [8,9],[9,10],[8,11],[11,12],[12,13],[8,14],[14,15],[15,16]]
LIMITATION = ('RTMPose 二维序列经 VideoPose3D 时序模型估计三维骨架；'
              '髋中心原点，全片固定腿长尺度，无需站立参考。'
              '深度为单目模型估计，无法恢复真实地面或整体位移。')


def estimate_skeleton(video, points, timestamps, output, pose3d=None,
                      correction=None, correction_metadata=None, original_pose=None):
    import cv2
    capture = cv2.VideoCapture(str(video))
    try:
        width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    finally:
        capture.release()
    aspect = width/height if width > 0 and height > 0 else 1
    if pose3d is None:
        if width <= 0 or height <= 0:
            raise RuntimeError('无法读取视频尺寸')
        pose3d = lift_pose(points, timestamps, width/height)
    xyz = np.asarray(pose3d, dtype=float).copy()
    if xyz.shape != (len(timestamps),17,3) or not np.isfinite(xyz).all():
        raise ValueError('VideoPose3D 输出格式无效')
    xyz -= xyz[:,[1,4]].mean(axis=1)[:,None]
    raw = xyz.copy() if original_pose is None else np.asarray(original_pose).copy()
    raw -= raw[:,[1,4]].mean(axis=1)[:,None]
    support, gaps = temporal_support(points, np.asarray(timestamps))
    if original_pose is None:
        xyz = smooth_supported(xyz, np.asarray(timestamps), support)
    quality = diagnostics(raw, points, aspect, support)
    quality['after_smoothing'] = diagnostics(xyz, points, aspect, support)
    quality['long_gaps'] = gaps
    quality['smoothing_rms_model_units'] = float(np.sqrt(np.mean((xyz-raw)**2)))
    lengths = np.concatenate([np.linalg.norm(xyz[:,h]-xyz[:,k],axis=1)
                             +np.linalg.norm(xyz[:,k]-xyz[:,a],axis=1)
                             for h,k,a in [(1,2,3),(4,5,6)]])
    scale = float(np.median(lengths))
    if not np.isfinite(scale) or scale <= .01:
        raise ValueError('VideoPose3D 腿长估计无效')
    camera_xyz = xyz.copy()
    if correction is not None:
        scale = correction['leg_length_m']/correction['meters_per_model_unit']
        xyz = xyz @ correction['rotation_camera_to_world'].T / scale
    else:
        xyz = raw / scale * [1,-1,-1]
    confidence = np.zeros((len(points),17))
    mapping = {1:24,2:26,3:28,4:23,5:25,6:27,11:11,12:13,13:15,14:12,15:14,16:16,10:0}
    for joint,common in mapping.items():
        confidence[:,joint] = np.nan_to_num(points[:,common,3],nan=0)
    confidence[:,0] = confidence[:,[1,4]].min(axis=1)
    confidence[:,8] = confidence[:,[11,14]].min(axis=1)
    confidence[:,7] = np.minimum(confidence[:,0],confidence[:,8])
    confidence[:,9] = confidence[:,8]
    valid = confidence >= .45
    valid[:,10] = confidence[:,10] >= .1
    if correction is not None:
        valid &= support[:,None]
        valid &= correction['valid'][:,None]
    else:
        xyz = raw / scale * [1,-1,-1]
    xyz[~valid] = np.nan
    original_display = xyz.copy()
    display_rotation = np.eye(3)
    display_method = 'standing_reference' if correction is not None else 'camera_original'
    display_reason = ''
    if correction is None:
        rotation,display_reason = display_alignment(raw, points, aspect)
        if rotation is not None:
            display_rotation = rotation
            xyz = xyz @ rotation.T
            display_method = 'fixed_2d_trunk_reference'
    constrained_xyz,plane_metadata = constrain_leg_planes(xyz,confidence)
    metadata = dict(status='available',version=5,model='VideoPose3D',
                    method='RTMPose 2D sequence → VideoPose3D',
                    coordinate_system='world_pelvis_relative' if correction is not None else 'camera_pelvis_relative',
                    units='estimated_leg_length',quality=quality,
                    leg_plane_constraint=plane_metadata,
                    correction=correction_metadata or {'status':'not_requested'},
                    display_alignment=dict(method=display_method,reason=display_reason,
                                           rotation=display_rotation.tolist()),
                    scale_method='clip_median_leg_length',sample_count=len(timestamps),
                    valid_sample_ratio=round(float(valid[:,0].mean()),3),limitations=LIMITATION)
    payload = dict(metadata,joint_names=H36M_NAMES,edges=EDGES,
                   timestamps=np.asarray(timestamps).tolist(),frame_indices=list(range(len(points))),
                   confidence=np.round(confidence,4).tolist(),
                   keypoints=[[[round(float(v),5) for v in j] if np.isfinite(j).all() else None
                               for j in frame] for frame in xyz])
    sagittal = dict(status='unavailable',reason=(correction_metadata or {}).get(
        'reason','需要同机位站立标定，未生成校正轨迹'))
    if correction is not None:
        ground_pose = camera_xyz @ correction['rotation_camera_to_world'].T
        leg_m = correction['leg_length_m']
        ground_pose *= correction['meters_per_model_unit']/leg_m
        ground_pose[:,:,1] += correction['hip_height_m'][:,None]/leg_m
        ground_valid = valid.copy()
        ground_valid[:,6] &= correction['validity']['left']
        ground_valid[:,3] &= correction['validity']['right']
        ground_pose[:,6,:2] = correction['trajectories']['left']
        ground_pose[:,3,:2] = correction['trajectories']['right']
        sagittal = dict(status='available',method='standing_reference',origin='pelvis_ground_projection',
                        valid_ratio=float(correction['valid'].mean()),
                        keypoints=[[[round(float(v),5) for v in j[:2]] if good else None
                                    for j,good in zip(frame,mask)]
                                   for frame,mask in zip(ground_pose,ground_valid)])
    payload['sagittal'] = sagittal
    plane_side=side_projection(constrained_xyz,plane_metadata)
    payload['plane_sagittal']=dict(
        status=plane_metadata['status'],reason=plane_metadata.get('reason'),
        method='per_frame_hip_plane',origin='pelvis',
        valid_ratio=float(np.isfinite(plane_side[:,[3,6]]).all(axis=(1,2)).mean()),
        keypoints=[[joint.tolist() if np.isfinite(joint).all() else None for joint in frame]
                   for frame in plane_side])
    payload['constrained_keypoints'] = [[[round(float(v),5) for v in j] if np.isfinite(j).all() else None
                                        for j in frame] for frame in constrained_xyz]
    payload['ground_reference'] = dict(
        method='standing_estimate' if correction is not None else 'display_reference',
        normal=[0,1,0],
        heights=(-correction['hip_height_m']/correction['leg_length_m']).tolist()
                if correction is not None else [-1.1]*len(timestamps),
        units='estimated_leg_length',
        limitation='Estimated standing ground; not camera-calibrated measurement'
                   if correction is not None else 'Illustrative plane only; no physical ground calibration')
    payload['original_keypoints'] = [[[round(float(v),5) for v in j] if np.isfinite(j).all() else None
                                      for j in frame] for frame in original_display]
    output = Path(output)
    np.savez_compressed(output/'skeleton3d.npz',keypoints=xyz,confidence=confidence,
                        constrained_keypoints=constrained_xyz,
                        plane_side_keypoints=plane_side,
                        leg_plane_normal=np.asarray(plane_metadata.get('normal',[np.nan]*3)),
                        leg_plane_normals=np.asarray([n if n is not None else [np.nan]*3 for n in plane_metadata.get('normals',[])],dtype=float),
                        raw_camera_keypoints=raw,smoothed_camera_keypoints=camera_xyz,
                        original_display_keypoints=original_display,
                        display_rotation=display_rotation,
                        rotated_raw_keypoints=raw @ correction['rotation_camera_to_world'].T if correction is not None else raw,
                        rotated_smoothed_keypoints=camera_xyz @ correction['rotation_camera_to_world'].T if correction is not None else camera_xyz,
                        temporal_valid=support,
                        rotation_camera_to_world=correction['rotation_camera_to_world'] if correction is not None else np.eye(3),
                        timestamps=timestamps,joint_names=np.asarray(H36M_NAMES),edges=np.asarray(EDGES))
    (output/'skeleton3d.json').write_text(json.dumps(payload,ensure_ascii=False,allow_nan=False,separators=(',',':')),'utf-8')
    return metadata
