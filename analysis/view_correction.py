"""Weak-perspective side-view estimates, anchored by an upright reference."""
from __future__ import annotations

import numpy as np

REQUIRED = [11, 12, 23, 24, 25, 26, 27, 28]
LIMITATIONS = ('固定机位、水平跑台、无变焦与明显横向移动；竖直方向由直立骨架近似估计。'
               '结果来自单目三维模型及弱透视假设，不是真实三维测量。关节角度仍为二维估计。')


def validate_calibration(value):
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError('站立标定必须为 JSON 对象')
    try:
        height = float(value['height_cm'])
        start, end = float(value['start_seconds']), float(value['end_seconds'])
        marker = float(value.get('marker_seconds', (start + end) / 2))
        head, ground = np.asarray(value['head'], dtype=float), np.asarray(value['ground'], dtype=float)
        if not (100 <= height <= 230 and 0 <= start < end and 1 <= end-start <= 10
                and start <= marker <= end):
            raise ValueError
        if head.shape != (2,) or ground.shape != (2,) or not np.isfinite([head, ground]).all():
            raise ValueError
        if np.any(head < 0) or np.any(head > 1) or np.any(ground < 0) or np.any(ground > 1):
            raise ValueError
        if ground[1] - head[1] < .15:
            raise ValueError
        source = value.get('source', 'video')
        facing = value.get('facing', 'right')
        if source not in ('video', 'reference') or facing not in ('left', 'right'):
            raise ValueError
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError('请提供有效身高（100–230 cm）、1–10 秒站立片段、片段内的标记时间和头顶/脚底点') from exc
    return dict(height_cm=height, start_seconds=start, end_seconds=end, marker_seconds=marker,
                head=head.tolist(), ground=ground.tolist(), source=source, facing=facing)


def _unit(vector):
    length = np.linalg.norm(vector)
    if not np.isfinite(length) or length < 1e-6:
        raise ValueError('身体方向无法确定，请重新选择全身可见的站立片段')
    return vector / length


def correct_trajectory(points, pose3d, timestamps, aspect, calibration,
                       reference_points=None, reference_pose3d=None, reference_times=None,
                       min_visibility=.45, temporal_valid=None, reference_valid=None):
    """Ground origins are expressed relative to the pelvis, not absolute camera roots."""
    calibration = validate_calibration(calibration)
    if calibration is None:
        raise ValueError('未提供站立标定')
    ref = points if reference_points is None else reference_points
    xyz = pose3d if reference_pose3d is None else reference_pose3d
    rt = timestamps if reference_times is None else reference_times
    if calibration['end_seconds'] > rt[-1] + np.median(np.diff(rt)) or calibration['start_seconds'] < rt[0]:
        raise ValueError('站立片段超出参考视频范围')
    selected = (rt >= calibration['start_seconds']) & (rt <= calibration['end_seconds'])
    visible_ref = (ref[:, REQUIRED, 3] >= min_visibility).all(axis=1)
    visible_ref &= np.isfinite(ref[:, REQUIRED, :2]).all(axis=(1, 2))
    usable = selected & visible_ref
    if reference_valid is not None:
        usable &= reference_valid
    if selected.sum() < 10 or usable.sum() < 10 or usable.sum() < selected.sum() * .7:
        raise ValueError('站立片段超出视频范围或髋、肩、膝、踝可见度不足')
    reference_leg = np.mean([np.median(np.linalg.norm(xyz[usable,h]-xyz[usable,k],axis=1)
                            + np.linalg.norm(xyz[usable,k]-xyz[usable,a],axis=1))
                            for h,k,a in [(1,2,3),(4,5,6)]])
    ankle_motion = max(np.linalg.norm(np.std(xyz[usable,a],axis=0)) for a in [3,6])
    if ankle_motion > .08 * reference_leg:
        raise ValueError('参考片段中脚踝移动过大，请选择直立站稳片段')
    up_samples = xyz[usable, 8] - (xyz[usable, 1] + xyz[usable, 4]) / 2
    up = _unit(np.median(up_samples, axis=0))
    lateral_samples = (xyz[usable, 1] - xyz[usable, 4] + xyz[usable, 14] - xyz[usable, 11]) / 2
    right = _unit(np.median(lateral_samples, axis=0))
    right = _unit(right - np.dot(right, up) * up)
    forward = _unit(np.cross(right, up))
    if forward[0] * (1 if calibration['facing'] == 'right' else -1) < 0:
        forward *= -1
    axes = np.column_stack([forward[:2], up[:2]])
    if np.linalg.norm(up[:2]) < .35 or np.linalg.cond(axes) > 15:
        raise ValueError('接近正面/背面或俯视角过大，无法稳定定位地面与侧面轨迹')
    image_ref = ref[:, :, :2] * [aspect, 1]
    marker_index = int(np.argmin(np.abs(rt - calibration['marker_seconds'])))
    if not usable[marker_index]:
        raise ValueError('标记画面的骨架不可用，请选择清晰的站立画面')
    head = np.asarray(calibration['head']) * [aspect, 1]
    ground = np.asarray(calibration['ground']) * [aspect, 1]
    projected_vertical = head-ground
    alignment = np.dot(_unit(projected_vertical), _unit(up[:2]))
    if alignment < np.cos(np.radians(20)):
        raise ValueError('头顶到脚底方向与直立骨架不一致，请站直并重新标记')
    # The local 3D/image similarity fit calibrates learned model units to stature.
    ratios = []
    for common_a, common_b, a, b in [(23,25,4,5), (25,27,5,6), (24,26,1,2), (26,28,2,3)]:
        screen = np.linalg.norm(image_ref[usable, common_a]-image_ref[usable, common_b],axis=1)
        model = np.linalg.norm(xyz[usable, a, :2]-xyz[usable, b, :2],axis=1)
        ratios.extend((screen[model > .02]/model[model > .02]).tolist())
    if not ratios:
        raise ValueError('站立尺度估计退化')
    screen_per_model = float(np.median(ratios))
    screen_per_meter = np.linalg.norm(projected_vertical) / (calibration['height_cm']/100 * np.linalg.norm(up[:2]))
    meters_per_model = screen_per_model/screen_per_meter
    ref_pelvis = image_ref[marker_index, [23,24]].mean(axis=0)
    # A weak-perspective origin anchor; head and ground must be in the runner's plane.
    height_ref = np.linalg.solve(axes, (ref_pelvis-ground)/screen_per_meter)[1]
    if not .25 <= height_ref <= calibration['height_cm']/100:
        raise ValueError('估计髋部高度不合理，请将脚底点标在身体正下方的支撑面')
    image_pelvis = (points[:,23,:2]+points[:,24,:2])/2 * [aspect,1]
    delta = np.linalg.solve(axes, ((image_pelvis-ref_pelvis)/screen_per_meter).T).T
    hip_height = height_ref + delta[:,1]
    metric_pose = pose3d * meters_per_model
    lengths = []
    for h,k,a in [(1,2,3),(4,5,6)]:
        lengths.append(np.median(np.linalg.norm(metric_pose[:,h]-metric_pose[:,k],axis=1)
                                 +np.linalg.norm(metric_pose[:,k]-metric_pose[:,a],axis=1)))
    leg = float(np.mean(lengths))
    if not np.isfinite(leg) or leg < .2 or leg > calibration['height_cm']/100:
        raise ValueError('三维腿长估计不合理')
    lateral_run = pose3d[:,1]-pose3d[:,4]+pose3d[:,14]-pose3d[:,11]
    norms = np.linalg.norm(lateral_run,axis=1)
    alignment_run = lateral_run @ right / np.maximum(norms,1e-6)
    valid = (points[:, [11,12,23,24], 3] >= min_visibility).all(axis=1)
    valid &= np.isfinite(image_pelvis).all(axis=1) & (alignment_run > np.cos(np.radians(45)))
    valid &= np.isfinite(hip_height) & (hip_height > .2) & (hip_height < calibration['height_cm']/100)
    if temporal_valid is not None:
        valid &= temporal_valid
    if valid.mean() < .7:
        raise ValueError('超过 30% 帧遮挡、身体转向或三维朝向不稳定，校正不可用')
    view_ground = _unit(np.array([0.,0.,1.])-up*up[2])
    yaw = float(np.degrees(np.arccos(np.clip(abs(np.dot(right,view_ground)),0,1))))
    if yaw > 75:
        raise ValueError('过于接近正面或背面，不输出侧面校正曲线')
    origins = -hip_height[:,None]*up
    # X forward, Y up, Z lateral: one proper rotation for the entire clip.
    lateral = _unit(np.cross(forward, up))
    rotation = np.stack([forward, up, lateral])
    trajectories, validity = {}, {}
    for side, common, joint in [('left',27,6),('right',28,3)]:
        displacement = metric_pose[:,joint]-origins
        trajectories[side] = np.column_stack([displacement @ forward, displacement @ up])/leg
        validity[side] = valid & (points[:,common,3] >= min_visibility)
        validity[side] &= np.isfinite(points[:,common,:2]).all(axis=1)
        # Never clamp to the floor: negative estimates are quality evidence.
        validity[side] &= trajectories[side][:,1] > -.05
        trajectories[side][~validity[side]] = np.nan
    if any(np.mean(validity[side]) < .5 for side in ('left', 'right')):
        raise ValueError('脚踝遮挡或三维估计异常超过 50%，校正不可用')
    return dict(trajectories=trajectories, validity=validity, pose3d_m=metric_pose,
                rotation_camera_to_world=rotation, valid=valid,
                leg_length_m=leg, meters_per_model_unit=meters_per_model,
                ground_origins_pelvis_relative_m=origins, hip_height_m=hip_height,
                metadata=dict(status='available', method='VideoPose3D + 站立参考弱透视估计',
                    deviation_from_side_degrees=round(yaw,1),
                    camera_elevation_degrees=round(float(np.degrees(np.arcsin(up[2]))),1),
                    units='leg_length', origin='pelvis_ground_projection',
                    leg_length_m=round(leg,4), meters_per_model_unit=round(meters_per_model,6),
                    height_reference_m=round(height_ref,4), valid_frame_ratio=round(float(valid.mean()),3),
                    axes_camera=dict(forward=forward.tolist(), up=up.tolist(), right=right.tolist()),
                    rotation_camera_to_world=rotation.tolist(),
                    rotation_determinant=float(np.linalg.det(rotation)),
                    calibration=calibration, limitations=LIMITATIONS))
