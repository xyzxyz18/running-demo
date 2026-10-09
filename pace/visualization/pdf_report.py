"""PDF report with measurements and analytical charts."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

LABELS = [
    ('视频时长', 'duration_seconds', '秒'), ('有效姿态帧占比', 'pose_detection_rate', '%'),
    ('检测步数', 'detected_steps', '步'), ('步频', 'cadence_steps_per_min', '步/分钟'),
    ('平均步时', 'step_time_seconds', '秒'), ('平均跨步时间', 'stride_time_seconds', '秒'),
    ('左跨步时间', 'left_stride_time_seconds', '秒'), ('右跨步时间', 'right_stride_time_seconds', '秒'),
    ('膝关节活动范围', 'knee_rom_degrees', '度'), ('髋关节活动范围', 'hip_rom_degrees', '度'),
    ('跨步时间左右差', 'stride_time_asymmetry_percent', '%'),
    ('平均着地偏移', 'mean_foot_strike_offset_body_ratio', '腿长'),
    ('脚踝轨迹周期偏差', 'foot_path_dispersion_body_ratio', '腿长'),
    ('左右脚踝平均轨迹差', 'left_right_mean_path_gap_body_ratio', '腿长'),
]


def _font():
    names = {font.name for font in font_manager.fontManager.ttflist}
    for name in ('PingFang SC', 'Noto Sans CJK SC', 'Microsoft YaHei', 'Hiragino Sans GB'):
        if name in names:
            return name
    return 'DejaVu Sans'


def create_pdf_report(path: Path, result: dict, times: np.ndarray, angles: dict,
                      clearances: dict, motion: dict, corrected_motion: dict = None) -> None:
    font = _font()
    plt.rcParams['font.family'] = font
    plt.rcParams['axes.unicode_minus'] = False
    metrics = result['metrics']
    with PdfPages(path) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69), facecolor='white')
        fig.text(.08, .94, '跑姿分析报告', fontsize=25, weight='bold')
        fig.text(.08, .90, f"视频时长 {metrics.get('duration_seconds', '—')} 秒    轨迹评价：{motion['assessment']}", fontsize=11, color='#526352')
        fig.text(.08, .875, f"平均轨迹纳入：左 {motion['left']['included_count']} 个周期 · 右 {motion['right']['included_count']} 个周期", fontsize=9, color='#526352')
        fig.text(.08, .85, '关键数据', fontsize=15, weight='bold')
        for i, (label, key, unit) in enumerate(LABELS):
            col, row = divmod(i, 7)
            x, y = .08 + col*.45, .81 - row*.067
            value = metrics.get(key)
            if key == 'pose_detection_rate' and value is not None:
                value = round(value * 100, 1)
            fig.text(x, y, label, fontsize=9, color='#5d6d62')
            fig.text(x, y-.027, '数据不足' if value is None else f'{value} {unit}', fontsize=14, weight='bold')
        fig.text(.08, .30, '分析结论', fontsize=15, weight='bold')
        observations = result.get('report', {}).get('observations', [])
        for i, item in enumerate(observations[:6]):
            import textwrap
            lines = textwrap.wrap(str(item), 48) or ['']
            fig.text(.08, .27-i*.037, '• ' + '\n  '.join(lines[:2]), fontsize=9, va='top')
        fig.text(.08, .045, result.get('disclaimer', ''), fontsize=8, color='#6b756c')
        fig.text(.92, .025, '1', ha='right', fontsize=8)
        pdf.savefig(fig); plt.close(fig)

        fig, axes = plt.subplots(3, 1, figsize=(8.27, 11.69), constrained_layout=True)
        palette = {'left': '#6c9626', 'right': '#008eaf'}
        for side in ('left', 'right'):
            axes[0].plot(times, angles[side]['knee'], color=palette[side],
                         label=f'{side.title()} knee flexion', lw=1)
            axes[0].plot(times, angles[side]['hip'], color=palette[side],
                         label=f'{side.title()} hip', lw=.8, alpha=.45)
            axes[1].plot(times, clearances[side], color=palette[side], label=f'{side.title()} ankle', lw=1)
            path_points = np.asarray(motion[side]['mean_path'], dtype=float)
            if path_points.ndim == 2 and len(path_points):
                axes[2].plot(path_points[:, 0], path_points[:, 1], color=palette[side],
                             label=f'{side.title()} ankle mean', lw=2)
                axes[2].scatter(path_points[0, 0], path_points[0, 1], color=palette[side], s=20)
        axes[0].set(ylabel='Angle (deg)', title='Joint angle curves (2D estimates)')
        axes[1].set(xlabel='Time (s)', ylabel='Clearance / leg length', title='Ankle clearance')
        axes[2].set(xlabel='Horizontal / leg length', ylabel='Vertical / leg length',
                    title='Mean ankle trajectory (dot = strike)')
        axes[2].set_aspect('equal', adjustable='datalim')
        for axis in axes:
            axis.grid(alpha=.2)
            axis.legend(fontsize=8)
        pdf.savefig(fig); plt.close(fig)

        correction = result.get('view_correction', {})
        if correction.get('status') in ('available', 'unavailable'):
            fig, axes = plt.subplots(2, 1, figsize=(8.27, 11.69), constrained_layout=True)
            if corrected_motion is not None:
                info = f"估计偏离正侧面 {correction['deviation_from_side_degrees']}°；有效帧 {correction['valid_frame_ratio']:.0%}"
            else:
                info = '校正不可用：' + correction.get('reason', '')
            import textwrap
            fig.suptitle('单目侧面轨迹估计\n' + '\n'.join(textwrap.wrap(info, 38)), fontsize=12)
            for axis, data, title in [(axes[0], motion, '原始二维轨迹\n原点为双髋中点'),
                                      (axes[1], corrected_motion, '估计的侧面轨迹\n原点为髋中点的地面投影')]:
                if data is not None:
                    for side in ('left', 'right'):
                        path_points = np.asarray(data[side]['mean_path'], dtype=float)
                        if path_points.ndim == 2 and len(path_points):
                            axis.plot(path_points[:,0], path_points[:,1], color=palette[side], label=side)
                    if axis.lines:
                        axis.legend()
                else:
                    axis.text(.5, .5, '校正不可用，保留原始轨迹', transform=axis.transAxes, ha='center')
                axis.axhline(0, color='#999999', lw=.6); axis.axvline(0, color='#999999', lw=.6)
                axis.set(title=title, xlabel='前后距离 / 腿长', ylabel='高度 / 腿长', aspect='equal')
                axis.grid(alpha=.2)
            fig.supxlabel('单目三维与弱透视近似；不是真实三维测量。站立脚踝中心高于地面。', fontsize=8)
            pdf.savefig(fig); plt.close(fig)
