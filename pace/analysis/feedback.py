"""Conservative, non-medical rule-based feedback."""

from __future__ import annotations

from typing import Dict, List

from pace.config import AnalysisConfig


def build_feedback(metrics: Dict[str, object], config: AnalysisConfig) -> List[str]:
    feedback: List[str] = []
    cadence = metrics.get("cadence_steps_per_min")
    if cadence is None:
        feedback.append("有效落脚事件不足，建议检查是否为固定机位的完整侧面视频。")
    elif cadence < config.cadence_low:
        feedback.append("当前检测步频偏低；请结合个人配速和训练目标解读。")
    elif cadence > config.cadence_high:
        feedback.append("当前检测步频较高；请确认事件标记是否与实际落脚一致。")
    else:
        feedback.append("检测到的步频位于常用观察区间内。")

    asymmetry = metrics.get("stride_time_asymmetry_percent")
    if asymmetry is not None and asymmetry > config.asymmetry_warning_percent:
        feedback.append("左右步幅周期存在较明显差异，建议用更长视频复核稳定性。")
    offset = metrics.get("mean_foot_strike_offset_body_ratio")
    if offset is not None and abs(offset) > config.overstride_warning_ratio:
        feedback.append("落脚点相对髋部的水平偏移较明显，建议人工查看标注视频。")
    feedback.append("结果仅用于运动观察，不构成医疗诊断或伤病风险判断。")
    return feedback

