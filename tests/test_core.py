import unittest

import numpy as np

from pace.biomechanics.angles import angle_series
from pace.biomechanics.gait_events import detect_foot_events, detect_ankle_events
from pace.biomechanics.foot_tracking import leg_length
from pace.analysis.metrics import compute_metrics
from pace.analysis.stability import foot_cycle_analysis


class CoreTests(unittest.TestCase):
    def test_angle(self):
        a = np.array([[1.0, 0.0]])
        b = np.array([[0.0, 0.0]])
        c = np.array([[0.0, 1.0]])
        self.assertAlmostEqual(float(angle_series(a, b, c)[0]), 90.0)

    def test_periodic_strikes(self):
        fps = 30.0
        t = np.arange(300) / fps
        y = 0.7 + 0.12 * np.cos(2 * np.pi * 1.5 * t)
        x = 0.5 + 0.08 * np.sin(2 * np.pi * 1.5 * t)
        events = detect_foot_events(x, y, fps)
        self.assertGreaterEqual(len(events.strikes), 10)
        intervals = np.diff(events.strikes) / fps
        self.assertAlmostEqual(float(np.median(intervals)), 1 / 1.5, delta=0.08)

    def test_ankle_clearance_crossings_ignore_long_occlusion(self):
        fps = 40
        time = np.arange(320) / fps
        ankle = np.zeros((len(time), 4))
        ankle[:, 1] = .8 - .12 * (1 - np.cos(2 * np.pi * 1.5 * time)) / 2
        ankle[:, 3] = 1
        ankle[100:140, 3] = 0
        events, clearance = detect_ankle_events(ankle, .45, fps)
        self.assertGreaterEqual(len(events.strikes), 8)
        self.assertTrue(np.isnan(clearance[110]))
        self.assertFalse(any(105 <= strike <= 135 for strike in events.strikes))

    def test_leg_length_corrects_image_aspect(self):
        points = np.zeros((3, 33, 4))
        points[:, :, 3] = 1
        points[:, 23, :2] = (.1, .2)
        points[:, 25, :2] = (.2, .2)
        points[:, 27, :2] = (.3, .2)
        self.assertAlmostEqual(leg_length(points, 'left', 2), .4)

    def test_missing_angles_are_json_safe(self):
        empty = np.full(30, np.nan)
        xy = np.full((30, 4), np.nan)
        result = compute_metrics(
            30.0, 30, [], [], empty, empty, empty, empty, xy, xy, xy, xy,
            1.0, 0.35, 2.0,
        )
        self.assertIsNone(result["knee_rom_degrees"])
        self.assertIsNone(result["hip_rom_degrees"])

    def test_foot_cycle_repeatability_and_insufficient_data(self):
        frames = 121
        points = np.full((frames, 33, 4), np.nan)
        phase = np.arange(frames) % 40 / 40
        for side, foot_index, hip_index in (("left", 27, 23), ("right", 28, 24)):
            points[:, hip_index, :2] = [0.5, 0.5]
            points[:, hip_index, 3] = 1
            points[:, foot_index, 0] = 0.5 + 0.15 * np.sin(2 * np.pi * phase)
            points[:, foot_index, 1] = 0.7 + 0.10 * np.cos(2 * np.pi * phase)
            points[:, foot_index, 3] = 1
        result = foot_cycle_analysis(points, {"left": [0, 40, 80, 120], "right": [0, 40, 80, 120]}, 0.2)
        self.assertEqual(result["version"], 6)
        self.assertEqual(result["left"]["cycle_count"], 3)
        self.assertEqual(result["right"]["drawable_count"], 3)
        self.assertEqual(len(result["left"]["mean_path"]), 64)
        self.assertEqual(result["left"]["included_count"], 1)
        self.assertEqual([cycle["status"] for cycle in result["left"]["cycles"]],
                         ["首尾排除", "纳入平均", "首尾排除"])
        self.assertTrue(all(cycle["deviation_body_ratio"] is not None for cycle in result["left"]["cycles"]))
        self.assertIsNone(result["overall_dispersion_body_ratio"])
        insufficient = foot_cycle_analysis(points, {"left": [0, 40], "right": []}, 0.2)
        self.assertEqual(insufficient["assessment"], "数据不足")

    def test_all_detected_cycles_remain_visible_and_report_individual_difference(self):
        frames = 121
        points = np.full((frames, 33, 4), np.nan)
        for hip in (23, 24):
            points[:, hip, :2] = [0.5, 0.5]
            points[:, hip, 3] = 1
        for foot in (27, 28):
            points[:, foot, 0] = 0.5 + 0.1 * np.sin(2 * np.pi * np.arange(frames) / 20)
            points[:, foot, 1] = 0.7 + 0.1 * np.cos(2 * np.pi * np.arange(frames) / 20)
            points[:, foot, 3] = 1
        points[20:40, 27, 0] += 0.06
        points[41:100, 28, 3] = 0
        strikes = {"left": [0, 20, 40, 100, 120], "right": [0, 20, 40, 100, 120]}
        result = foot_cycle_analysis(points, strikes, 0.2)
        self.assertEqual(result["left"]["cycle_count"], 4)
        self.assertEqual(result["left"]["cycles"][2]["end_frame"], 100)
        self.assertGreater(result["left"]["cycles"][1]["deviation_body_ratio"], 0)
        self.assertEqual(result["right"]["cycle_count"], 4)
        self.assertEqual(result["right"]["cycles"][2]["status"], "无法绘制")

    def test_two_cycles_do_not_form_a_reference_mean(self):
        frames = 41
        points = np.full((frames, 33, 4), np.nan)
        for hip in (23, 24):
            points[:, hip, :2] = [0.5, 0.5]
            points[:, hip, 3] = 1
        for foot in (27, 28):
            points[:, foot, 0] = 0.5 + 0.1 * (np.arange(frames) % 20) / 20
            points[:, foot, 1] = 0.7
            points[:, foot, 3] = 1
        points[21:31, 27, 3] = 0
        points[31:41, 27, 0] += 0.1
        result = foot_cycle_analysis(points, {"left": [0, 20, 40], "right": []}, 0.2)
        self.assertEqual(result["left"]["drawable_count"], 2)
        self.assertEqual(result["left"]["included_count"], 0)
        self.assertEqual(result["left"]["mean_path"], [])

    def test_edge_and_middle_outliers_do_not_pull_mean(self):
        frames = 141
        points = np.full((frames, 33, 4), np.nan)
        positions = np.arange(frames)
        for hip in (23, 24):
            points[:, hip, :2] = [0.5, 0.5]
            points[:, hip, 3] = 1
        for foot in (27, 28):
            points[:, foot, 0] = .5 + .1 * np.sin(2 * np.pi * positions / 20)
            points[:, foot, 1] = .7 + .1 * np.cos(2 * np.pi * positions / 20)
            points[:, foot, 3] = 1
            for cycle in (0, 3, 6):
                points[cycle*20+1:(cycle+1)*20, foot, 0] += .35
        strikes = list(range(0, frames, 20))
        result = foot_cycle_analysis(points, {"left": strikes, "right": strikes}, .2)
        left = result["left"]
        self.assertEqual(left["included_count"], 4)
        self.assertEqual([cycle["status"] for cycle in left["cycles"]],
                         ["首尾排除", "纳入平均", "纳入平均", "轨迹异常",
                          "纳入平均", "纳入平均", "首尾排除"])
        self.assertAlmostEqual(left["mean_path"][16][0], .5, delta=.1)
        self.assertLess(left["dispersion_body_ratio"], .02)

    def test_local_ground_survives_vertical_drift(self):
        fps = 40.0
        t = np.arange(240) / fps
        y = 0.7 + 0.09 * np.cos(2 * np.pi * 2 * t) - 0.04 * t / t[-1]
        events = detect_foot_events(np.zeros_like(y), y, fps)
        self.assertGreaterEqual(len(events.strikes), 9)


if __name__ == "__main__":
    unittest.main()
