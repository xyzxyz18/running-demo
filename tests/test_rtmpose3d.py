import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import main
from config import AnalysisConfig
from pose.backends import COCO_TO_LANDMARKS
from pose import rtmpose3d as rtm3d
from tests.test_view_correction import scene


def simcc_outputs():
    # Coherent left/right leg plus depth: equal axis units are essential.
    coordinates = np.full((133, 3), [144, 160, 144])
    coordinates[11] = [125, 170, 130]; coordinates[12] = [163, 170, 158]
    coordinates[13] = [125, 230, 110]; coordinates[14] = [163, 230, 180]
    coordinates[15] = [125, 310, 100]; coordinates[16] = [163, 310, 190]
    outputs = [np.zeros((1, 133, size)) for size in [576, 768, 576]]
    for joint in range(133):
        for axis in range(3):
            outputs[axis][0, joint, int(coordinates[joint, axis] * 2)] = .8
    return outputs


class RTMPose3DTests(unittest.TestCase):
    def test_decode_retains_depth_and_common_units(self):
        xyz, confidence = rtm3d.decode_simcc(simcc_outputs())
        np.testing.assert_array_equal(xyz[23], [0, 0, 0])
        self.assertEqual(xyz.shape, (25, 3))
        self.assertGreater(abs(xyz[15, 2] - xyz[16, 2]), .2)
        self.assertLess(xyz[15, 1], 0)
        # 90 decoded depth bins are not 90 image pixels: verify metric decoder.
        self.assertAlmostEqual(xyz[15,2]-xyz[16,2],90/144*rtm3d.Z_RANGE)
        self.assertTrue(np.isfinite(confidence).all())

    def test_low_confidence_not_drawn_or_invented(self):
        outputs = simcc_outputs()
        outputs[0][0, 15] *= .1
        xyz, _ = rtm3d.decode_simcc(outputs)
        self.assertTrue(np.isnan(xyz[15]).all())
        outputs[0][0, 11] *= .1
        self.assertIsNone(rtm3d.decode_simcc(outputs))

    def test_camera_xy_uses_original_image_and_consistent_depth(self):
        outputs=simcc_outputs()
        xyz,_=rtm3d.decode_simcc(outputs,center=(400,500),scale=(288,384),image_size=(1000,1000))
        depths=(np.array([130,158,100])/144-1)*rtm3d.Z_RANGE+rtm3d.ROOT_DEPTH
        xy=np.array([[125,170],[163,170],[125,310]])+[400-144,500-192]
        camera=(xy-[500,500])/[1145.04940459,1143.78109572]*depths[:,None]
        pelvis=camera[:2].mean(axis=0)
        np.testing.assert_allclose(xyz[15,:2],(camera[2]-pelvis)*[1,-1])

    def test_bbox_reuses_visible_rtmpose_runner(self):
        points = np.full((33, 4), np.nan)
        points[COCO_TO_LANDMARKS, :2] = np.column_stack([np.linspace(.2, .6, 17), np.linspace(.1, .9, 17)])
        points[COCO_TO_LANDMARKS, 3] = .9
        bbox = rtm3d.runner_bbox(points, 1000, 800)
        self.assertLess(bbox[0], 200); self.assertGreater(bbox[2], 600)
        points[23, 3] = 0
        self.assertIsNone(rtm3d.runner_bbox(points, 1000, 800))

    def test_sample_timestamps_and_missing_pose_export(self):
        class Capture:
            def isOpened(self): return True
            def read(self): return True, np.zeros((100, 100, 3), np.uint8)
            def release(self): pass
        class Model:
            def process(self, *args): return rtm3d.decode_simcc(simcc_outputs())
        points = np.full((10, 33, 4), np.nan)
        times = np.arange(10) / 30
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(rtm3d.cv2, 'VideoCapture', return_value=Capture()), \
             patch.object(rtm3d, '_MODEL', Model()), \
             patch.object(rtm3d, 'runner_bbox', side_effect=[np.zeros(4), None, np.zeros(4), np.zeros(4)]), \
             patch.dict('os.environ', {'PACE_POSE3D_FPS': '10'}):
            result = rtm3d.estimate_skeleton('source.mp4', points, times, folder)
            data = json.loads((Path(folder) / 'skeleton3d.json').read_text())
            self.assertEqual(data['frame_indices'], [0, 3, 6, 9])
            np.testing.assert_allclose(data['timestamps'], times[[0, 3, 6, 9]])
            self.assertTrue(all(p is None for p in data['keypoints'][1]))
            self.assertEqual(result['valid_sample_ratio'], .75)
            self.assertEqual(result['scale_method'], 'clip_median_leg_length')
            self.assertEqual(result['coordinate_system'], 'camera_pelvis_relative')
            with np.load(Path(folder) / 'skeleton3d.npz', allow_pickle=False) as artifact:
                self.assertEqual(artifact['keypoints'].shape, (4, 25, 3))

    def test_3d_runs_without_calibration_and_failure_keeps_2d(self):
        p, _, t, *_ = scene()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); source = root / 'source.mp4'; source.write_bytes(b'placeholder')
            (root / 'skeleton3d.json').write_text('stale')
            with patch.object(main, 'extract_pose', return_value=(p, 30, 640, 360, t)), \
                 patch.object(main, 'estimate_skeleton', side_effect=RuntimeError('test 3d failure')) as estimator, \
                 patch.object(main, 'save_annotated_video'), patch.object(main, 'create_browser_video'), \
                 patch.object(main, 'create_report'), patch.object(main, 'create_pdf_report'):
                result = main.analyze(source, root, AnalysisConfig())
            estimator.assert_called_once()
            self.assertEqual(result['view_correction']['status'], 'not_requested')
            self.assertEqual(result['skeleton3d']['status'], 'unavailable')
            self.assertIn('foot_motion', result)
            self.assertTrue((root / 'timeline.json').exists())
            self.assertFalse((root / 'skeleton3d.json').exists())
