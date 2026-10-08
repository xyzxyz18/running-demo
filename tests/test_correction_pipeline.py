import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import main
from config import AnalysisConfig
from tests.test_view_correction import scene


class CorrectionPipelineTests(unittest.TestCase):
    def test_lifting_failure_keeps_original_analysis(self):
        p,xyz,t,aspect,c,ref,refxyz,*_ = scene()
        c['source'] = 'video'
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); source = root / 'source.mp4'; source.write_bytes(b'placeholder')
            with patch.object(main, 'extract_pose', return_value=(p,30,640,360,t)), \
                 patch.object(main, 'lift_pose', side_effect=RuntimeError('test model failure')), \
                 patch.object(main, 'save_annotated_video'), patch.object(main, 'create_browser_video'), \
                 patch.object(main, 'create_report'), patch.object(main, 'create_pdf_report'):
                result = main.analyze(source, root, AnalysisConfig(), calibration=c)
            self.assertEqual(result['view_correction']['status'], 'unavailable')
            self.assertNotIn('foot_motion_corrected', result)
            timeline = json.loads((root/'timeline.json').read_text())
            self.assertIn('foot_motion', timeline)
            self.assertNotIn('foot_motion_corrected', timeline)
            self.assertFalse((root/'pose3d.npz').exists())
            json.dumps(result, allow_nan=False)

    def test_success_exports_original_and_corrected_data(self):
        p,xyz,t,aspect,c,ref,refxyz,*_ = scene(35,10)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); source = root/'source.mp4'; source.write_bytes(b'placeholder')
            reference = root/'reference.mp4'; reference.write_bytes(b'placeholder')
            with patch.object(main, 'extract_pose', side_effect=[(p,30,640,360,t),(ref,30,640,360,t)]), \
                 patch.object(main, 'lift_pose', side_effect=[xyz,refxyz]), \
                 patch.object(main, 'save_annotated_video'), patch.object(main, 'create_browser_video'), \
                 patch.object(main, 'create_report'), patch.object(main, 'create_pdf_report'):
                result = main.analyze(source, root, AnalysisConfig(), calibration=c, reference_video=reference)
            self.assertEqual(result['view_correction']['status'], 'available')
            self.assertIn('foot_motion_corrected', result)
            self.assertIn('foot_motion_plane',result)
            self.assertEqual(result['foot_motion_plane']['version'],6)
            timeline = json.loads((root/'timeline.json').read_text())
            self.assertEqual(timeline['foot_motion_corrected']['coordinate_system'], 'estimated_sagittal_ground')
            self.assertIn('foot_motion', timeline)
            self.assertEqual(timeline['foot_motion_plane']['coordinate_system'],'hip_plane_pelvis_relative')
            with np.load(root/'pose3d.npz', allow_pickle=False) as data:
                self.assertEqual(data['camera_pose_pelvis_relative_m'].shape, (121,17,3))
                self.assertEqual(data['left_trajectory'].shape, (121,2))
                self.assertGreater(data['left_trajectory'][0,1],0)
            json.dumps(result, allow_nan=False)
