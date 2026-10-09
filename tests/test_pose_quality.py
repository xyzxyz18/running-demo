import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pace.pose.quality import fill_for_model, temporal_support, smooth_supported, diagnostics, display_alignment
from pace.pose.skeleton3d import estimate_skeleton, EDGES
from pace.analysis.view_correction import correct_trajectory
from tests.test_view_correction import scene


class PoseQualityTests(unittest.TestCase):
    def test_fixed_display_rotation_preserves_shape_and_observed_lean(self):
        p,pose,t,aspect,*_ = scene(35,12,8)
        rotation,reason=display_alignment(pose,p,aspect)
        self.assertIsNotNone(rotation)
        np.testing.assert_allclose(rotation@rotation.T,np.eye(3),atol=1e-10)
        self.assertAlmostEqual(np.linalg.det(rotation),1)
        displayed=pose*[1,-1,-1]
        adjusted=displayed@rotation.T
        for a,b in EDGES:
            np.testing.assert_allclose(np.linalg.norm(displayed[:,a]-displayed[:,b],axis=1),
                                       np.linalg.norm(adjusted[:,a]-adjusted[:,b],axis=1),atol=1e-10)
        target=(p[:,[11,12],:2].mean(axis=1)-p[:,[23,24],:2].mean(axis=1))*[aspect,-1]
        trunk=np.median(adjusted[:,8]-adjusted[:,[1,4]].mean(axis=1),axis=0)
        expected=np.r_[np.median(target,axis=0),0.]
        np.testing.assert_allclose(trunk/np.linalg.norm(trunk),expected/np.linalg.norm(expected),atol=1e-10)

    def test_short_interpolation_and_long_model_padding(self):
        t = np.arange(20)/50
        values = np.arange(20,dtype=float)
        values[2:4] = np.nan
        values[7:15] = np.nan
        result = fill_for_model(values,t)
        np.testing.assert_allclose(result[2:4],[2,3])
        self.assertEqual(result[8],6)
        self.assertEqual(result[14],15)

    def test_temporal_halo_and_no_smoothing_across_gap(self):
        t = np.arange(1000)/50
        points = np.ones((1000,33,4))
        points[400:420,27,3] = 0
        valid,gaps = temporal_support(points,t)
        self.assertEqual(len(gaps),1)
        self.assertFalse(valid[280:540].any())
        self.assertTrue(valid[:270].all())
        pose = np.zeros((1000,17,3)); pose[420:] = 100
        smoothed = smooth_supported(pose,t,valid)
        np.testing.assert_array_equal(smoothed[~valid],pose[~valid])

    def test_world_rotation_and_same_ground_trajectory(self):
        for angles in [(0,0,0),(35,12,8)]:
            p,pose,t,aspect,c,ref,refpose,*_ = scene(*angles)
            correction = correct_trajectory(p,pose,t,aspect,c,ref,refpose,t)
            rot = correction['rotation_camera_to_world']
            np.testing.assert_allclose(rot@rot.T,np.eye(3),atol=1e-10)
            self.assertAlmostEqual(np.linalg.det(rot),1)
            world = pose@rot.T
            for a,b in EDGES:
                np.testing.assert_allclose(np.linalg.norm(world[:,a]-world[:,b],axis=1),
                                           np.linalg.norm(pose[:,a]-pose[:,b],axis=1),atol=1e-10)
            with tempfile.TemporaryDirectory() as folder:
                estimate_skeleton('unused',p,t,folder,pose3d=pose,correction=correction,
                                  correction_metadata=correction['metadata'],original_pose=pose)
                data = json.loads((Path(folder)/'skeleton3d.json').read_text())
                self.assertEqual(data['coordinate_system'],'world_pelvis_relative')
                path = np.array([frame[6] for frame in data['sagittal']['keypoints']])
                np.testing.assert_allclose(path,correction['trajectories']['left'],atol=1e-5)
                self.assertGreater(data['sagittal']['keypoints'][0][4][1],0)
                with np.load(Path(folder)/'skeleton3d.npz') as artifact:
                    self.assertIn('raw_camera_keypoints',artifact.files)

    def test_no_reference_does_not_create_projection(self):
        p,pose,t,*_ = scene()
        with tempfile.TemporaryDirectory() as folder:
            estimate_skeleton('unused',p,t,folder,pose3d=pose)
            data = json.loads((Path(folder)/'skeleton3d.json').read_text())
            self.assertEqual(data['sagittal']['status'],'unavailable')
            self.assertEqual(data['coordinate_system'],'camera_pelvis_relative')

    def test_running_lean_and_hip_bounce_survive_reference_rotation(self):
        p,pose,t,aspect,c,ref,refpose,up,forward,rise = scene(35,12,8)
        pose[:,8] += .12*forward
        result = correct_trajectory(p,pose,t,aspect,c,ref,refpose,t)
        world = pose@result['rotation_camera_to_world'].T
        self.assertGreater(np.mean(world[:,8,0]),.1)
        np.testing.assert_allclose(result['hip_height_m'],.95+rise,atol=1e-8)
        with self.assertRaisesRegex(ValueError,'30%'):
            correct_trajectory(p,pose,t,aspect,c,ref,refpose,t,
                               temporal_valid=np.zeros(len(t),bool))

    def test_reprojection_matches_observations(self):
        p,pose,t,aspect,*_ = scene()
        result = diagnostics(pose,p,aspect,np.ones(len(t),bool))
        self.assertLess(result['weak_perspective_reprojection_rmse_image_height'],1e-10)
        changed = pose.copy(); changed[:,6,0] += .3
        self.assertGreater(diagnostics(changed,p,aspect,np.ones(len(t),bool))[
            'weak_perspective_reprojection_rmse_image_height'],.01)
