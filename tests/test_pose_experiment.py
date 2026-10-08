import unittest
import numpy as np
from analysis.pose_experiment import align_world, smooth_pose, knee_angles, quality_2d


class PoseExperimentTests(unittest.TestCase):
    def scene(self):
        rng=np.random.default_rng(8)
        p=rng.normal(0,.1,(45,17,3))
        p[:,4]=[-.1,0,0];p[:,1]=[.1,0,0]
        p[:,8]=[0,-.6,0];p[:,11]=[-.2,-.6,0];p[:,14]=[.2,-.6,0]
        p[:,3]=[.1,.8,0];p[:,6]=[-.1,.8,0]
        return p,np.arange(45)/30

    def test_rotation_preserves_lengths_angles_and_handedness(self):
        p,t=self.scene();world,meta=align_world(p,{'running_direction':[1,0]})
        r=np.asarray(meta['rotation_camera_to_world'])
        np.testing.assert_allclose(r@r.T,np.eye(3),atol=1e-10)
        self.assertAlmostEqual(np.linalg.det(r),1)
        np.testing.assert_allclose(np.linalg.norm(p[:,4]-p[:,5],axis=1),np.linalg.norm(world[:,4]-world[:,5],axis=1))
        for side in ['left','right']:np.testing.assert_allclose(knee_angles(p,side),knee_angles(world,side),atol=1e-9)

    def test_temporal_processing_commutes_with_fixed_rotation(self):
        p,t=self.scene();world,meta=align_world(p);r=np.asarray(meta['rotation_camera_to_world'])
        actual=smooth_pose(world,t)
        expected=smooth_pose(p,t)@r.T;expected[:,:,1]-=meta['ankle_low_reference']
        np.testing.assert_allclose(actual,expected,atol=1e-10)

    def test_quality_gate_rejects_occluded_ankles(self):
        p=np.ones((30,17,3));p[:,15:,2]=0
        self.assertFalse(quality_2d(p)['passed'])


if __name__=='__main__':unittest.main()
