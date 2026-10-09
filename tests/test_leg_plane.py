import unittest
import numpy as np
from pace.pose.leg_plane import constrain_leg_planes, side_projection


class LegPlaneTests(unittest.TestCase):
    def test_projection_hip_anchors_and_parallel_planes(self):
        pose=np.zeros((40,17,3));confidence=np.ones((40,17))
        pose[:,4]=[-.2,0,0];pose[:,1]=[.2,0,0]
        pose[:,11]=[-.3,.5,0];pose[:,14]=[.3,.5,0]
        pose[:,5]=[-.4,-.5,.1];pose[:,6]=[-.5,-1,.3]
        pose[:,2]=[.5,-.5,-.2];pose[:,3]=[.6,-1,-.3]
        original=pose.copy()
        result,meta=constrain_leg_planes(pose,confidence)
        self.assertEqual(meta['status'],'available')
        normal=np.asarray(meta['normals'])
        for hip,knee,ankle in [(4,5,6),(1,2,3)]:
            np.testing.assert_allclose(np.sum((result[:,[knee,ankle]]-result[:,hip,None])*normal[:,None],axis=2),0,atol=1e-10)
            np.testing.assert_array_equal(result[:,hip],original[:,hip])
        np.testing.assert_array_equal(pose,original)
        self.assertGreater(meta['mean_bone_length_change_ratio'],0)
        projected=side_projection(result,meta)
        np.testing.assert_allclose(projected[:,0],0,atol=1e-10)
        np.testing.assert_allclose(projected[:,6,1],result[:,6,1],atol=1e-10)
        np.testing.assert_allclose(projected[:,6,0],result[:,6,2]*-1,atol=1e-10)
        pose[10,6]=np.nan;pose[11,4]=np.nan
        result,meta=constrain_leg_planes(pose,confidence)
        self.assertTrue(np.isnan(result[10,6]).all())
        self.assertTrue(np.isnan(result[11,5:7]).all())

    def test_degenerate_or_insufficient_does_not_invent_axis(self):
        pose=np.zeros((40,17,3));confidence=np.ones((40,17))
        result,meta=constrain_leg_planes(pose,confidence)
        self.assertEqual(meta['status'],'unavailable')
        np.testing.assert_array_equal(result,pose)

    def test_tilted_hips_unchanged_and_planes_perpendicular_to_each_frame_hipline(self):
        pose=np.zeros((40,17,3));confidence=np.ones((40,17))
        pose[:,4]=[-.2,-.1,-.1];pose[:,1]=[.2,.1,.1]
        pose[:,11]=[-.3,.4,-.1];pose[:,14]=[.3,.6,.1]
        pose[:,5]=[-.4,-.5,.1];pose[:,6]=[-.5,-1,.3]
        pose[:,2]=[.5,-.5,-.2];pose[:,3]=[.6,-1,-.3]
        result,meta=constrain_leg_planes(pose,confidence)
        normal=np.asarray(meta['normals'])
        np.testing.assert_array_equal(result[:,[1,4]],pose[:,[1,4]])
        np.testing.assert_allclose(result[:,[1,4]].mean(axis=1),pose[:,[1,4]].mean(axis=1))
        hipline=result[:,1]-result[:,4]
        np.testing.assert_allclose(np.cross(hipline,normal),0,atol=1e-10)
        np.testing.assert_allclose(np.linalg.norm(hipline,axis=1),np.linalg.norm(pose[:,1]-pose[:,4],axis=1))
        for h,k,a in [(4,5,6),(1,2,3)]:
            np.testing.assert_allclose(np.sum((result[:,[k,a]]-result[:,h,None])*normal[:,None],axis=2),0,atol=1e-10)
        self.assertLess(meta['max_plane_residual_leg_ratio'],1e-10)
        pose[20,1]=[0,.2,.3]
        result,meta=constrain_leg_planes(pose,confidence)
        self.assertFalse(np.allclose(meta['normals'][19],meta['normals'][20]))
        np.testing.assert_array_equal(result[:,[1,4]],pose[:,[1,4]])
