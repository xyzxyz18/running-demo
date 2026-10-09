import unittest
import numpy as np

from pace.analysis.view_correction import correct_trajectory, validate_calibration
from pace.analysis.stability import foot_cycle_analysis


def scene(yaw=0, pitch=0, roll=0):
    a,b,c = np.radians([yaw,pitch,roll])
    ry=np.array([[np.cos(a),0,np.sin(a)],[0,1,0],[-np.sin(a),0,np.cos(a)]])
    rx=np.array([[1,0,0],[0,np.cos(b),-np.sin(b)],[0,np.sin(b),np.cos(b)]])
    rz=np.array([[np.cos(c),-np.sin(c),0],[np.sin(c),np.cos(c),0],[0,0,1]])
    rot=rz @ rx @ ry
    up=rot @ [0,-1,0]; right=rot @ [0,0,1]; forward=rot @ [1,0,0]
    n=121; t=np.arange(n)/30; aspect=16/9
    xyz=np.zeros((n,17,3))
    for h,k,a,s in [(1,2,3,1),(4,5,6,-1)]:
        xyz[:,h]=right*.1*s
        xyz[:,k]=right*.1*s-up*.4
        xyz[:,a]=right*.1*s-up*.85
    xyz[:,8]=up*.45; xyz[:,10]=up*.7
    xyz[:,11]=up*.45-right*.2; xyz[:,14]=up*.45+right*.2
    pelvis=np.tile([aspect*.5,.45],(n,1))
    def screen(poses,root):
        points=np.full((n,33,4),np.nan)
        for idx,joint in [(11,11),(12,14),(13,12),(14,15),(15,13),(16,16),
                          (23,4),(24,1),(25,5),(26,2),(27,6),(28,3)]:
            points[:,idx,:2]=(poses[:,joint,:2]*.4+root)/[aspect,1]
            points[:,idx,3]=1
        return points
    ref=screen(xyz,pelvis); refxyz=xyz.copy()
    rise=.03*np.sin(t*4); travel=.04*np.cos(t*3)
    root=pelvis+.4*(rise[:,None]*up[:2]+travel[:,None]*forward[:2])
    phase=np.arange(n)%30/30
    for ankle,amp in [(3,.25),(6,.2)]:
        xyz[:,ankle]+=amp*np.sin(phase*2*np.pi)[:,None]*forward
        xyz[:,ankle]+=.08*(1-np.cos(phase*2*np.pi))[:,None]*up
    points=screen(xyz,root)
    head=(pelvis[0]+.75*.4*up[:2])/[aspect,1]
    ground=(pelvis[0]-.95*.4*up[:2])/[aspect,1]
    calibration=dict(height_cm=170,start_seconds=0,end_seconds=3,marker_seconds=1,
                     head=head.tolist(),ground=ground.tolist(),source='reference',facing='right')
    return points,xyz,t,aspect,calibration,ref,refxyz,up,forward,rise


class ViewCorrectionTests(unittest.TestCase):
    def test_yaw_pitch_roll_recover_same_paths_and_retain_hip_bounce(self):
        baseline=None
        for angles in [(0,0,0),(30,0,0),(55,12,8)]:
            p,xyz,t,aspect,c,ref,refxyz,up,forward,rise=scene(*angles)
            result=correct_trajectory(p,xyz,t,aspect,c,ref,refxyz,t)
            path=result['trajectories']['left']
            self.assertTrue(result['validity']['left'].all())
            self.assertAlmostEqual(result['metadata']['height_reference_m'],.95,places=4)
            np.testing.assert_allclose(result['hip_height_m'],.95+rise,atol=1e-8)
            self.assertGreater(path[0,1],0)  # Ankle is not a ground-contact keypoint.
            if baseline is None: baseline=path
            else: np.testing.assert_allclose(path,baseline,atol=1e-8)
            self.assertFalse(np.allclose(path,result['trajectories']['right']))

    def test_near_frontal_and_bad_reference_are_rejected(self):
        p,xyz,t,aspect,c,ref,refxyz,*_=scene(88)
        with self.assertRaises(ValueError): correct_trajectory(p,xyz,t,aspect,c,ref,refxyz,t)
        p,xyz,t,aspect,c,ref,refxyz,*_=scene()
        ref[:,23,3]=0
        with self.assertRaises(ValueError): correct_trajectory(p,xyz,t,aspect,c,ref,refxyz,t)

    def test_orientation_flip_is_rejected(self):
        p,xyz,t,aspect,c,ref,refxyz,*_=scene()
        xyz[:,[1,4]]=xyz[:,[4,1]].copy(); xyz[:,[11,14]]=xyz[:,[14,11]].copy()
        with self.assertRaises(ValueError): correct_trajectory(p,xyz,t,aspect,c,ref,refxyz,t)

    def test_missing_ankle_does_not_create_a_corrected_path(self):
        p,xyz,t,aspect,c,ref,refxyz,*_=scene()
        p[40:55,27,3]=0
        result=correct_trajectory(p,xyz,t,aspect,c,ref,refxyz,t)
        self.assertTrue(np.isnan(result['trajectories']['left'][40:55]).all())
        motion=foot_cycle_analysis(p,{'left':[0,30,60,90,120],'right':[0,30,60,90,120]},1,
                                  timestamps=t,trajectories=result['trajectories'],validity=result['validity'])
        self.assertEqual(motion['left']['cycles'][1]['status'],'可见度不足')

    def test_running_reference_and_out_of_range_clip_are_rejected(self):
        p,xyz,t,aspect,c,ref,refxyz,*_=scene()
        c['end_seconds']=9
        with self.assertRaises(ValueError): correct_trajectory(p,xyz,t,aspect,c,ref,refxyz,t)
        c['end_seconds']=3
        refxyz[:,3,0] += .2*np.sin(t*8)
        with self.assertRaises(ValueError): correct_trajectory(p,xyz,t,aspect,c,ref,refxyz,t)

    def test_calibration_validation(self):
        for value in [{'height_cm':float('nan')}, [], {'height_cm':170}]:
            with self.assertRaises(ValueError):validate_calibration(value)
        self.assertIsNone(validate_calibration(None))
