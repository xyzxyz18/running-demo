import unittest

import numpy as np

from pace.pose.backends import COCO_TO_LANDMARKS, coco_landmarks, validate_model


class ModelMappingTests(unittest.TestCase):
    def test_mapping_preserves_joints_and_does_not_invent_feet(self):
        xy = np.arange(34).reshape(17, 2) / 40
        points = coco_landmarks(xy, np.ones(17)).landmarks
        np.testing.assert_allclose(points[COCO_TO_LANDMARKS, :2], xy)
        self.assertTrue(np.isnan(points[[29, 30, 31, 32], :3]).all())
        self.assertTrue((points[[29, 30, 31, 32], 3] == 0).all())

    def test_low_confidence_is_not_detected(self):
        self.assertIsNone(coco_landmarks(np.zeros((17, 2)), np.zeros(17)).landmarks)

    def test_unknown_model_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_model('unknown')
        for removed in ('mediapipe', 'movenet'):
            with self.assertRaises(ValueError):
                validate_model(removed)


