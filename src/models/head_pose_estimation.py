"""
model: head-pose-estimation-adas-0001
input: BxCxHxW
input shape: (1, 3, 60, 60)
output: angle_y_fc, angle_p_fc, angle_r_fc
output shape: (1, 1), (1, 1), (1, 1)
"""

import numpy as np

from .base_model import BaseModel


class HeadPoseDetect(BaseModel):
    model_name = "head-pose-estimation-adas-0001"

    def preprocess_output(self, image, output_blobs):
        angle_y_fc = output_blobs["angle_y_fc"].buffer
        angle_p_fc = output_blobs["angle_p_fc"].buffer
        angle_r_fc = output_blobs["angle_r_fc"].buffer
        return np.column_stack((angle_y_fc, angle_p_fc, angle_r_fc))
