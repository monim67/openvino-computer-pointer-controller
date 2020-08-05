"""
model: landmarks-regression-retail-0009
input: BxCxHxW
input shape: (1, 3, 48, 48)
output: (point_x, point_y) * 5
output shape: (1, 10, 1, 1)
"""

import numpy as np

from .base_model import BaseModel


class FacialLandmarkDetect(BaseModel):
    model_name = "landmarks-regression-retail-0009"

    def get_coords(self, buffer):
        shape_5_2 = buffer.reshape((-1, 2))
        shape_5_4 = np.column_stack((shape_5_2, shape_5_2))
        shape_5_4 += (-0.11, -0.08, 0.11, 0.08)
        shape_5_4[shape_5_4 < 0] = 0
        return shape_5_4
