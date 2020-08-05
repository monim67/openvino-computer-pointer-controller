"""
model: face-detection-adas-binary-0001
input: BxCxHxW
input shape: (1, 3, 384, 672)
output: (image_id, label, conf, x_min, y_min, x_max, y_max)
output shape: (1, 1, N, 7)
"""

from .base_model import BaseModel


class FaceDetect(BaseModel):
    model_name = "face-detection-adas-binary-0001"
    precision_directory_dict = {
        "FP32": "FP32-INT1",
        "FP16": "FP32-INT1",
        "INT8": "FP32-INT1",
    }
