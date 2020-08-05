"""
model: gaze-estimation-adas-0002
input: BxCxHxW, BxCxHxW, angle
input shape: (1, 3, 60, 60), (1, 3, 60, 60), (1, 3)
output: angle
output shape: (1, 3)
"""
import cv2
import numpy as np

from .base_model import BaseModel
from .face_detection import FaceDetect
from .facial_landmarks_detection import FacialLandmarkDetect
from .head_pose_estimation import HeadPoseDetect


class GazeDetect(BaseModel):
    model_name = "gaze-estimation-adas-0002"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        model_kwargs = dict(
            model_dir=self.model_dir, device=self.device, precision=self.precision
        )
        self.face_detect = FaceDetect(**model_kwargs)
        self.facial_landmark_detect = FacialLandmarkDetect(**model_kwargs)
        self.head_pose_detect = HeadPoseDetect(**model_kwargs)

    def load_model(self):
        super().load_model()
        self.face_detect.load_model()
        self.facial_landmark_detect.load_model()
        self.head_pose_detect.load_model()

    def predict(self, image):
        face_coords = self.face_detect.predict(image)
        if not len(face_coords):
            return print('No face detected!!!')
        face_image = next(self.face_detect.yield_output_images(image, face_coords))

        head_pose_request = self.head_pose_detect.predict_async(face_image)

        landmark_coords = self.facial_landmark_detect.predict(face_image)
        image_generator = self.face_detect.yield_output_images(
            face_image, landmark_coords
        )
        left_eye_image = next(image_generator)
        right_eye_image = next(image_generator)

        head_pose_angles = self.head_pose_detect.extract_output(
            face_image, head_pose_request
        )

        return self.predict_by_inputs(left_eye_image, right_eye_image, head_pose_angles)

    def predict_by_inputs(self, left_eye_image, right_eye_image, head_pose_angles):
        image_input_shape = (1, 3, 60, 60)
        inputs = dict(
            **self.preprocess_input(
                left_eye_image,
                input_name="left_eye_image",
                input_shape=image_input_shape,
            ),
            **self.preprocess_input(
                right_eye_image,
                input_name="right_eye_image",
                input_shape=image_input_shape,
            ),
            head_pose_angles=head_pose_angles,
        )
        request = self.net.start_async(request_id=0, inputs=inputs)
        if request.wait(-1) == 0:
            output_buffer = request.output_blobs[self.output_name].buffer
        return output_buffer

    def predict_async(self, image):
        raise NotImplementedError("Async prediction not available for this model")

    def extract_output(self, image, request):
        raise NotImplementedError("Async prediction not available for this model")
