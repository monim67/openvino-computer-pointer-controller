import cv2
import os

from openvino.inference_engine import IECore


class BaseModel:
    """
    Abstract class for OpenVino models
    """

    model_name = None
    confidence_index = 2
    precision_directory_dict = {"FP32": "FP32", "FP16": "FP16", "INT8": "FP16-INT8"}

    def __init__(self, model_dir, device="CPU", precision="FP32", threshold=0.60):
        model_path = os.path.join(
            model_dir,
            self.model_name,
            self.precision_directory_dict[precision],
            self.model_name,
        )
        self.model_dir = model_dir
        self.model_weights = f"{model_path}.bin"
        self.model_structure = f"{model_path}.xml"
        self.device = device
        self.precision = precision
        self.threshold = threshold

        try:
            self.core = IECore()
            self.model = self.core.read_network(
                model=self.model_structure, weights=self.model_weights
            )
        except Exception as e:
            raise ValueError(
                "Could not Initialise the network. Have you enterred the correct model path?"
            )

        self.input_name = next(iter(self.model.input_info))
        self.input_shape = self.model.input_info[self.input_name].input_data.shape
        self.output_name = next(iter(self.model.outputs))
        self.output_shape = self.model.outputs[self.output_name].shape

    def load_model(self):
        self.net = self.core.load_network(
            network=self.model, device_name=self.device, num_requests=1
        )

    def predict(self, image):
        request = self.predict_async(image)
        return self.extract_output(image, request)

    def predict_async(self, image):
        inputs = self.preprocess_input(image)
        return self.net.start_async(request_id=0, inputs=inputs)

    def extract_output(self, image, request):
        if request.wait(-1) == 0:
            outputs = self.preprocess_output(image, request.output_blobs)
        return outputs

    def preprocess_input(self, image, input_name=None, input_shape=None):
        _, _, height, width = input_shape or self.input_shape
        frame = cv2.resize(image, (width, height))
        frame = frame.transpose((2, 0, 1))
        frame = frame.reshape(1, *frame.shape)
        return {input_name or self.input_name: frame}

    def preprocess_output(self, image, output_blobs):
        coords = self.get_coords(output_blobs[self.output_name].buffer)
        return self.preprocess_coords(image, coords)

    def get_coords(self, buffer):
        outputs = buffer[0][0]
        confident_outputs = outputs[outputs[:, self.confidence_index] > self.threshold]
        return confident_outputs[:, 3:]

    def preprocess_coords(self, image, coords):
        height, width, _ = image.shape
        coords = coords * ((width, height) * 2)
        return coords.astype(int)

    def draw_output_boxes(self, image, coords, size=3):
        for x0, y0, x1, y1 in coords:
            cv2.rectangle(image, (x0, y0), (x1, y1), (0, 0, 255), size)

    def yield_output_images(self, image, coords):
        for x0, y0, x1, y1 in coords:
            yield image[y0:y1, x0:x1]
