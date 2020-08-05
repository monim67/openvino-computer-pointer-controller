import os
from contexttimer import Timer

from app_args import build_argparser
from input_feeder import InputFeeder
from models.gaze_estimation import GazeDetect
from mouse_controller import MouseController


def main(model_dir, device, precision, input_type, input_file, inspect):
    mouse_controller = MouseController("medium", "fast")
    input_feeder = InputFeeder(input_type=input_type, input_file=input_file)
    input_feeder.load_data()

    gaze_detect = GazeDetect(model_dir=model_dir, device=device, precision=precision)
    gaze_detect.load_model()

    for image in input_feeder.next_batch():
        with Timer() as t:
            outputs = gaze_detect.predict(image)
        if outputs is not None:
            angle_y_fc, angle_p_fc, angle_r_fc = outputs.reshape(3)
            mouse_controller.move(-angle_y_fc, angle_p_fc)
            print(
                f"Mouse move x: {-angle_y_fc}, y: {angle_p_fc}, execution time: {t.elapsed}"
            )


if __name__ == "__main__":
    ARGS = build_argparser().parse_args()
    main(
        model_dir=os.path.expandvars(ARGS.model_dir),
        device=ARGS.device,
        precision=ARGS.precision,
        input_type=ARGS.input_type,
        input_file=ARGS.input_file,
        inspect=ARGS.inspect,
    )
