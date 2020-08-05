from argparse import ArgumentParser


def build_argparser():
    parser = ArgumentParser()
    parser.add_argument(
        "--model-dir",
        type=str,
        default="${USERPROFILE}/Documents/Intel/OpenVINO/openvino_models/models/intel",
        help="Path to intel model deirectory e.g ...openvino_models/models/intel",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="CPU",
        help="Device for running inference e.g CPU GPU etc.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="FP32",
        help="Precision FP32, FP16 or INT8 only.",
    )
    parser.add_argument(
        "--input-type",
        type=str,
        default="cam",
        help="Input Type cam, video or image only.",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        required=False,
        help="Input video or image file absolute path.",
    )
    parser.add_argument(
        "--inspect",
        default=False,
        action="store_true",
        help="Inspect outputs of the intermediate models.",
    )
    return parser
