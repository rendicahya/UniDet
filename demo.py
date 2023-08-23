# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import json
import tqdm
import pathlib

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from moviepy.editor import ImageSequenceClip

from unidet.predictor import UnifiedVisualizationDemo
from unidet.config import add_unidet_config

# constants
WINDOW_NAME = "Unified detections"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_unidet_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin models")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--batch-video-input",
        help="A directory containing the input videos.",
    )
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument('--parallel',
        action='store_true',
        help='Perform processing in parallel'
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)
    demo = UnifiedVisualizationDemo(cfg, parallel=args.parallel)

    if args.input:
        if len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
        for path in tqdm.tqdm(args.input, disable=not args.output):
            # use PIL, to be consistent with evaluation
            img = read_image(path, format="BGR")
            start_time = time.time()
            predictions, visualized_output = demo.run_on_image(img)
            logger.info(
                "{}: {} in {:.2f}s".format(
                    path,
                    "detected {} instances".format(len(predictions["instances"]))
                    if "instances" in predictions
                    else "finished",
                    time.time() - start_time,
                )
            )

            if args.output:
                if os.path.isdir(args.output):
                    assert os.path.isdir(args.output), args.output
                    out_filename = os.path.join(args.output, os.path.basename(path))
                else:
                    assert len(args.input) == 1, "Please specify a directory with args.output"
                    out_filename = args.output
                visualized_output.save(out_filename)
            else:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
                if cv2.waitKey(0) == 27:
                    break  # esc to quit
    elif args.webcam:
        assert args.input is None, "Cannot have both --input and --webcam!"
        assert args.output is None, "output not yet supported with --webcam!"
        cam = cv2.VideoCapture(0)
        for vis, _ in tqdm.tqdm(demo.run_on_video(cam)):
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW_NAME, vis)
            if cv2.waitKey(1) == 27:
                break  # esc to quit
        cam.release()
        cv2.destroyAllWindows()
    elif args.video_input:
        video = cv2.VideoCapture(args.video_input)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        basename = os.path.basename(args.video_input)

        if args.output:
            if os.path.isdir(args.output):
                output_fname = os.path.join(args.output, basename)
                output_fname = os.path.splitext(output_fname)[0] + ".mp4"
            else:
                output_fname = args.output
            assert not os.path.isfile(output_fname), output_fname
            output_file = cv2.VideoWriter(
                filename=output_fname,
                # some installation of opencv may not support x264 (due to its license),
                # you can try other format (e.g. MPEG)
                fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
                fps=float(frames_per_second),
                frameSize=(width, height),
                isColor=True,
            )
        assert os.path.isfile(args.video_input)
        for vis_frame, _ in tqdm.tqdm(demo.run_on_video(video), total=num_frames):
            if args.output:
                output_file.write(vis_frame)
            else:
                cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
                cv2.imshow(basename, vis_frame)
                if cv2.waitKey(1) == 27:
                    break  # esc to quit
        video.release()
        if args.output:
            output_file.release()
        else:
            cv2.destroyAllWindows()
    elif args.batch_video_input:
        input_path = pathlib.Path(args.batch_video_input)
        assert input_path.exists(), "Video input path not exist"
        assert os.path.isdir(args.output), "Output path must be a directory"
        num_video = sum(1 for f in input_path.glob("**/*") if f.is_file())
        with tqdm.tqdm(total=num_video) as bar:
            for action in input_path.iterdir():
                for file in action.iterdir():
                    bar.set_description(file.name)
                    video = cv2.VideoCapture(str(file))
                    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = video.get(cv2.CAP_PROP_FPS)
                    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                    output_frames = []
                    det_data = {}
                    gen = demo.run_on_video(video)
                    for i, (f, pred) in enumerate(gen):
                        bar.set_description(f'{file.name} ({i}/{num_frames})')
                        viz = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                        output_frames.append(viz)
                        det_data.update({
                            i: [(pred_box.tolist(), score.tolist(), pred_class.tolist()) for pred_box, score, pred_class in zip(pred.pred_boxes.tensor, pred.scores, pred.pred_classes)]
                        })
                    video.release()
                    output_fname = pathlib.Path(args.output) / action.name / file.with_suffix('.mp4').name
                    output_fname.parent.mkdir(parents=True, exist_ok=True)
                    ImageSequenceClip(output_frames, fps=fps).write_videofile(str(output_fname), audio=False, logger=None)
                    output_json_path = pathlib.Path(args.output + '-json') / action.name / file.with_suffix('.json').name
                    output_json_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(output_json_path, "w") as json_file:
                        json.dump(det_data, json_file)
                    bar.update(1)
