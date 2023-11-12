'''
This script detects objects in videos and
generates videos as well as JSON files.
'''

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import cv2
import tqdm
from utils.config import Config
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from unidet.config import add_unidet_config
from unidet.predictor import UnifiedVisualizationDemo
from utils import *


def setup_cfg(args):
    cfg = get_cfg()

    add_unidet_config(cfg)

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = (
        args.confidence_threshold
    )
    cfg.freeze()

    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin models")
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )

    return parser


conf = Config("config.json")
input_path = Path(conf.ucf101.path)
output_path = Path(conf.unidet.output)
config_file = Path(conf.unidet.config)
checkpoint = Path(conf.unidet.checkpoint)

assert_file(config_file)
assert_dir(input_path)
assert_dir(output_path)
assert_file(checkpoint)

parallel = False

mp.set_start_method("spawn", force=True)
args = get_parser().parse_args()
setup_logger(name="fvcore")
logger = setup_logger()
logger.info("Arguments: " + str(args))

args.config_file = conf.unidet.config
args.confidence_threshold = conf.unidet.confidence
args.parallel = parallel
args.opts = ["MODEL.WEIGHTS", conf.unidet.checkpoint]

cfg = setup_cfg(args)
demo = UnifiedVisualizationDemo(cfg, parallel=parallel)
n_videos = sum(1 for f in input_path.glob(f"**/*{conf.ucf101.ext}"))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

with tqdm.tqdm(total=n_videos) as bar:
    for action in input_path.iterdir():
        for file in action.iterdir():
            bar.set_description(file.name)

            input_video = cv2.VideoCapture(str(file))
            width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(input_video.get(cv2.CAP_PROP_FPS))
            n_frames = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))
            detection_data = {}
            gen = demo.run_on_video(input_video)
            output_video_path = (
                output_path / action.name / file.with_suffix(".mp4").name
            )

            output_video_path.parent.mkdir(parents=True, exist_ok=True)

            video_writer = cv2.VideoWriter(
                str(output_video_path),
                fourcc,
                fps,
                (width, height),
            )

            for i, (viz, pred) in enumerate(gen):
                bar.set_description(f"{file.name} ({i}/{n_frames})")
                video_writer.write(viz)

                detection_data.update(
                    {
                        i: [
                            (pred_box.tolist(), score.tolist(), pred_class.tolist())
                            for pred_box, score, pred_class in zip(
                                pred.pred_boxes.tensor,
                                pred.scores,
                                pred.pred_classes,
                            )
                        ]
                    }
                )

            input_video.release()
            video_writer.release()

            output_json_path = (
                output_path / action.name / file.with_suffix(".json").name
            )

            output_json_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_json_path, "w") as json_file:
                json.dump(detection_data, json_file)

            bar.update(1)
