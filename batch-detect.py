"""
This script detects objects in videos and generates the detection in JSON files and optionally outputs the videos .
This script can only be run after obtaining the relevancy lists via relevancy.py.
"""

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import cv2
import tqdm
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from unidet.config import add_unidet_config
from unidet.predictor import UnifiedVisualizationDemo
from utils.config import Config
from utils.file_utils import *


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


script_config = "../intercutmix/config.json"
conf = Config(script_config)
output_video_dir = Path(conf.unidet.detect.output.video.path)
output_json_dir = Path(conf.unidet.detect.output.json)

assert_file(conf.unidet.detect.config, "Configuration", ".yaml")
assert_file(conf.unidet.detect.checkpoint, "Checkpoint", ".pth")

if conf.unidet.detect.output.video:
    dataset_path = Path(conf.ucf101.path)
    assert_dir(dataset_path, "Dataset path")

mp.set_start_method("spawn", force=True)

args = argparse.ArgumentParser()
args.config_file = conf.unidet.detect.config
args.confidence_threshold = conf.unidet.detect.confidence
args.parallel = conf.unidet.detect.parallel
args.opts = ["MODEL.WEIGHTS", conf.unidet.detect.checkpoint]

setup_logger(name="fvcore")
logger = setup_logger()
logger.info("Arguments: " + str(args))

cfg = setup_cfg(args)
demo = UnifiedVisualizationDemo(cfg, parallel=conf.unidet.detect.parallel)
n_videos = count_files(dataset_path, ext=conf.ucf101.ext)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

with tqdm(total=n_videos) as bar:
    for action in dataset_path.iterdir():
        for file in action.iterdir():
            bar.set_description(file.name)

            input_video = cv2.VideoCapture(str(file))
            width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(input_video.get(cv2.CAP_PROP_FPS))
            n_frames = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))
            detection_data = {}
            gen = demo.run_on_video(input_video)

            if conf.unidet.detect.output.video:
                output_video_path = (
                    output_video_dir / action.name / file.with_suffix(".mp4").name
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

                if conf.unidet.detect.output.video:
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

            if conf.unidet.detect.output.video:
                video_writer.release()

            output_json_path = (
                output_json_dir / action.name / file.with_suffix(".json").name
            )

            output_json_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_json_path, "w") as json_file:
                json.dump(detection_data, json_file)

            bar.update(1)
