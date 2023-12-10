"""
This script detects objects in videos and stores the detection results in JSON files and optionally outputs bounding box-annotated videos.
This script can only be run after obtaining the relevancy lists via relevancy.py.
This script uses GPU and takes several seconds to process a short video.
"""

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import cv2
from assertpy.assertpy import assert_that
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from python_config import Config
from python_file import count_files
from python_video import frames_to_video, video_info, video_writer_like
from tqdm import tqdm
from unidet.config import add_unidet_config
from unidet.predictor import UnifiedVisualizationDemo


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


conf = Config("../config.json")

dataset_path = Path(conf.unidet.detect.dataset.path)
output_video_dir = Path(conf.unidet.detect.output.video.path)
output_json_dir = Path(conf.unidet.detect.output.json)

assert_that(dataset_path).is_directory().is_readable()
assert_that(conf.unidet.detect.config).is_file().is_readable()
assert_that(conf.unidet.detect.checkpoint).is_file().is_readable()

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
n_videos = count_files(dataset_path, ext=conf.unidet.detect.dataset.ext)
bar = tqdm(total=n_videos)

for action in dataset_path.iterdir():
    for file in action.iterdir():
        bar.set_description(file.name)

        input_video = cv2.VideoCapture(str(file))
        info = video_info(file)
        n_frames = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))
        detection_data = {}
        output_frames = []
        gen = demo.run_on_video(input_video)

        for i, (viz, pred) in enumerate(gen):
            bar.set_description(f"{file.name} ({i}/{n_frames})")

            if conf.unidet.detect.output.video.generate:
                output_frames.append(viz)

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

        output_json_path = (
            output_json_dir / action.name / file.with_suffix(".json").name
        )

        output_json_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_json_path, "w") as json_file:
            json.dump(detection_data, json_file)

        if conf.unidet.detect.output.video.generate:
            output_video_path = (
                output_video_dir / action.name / file.with_suffix(".mp4").name
            )

            output_video_path.parent.mkdir(parents=True, exist_ok=True)
            frames_to_video(
                output_frames, output_video_path, conf.unidet.detect.output.video.writer
            )

        bar.update(1)

bar.close()
