import sys

sys.path.append(".")

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import click
import cv2
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from tqdm import tqdm
from unidet.config import add_unidet_config
from unidet.predictor import UnifiedVisualizationDemo

from assertpy.assertpy import assert_that
from config import settings as conf
from python_video import frames_to_video


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


ROOT = Path.cwd()
DATASET = conf.active.dataset
DETECTOR = conf.active.detector
VIDEO_IN_DIR = ROOT / conf[DATASET].path
GENERATE_VIDEOS = conf.unidet.detect.generate_videos
DET_CONF = conf.unidet.detect.confidence
VIDEO_OUT_DIR = ROOT / f"data/{DATASET}/{DETECTOR}/detect/{DET_CONF}/videos"
JSON_OUT_DIR = ROOT / f"data/{DATASET}/{DETECTOR}/detect/{DET_CONF}/json"
VIDEO_EXT = conf[DATASET].ext

unidet_dir = ROOT / "UniDet"
unidet_config = unidet_dir / conf.unidet.detect.config
unidet_checkpoint = unidet_dir / conf.unidet.detect.checkpoint

assert_that(VIDEO_IN_DIR).is_directory().is_readable()
assert_that(unidet_config).is_file().is_readable()
assert_that(unidet_checkpoint).is_file().is_readable()

mp.set_start_method("spawn", force=True)

args = argparse.ArgumentParser()
args.config_file = unidet_config
args.confidence_threshold = DET_CONF
args.parallel = conf.unidet.detect.parallel
args.opts = ["MODEL.WEIGHTS", str(unidet_checkpoint)]

setup_logger(name="fvcore")
logger = setup_logger()

print("Input:", VIDEO_IN_DIR.relative_to(ROOT))
print("Output JSON:", JSON_OUT_DIR.relative_to(ROOT))

if GENERATE_VIDEOS:
    print("Output video:", VIDEO_OUT_DIR.relative_to(ROOT))

if not click.confirm("\nDo you want to continue?", show_default=True):
    exit("Aborted.")

cfg = setup_cfg(args)
demo = UnifiedVisualizationDemo(cfg, parallel=conf.unidet.detect.parallel)
n_videos = conf[DATASET].n_videos
bar = tqdm(total=n_videos, dynamic_ncols=True)

for file in VIDEO_IN_DIR.glob(f"**/*{VIDEO_EXT}"):
    action = file.parent.name
    json_out_path = JSON_OUT_DIR / action / file.with_suffix(".json").name

    if json_out_path.exists() and json_out_path.stat().st_size:
        bar.update(1)
        continue

    video_in = cv2.VideoCapture(str(file))
    n_frames = int(video_in.get(cv2.CAP_PROP_FRAME_COUNT))
    gen = demo.run_on_video(video_in)
    detection_data = {}
    out_frames = []

    for i, (viz, pred) in enumerate(gen):
        bar.set_description(f"({i}/{n_frames})")

        if GENERATE_VIDEOS:
            rgb = cv2.cvtColor(viz, cv2.COLOR_BGR2RGB)
            out_frames.append(rgb)

        detection_data.update(
            {
                i: [
                    (
                        [round(i) for i in pred_box.tolist()],
                        round(score.tolist(), 3),
                        pred_class.tolist(),
                    )
                    for pred_box, score, pred_class in zip(
                        pred.pred_boxes.tensor,
                        pred.scores,
                        pred.pred_classes,
                    )
                ]
            }
        )

    video_in.release()
    json_out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(json_out_path, "w") as json_file:
        json.dump(detection_data, json_file)

    if GENERATE_VIDEOS:
        video_out_path = VIDEO_OUT_DIR / action / file.with_suffix(".mp4").name

        video_out_path.parent.mkdir(parents=True, exist_ok=True)
        frames_to_video(out_frames, video_out_path, conf.active.video.writer)

    bar.update(1)

bar.close()
