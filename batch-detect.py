import argparse
import json
import multiprocessing as mp
from pathlib import Path

import cv2
from assertpy.assertpy import assert_that
from config import settings as conf
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from python_file import count_files
from python_video import frames_to_video, video_info
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


root = Path.cwd().parent
video_in_dir = root / conf[conf.active.dataset].path
generate_video = conf.unidet.detect.generate_videos
video_out_dir = (
    root / "data" / conf.active.dataset / conf.active.detector / "detect" / "videos"
)
json_out_dir = (
    root / "data" / conf.active.dataset / conf.active.detector / "detect" / "json"
)
video_ext = conf[conf.active.dataset].ext

assert_that(video_in_dir).is_directory().is_readable()
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
n_videos = count_files(video_in_dir, ext=video_ext)
bar = tqdm(total=n_videos)

for file in video_in_dir.glob(f"**/*{video_ext}"):
    action = file.parent.name
    video_in = cv2.VideoCapture(str(file))
    gen = demo.run_on_video(video_in)
    n_frames = int(video_in.get(cv2.CAP_PROP_FRAME_COUNT))
    detection_data = {}
    out_frames = []

    for i, (viz, pred) in enumerate(gen):
        bar.set_description(f"{file.name[:50].ljust(50)} ({i}/{n_frames})")

        if generate_video:
            rgb = cv2.cvtColor(viz, cv2.COLOR_BGR2RGB)
            out_frames.append(rgb)

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

    json_out_path = json_out_dir / action / file.with_suffix(".json").name

    video_in.release()
    json_out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(json_out_path, "w") as json_file:
        json.dump(detection_data, json_file)

    if generate_video:
        video_out_path = video_out_dir / action / file.with_suffix(".mp4").name

        video_out_path.parent.mkdir(parents=True, exist_ok=True)
        frames_to_video(out_frames, video_out_path, conf.active.video.writer)

    bar.update(1)

bar.close()
