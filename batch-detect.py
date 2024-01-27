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
project_root = Path.cwd().parent
video_root = project_root / conf.unidet.detect.video.path
video_out_dir = project_root / conf.unidet.detect.output.video.path
json_out_dir = project_root / conf.unidet.detect.output.json
video_ext = conf.unidet.detect.video.ext

assert_that(video_root).is_directory().is_readable()
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
n_videos = count_files(video_root, ext=video_ext)
bar = tqdm(total=n_videos)

for file in video_root.glob(f"**/*.{video_ext}"):
    bar.set_description(file.name)

    action = file.parent.name
    video_in = cv2.VideoCapture(str(file))
    info = video_info(file)
    n_frames = int(video_in.get(cv2.CAP_PROP_FRAME_COUNT))
    detection_data = {}
    out_frames = []
    gen = demo.run_on_video(video_in)

    for i, (viz, pred) in enumerate(gen):
        bar.set_description(f"{file.name} ({i}/{n_frames})")

        if conf.unidet.detect.output.video.generate:
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

    if conf.unidet.detect.output.video.generate:
        video_out_path = video_out_dir / action / file.with_suffix(".mp4").name

        video_out_path.parent.mkdir(parents=True, exist_ok=True)
        frames_to_video(
            out_frames, video_out_path, conf.unidet.detect.output.video.writer
        )

    bar.update(1)

bar.close()
