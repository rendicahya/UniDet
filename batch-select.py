import sys

sys.path.append(".")

import json
import pickle
import random
from pathlib import Path

import click
import cv2
import mmcv
import numpy as np
from tqdm import tqdm

from assertpy.assertpy import assert_that
from config import settings as conf
from python_video import frames_to_video, video_frames

ROOT = Path.cwd()
DATASET = conf.active.dataset
METHOD = conf.active.mode
USE_REPP = conf.active.use_REPP
DETECTOR = conf.active.detector
RELEV_MODEL = conf.active.relevancy.method
RELEV_THRESH = str(conf.active.relevancy.threshold)
DET_CONF = str(conf.unidet.detect.confidence)
VIDEO_IN_DIR = ROOT / conf[DATASET].path
UNIDET_JSON_DIR = ROOT / "data" / DATASET / DETECTOR / "detect" / DET_CONF / "json"
RELEV_OBJECT_JSON = (
    ROOT / f"data/relevancy/{DETECTOR}/{DATASET}/ids/{RELEV_MODEL}/{RELEV_THRESH}.json"
)

CONFIDENCE_THRESH = conf.unidet.select.confidence
GENERATE_VIDEO = conf.unidet.select.output.video
ENABLE_DUMP = conf.unidet.select.output.dump
GENERATE_MASK = conf.unidet.select.output.mask
unified_label = "UniDet/datasets/label_spaces/learned_mAP.json"
mid_dir = ROOT / "data" / DATASET / DETECTOR / str(CONFIDENCE_THRESH) / METHOD

if METHOD in ("allcutmix", "actorcutmix"):
    dump_out_dir = mid_dir / "dump"
    mask_out_dir = mid_dir / "mask"
    video_out_dir = mid_dir / "videos"
else:
    dump_out_dir = mid_dir / "dump" / RELEV_MODEL / RELEV_THRESH
    mask_out_dir = mid_dir / "mask" / RELEV_MODEL / RELEV_THRESH
    video_out_dir = mid_dir / "videos" / RELEV_MODEL / RELEV_THRESH

print("Input:", UNIDET_JSON_DIR.relative_to(ROOT))
print(f"Dump output: {dump_out_dir.relative_to(ROOT)} ({ENABLE_DUMP})")
print(f"Mask output: {mask_out_dir.relative_to(ROOT)} ({GENERATE_MASK})")
print(f"Video output: {video_out_dir.relative_to(ROOT)} ({GENERATE_VIDEO})")

assert_that(METHOD).is_in("actorcutmix", "allcutmix", "intercutmix")
assert_that(VIDEO_IN_DIR).is_directory().is_readable()
assert_that(UNIDET_JSON_DIR).is_directory().is_readable()
assert_that(RELEV_OBJECT_JSON).is_file().is_readable()
assert_that(unified_label).is_file().is_readable()

if not click.confirm("\nDo you want to continue?", show_default=True):
    exit("Aborted.")

with open(unified_label, "r") as f:
    unified_label_file = json.load(f)

thing_classes = [
    "{}".format([xx for xx in x["name"].split("_") if xx != ""][0])
    for x in unified_label_file["categories"]
]

with open(RELEV_OBJECT_JSON, "r") as f:
    relevant_ids = json.load(f)

colors = [
    (random.randint(0, 200), random.randint(0, 200), random.randint(0, 200))
    for _ in range(len(thing_classes))
]

common_obj = conf.unidet.select.common_objects
common_ids = [thing_classes.index(i) for i in common_obj]
n_files = conf[DATASET].n_videos
bar = tqdm(total=n_files, dynamic_ncols=True)
font, font_size, font_weight = cv2.FONT_HERSHEY_PLAIN, 1.2, 1

for action in UNIDET_JSON_DIR.iterdir():
    if METHOD == "actorcutmix":
        target_obj = common_ids
    elif METHOD == "intercutmix":
        target_obj = [*relevant_ids[action.name.replace('_', ' ')], *common_ids]

    for file in action.iterdir():
        video_path = (
            VIDEO_IN_DIR / action.name / file.with_suffix(conf[DATASET].ext).name
        )
        vid_info = mmcv.VideoReader(str(video_path))
        iw, ih = vid_info.resolution

        if GENERATE_VIDEO:
            in_frames = video_frames(video_path, reader=conf.active.video.reader)
            out_frames = []

        if ENABLE_DUMP:
            video_dets = {}

        if GENERATE_MASK:
            n_frames = vid_info.frame_cnt
            mask_cube = np.zeros((n_frames, ih, iw), np.uint8)

        with open(file, "r") as f:
            json_data = json.load(f)

        for i, boxes in json_data.items():
            if ENABLE_DUMP:
                frame_dets = []
                width_diff = max(0, (ih - iw) // 2)
                height_diff = max(0, (iw - ih) // 2)
                image_id = "%06d" % int(i)

                for box, confidence, class_id in boxes:
                    if METHOD in ("actorcutmix", "intercutmix") and (
                        confidence < CONFIDENCE_THRESH or class_id not in target_obj
                    ):
                        continue

                    x1, y1, x2, y2 = box
                    y_min, x_min, y_max, x_max = y1, x1, y2, x2
                    y_min, x_min = max(0, y_min), max(0, x_min)
                    y_max, x_max = min(ih, y_max), min(iw, x_max)
                    width, height = x_max - x_min, y_max - y_min

                    if width <= 0 or height <= 0:
                        continue

                    bbox_center = (
                        (x_min + width_diff + width / 2) / max(iw, ih),
                        (y_min + height_diff + height / 2) / max(iw, ih),
                    )

                    det_box = {
                        "image_id": image_id,
                        "bbox": [x_min, y_min, width, height],
                        "scores": [confidence],
                        "bbox_center": bbox_center,
                    }

                    frame_dets.append(det_box)

                video_dets[image_id] = frame_dets

            if GENERATE_MASK:
                for box, confidence, class_id in boxes:
                    if METHOD in ("actorcutmix", "intercutmix") and (
                        confidence < CONFIDENCE_THRESH or class_id not in target_obj
                    ):
                        continue

                    if int(i) < n_frames:
                        x1, y1, x2, y2 = [round(b) for b in box]
                        mask_cube[int(i), y1:y2, x1:x2] = 255

            if GENERATE_VIDEO:
                frame = next(in_frames)

                for box, confidence, class_id in boxes:
                    if METHOD in ("actorcutmix", "intercutmix") and (
                        confidence < CONFIDENCE_THRESH or class_id not in target_obj
                    ):
                        continue

                    if class_id > len(thing_classes) - 1:
                        continue

                    x1, y1, x2, y2 = [round(i) for i in box]
                    text = f"{thing_classes[class_id]} {confidence:.02}"
                    text_size = cv2.getTextSize(text, font, font_size, font_weight)[0]
                    text_width, text_height = text_size[:2]
                    box_thickness = 2

                    cv2.rectangle(
                        frame, (x1, y1), (x2, y2), colors[class_id], box_thickness
                    )

                    cv2.rectangle(
                        frame,
                        (x1 - 1, y1 - int(text_height * 2)),
                        (x1 + int(text_width * 1.1), y1),
                        colors[class_id],
                        cv2.FILLED,
                    )

                    cv2.putText(
                        frame,
                        text,
                        (x1 + 3, y1 - 5),
                        font,
                        font_size,
                        (255, 255, 255),
                        font_weight,
                    )

                out_frames.append(frame)

        if ENABLE_DUMP:
            out_dump_dir = dump_out_dir / action.name / file.with_suffix(".pckl").name

            out_dump_dir.parent.mkdir(parents=True, exist_ok=True)
            with open(out_dump_dir, "wb") as f:
                pickle.dump((file.name, video_dets), f)

        if GENERATE_MASK:
            out_mask_path = mask_out_dir / action.name / file.stem

            out_mask_path.parent.mkdir(exist_ok=True, parents=True)
            np.savez_compressed(out_mask_path, mask_cube)

        if GENERATE_VIDEO:
            out_video_path = video_out_dir / action.name / file.with_suffix(".mp4").name

            out_video_path.parent.mkdir(parents=True, exist_ok=True)
            frames_to_video(
                out_frames,
                out_video_path,
                writer=conf.active.video.writer,
            )

        bar.update(1)

bar.close()
