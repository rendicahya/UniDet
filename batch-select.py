import sys

sys.path.append(".")

import json
import pickle
import random
from pathlib import Path

import click
import cv2
import numpy as np
from assertpy.assertpy import assert_that
from config import settings as conf
from python_file import count_files
from python_video import frames_to_video, video_frames, video_info
from tqdm import tqdm

root = Path.cwd()
dataset = conf.active.dataset
method = conf.active.mode
detector = conf.active.detector
relevancy_model = conf.active.relevancy.method
relevancy_thresh = str(conf.active.relevancy.threshold)
object_conf = str(conf.unidet.detect.confidence)
video_in_dir = root / conf[dataset].path
unidet_json_dir = root / f"data/{dataset}/{detector}/detect/{object_conf}/json"
relevant_object_json = (
    root
    / f"data/relevancy/{detector}/{dataset}/ids/{relevancy_model}/{relevancy_thresh}.json"
)

object_selection = conf.active.object_selection
confidence_thres = conf.unidet.select.confidence
generate_video = conf.unidet.select.output.video
enable_dump = conf.unidet.select.output.dump
generate_mask = conf.unidet.select.output.mask
unified_label = "UniDet/datasets/label_spaces/learned_mAP.json"

mode = "select" if object_selection else "detect"
mode_dir = root / "data" / dataset / detector / mode

if mode == "detect":
    dump_out_dir = mode_dir / "dump"
    mask_out_dir = mode_dir / "mask"
    video_out_dir = mode_dir / "videos"
elif mode == "select":
    dump_out_dir = mode_dir / method / object_conf / "dump"
    mask_out_dir = mode_dir / method / object_conf / "mask"
    video_out_dir = mode_dir / method / object_conf / "videos"

    if method == "intercutmix":
        dump_out_dir = dump_out_dir / relevancy_model / relevancy_thresh
        mask_out_dir = mask_out_dir / relevancy_model / relevancy_thresh
        video_out_dir = video_out_dir / relevancy_model / relevancy_thresh

print("Input:", unidet_json_dir.relative_to(root))
print(f"Dump output: {dump_out_dir.relative_to(root)} ({enable_dump})")
print(f"Mask output: {mask_out_dir.relative_to(root)} ({generate_mask})")
print(f"Video output: {video_out_dir.relative_to(root)} ({generate_video})")

assert_that(method).is_in("actorcutmix", "intercutmix")
assert_that(video_in_dir).is_directory().is_readable()
assert_that(unidet_json_dir).is_directory().is_readable()
assert_that(relevant_object_json).is_file().is_readable()
assert_that(unified_label).is_file().is_readable()

if not click.confirm("\nDo you want to continue?", show_default=True):
    exit("Aborted.")

with open(unified_label, "r") as f:
    unified_label_file = json.load(f)

thing_classes = [
    "{}".format([xx for xx in x["name"].split("_") if xx != ""][0])
    for x in unified_label_file["categories"]
]

with open(relevant_object_json, "r") as f:
    relevant_ids = json.load(f)

colors = [
    (random.randint(0, 200), random.randint(0, 200), random.randint(0, 200))
    for _ in range(len(thing_classes))
]

common_obj = conf.unidet.select.common_objects
common_ids = [thing_classes.index(i) for i in common_obj]
n_files = count_files(video_in_dir, ext=conf[dataset].ext)
bar = tqdm(total=n_files)
font, font_size, font_weight = cv2.FONT_HERSHEY_PLAIN, 1.2, 1

for action in unidet_json_dir.iterdir():
    if method == "actorcutmix":
        target_obj = common_ids
    elif method == "intercutmix":
        target_obj = [*relevant_ids[action.name], *common_ids]

    for file in action.iterdir():
        video_path = (
            video_in_dir / action.name / file.with_suffix(conf[dataset].ext).name
        )
        vid_info = video_info(video_path)
        iw, ih = vid_info["width"], vid_info["height"]

        if generate_video:
            in_frames = video_frames(video_path, reader=conf.active.video.reader)
            out_frames = []

        if enable_dump:
            video_dets = {}

        if generate_mask:
            n_frames = vid_info["n_frames"]
            mask_cube = np.zeros((n_frames, ih, iw), np.uint8)

        with open(file, "r") as f:
            json_data = json.load(f)

        for i, boxes in json_data.items():
            if enable_dump:
                frame_dets = []
                width_diff = max(0, (ih - iw) // 2)
                height_diff = max(0, (iw - ih) // 2)
                image_id = "%06d" % int(i)

                for box, confidence, class_id in boxes:
                    if object_selection and (
                        confidence < confidence_thres or class_id not in target_obj
                    ):
                        continue

                    x1, y1, x2, y2 = [round(i) for i in box]
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

            if generate_mask:
                for box, confidence, class_id in boxes:
                    if object_selection and (
                        confidence < confidence_thres or class_id not in target_obj
                    ):
                        continue

                    if int(i) < n_frames:
                        x1, y1, x2, y2 = [round(b) for b in box]
                        mask_cube[int(i), y1:y2, x1:x2] = 255

            if generate_video:
                frame = next(in_frames)

                for box, confidence, class_id in boxes:
                    if object_selection and (
                        confidence < confidence_thres or class_id not in target_obj
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

        if enable_dump:
            out_dump_dir = dump_out_dir / action.name / file.with_suffix(".pckl").name

            out_dump_dir.parent.mkdir(parents=True, exist_ok=True)
            with open(out_dump_dir, "wb") as f:
                pickle.dump((file.name, video_dets), f)

        if generate_mask:
            out_mask_path = mask_out_dir / action.name / file.stem

            out_mask_path.parent.mkdir(exist_ok=True, parents=True)
            np.savez_compressed(out_mask_path, mask_cube)

        if generate_video:
            out_video_path = video_out_dir / action.name / file.with_suffix(".mp4").name

            out_video_path.parent.mkdir(parents=True, exist_ok=True)
            frames_to_video(
                out_frames,
                out_video_path,
                writer=conf.active.video.writer,
            )

        bar.update(1)

bar.close()
