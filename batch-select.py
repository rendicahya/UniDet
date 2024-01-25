import json
import pickle
import random
from pathlib import Path

import cv2
import numpy as np
from assertpy.assertpy import assert_that
from python_config import Config
from python_file import count_files
from python_video import frames_to_video, video_frames
from tqdm import tqdm

conf = Config("../config.json")
dataset_dir = Path.cwd().parent / conf.unidet.select.dataset.path
unidet_json_dir = Path.cwd().parent / conf.unidet.select.json
relevant_object_json = Path.cwd().parent / conf.relevancy.json
confidence_thres = conf.unidet.select.confidence
unified_label = "datasets/label_spaces/learned_mAP.json"
common_obj = conf.unidet.select.common_objects

assert_that(conf.unidet.select.mode).is_in("actorcutmix", "intercutmix")
assert_that(dataset_dir).is_directory().is_readable()
assert_that(unidet_json_dir).is_directory().is_readable()
assert_that(relevant_object_json).is_file().is_readable()
assert_that(unified_label).is_file().is_readable()
assert_that(common_obj).is_type_of(list)
print("Mode:", conf.unidet.select.mode)

n_files = count_files(dataset_dir, ext=conf.ucf101.ext)

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

common_ids = [thing_classes.index(i) for i in common_obj]
bar = tqdm(total=n_files)
font, font_size, font_weight = cv2.FONT_HERSHEY_PLAIN, 1.2, 1

for action in dataset_dir.iterdir():
    if conf.unidet.select.mode == "actorcutmix":
        target_obj = common_ids
    elif conf.unidet.select.mode == "intercutmix":
        target_obj = [*relevant_ids[action.name], *common_ids]

    for file in action.iterdir():
        if file.suffix != conf.unidet.select.dataset.ext:
            continue

        bar.set_description(file.name)

        input_frames = video_frames(file, reader=conf.unidet.select.video_reader)

        if conf.unidet.select.output.video.generate:
            output_frames = []

        if conf.unidet.select.output.repp.enabled:
            video_dets = {}

        json_path = unidet_json_dir / action.name / file.with_suffix(".json").name

        if not json_path.exists():
            print("JSON file not found:", json_path.name)
            continue

        with open(json_path, "r") as f:
            box_data = json.load(f)

        for i, frame in enumerate(input_frames):
            if str(i) not in box_data.keys():
                continue

            # Robust-and-efficient-post-processing-for-video-object-detection (REPP)
            if conf.unidet.select.output.repp.enabled:
                frame_dets = []
                ih, iw = frame.shape[:-1]
                width_diff = max(0, (ih - iw) // 2)
                height_diff = max(0, (iw - ih) // 2)

                for box, confidence, class_id in box_data[str(i)]:
                    if confidence < confidence_thres or class_id not in target_obj:
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
                        "image_id": f"{i:06}",
                        "bbox": [x_min, y_min, width, height],
                        "scores": [confidence],
                        "bbox_center": bbox_center,
                    }

                    frame_dets.append(det_box)

                video_dets[f"{i:06}"] = frame_dets

            if conf.unidet.select.output.mask.generate:
                mask_ext = conf.unidet.select.output.mask.ext
                output_mask_dir = Path(conf.unidet.select.output.mask.path)
                mask = np.zeros(frame.shape)
                output_mask_path = (
                    output_mask_dir / action.name / file.stem / (f"%05d{mask_ext}" % i)
                )

                output_mask_path.parent.mkdir(exist_ok=True, parents=True)

                for box, confidence, class_id in box_data[str(i)]:
                    if confidence < confidence_thres or class_id not in target_obj:
                        continue

                    x1, y1, x2, y2 = [round(i) for i in box]
                    mask[y1:y2, x1:x2] = 255

                cv2.imwrite(str(output_mask_path), mask)

            if conf.unidet.select.output.video.generate:
                for box, confidence, class_id in box_data[str(i)]:
                    if confidence < confidence_thres or class_id not in target_obj:
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

                output_frames.append(frame)

        if conf.unidet.select.output.video.generate:
            output_video_dir = Path.cwd().parent / conf.unidet.select.output.video.path
            output_video_path = (
                output_video_dir / action.name / file.with_suffix(".mp4").name
            )

            output_video_path.parent.mkdir(parents=True, exist_ok=True)

            frames_to_video(
                output_frames,
                output_video_path,
                writer=conf.unidet.select.output.video.writer,
            )

        if conf.unidet.select.output.repp.enabled:
            output_repp_dir = Path.cwd().parent / conf.unidet.select.output.repp.path
            output_repp_path = (
                output_repp_dir / action.name / file.with_suffix(".pckl").name
            )

            output_repp_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_repp_path, "wb") as f:
                pickle.dump((file.name, video_dets), f)

        bar.update(1)

bar.close()
