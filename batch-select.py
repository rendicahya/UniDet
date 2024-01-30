import json
import pickle
import random
from pathlib import Path

import cv2
import numpy as np
from assertpy.assertpy import assert_that
from python_config import Config
from python_file import count_files
from python_video import frames_to_video, video_frames, video_info
from tqdm import tqdm

conf = Config("../config.json")
project_root = Path.cwd().parent
video_root = project_root / conf.unidet.select.video.path
unidet_json_root = project_root / conf.unidet.select.json
relevant_object_json = project_root / conf.relevancy.json
confidence_thres = conf.unidet.select.confidence
generate_video = conf.unidet.select.output.video.generate
generate_mask = conf.unidet.select.output.mask.generate
bundle_mask = conf.unidet.select.output.mask.bundle
enable_dump = conf.unidet.select.output.dump.enabled
out_mask_dir = Path.cwd().parent / conf.unidet.select.output.mask.path
unified_label = "datasets/label_spaces/learned_mAP.json"
common_obj = conf.unidet.select.common_objects

assert_that(conf.unidet.select.mode).is_in("actorcutmix", "intercutmix")
assert_that(video_root).is_directory().is_readable()
assert_that(unidet_json_root).is_directory().is_readable()
assert_that(relevant_object_json).is_file().is_readable()
assert_that(unified_label).is_file().is_readable()
assert_that(common_obj).is_type_of(list)

n_files = count_files(video_root, ext=conf.ucf101.ext)

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

for action in unidet_json_root.iterdir():
    if conf.unidet.select.mode == "actorcutmix":
        target_obj = common_ids
    elif conf.unidet.select.mode == "intercutmix":
        target_obj = [*relevant_ids[action.name], *common_ids]

    for file in action.iterdir():
        bar.set_description(file.stem)

        video_path = (
            video_root
            / action.name
            / file.with_suffix(conf.unidet.select.video.ext).name
        )
        vid_info = video_info(video_path)
        iw, ih = vid_info["width"], vid_info["height"]

        if generate_video:
            in_frames = video_frames(video_path, reader=conf.unidet.select.video.reader)
            out_frames = []

        if conf.unidet.select.output.dump.enabled:
            video_dets = {}

        if generate_mask and bundle_mask:
            n_frames = vid_info["n_frames"]
            mask_bundle = np.zeros((n_frames, ih, iw), np.uint8)
            out_mask_path = out_mask_dir / action.name / file.stem

        with open(file, "r") as f:
            json_data = json.load(f)

        for i, boxes in json_data.items():
            if enable_dump:
                frame_dets = []
                width_diff = max(0, (ih - iw) // 2)
                height_diff = max(0, (iw - ih) // 2)
                image_id = "%06d" % int(i)

                for box, confidence, class_id in boxes:
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
                        "image_id": image_id,
                        "bbox": [x_min, y_min, width, height],
                        "scores": [confidence],
                        "bbox_center": bbox_center,
                    }

                    frame_dets.append(det_box)

                video_dets[image_id] = frame_dets

            if generate_mask:
                if not bundle_mask:
                    mask = np.zeros((ih, iw), np.uint8)
                    out_mask_path = (
                        out_mask_dir / action.name / file.stem / ("%05d.png" % int(i))
                    )

                out_mask_path.parent.mkdir(exist_ok=True, parents=True)

                for box, confidence, class_id in boxes:
                    if confidence < confidence_thres or class_id not in target_obj:
                        continue

                    x1, y1, x2, y2 = [round(b) for b in box]

                    if bundle_mask:
                        mask_bundle[int(i), y1:y2, x1:x2] = 255
                    else:
                        mask[y1:y2, x1:x2] = 255

                if not bundle_mask:
                    cv2.imwrite(str(out_mask_path), mask)

            if generate_video:
                frame = next(in_frames)

                for box, confidence, class_id in boxes:
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

                out_frames.append(frame)

        if generate_video:
            out_video_dir = Path.cwd().parent / conf.unidet.select.output.video.path
            out_video_path = out_video_dir / action.name / file.with_suffix(".mp4").name

            out_video_path.parent.mkdir(parents=True, exist_ok=True)

            frames_to_video(
                out_frames,
                out_video_path,
                writer=conf.unidet.select.output.video.writer,
            )

        if generate_mask and bundle_mask:
            np.savez_compressed(out_mask_path, mask_bundle)

        if enable_dump:
            out_dump_dir = Path.cwd().parent / conf.unidet.select.output.dump.path
            out_dump_dir = out_dump_dir / action.name / file.with_suffix(".pckl").name

            out_dump_dir.parent.mkdir(parents=True, exist_ok=True)

            with open(out_dump_dir, "wb") as f:
                pickle.dump((file.name, video_dets), f)

        bar.update(1)

bar.close()
