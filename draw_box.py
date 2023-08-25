import json
import pathlib
import random

import click
import cv2
from moviepy.editor import ImageSequenceClip, VideoFileClip
from tqdm import tqdm


@click.command()
@click.argument(
    "dataset-path",
    nargs=1,
    required=True,
    type=click.Path(
        file_okay=False,
        dir_okay=True,
        exists=True,
        readable=True,
        path_type=pathlib.Path,
    ),
)
@click.argument(
    "bbox-json-path",
    nargs=1,
    required=True,
    type=click.Path(
        file_okay=False,
        dir_okay=True,
        exists=True,
        readable=True,
        path_type=pathlib.Path,
    ),
)
@click.argument(
    "output-path",
    nargs=1,
    required=True,
    type=click.Path(
        file_okay=False,
        dir_okay=True,
        exists=True,
        readable=True,
        path_type=pathlib.Path,
    ),
)
@click.argument(
    "relevant-object-json",
    nargs=1,
    required=True,
    type=click.Path(
        file_okay=True,
        dir_okay=False,
        exists=True,
        readable=True,
        path_type=pathlib.Path,
    ),
)
@click.option("--threshold", nargs=1, required=False, type=float, default=0.2)
def main(dataset_path, bbox_json_path, output_path, relevant_object_json, threshold):
    n_files = sum(1 for f in dataset_path.glob("**/*") if f.is_file())

    with open("datasets/label_spaces/learned_mAP.json", "r") as f:
        unified_label_file = json.load(f)

    with open(relevant_object_json, "r") as f:
        relevant_ids = json.load(f)

    thing_classes = [
        "{}".format([xx for xx in x["name"].split("_") if xx != ""][0])
        for x in unified_label_file["categories"]
    ]

    colors = [
        (random.randint(0, 200), random.randint(0, 200), random.randint(0, 200))
        for _ in range(len(thing_classes))
    ]

    common_obj = "Person", "Man", "Woman"
    common_ids = [thing_classes.index(i) for i in thing_classes if i in common_obj]

    with tqdm(total=n_files) as bar:
        for action in dataset_path.iterdir():
            relevant_objects = [*relevant_ids[action.name], *common_ids]

            for file in action.iterdir():
                bar.set_description(file.name)

                video = VideoFileClip(str(file))
                output_frames = []
                output_video_path = (
                    output_path / action.name / file.with_suffix(".mp4").name
                )

                output_video_path.parent.mkdir(parents=True, exist_ok=True)
                json_file = (
                    bbox_json_path / action.name / file.with_suffix(".json").name
                )

                if not json_file.exists():
                    print("JSON file not found:", json_file.name)
                    continue

                with open(json_file, "r") as f:
                    box_data = json.load(f)

                for i, frame in enumerate(video.iter_frames()):
                    if str(i) not in box_data.keys():
                        continue

                    # output_frame = np.zeros(frame.shape)

                    for box, confidence, class_id in box_data[str(i)]:
                        if confidence < threshold or class_id not in relevant_objects:
                            continue

                        x1, y1, x2, y2 = [round(i) for i in box]
                        text = f"{thing_classes[class_id]} {confidence:.02}"
                        font = cv2.FONT_HERSHEY_PLAIN
                        font_scale = 1.2
                        text_thickness = 1
                        text_size = cv2.getTextSize(
                            text, cv2.FONT_HERSHEY_DUPLEX, font_scale, text_thickness
                        )[0]
                        text_width, text_height = text_size[:2]
                        text_x = x1
                        text_y = y1
                        box_thickness = 2

                        cv2.rectangle(
                            frame, (x1, y1), (x2, y2), colors[class_id], box_thickness
                        )
                        cv2.rectangle(
                            frame,
                            (text_x - 1, text_y - int(text_height * 0.9)),
                            (text_x + int(text_width * 0.55), text_y),
                            colors[class_id],
                            cv2.FILLED,
                        )
                        cv2.putText(
                            frame,
                            text,
                            (x1 + 3, y1 - 5),
                            font,
                            font_scale,
                            (255, 255, 255),
                            text_thickness,
                        )

                        # output_frame[y1:y2, x1:x2] = frame[y1:y2, x1:x2]

                    output_frames.append(frame)

                ImageSequenceClip(output_frames, fps=video.fps).write_videofile(
                    str(output_video_path), audio=False, logger=None
                )
                video.close()
                bar.update(1)


if __name__ == "__main__":
    main()
