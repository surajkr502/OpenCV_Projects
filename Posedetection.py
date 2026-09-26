import cv2
import sys
from pathlib import Path
from typing import cast

from ultralytics import YOLO
from ultralytics.engine.results import Results


MODEL_PATH = "yolov8n-pose.pt"
CAMERA_INDEX = 0
CONF_THRESHOLD = 0.5


KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


def main() -> None:
    model_path = Path(MODEL_PATH)

    if not model_path.exists():
        print(f'Error: model file not found at "{model_path}"')
        sys.exit(1)

    model = YOLO(str(model_path))

    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("Error: Cannot open webcam")
        sys.exit(1)

    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("Failed to capture frame")
                break

            frame_count += 1

            results = model(
                frame,
                conf=CONF_THRESHOLD,
                verbose=False,
            )

            result = cast(Results, next(iter(results)))

            annotated_frame = result.plot()

            keypoints = result.keypoints

            if keypoints is not None and keypoints.xy is not None:

                for person_idx, person_kpts in enumerate(keypoints.xy):

                    if len(person_kpts) > 0:

                        nose_x, nose_y = person_kpts[0]

                        if nose_x > 0 and nose_y > 0:

                            print(
                                f"Person {person_idx}: "
                                f"nose at "
                                f"({nose_x:.0f}, {nose_y:.0f})"
                            )

            cv2.imshow(
                "YOLO Pose Detection",
                annotated_frame,
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()

        cv2.destroyAllWindows()

        print(
            f"Done. Processed {frame_count} frames."
        )


if __name__ == "__main__":
    main()