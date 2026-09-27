import cv2
import sys
from pathlib import Path


CAMERA_INDEX = 0
CONFIDENCE_SCALE_FACTOR = 1.1
MIN_NEIGHBORS = 5
MIN_FACE_SIZE = (30, 30)

CASCADE_FILE = "haarcascade_frontalface_default.xml"

WINDOW_NAME = "Face Detection - OpenCV"


def load_face_detector() -> cv2.CascadeClassifier:
    """
    Load the Haar Cascade face detection model.
    """

    project_dir = Path(__file__).resolve().parent

    cascade_path = project_dir / CASCADE_FILE

    if not cascade_path.exists():
        print(f'Error: Cascade file not found at "{cascade_path}"')
        sys.exit(1)

    face_cascade = cv2.CascadeClassifier(str(cascade_path))

    if face_cascade.empty():
        print(f'Error: Could not load cascade file "{cascade_path}"')
        sys.exit(1)

    print(f"Face detector loaded successfully:")
    print(cascade_path)

    return face_cascade


def detect_faces(
    frame,
    face_cascade: cv2.CascadeClassifier
):
    """
    Detect faces in a single video frame.
    """

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=CONFIDENCE_SCALE_FACTOR,
        minNeighbors=MIN_NEIGHBORS,
        minSize=MIN_FACE_SIZE
    )

    return faces



def draw_faces(frame, faces):
    """
    Draw rectangles and labels around detected faces.
    """

    for face_number, (x, y, w, h) in enumerate(faces, start=1):

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        label = f"Face {face_number}"

        cv2.putText(
            frame,
            label,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    return frame



def main() -> None:



    face_cascade = load_face_detector()

  

    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("Error: Cannot open webcam")
        sys.exit(1)

    print("Webcam started successfully.")
    print("Press 'q' to quit.")



    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL
    )

    cv2.setWindowProperty(
        WINDOW_NAME,
        cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_FULLSCREEN
    )



    frame_count = 0
    total_faces_detected = 0


    try:

        while True:

            ret, frame = cap.read()

            if not ret:
                print("Error: Failed to capture frame")
                break

            frame_count += 1


            faces = detect_faces(
                frame,
                face_cascade
            )

            face_count = len(faces)

            total_faces_detected += face_count


            frame = draw_faces(
                frame,
                faces
            )



            cv2.putText(
                frame,
                f"Faces: {face_count}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )


            cv2.imshow(
                WINDOW_NAME,
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Stopped by user.")
                break

    finally:


        cap.release()

        cv2.destroyAllWindows()

        print()
        print("=" * 50)
        print("FACE DETECTION COMPLETED")
        print("=" * 50)
        print(f"Frames processed        : {frame_count}")
        print(f"Total face detections   : {total_faces_detected}")
        print("=" * 50)


if __name__ == "__main__":
    main()




#      PS C:\Users\sawsu\OneDrive\Desktop\OpenCV_Projects> Invoke-WebRequest `
# >>     -Uri "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml" `
# >>     -OutFile ".\haarcascade_frontalface_default.xml"
# >> 