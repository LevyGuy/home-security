import cv2
import datetime
import numpy
from time import sleep

MOTION_RECORD_TIME = datetime.timedelta(seconds=20)
TIMEOUT_START = 10


def have_motion(frame1, frame2):
    if frame1 is None or frame2 is None:
        return False
    delta = cv2.absdiff(frame1, frame2)
    thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
    return numpy.sum(thresh) > 0


def main():
    cap = cv2.VideoCapture(0)
    frame_size = (int(cap.get(3)), int(cap.get(4)))
    # https://stackoverflow.com/a/60757696
    fourcc = cv2.VideoWriter_fourcc(*'X264')

    prev_frame = None
    last_motion = None
    motion_file = None

    sleep(TIMEOUT_START)

    while cap.isOpened():
        now = datetime.datetime.now()
        success, frame = cap.read()
        assert success, "failed reading frame"

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.GaussianBlur(frame_gray, (21, 21), 0)

        if have_motion(prev_frame, frame_gray):
            if motion_file is None:
                motion_filename = now.strftime("%H-%M-%S.mp4")
                motion_file = cv2.VideoWriter(motion_filename, fourcc, 20.0, frame_size)
            last_motion = now
            print("Motion!", last_motion)

        if motion_file is not None:
            motion_file.write(frame)
            if now - last_motion > MOTION_RECORD_TIME:
                motion_file.release()
                motion_file = None

        prev_frame = frame_gray
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
