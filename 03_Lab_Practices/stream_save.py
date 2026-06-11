import time
import cv2
from djitellopy import Tello

def main():
    tello = Tello()
    cap = None
    out = None

    try:
        tello.connect()
        print("Battery:", tello.get_battery())

        tello.streamoff()
        time.sleep(1)

        tello.set_video_resolution(Tello.RESOLUTION_480P)
        tello.set_video_fps(Tello.FPS_15)
        tello.set_video_bitrate(Tello.BITRATE_1MBPS)

        tello.streamon()
        time.sleep(3)

        url = "udp://@0.0.0.0:11111?overrun_nonfatal=1&fifo_size=5000000"
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

        if not cap.isOpened():
            raise RuntimeError("Could not open UDP stream.")

        ok, frame = False, None
        start = time.time()
        while time.time() - start < 10:
            ok, frame = cap.read()
            if ok and frame is not None:
                break
            time.sleep(0.03)

        if not ok or frame is None:
            raise RuntimeError("No valid frame received.")

        h, w, _ = frame.shape

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter("tello_stream.mp4", fourcc, 15, (w, h))

        if not out.isOpened():
            raise RuntimeError("Could not open MP4 writer.")

        print("Recording to tello_stream.mp4")
        print("Press q to stop correctly")

        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            cv2.imshow("Tello Stream", frame)
            out.write(frame)

            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

        print("Closing file correctly...")
        time.sleep(1)

    except Exception as e:
        print("ERROR:", e)

    finally:
        if cap is not None:
            cap.release()

        if out is not None:
            out.release()

        cv2.destroyAllWindows()
        time.sleep(1)

        try:
            tello.streamoff()
        except:
            pass

        try:
            tello.end()
        except:
            pass

        print("Saved as tello_stream.mp4")

if __name__ == "__main__":
    main()