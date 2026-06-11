import time
import cv2
from djitellopy import Tello

def main():
    tello = Tello()

    try:
        tello.connect()
        print("Battery:", tello.get_battery())

        tello.streamoff()
        time.sleep(1)

        # Safer stream settings first
        tello.set_video_resolution(Tello.RESOLUTION_480P)
        tello.set_video_fps(Tello.FPS_15)
        tello.set_video_bitrate(Tello.BITRATE_1MBPS)

        tello.streamon()
        time.sleep(3)

        # Queue helps when frames arrive faster than your loop consumes them
        frame_read = tello.get_frame_read(with_queue=True, max_queue_len=64)

        # Warm up and skip bad startup packets
        for _ in range(60):
            frame = frame_read.frame
            time.sleep(0.03)

        while True:
            frame = frame_read.frame
            if frame is None:
                continue

            # Fix channel order if colors look wrong
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            cv2.imshow("Tello Stream", frame)

            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

    except Exception as e:
        print("ERROR:", e)

    finally:
        cv2.destroyAllWindows()
        try:
            tello.streamoff()
        except:
            pass
        try:
            tello.end()
        except:
            pass

if __name__ == "__main__":
    main()