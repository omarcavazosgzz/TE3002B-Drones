import time
import cv2
from djitellopy import Tello


# =========================
# CONFIGURATION
# =========================
VIDEO_NAME = "practice3_flight.mp4"
RECORD_SECONDS = 30

# Set True for circular trajectory while recording.
# Set False if you only want hover + recording.
DO_CIRCLE = True

# RC values for circular trajectory
# send_rc_control(left_right, forward_backward, up_down, yaw)
CIRCLE_FORWARD_SPEED = 20
CIRCLE_YAW_SPEED = 25

# Video settings
FPS = 15
FIFO_SIZE = 5000000


def safe_land(tello, retries=6):
    """
    Stop motion first, then try landing multiple times.
    This avoids the issue where land sometimes returns 'error'
    on the first attempts.
    """
    try:
        tello.send_rc_control(0, 0, 0, 0)
        time.sleep(2)
    except:
        pass

    for i in range(retries):
        try:
            print(f"Landing attempt {i + 1}...")
            tello.land()
            print("Landing successful.")
            return True
        except Exception as e:
            print("Land failed:", e)
            time.sleep(2)

    print("Could not land through SDK after retries.")
    print("Use the app/manual control if needed.")
    return False


def main():
    tello = Tello()
    cap = None
    out = None
    already_landed = False

    try:
        # =========================
        # CONNECT
        # =========================
        tello.connect()
        print("Battery:", tello.get_battery())

        time.sleep(3)

        # =========================
        # TAKEOFF
        # =========================
        tello.takeoff()
        time.sleep(2)

        # Small rise
        tello.send_rc_control(0, 0, 20, 0)
        time.sleep(6)

        # Stabilize
        tello.send_rc_control(0, 0, 0, 0)
        time.sleep(1)

        # =========================
        # VIDEO STREAM SETUP
        # =========================
        try:
            tello.streamoff()
        except:
            pass

        time.sleep(1)

        tello.set_video_resolution(Tello.RESOLUTION_480P)
        tello.set_video_fps(Tello.FPS_15)
        tello.set_video_bitrate(Tello.BITRATE_1MBPS)

        tello.streamon()
        time.sleep(3)

        # Direct UDP stream through OpenCV/FFmpeg.
        # This avoids djitellopy's frame_read decoder issues.
        url = f"udp://@0.0.0.0:11111?overrun_nonfatal=1&fifo_size={FIFO_SIZE}"
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

        if not cap.isOpened():
            raise RuntimeError("Could not open UDP video stream.")

        # =========================
        # WAIT FOR FIRST VALID FRAME
        # =========================
        ok, frame = False, None
        start_wait = time.time()

        while time.time() - start_wait < 10:
            ok, frame = cap.read()
            if ok and frame is not None:
                break
            time.sleep(0.03)

        if not ok or frame is None:
            raise RuntimeError("No valid frame received from drone.")

        h, w, _ = frame.shape
        print(f"Frame size: {w}x{h}")

        # =========================
        # VIDEO WRITER
        # =========================
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(VIDEO_NAME, fourcc, FPS, (w, h))

        if not out.isOpened():
            raise RuntimeError("Could not open MP4 writer.")

        print(f"Recording to {VIDEO_NAME}")
        print("Do not press Ctrl+C. Let the script finish.")

        # =========================
        # RECORDING LOOP
        # =========================
        record_start = time.time()

        while time.time() - record_start < RECORD_SECONDS:
            ok, frame = cap.read()

            if not ok or frame is None:
                continue

            out.write(frame)

            if DO_CIRCLE:
                # Circular movement:
                # forward velocity + yaw velocity
                tello.send_rc_control(
                    0,
                    CIRCLE_FORWARD_SPEED,
                    0,
                    CIRCLE_YAW_SPEED
                )
            else:
                # Hover
                tello.send_rc_control(0, 0, 0, 0)

            time.sleep(0.03)

        # =========================
        # STOP MOVEMENT
        # =========================
        print("Stopping movement...")
        tello.send_rc_control(0, 0, 0, 0)
        time.sleep(2)

        # =========================
        # CLOSE VIDEO BEFORE LANDING
        # =========================
        print("Closing video file...")

        if cap is not None:
            cap.release()
            cap = None

        if out is not None:
            out.release()
            out = None

        time.sleep(1)

        # =========================
        # STOP STREAM BEFORE LANDING
        # =========================
        try:
            tello.streamoff()
        except:
            pass

        time.sleep(1)

        # =========================
        # LAND SAFELY
        # =========================
        already_landed = safe_land(tello)

    except Exception as e:
        print("ERROR:", e)

        try:
            tello.send_rc_control(0, 0, 0, 0)
        except:
            pass

        if not already_landed:
            safe_land(tello)

    finally:
        # =========================
        # FINAL CLEANUP
        # =========================
        if cap is not None:
            cap.release()

        if out is not None:
            out.release()

        try:
            tello.streamoff()
        except:
            pass

        try:
            tello.end()
        except:
            pass

        print(f"Saved as {VIDEO_NAME}")


if __name__ == "__main__":
    main()