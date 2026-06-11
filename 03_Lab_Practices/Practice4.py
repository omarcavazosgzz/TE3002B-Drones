import time
import cv2
import numpy as np
from djitellopy import Tello

# =========================
# MODE
# =========================
# False = standby detection only, no flying
# True  = takeoff and track green object
FLY_MODE = True

# =========================
# RECORDING
# =========================
SAVE_VIDEO = True
VIDEO_NAME = "practice4_green_detection.mp4"
FPS = 15

# =========================
# GREEN COLOR SETTINGS
# =========================
# Phone color: #00FF00
LOWER_GREEN = np.array([40, 70, 70])
UPPER_GREEN = np.array([85, 255, 255])

# Tighter range from the PDF, if needed:
# LOWER_GREEN = np.array([50, 100, 100])
# UPPER_GREEN = np.array([70, 255, 255])

MIN_AREA = 250

# =========================
# FRAME SETTINGS
# =========================
FRAME_W = 480
FRAME_H = 360
CENTER_X = FRAME_W // 2
CENTER_Y = FRAME_H // 2

DEADZONE_X = 45
DEADZONE_Y = 40

# =========================
# PD CONTROL SETTINGS
# =========================
Kp_yaw = 0.35
Kd_yaw = 0.09

Kp_z = 0.45
Kd_z = 0.06

MAX_YAW = 25
MAX_UD = 15

TRACK_SECONDS = 20


def safe_land(tello, retries=6):
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

    print("Could not land through SDK. Use app/manual control if needed.")
    return False


def main():
    tello = Tello()
    cap = None
    out = None
    already_landed = False

    prev_error_yaw = 0
    prev_error_z = 0

    try:
        # =========================
        # CONNECT
        # =========================
        tello.connect()
        print("Battery:", tello.get_battery())

        # =========================
        # START VIDEO STREAM
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

        # Direct UDP stream through OpenCV.
        # This avoids the get_frame_read() decoder crash you were getting.
        url = "udp://@0.0.0.0:11111?overrun_nonfatal=1&fifo_size=5000000"
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

        if not cap.isOpened():
            raise RuntimeError("Could not open UDP video stream.")

        # =========================
        # WAIT FOR FIRST FRAME
        # =========================
        first_frame = None
        start_wait = time.time()

        while time.time() - start_wait < 10:
            ok, frame = cap.read()
            if ok and frame is not None:
                first_frame = cv2.resize(frame, (FRAME_W, FRAME_H))
                break
            time.sleep(0.03)

        if first_frame is None:
            raise RuntimeError("No valid frame received from drone.")

        # =========================
        # VIDEO WRITER
        # =========================
        if SAVE_VIDEO:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(VIDEO_NAME, fourcc, FPS, (FRAME_W, FRAME_H))

            if not out.isOpened():
                raise RuntimeError("Could not open video writer.")

            print(f"Recording to {VIDEO_NAME}")

        # =========================
        # TAKEOFF IF ENABLED
        # =========================
        if FLY_MODE:
            print("Taking off...")
            tello.takeoff()
            time.sleep(2)

            # Small rise, then stabilize
            tello.send_rc_control(0, 0, 15, 0)
            time.sleep(1.5)
            tello.send_rc_control(0, 0, 0, 0)
            time.sleep(1)

            print("Tracking started. Move green phone screen slowly.")
        else:
            print("Standby mode: no flying.")
            print("Show green phone screen: #00FF00")

        print("Press q in the camera window to stop.")

        start_time = time.time()

        while True:
            if FLY_MODE and time.time() - start_time > TRACK_SECONDS:
                break

            ok, frame = cap.read()

            if not ok or frame is None:
                if FLY_MODE:
                    tello.send_rc_control(0, 0, 0, 0)
                continue

            frame = cv2.resize(frame, (FRAME_W, FRAME_H))

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)

            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.erode(mask, kernel, iterations=1)
            mask = cv2.dilate(mask, kernel, iterations=2)

            contours, _ = cv2.findContours(
                mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            detected = False
            yaw_speed = 0
            ud_speed = 0

            if contours:
                largest = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest)

                if area > MIN_AREA:
                    detected = True

                    x, y, w, h = cv2.boundingRect(largest)
                    cx = x + w // 2
                    cy = y + h // 2

                    error_yaw = cx - CENTER_X
                    error_z = CENTER_Y - cy

                    derivative_yaw = error_yaw - prev_error_yaw
                    derivative_z = error_z - prev_error_z

                    if abs(error_yaw) > DEADZONE_X:
                        yaw_speed = int(Kp_yaw * error_yaw + Kd_yaw * derivative_yaw)

                    if abs(error_z) > DEADZONE_Y:
                        ud_speed = int(Kp_z * error_z + Kd_z * derivative_z)

                    yaw_speed = int(np.clip(yaw_speed, -MAX_YAW, MAX_YAW))
                    ud_speed = int(np.clip(ud_speed, -MAX_UD, MAX_UD))

                    prev_error_yaw = error_yaw
                    prev_error_z = error_z

                    if FLY_MODE:
                        # send_rc_control(left_right, forward_back, up_down, yaw)
                        tello.send_rc_control(0, 0, ud_speed, yaw_speed)

                    # Draw detection
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.circle(frame, (cx, cy), 6, (255, 0, 0), -1)

                    cv2.putText(
                        frame,
                        "GREEN DETECTED",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 255, 0),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"cx={cx} cy={cy} area={int(area)}",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 0),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"yaw={yaw_speed} ud={ud_speed}",
                        (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 0),
                        2
                    )

                    print(
                        f"Green detected | cx={cx}, cy={cy}, area={area:.0f}, "
                        f"yaw={yaw_speed}, ud={ud_speed}"
                    )

                else:
                    if FLY_MODE:
                        tello.send_rc_control(0, 0, 0, 0)

                    cv2.putText(
                        frame,
                        f"Target too small: area={int(area)}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 0, 255),
                        2
                    )

            if not detected:
                if FLY_MODE:
                    tello.send_rc_control(0, 0, 0, 0)

                cv2.putText(
                    frame,
                    "GREEN TARGET LOST",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 0, 255),
                    2
                )

            # Draw center lines and deadzone
            cv2.line(frame, (CENTER_X, 0), (CENTER_X, FRAME_H), (255, 255, 255), 1)
            cv2.line(frame, (0, CENTER_Y), (FRAME_W, CENTER_Y), (255, 255, 255), 1)

            cv2.rectangle(
                frame,
                (CENTER_X - DEADZONE_X, CENTER_Y - DEADZONE_Y),
                (CENTER_X + DEADZONE_X, CENTER_Y + DEADZONE_Y),
                (255, 255, 0),
                1
            )

            # Show windows
            cv2.imshow("Tello Green Tracking Camera", frame)
            cv2.imshow("Green Mask", mask)

            # Record annotated camera view
            if SAVE_VIDEO and out is not None:
                out.write(frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Manual stop with q.")
                break

            time.sleep(0.05)

        # =========================
        # FINISH FLIGHT
        # =========================
        if FLY_MODE:
            print("Tracking finished. Stabilizing...")
            tello.send_rc_control(0, 0, 0, 0)
            time.sleep(2)

            already_landed = safe_land(tello)

    except Exception as e:
        print("ERROR:", e)

        try:
            tello.send_rc_control(0, 0, 0, 0)
        except:
            pass

        if FLY_MODE and not already_landed:
            safe_land(tello)

    finally:
        if cap is not None:
            cap.release()

        if out is not None:
            out.release()

        cv2.destroyAllWindows()

        try:
            tello.streamoff()
        except:
            pass

        try:
            tello.end()
        except:
            pass

        print("Closed cleanly.")

        if SAVE_VIDEO:
            print(f"Saved as {VIDEO_NAME}")


if __name__ == "__main__":
    main()