import time
import cv2
import numpy as np
from djitellopy import Tello

# Target color: bright magenta / hot pink
# Show this on your phone: #FF00FF
LOWER_MAGENTA = np.array([135, 80, 80])
UPPER_MAGENTA = np.array([170, 255, 255])

MIN_AREA = 700

# Optional evidence recording
SAVE_VIDEO = True
VIDEO_NAME = "practice4_color_detection.mp4"
FPS = 30


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

        # Use OpenCV directly instead of get_frame_read()
        url = "udp://@0.0.0.0:11111?overrun_nonfatal=1&fifo_size=5000000"
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

        if not cap.isOpened():
            raise RuntimeError("Could not open UDP video stream.")

        print("Show a full-screen magenta color on your phone: #FF00FF")
        print("Press q to quit.")

        first_frame_received = False

        while True:
            ok, frame = cap.read()

            if not ok or frame is None:
                continue

            frame = cv2.resize(frame, (480, 360))

            if not first_frame_received:
                h, w, _ = frame.shape

                if SAVE_VIDEO:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    out = cv2.VideoWriter(VIDEO_NAME, fourcc, FPS, (w, h))

                    if not out.isOpened():
                        raise RuntimeError("Could not open video writer.")

                    print(f"Recording evidence to {VIDEO_NAME}")

                first_frame_received = True

            # Convert BGR to HSV for color detection
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Create magenta mask
            mask = cv2.inRange(hsv, LOWER_MAGENTA, UPPER_MAGENTA)

            # Clean mask
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.erode(mask, kernel, iterations=1)
            mask = cv2.dilate(mask, kernel, iterations=2)

            contours, _ = cv2.findContours(
                mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            detected = False

            if contours:
                largest = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest)

                if area > MIN_AREA:
                    detected = True

                    x, y, w, h = cv2.boundingRect(largest)
                    cx = x + w // 2
                    cy = y + h // 2

                    object_region_bgr = frame[y:y+h, x:x+w]
                    object_region_hsv = hsv[y:y+h, x:x+w]

                    avg_bgr = cv2.mean(object_region_bgr)[:3]
                    avg_hsv = cv2.mean(object_region_hsv)[:3]

                    # Draw box and center
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

                    cv2.putText(
                        frame,
                        "MAGENTA DETECTED",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"Center: ({cx}, {cy}) Area: {int(area)}",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 0),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"HSV avg: ({avg_hsv[0]:.0f}, {avg_hsv[1]:.0f}, {avg_hsv[2]:.0f})",
                        (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 0),
                        2
                    )

                    print(
                        f"Detected | center=({cx},{cy}) | area={area:.0f} | "
                        f"BGR=({avg_bgr[0]:.0f},{avg_bgr[1]:.0f},{avg_bgr[2]:.0f}) | "
                        f"HSV=({avg_hsv[0]:.0f},{avg_hsv[1]:.0f},{avg_hsv[2]:.0f})"
                    )

            if not detected:
                cv2.putText(
                    frame,
                    "Magenta not detected",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

            cv2.imshow("Tello Camera - Magenta Detection", frame)
            cv2.imshow("Mask", mask)

            if SAVE_VIDEO and out is not None:
                out.write(frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except Exception as e:
        print("ERROR:", e)

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