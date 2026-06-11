from djitellopy import Tello
import cv2
import numpy as np
import time
import logging

Tello.LOGGER.setLevel(logging.ERROR)

# -----------------------------
# CAMERA / CONTROL SETTINGS
# -----------------------------
FRAME_W = 640
FRAME_H = 480

# Press C while running if colors look wrong.
SOURCE_IS_RGB = False

RC_INTERVAL = 0.05
MIN_BATTERY = 25

# If target is lost, the drone searches briefly instead of landing.
LOST_TARGET_TIMEOUT = 4.0
SEARCH_YAW_SPEED = 25


def nothing(x):
    pass


def clamp(value, min_value, max_value):
    return int(max(min_value, min(max_value, value)))


def create_trackbars():
    cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Controls", 520, 560)

    # Light green shirt defaults
    cv2.createTrackbar("H min", "Controls", 35, 179, nothing)
    cv2.createTrackbar("H max", "Controls", 85, 179, nothing)
    cv2.createTrackbar("S min", "Controls", 25, 255, nothing)
    cv2.createTrackbar("S max", "Controls", 255, 255, nothing)
    cv2.createTrackbar("V min", "Controls", 80, 255, nothing)
    cv2.createTrackbar("V max", "Controls", 255, 255, nothing)

    # Detection / distance
    cv2.createTrackbar("Min Area", "Controls", 2000, 25000, nothing)

    # Higher target area = follows closer
    cv2.createTrackbar("Target Area", "Controls", 22000, 70000, nothing)

    # Speed limits
    cv2.createTrackbar("Max FB", "Controls", 60, 100, nothing)
    cv2.createTrackbar("Max UD", "Controls", 30, 100, nothing)
    cv2.createTrackbar("Max Yaw", "Controls", 60, 100, nothing)

    # Controller gains
    cv2.createTrackbar("K yaw x100", "Controls", 14, 60, nothing)
    cv2.createTrackbar("K ud x100", "Controls", 8, 60, nothing)
    cv2.createTrackbar("K fb x1000", "Controls", 6, 80, nothing)

    # Dead zones
    cv2.createTrackbar("Dead X px", "Controls", 30, 200, nothing)
    cv2.createTrackbar("Dead Y px", "Controls", 45, 200, nothing)
    cv2.createTrackbar("Dead Area", "Controls", 3000, 20000, nothing)


def get_trackbars():
    return {
        "h_min": cv2.getTrackbarPos("H min", "Controls"),
        "h_max": cv2.getTrackbarPos("H max", "Controls"),
        "s_min": cv2.getTrackbarPos("S min", "Controls"),
        "s_max": cv2.getTrackbarPos("S max", "Controls"),
        "v_min": cv2.getTrackbarPos("V min", "Controls"),
        "v_max": cv2.getTrackbarPos("V max", "Controls"),

        "min_area": max(100, cv2.getTrackbarPos("Min Area", "Controls")),
        "target_area": max(1000, cv2.getTrackbarPos("Target Area", "Controls")),

        "max_fb": cv2.getTrackbarPos("Max FB", "Controls"),
        "max_ud": cv2.getTrackbarPos("Max UD", "Controls"),
        "max_yaw": cv2.getTrackbarPos("Max Yaw", "Controls"),

        "k_yaw": cv2.getTrackbarPos("K yaw x100", "Controls") / 100.0,
        "k_ud": cv2.getTrackbarPos("K ud x100", "Controls") / 100.0,
        "k_fb": cv2.getTrackbarPos("K fb x1000", "Controls") / 1000.0,

        "dead_x": cv2.getTrackbarPos("Dead X px", "Controls"),
        "dead_y": cv2.getTrackbarPos("Dead Y px", "Controls"),
        "dead_area": cv2.getTrackbarPos("Dead Area", "Controls"),
    }


def detect_green_shirt(frame, source_is_rgb, values):
    if source_is_rgb:
        display = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    else:
        display = frame.copy()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower = np.array([values["h_min"], values["s_min"], values["v_min"]])
    upper = np.array([values["h_max"], values["s_max"], values["v_max"]])

    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    target = None

    if len(contours) > 0:
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        if area >= values["min_area"]:
            x, y, w, h = cv2.boundingRect(largest)

            cx = x + w // 2
            cy = y + h // 2

            target = {
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "cx": cx,
                "cy": cy,
                "area": area,
            }

            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(display, (cx, cy), 6, (0, 255, 0), -1)

            cv2.putText(
                display,
                f"Area: {int(area)}",
                (x, max(20, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

    return display, mask, target


def compute_control(target, values):
    if target is None:
        return 0, 0, 0, 0, None

    center_x = FRAME_W // 2
    center_y = FRAME_H // 2

    error_x = target["cx"] - center_x
    error_y = target["cy"] - center_y
    error_area = values["target_area"] - target["area"]

    # Yaw: positive = rotate right
    if abs(error_x) < values["dead_x"]:
        yaw = 0
    else:
        yaw = values["k_yaw"] * error_x
        yaw = clamp(yaw, -values["max_yaw"], values["max_yaw"])

    # Up/down: positive = up
    if abs(error_y) < values["dead_y"]:
        ud = 0
    else:
        ud = -values["k_ud"] * error_y
        ud = clamp(ud, -values["max_ud"], values["max_ud"])

    # Forward/back: positive = forward
    # Bigger target_area means the drone wants the shirt to look bigger, so it follows closer.
    if abs(error_area) < values["dead_area"]:
        fb = 0
    else:
        fb = values["k_fb"] * error_area
        fb = clamp(fb, -values["max_fb"], values["max_fb"])

    lr = 0

    debug = {
        "error_x": error_x,
        "error_y": error_y,
        "error_area": error_area,
        "lr": lr,
        "fb": fb,
        "ud": ud,
        "yaw": yaw,
    }

    return lr, fb, ud, yaw, debug


def main():
    global SOURCE_IS_RGB

    create_trackbars()

    tello = Tello()

    print("Connecting...")
    tello.connect()

    battery = tello.get_battery()
    print(f"Battery: {battery}%")

    if battery < MIN_BATTERY:
        print(f"Battery too low: {battery}%. Charge first.")
        tello.end()
        return

    print("Starting stream...")
    tello.streamoff()
    time.sleep(1)
    tello.streamon()
    time.sleep(2)

    frame_reader = tello.get_frame_read()

    is_flying = False
    follow_enabled = False

    last_rc_time = time.time()
    last_seen_time = time.time()
    last_battery_check = time.time()

    # Used for L twice to land
    land_press_time = 0

    # Used for corner search
    last_seen_x_error = 0
    last_known_yaw_direction = 1

    print("")
    print("Controls:")
    print("T = takeoff")
    print("F = toggle follow")
    print("SPACE = hover / stop follow")
    print("L twice = land")
    print("C = toggle RGB/BGR")
    print("ESC = safe quit")
    print("X = emergency motor stop")
    print("")
    print("Distance tuning:")
    print("Increase Target Area to follow closer.")
    print("Increase Max FB / K fb x1000 to move forward faster.")
    print("Increase Max Yaw / K yaw x100 to handle corners better.")
    print("")

    while True:
        frame = frame_reader.frame

        if frame is None:
            continue

        frame = cv2.resize(frame, (FRAME_W, FRAME_H))

        now = time.time()

        if now - last_battery_check > 2.0:
            try:
                battery = tello.get_battery()
            except Exception:
                pass
            last_battery_check = now

        values = get_trackbars()

        display, mask, target = detect_green_shirt(frame, SOURCE_IS_RGB, values)

        lr, fb, ud, yaw, debug = compute_control(target, values)

        # Update last seen info
        if target is not None:
            last_seen_time = now

            if debug is not None:
                last_seen_x_error = debug["error_x"]

                if last_seen_x_error > 0:
                    last_known_yaw_direction = 1
                elif last_seen_x_error < 0:
                    last_known_yaw_direction = -1

        time_since_seen = now - last_seen_time

        # If lost for too long, stop following but keep hovering.
        if follow_enabled and target is None and time_since_seen > LOST_TARGET_TIMEOUT:
            print("Target lost too long. Follow disabled. Hovering.")
            follow_enabled = False
            if is_flying:
                tello.send_rc_control(0, 0, 0, 0)

        # Draw center lines
        cv2.line(display, (FRAME_W // 2, 0), (FRAME_W // 2, FRAME_H), (255, 255, 255), 1)
        cv2.line(display, (0, FRAME_H // 2), (FRAME_W, FRAME_H // 2), (255, 255, 255), 1)

        status = "FOLLOW ON" if follow_enabled else "FOLLOW OFF"
        flying_status = "FLYING" if is_flying else "LANDED"
        color_mode = "RGB" if SOURCE_IS_RGB else "BGR"

        if follow_enabled:
            status_color = (0, 255, 0)
        else:
            status_color = (0, 0, 255)

        cv2.putText(
            display,
            f"{status} | {flying_status}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            status_color,
            2,
        )

        cv2.putText(
            display,
            f"Battery: {battery}% | Source: {color_mode}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
        )

        if target is None:
            cv2.putText(
                display,
                f"NO TARGET | lost: {time_since_seen:.1f}s",
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

            if follow_enabled:
                cv2.putText(
                    display,
                    "SEARCHING LAST DIRECTION",
                    (20, 145),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 165, 255),
                    2,
                )

        else:
            cv2.putText(
                display,
                f"CMD fb:{fb} ud:{ud} yaw:{yaw}",
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 0),
                2,
            )

            if debug is not None:
                cv2.putText(
                    display,
                    f"err_x:{int(debug['error_x'])} err_y:{int(debug['error_y'])} err_area:{int(debug['error_area'])}",
                    (20, 145),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2,
                )

        cv2.putText(
            display,
            "T takeoff | F follow | SPACE stop | L twice land | ESC quit",
            (20, FRAME_H - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
        )

        cv2.imshow("Green Shirt Follower CLOSE + CORNERS", display)
        cv2.imshow("Green Mask", mask)

        key = cv2.waitKey(1) & 0xFF

        # -----------------------------
        # KEYBOARD CONTROLS
        # -----------------------------
        if key == 27:  # ESC
            print("Exiting safely...")
            break

        elif key == ord("t"):
            if not is_flying:
                if battery < MIN_BATTERY:
                    print(f"Battery too low: {battery}%. Charge first.")
                else:
                    print("Takeoff")
                    tello.takeoff()
                    is_flying = True
                    follow_enabled = False
                    time.sleep(1)

        elif key == ord("f"):
            if is_flying:
                follow_enabled = not follow_enabled
                last_seen_time = now
                print(f"Follow enabled: {follow_enabled}")
            else:
                print("Takeoff first before enabling follow.")

        elif key == ord(" "):
            print("Hover / stop follow")
            follow_enabled = False
            if is_flying:
                tello.send_rc_control(0, 0, 0, 0)

        elif key == ord("l"):
            if now - land_press_time < 2.0:
                if is_flying:
                    print("Landing confirmed")
                    follow_enabled = False
                    tello.send_rc_control(0, 0, 0, 0)
                    time.sleep(0.3)
                    tello.land()
                    is_flying = False
                land_press_time = 0
            else:
                print("Press L again within 2 seconds to land.")
                land_press_time = now

        elif key == ord("c"):
            SOURCE_IS_RGB = not SOURCE_IS_RGB
            print(f"SOURCE_IS_RGB changed to: {SOURCE_IS_RGB}")

        elif key == ord("x"):
            print("EMERGENCY STOP")
            tello.emergency()
            is_flying = False
            follow_enabled = False
            break

        # -----------------------------
        # SEND RC COMMANDS
        # -----------------------------
        if now - last_rc_time >= RC_INTERVAL:
            if is_flying and follow_enabled and target is not None:
                # Normal tracking
                tello.send_rc_control(lr, fb, ud, yaw)

            elif is_flying and follow_enabled and target is None:
                # Corner-search behavior.
                # Rotate toward the side where target was last seen.
                search_yaw = last_known_yaw_direction * SEARCH_YAW_SPEED
                tello.send_rc_control(0, 0, 0, search_yaw)

            elif is_flying:
                # Hover
                tello.send_rc_control(0, 0, 0, 0)

            last_rc_time = now

    # -----------------------------
    # CLEAN EXIT
    # -----------------------------
    print("Stopping...")

    follow_enabled = False

    if is_flying:
        tello.send_rc_control(0, 0, 0, 0)
        time.sleep(0.3)
        tello.land()

    tello.streamoff()
    tello.end()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
