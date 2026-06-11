from djitellopy import Tello
import cv2
import time
import logging

Tello.LOGGER.setLevel(logging.ERROR)

SPEED = 30

tello = Tello()

print("Connecting...")
tello.connect()
print(f"Battery: {tello.get_battery()}%")

print("Starting stream...")
tello.streamoff()
time.sleep(1)
tello.streamon()
time.sleep(2)

frame_reader = tello.get_frame_read()

print("Controls:")
print("T = takeoff")
print("L = land")
print("W/S = forward/back")
print("A/D = left/right")
print("R/F = up/down")
print("Q/E = rotate left/right")
print("SPACE = stop")
print("ESC = quit")

is_flying = False

while True:
    frame = frame_reader.frame

    if frame is not None:
        frame = cv2.resize(frame, (640, 480))

        cv2.putText(frame, f"Battery: {tello.get_battery()}%", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.putText(frame, "T takeoff | L land | ESC quit", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Tello Keyboard Control", frame)

    key = cv2.waitKey(1) & 0xFF

    lr = 0
    fb = 0
    ud = 0
    yaw = 0

    if key == 27:  # ESC
        break

    elif key == ord("t"):
        if not is_flying:
            print("Takeoff")
            tello.takeoff()
            is_flying = True

    elif key == ord("l"):
        if is_flying:
            print("Landing")
            tello.land()
            is_flying = False

    elif key == ord("w"):
        fb = SPEED

    elif key == ord("s"):
        fb = -SPEED

    elif key == ord("a"):
        lr = -SPEED

    elif key == ord("d"):
        lr = SPEED

    elif key == ord("r"):
        ud = SPEED

    elif key == ord("f"):
        ud = -SPEED

    elif key == ord("q"):
        yaw = -SPEED

    elif key == ord("e"):
        yaw = SPEED

    elif key == 32:  # SPACE
        lr = fb = ud = yaw = 0

    if is_flying:
        tello.send_rc_control(lr, fb, ud, yaw)

print("Stopping...")

if is_flying:
    tello.land()

tello.streamoff()
tello.end()
cv2.destroyAllWindows()
print("Done.")
