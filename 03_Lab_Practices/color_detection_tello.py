#!/usr/bin/env python3

import time
import cv2
import numpy as np

from djitellopy import Tello

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from std_msgs.msg import Bool

from cv_bridge import CvBridge


class TelloColorDetector(Node):
    def __init__(self):
        super().__init__("tello_color_detector")

        # -----------------------------
        # Parameters
        # -----------------------------
        self.declare_parameter("target_h_min", 135)
        self.declare_parameter("target_s_min", 80)
        self.declare_parameter("target_v_min", 80)

        self.declare_parameter("target_h_max", 170)
        self.declare_parameter("target_s_max", 255)
        self.declare_parameter("target_v_max", 255)

        self.declare_parameter("min_area", 700)
        self.declare_parameter("publish_fps", 15.0)
        self.declare_parameter("resize_width", 480)
        self.declare_parameter("resize_height", 360)

        self.h_min = self.get_parameter("target_h_min").value
        self.s_min = self.get_parameter("target_s_min").value
        self.v_min = self.get_parameter("target_v_min").value

        self.h_max = self.get_parameter("target_h_max").value
        self.s_max = self.get_parameter("target_s_max").value
        self.v_max = self.get_parameter("target_v_max").value

        self.min_area = self.get_parameter("min_area").value
        self.publish_fps = self.get_parameter("publish_fps").value
        self.resize_width = self.get_parameter("resize_width").value
        self.resize_height = self.get_parameter("resize_height").value

        self.lower_color = np.array([self.h_min, self.s_min, self.v_min])
        self.upper_color = np.array([self.h_max, self.s_max, self.v_max])

        # -----------------------------
        # ROS publishers
        # -----------------------------
        self.bridge = CvBridge()

        self.image_pub = self.create_publisher(Image, "/tello/image_raw", 10)
        self.mask_pub = self.create_publisher(Image, "/tello/mask", 10)
        self.center_pub = self.create_publisher(Point, "/tello/target_center", 10)
        self.detected_pub = self.create_publisher(Bool, "/tello/target_detected", 10)

        # -----------------------------
        # Tello setup
        # -----------------------------
        self.get_logger().info("Connecting to Tello...")

        self.tello = Tello()
        self.tello.connect()

        battery = self.tello.get_battery()
        self.get_logger().info(f"Battery: {battery}%")

        self.get_logger().info("Starting video stream...")

        self.tello.streamoff()
        time.sleep(1)

        self.tello.set_video_resolution(Tello.RESOLUTION_480P)
        self.tello.set_video_fps(Tello.FPS_15)
        self.tello.set_video_bitrate(Tello.BITRATE_1MBPS)

        self.tello.streamon()
        time.sleep(3)

        # OpenCV UDP stream. Usually lower delay than get_frame_read()
        self.url = "udp://@0.0.0.0:11111?overrun_nonfatal=1&fifo_size=5000000"
        self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)

        if not self.cap.isOpened():
            raise RuntimeError("Could not open Tello UDP video stream.")

        self.get_logger().info("Tello video stream opened.")
        self.get_logger().info("Show full-screen magenta on your phone: #FF00FF")
        self.get_logger().info("Publishing:")
        self.get_logger().info("  /tello/image_raw")
        self.get_logger().info("  /tello/mask")
        self.get_logger().info("  /tello/target_center")
        self.get_logger().info("  /tello/target_detected")

        timer_period = 1.0 / self.publish_fps
        self.timer = self.create_timer(timer_period, self.process_frame)

    def process_frame(self):
        ok, frame = self.cap.read()

        if not ok or frame is None:
            self.get_logger().warn("No frame received.")
            return

        frame = cv2.resize(frame, (self.resize_width, self.resize_height))

        # -------------------------------------------------
        # IMPORTANT:
        # OpenCV frames are BGR, so convert BGR to HSV.
        # Do NOT use RGB2HSV here.
        # -------------------------------------------------
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, self.lower_color, self.upper_color)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        detected = False
        center_msg = Point()
        center_msg.x = -1.0
        center_msg.y = -1.0
        center_msg.z = 0.0

        if contours:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)

            if area > self.min_area:
                detected = True

                x, y, w, h = cv2.boundingRect(largest)
                cx = x + w // 2
                cy = y + h // 2

                center_msg.x = float(cx)
                center_msg.y = float(cy)
                center_msg.z = float(area)

                object_region_bgr = frame[y:y+h, x:x+w]
                object_region_hsv = hsv[y:y+h, x:x+w]

                avg_bgr = cv2.mean(object_region_bgr)[:3]
                avg_hsv = cv2.mean(object_region_hsv)[:3]

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

                self.get_logger().info(
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

        detected_msg = Bool()
        detected_msg.data = detected

        now = self.get_clock().now().to_msg()

        image_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        image_msg.header.stamp = now
        image_msg.header.frame_id = "tello_camera"

        mask_msg = self.bridge.cv2_to_imgmsg(mask, encoding="mono8")
        mask_msg.header.stamp = now
        mask_msg.header.frame_id = "tello_camera"

        self.image_pub.publish(image_msg)
        self.mask_pub.publish(mask_msg)
        self.center_pub.publish(center_msg)
        self.detected_pub.publish(detected_msg)

    def destroy_node(self):
        self.get_logger().info("Closing Tello camera node...")

        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass

        try:
            self.tello.streamoff()
        except Exception:
            pass

        try:
            self.tello.end()
        except Exception:
            pass

        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    node = None

    try:
        node = TelloColorDetector()
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    except Exception as e:
        print("ERROR:", e)

    finally:
        if node is not None:
            node.destroy_node()

        rclpy.shutdown()
        print("Closed cleanly.")


if __name__ == "__main__":
    main()