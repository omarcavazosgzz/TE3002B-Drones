#!/usr/bin/env python3

from djitellopy import Tello
import cv2
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class TelloCameraPublisher(Node):
    def __init__(self):
        super().__init__("tello_camera_publisher")

        self.publisher = self.create_publisher(Image, "/tello/image_raw", 10)
        self.bridge = CvBridge()

        self.get_logger().info("Connecting to Tello...")
        self.tello = Tello()
        self.tello.connect()

        battery = self.tello.get_battery()
        self.get_logger().info(f"Tello battery: {battery}%")

        self.get_logger().info("Starting video stream...")
        self.tello.streamoff()
        time.sleep(1)

        self.tello.set_video_resolution(Tello.RESOLUTION_480P)
        self.tello.set_video_fps(Tello.FPS_15)
        self.tello.set_video_bitrate(Tello.BITRATE_1MBPS)

        self.tello.streamon()
        time.sleep(2)

        self.frame_reader = self.tello.get_frame_read()

        self.timer = self.create_timer(1.0 / 15.0, self.publish_frame)

    def publish_frame(self):
        frame = self.frame_reader.frame

        if frame is None:
            self.get_logger().warn("No frame received")
            return

        # Optional: resize even smaller to reduce latency
        frame = cv2.resize(frame, (480, 360))

        msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "tello_camera"

        self.publisher.publish(msg)

    def destroy_node(self):
        self.get_logger().info("Stopping Tello stream...")
        try:
            self.tello.streamoff()
            self.tello.end()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    node = TelloCameraPublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
