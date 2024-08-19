#!/usr/bin/env python3

# Copyright (c) 2024 Smart Rollerz e.V. All rights reserved.

import os

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from sensor_msgs.msg import Image
from timing.timer import Timer

from camera_preprocessing.transformation.birds_eyed_view import Birdseye
from camera_preprocessing.transformation.calibration import Calibration
from camera_preprocessing.transformation.distortion import Distortion


class CameraPreprocessing(Node):
    """CameraPreprocessing Node."""

    def __init__(self):
        super().__init__("camera_preprocessing")

        # Get parameters from the parameter server
        package_share_path = get_package_share_directory("camera_preprocessing")
        self.get_logger().info(f"Package Path: {package_share_path}")

        debug = self.declare_parameter("debug", False).value
        calibration_file = os.path.join(
            package_share_path,
            self.declare_parameter("calibration_file", "config/calib.bin").value,
        )
        config_file = os.path.join(
            package_share_path,
            self.declare_parameter("config_file", "config/config.yaml").value,
        )
        calibration_images_path = os.path.join(
            package_share_path,
            self.declare_parameter(
                "calibration_images_path", "img/calib/Neue_3MP_Kamera/"
            ).value,
        )
        position_calib_img_path = os.path.join(
            package_share_path,
            self.declare_parameter(
                "position_calib_img_path", "img/position/chessboard.png"
            ).value,
        )
        num_skip_frames = self.declare_parameter("num_skip_frames", 1).value
        subscriber_topic = self.declare_parameter(
            "subscriber_topic", "/camera/image_raw"
        ).value
        undistorted_publisher_topic = self.declare_parameter(
            "undistorted_publisher_topic", "/camera/undistorted"
        ).value
        birds_eye_publisher_topic = self.declare_parameter(
            "birds_eye_publisher_topic", "/camera/birds_eye"
        ).value

        self.frame_counter = 0
        self.num_skip_frames = num_skip_frames
        self.bridge = CvBridge()

        # Load distortion and bird's eye view transformations
        self.calibration = Calibration(
            calibration_file,
            config_file,
            calibration_images_path,
            position_calib_img_path,
            debug,
        )
        self.calibration.setup()
        self.distorter = Distortion(self.calibration, debug)
        self.bird = Birdseye(self.calibration, self.distorter, debug)

        # Initialize subscribers and publishers
        self.image_subscriber = self.create_subscription(
            Image, subscriber_topic, self.on_raw_image, 10
        )
        self.undistorted_image_publisher = self.create_publisher(
            Image, undistorted_publisher_topic, 10
        )
        self.birds_eye_publisher = self.create_publisher(
            Image, birds_eye_publisher_topic, 10
        )

        self.get_logger().info("CameraPreprocessing node initialized")

    def adjust_brightness(self, image, target_brightness=127):
        avg_brightness = np.mean(image)
        adjustment_factor = target_brightness / avg_brightness
        adjusted_image = cv2.convertScaleAbs(image, alpha=adjustment_factor, beta=0)
        return adjusted_image

    def on_raw_image(self, image):
        if self.frame_counter < self.num_skip_frames:
            self.frame_counter += 1
            return
        self.frame_counter = 0

        with Timer(name="msg_transform", filter_strength=40):
            try:
                cv_img = self.bridge.imgmsg_to_cv2(image, "bgr8")
                cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            except CvBridgeError as e:
                self.get_logger().error(f"CvBridge Error: {e}")
                return

        with Timer(name="downscale", filter_strength=40):
            cv_img = self.calibration.downscale_image(cv_img)

        with Timer(name="Thresh", filter_strength=40):
            cv_img = self.adjust_brightness(cv_img, 50)

        with Timer(name="undistortion", filter_strength=40):
            img_undistorted = self.distorter.undistort_image(cv_img)
        self.publish_image_undistorted(img_undistorted)

        with Timer(name="birds_eye", filter_strength=40):
            img_birds_eye = self.bird.transform_img(img_undistorted)

        with Timer(name="publish", filter_strength=40):
            self.publish_image_undistorted(img_undistorted)
            self.publish_image_birds_eye(img_birds_eye)

        Timer().print()

    def publish_image_undistorted(self, image_cv2):
        image_msg = self.bridge.cv2_to_imgmsg(image_cv2, "8UC1")
        self.undistorted_image_publisher.publish(image_msg)

    def publish_image_birds_eye(self, image_cv2):
        image_msg = self.bridge.cv2_to_imgmsg(image_cv2, "8UC1")
        self.birds_eye_publisher.publish(image_msg)


def main(args=None):
    rclpy.init(args=args)
    camera_preprocessing = CameraPreprocessing()
    try:
        rclpy.spin(camera_preprocessing)
    except KeyboardInterrupt:
        pass
    finally:
        camera_preprocessing.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
