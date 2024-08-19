#!/usr/bin/env python3

# Copyright (c) 2024 Smart Rollerz e.V. All rights reserved.

import os

import cv2
import rclpy
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from sensor_msgs.msg import Image

from camera_preprocessing.transformation.calibration import Calibration


class CameraCalibration(Node):
    """
    CameraCalibration Node.

    Arguments:
        Node -- The ROS node class.
    """

    def __init__(self):
        """Initialize the CameraCalibration node."""
        super().__init__("camera_calibration")

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
        chessboard_path = self.declare_parameter("chessboard_path", "none").value
        subscriber_topic = self.declare_parameter(
            "subscriber_topic", "/camera/image_raw"
        ).value
        self.bridge = CvBridge()
        self.lock = False

        # Set the path to the chessboard image
        self.chessboard_path = (
            "/tmp/chessboard.png" if chessboard_path == "none" else chessboard_path
        )

        # Load distortion and bird's eye view transformations
        self.calibration = Calibration(
            calibration_file,
            config_file,
            calibration_images_path,
            self.chessboard_path,
            debug,
        )

        # Check if the chessboard image is provided
        if chessboard_path != "none":
            ret = self.calibrate()
            if ret:
                rclpy.shutdown()
                exit(0)
            else:
                rclpy.shutdown()
                exit(1)
        else:
            # Initialize subscribers
            self.calibration.load_config(config_file)
            self.calibration.calibrate_distortion(resize=False)
            self.image_subscriber = self.create_subscription(
                Image, subscriber_topic, self.on_raw_image, 10
            )

    def calibrate(self) -> bool:
        """
        Calibrate the camera.

        Returns:
            bool -- True if the camera was calibrated successfully, False otherwise.
        """
        self.calibration.setup(force_recalibration=True)

        if self.calibration.all_calibrated:
            self.get_logger().info(str(self.calibration))
            self.get_logger().info("Camera calibrated successfully.")
            return True
        else:
            self.get_logger().info(str(self.calibration))
            self.get_logger().error("Failed to calibrate the camera.")
            return False

    def on_raw_image(self, image):
        """
        Callback function for the raw image subscriber.

        Arguments:
            image -- The raw image message.
        """
        if self.lock:
            return
        self.lock = True
        try:
            cv_img = self.bridge.imgmsg_to_cv2(image, "bgr8")
            found = self.calibration.find_chessboard(cv_img)
            if found:
                self.get_logger().info("Chessboard found in the image.")
                cv2.imwrite(self.chessboard_path, cv_img)
                ret = self.calibrate()
                os.remove(self.chessboard_path)
                if ret:
                    rclpy.shutdown()
                    exit(0)
            else:
                self.lock = False
                self.get_logger().info("Chessboard not found in the image.")
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge Error: {e}")
            self.lock = False


def main(args=None):
    """
    Main function to start the CameraCalibration node.

    Keyword Arguments:
        args -- Launch arguments (default: {None})
    """
    rclpy.init(args=args)
    camera_calibration = CameraCalibration()
    try:
        rclpy.spin(camera_calibration)
    except KeyboardInterrupt:
        pass
    finally:
        camera_calibration.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
