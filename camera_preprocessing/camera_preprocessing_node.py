#!/usr/bin/env python3

# Copyright (c) 2024 Smart Rollerz e.V. All rights reserved.

from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge, CvBridgeError
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import Image
from timing.timer import Timer

from camera_preprocessing.transformation.birds_eyed_view import Birdseye
from camera_preprocessing.transformation.calibration import Calibration
from camera_preprocessing.transformation.coordinate_transform import (
    CoordinateTransform,
)
from camera_preprocessing.transformation.distortion import Distortion
from camera_preprocessing.transformation.region_birdseye import (
    BEV_TOPIC_PREFIX,
    RegionBirdseye,
)

if TYPE_CHECKING:
    from rclpy.publisher import Publisher


class CameraPreprocessing(Node):
    """CameraPreprocessing Node."""

    def __init__(self):
        """Initialize the CameraPreprocessing node."""
        super().__init__("camera_preprocessing")

        # Get parameters from the parameter server
        package_share_path = get_package_share_directory("camera_preprocessing")
        self.get_logger().info(f"Package Path: {package_share_path}")

        debug = self.declare_parameter("debug", value=False).value
        calibration_file = str(
            Path(package_share_path)
            / self.declare_parameter("calibration_file", "config/calib.bin").value
        )
        config_file = str(
            Path(package_share_path)
            / self.declare_parameter("config_file", "config/config.yaml").value
        )
        calibration_images_path = str(
            Path(package_share_path)
            / self.declare_parameter(
                "calibration_images_path", "img/calib/Neue_3MP_Kamera/"
            ).value
        )
        position_calib_img_path = str(
            Path(package_share_path)
            / self.declare_parameter(
                "position_calib_img_path", "img/position/chessboard.png"
            ).value
        )
        num_skip_frames = self.declare_parameter("num_skip_frames", 1).value
        subscriber_topic = self.declare_parameter(
            "image_topic", "/camera/image_raw"
        ).value
        undistorted_publisher_topic = self.declare_parameter(
            "undistorted_publisher_topic", "/camera/undistorted"
        ).value
        birds_eye_publisher_topic = self.declare_parameter(
            "birds_eye_publisher_topic", "/camera/birds_eye"
        ).value

        # Additional bird's eye views of freely chosen regions on the ground, each
        # one on its own topic <bev_view_topic_prefix>/x_.._../y_.._../r.. (see
        # region_birdseye.bev_topic_name). Names of the views in bev_views, their
        # region in bev_view.<name>.x_range, .y_range and .resolution.
        self.bev_view_topic_prefix = self.declare_parameter(
            "bev_view_topic_prefix", BEV_TOPIC_PREFIX
        ).value
        bev_view_names = self.declare_parameter(
            "bev_views", Parameter.Type.STRING_ARRAY
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

        # Additional bird's eye views, use the same calibration as the regular one
        self.coordinate_transform = CoordinateTransform(
            calib=self.calibration, debug=debug
        )
        self.bev_views: list[tuple[RegionBirdseye, Publisher]] = []
        for name in bev_view_names or []:
            if name:  # [""]: no additional view
                self.add_bev_view(**self.bev_view_config(name))

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

    def bev_view_config(self, name: str) -> dict[str, list[float] | float]:
        """Declare and read the parameters of an additional bird's eye view.

        Args:
            name: Name of the view in bev_views.

        Returns:
            Keyword arguments of add_bev_view (x_range, y_range, resolution).

        Raises:
            ValueError: If a parameter of the view is missing.
        """
        # Dynamic typing: integers in the yaml (e.g. [-150, 2000]) are accepted
        number = ParameterDescriptor(dynamic_typing=True)
        config = {}
        for key in ("x_range", "y_range", "resolution"):
            value = self.declare_parameter(f"bev_view.{name}.{key}", None, number).value
            if value is None:
                raise ValueError(
                    f"Parameter bev_view.{name}.{key} of the bird's eye view '{name}' "
                    "is missing"
                )
            config[key] = value
        return config

    def add_bev_view(
        self,
        x_range: tuple[float, float] | list[float],
        y_range: tuple[float, float] | list[float],
        resolution: float,
    ) -> str:
        """Publish an additional bird's eye view of a rectangle on the ground.

        The view is computed from the undistorted image of every frame and published
        on <bev_view_topic_prefix>/x_<start>_<end>/y_<start>_<end>/r<resolution>
        (mm, "-" written as "m", "." as "p"), e.g.
        add_bev_view((-150, 2000), (-1200, 1200), 3.0) publishes
        /camera/image/bev/x_m150_2000/y_m1200_1200/r3.

        Args:
            x_range: Region on the ground in driving direction (min, max) in mm.
            y_range: Region on the ground to the left (min, max) in mm.
            resolution: Size of a pixel in mm.

        Returns:
            The topic of the view.
        """
        view = RegionBirdseye(
            self.coordinate_transform,
            x_range=(float(x_range[0]), float(x_range[1])),
            y_range=(float(y_range[0]), float(y_range[1])),
            resolution=float(resolution),
        )
        topic = view.topic_name(self.bev_view_topic_prefix)
        if any(publisher.topic_name == topic for _, publisher in self.bev_views):
            self.get_logger().warning(f"Bird's eye view {topic} exists already")
            return topic
        self.bev_views.append((view, self.create_publisher(Image, topic, 10)))
        self.get_logger().info(
            f"Bird's eye view x={list(view.x_range)} mm, y={list(view.y_range)} mm, "
            f"{view.resolution:g} mm/px -> {view.size[0]}x{view.size[1]} px on {topic}"
        )
        return topic

    def adjust_brightness(
        self, image: np.ndarray, target_brightness: float = 127
    ) -> np.ndarray:
        """Adjust the brightness of the given image.

        Args:
            image: Image to adjust the brightness of.
            target_brightness: Target brightness value [0,255] (default: {127})

        Returns:
            Image with adjusted brightness.
        """
        avg_brightness = np.mean(image)
        adjustment_factor = target_brightness / avg_brightness
        adjusted_image = cv2.convertScaleAbs(image, alpha=adjustment_factor, beta=0)
        return adjusted_image

    def on_raw_image(self, image: Image) -> None:
        """Callback function for the raw image subscriber.

        Args:
            image: Raw image from the camera.
        """
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

        img_bev_views = []
        for view, _ in self.bev_views:
            with Timer(name=f"bev {view.size[0]}x{view.size[1]}", filter_strength=40):
                img_bev_views.append(view.warp(img_undistorted))

        with Timer(name="publish", filter_strength=40):
            self.publish_image_undistorted(img_undistorted)
            self.publish_image_birds_eye(img_birds_eye)
            for (_, publisher), img_view in zip(
                self.bev_views, img_bev_views, strict=True
            ):
                publisher.publish(
                    self.bridge.cv2_to_imgmsg(img_view, "8UC1", header=image.header)
                )

        Timer().print()

    def publish_image_undistorted(self, image_cv2: np.ndarray) -> None:
        """Publish the undistorted image.

        Args:
            image_cv2: Image to publish in OpenCV format.
        """
        image_msg = self.bridge.cv2_to_imgmsg(image_cv2, "8UC1")
        self.undistorted_image_publisher.publish(image_msg)

    def publish_image_birds_eye(self, image_cv2: np.ndarray) -> None:
        """Publish the bird's eye view image.

        Args:
            image_cv2: Image to publish in OpenCV format.
        """
        image_msg = self.bridge.cv2_to_imgmsg(image_cv2, "8UC1")
        self.birds_eye_publisher.publish(image_msg)


def main(args: list[str] | None = None) -> None:
    """Main function to start the CameraPreprocessing node.

    Args:
        args: Launch Arguments (default: {None})
    """
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
