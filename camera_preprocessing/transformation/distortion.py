#!/usr/bin/env python3
# Copyright (c) 2024 Smart Rollerz e.V. All rights reserved.


import cv2
import numpy as np
from rclpy.logging import get_logger

from camera_preprocessing.transformation.calibration import Calibration

DEBUG = False


class Distortion:
    """Class for camera distortion correction."""

    def __init__(self, calibration: Calibration, debug=DEBUG):
        self.calibration = calibration
        self.logger = get_logger("distortion_logger")
        self.precompute_undistortion_map()
        self.logger.info("Distortion class initialized")

    def recalibrate(self):
        """
        Recalibrate the camera.
        """
        if not self.calibration.all_calibrated:
            self.calibration.setup()
        if self.calibration.all_calibrated:
            self.logger.warn("Camera is already calibrated.")

        self.calibration.calibrate_distortion()
        self.precompute_undistortion_map()

    def undistort(self, img: np.ndarray) -> np.ndarray:
        """Undistorts the image. Deprecated, use undistort_image instead.

        Args:
            img (np.ndarray): The image to undistort.

        Returns:
            np.ndarray: The undistorted image.
        """
        self.logger.warn("Using deprecated method. Use undistort_image instead.")
        return self.undistort_image(img)

    def undistort_image(self, src):
        """
        Correct distortion in the input image.

        Args:
            src (numpy.ndarray): Input image.

        Returns:
            numpy.ndarray: Undistorted image.
        """
        if self.map1 is None or self.map2 is None:
            raise RuntimeError("Camera is not calibrated.")
        dst = cv2.remap(src, self.map1, self.map2, cv2.INTER_AREA)
        return dst

    def precompute_undistortion_map(self):
        """
        Precompute the undistortion map.

        Args:
            image_size (tuple): Size of the input image.
        """
        if (
            not self.calibration.is_distortion_calibrated
            or not self.calibration.is_config_loaded
        ):
            raise RuntimeError("Camera is not calibrated.")

        image_size = self.calibration.config["target_size"]
        internal_camera_matrix = self.calibration._calib_distortion[
            "internal_camera_matrix"
        ]
        new_camera_matrix = self.calibration._calib_distortion["new_camera_matrix"]
        dist_coeffs = self.calibration._calib_distortion["dist_coeffs"]

        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            new_camera_matrix, dist_coeffs, None, None, image_size, cv2.CV_16SC2
        )
