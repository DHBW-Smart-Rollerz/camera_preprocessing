#!/usr/bin/env python3
# Copyright (c) 2024 Smart Rollerz e.V. All rights reserved.
import cv2
from rclpy.logging import get_logger

from camera_preprocessing.transformation.calibration import Calibration
from camera_preprocessing.transformation.distortion import Distortion

DEBUG = False


class BirdseyedviewTransformation:
    """Deprecated class for bird's eye view transformation."""

    def __init__(self, **kwargs):
        """Initialize the BirdseyedviewTransformation class."""
        self.logger = get_logger("birdseyedview_transformation_logger")
        self.logger.warn("Deprecated class. Use Birdseye instead.")
        self.calibration = Calibration(debug=DEBUG)
        self.calibration.setup()
        self.bird = Birdseye(
            self.calibration, Distortion(self.calibration, DEBUG), DEBUG
        )

    def undistorted_to_birdseyedview(self, img):
        """
        Transform undistorted image to birds eye view. Deprecated, use Birdseye.transform_img() instead.

        Args:
            img (np.array): Undistorted image.

        Returns:
            np.array: Transformed image.
        """
        self.logger.warn(
            "Using deprecated method. Use Birdseye.transform_img() instead."
        )
        return self.bird.transform_img(img)


class Birdseye:
    """Class for bird's eye view transformation."""

    def __init__(
        self, calibration: Calibration, distortion: Distortion, debug: bool = DEBUG
    ) -> None:
        """
        Initialize the Birdseye class.

        Arguments:
            calibration -- Calibration object.
            distortion -- Distortion object.

        Keyword Arguments:
            debug -- Enable debug (default: {DEBUG})
        """
        self._distortion = distortion
        self._calibration = calibration
        self._debug = debug

    def recalibrate(self):
        """Recalibrate the camera."""
        if not self.calibration.all_calibrated:
            self.calibration.setup()
        if self.calibration.all_calibrated:
            self.logger.warn("Camera is already calibrated.")

        self._calibration.calibrate_birds_eye()

    def transform_img(self, img):
        """
        Transform the given image to birds eye view.

        Arguments:
            img -- Image to transform.

        Returns:
            Transformed image.
        """
        target_size = self._calibration._calib_birds_eye["target_size"]
        target_size = (int(target_size[0]), int(target_size[1]))
        if img.shape[0] != target_size[1] or img.shape[0] != target_size[1]:
            self.logger.warn("Resizing image to target size.")
            img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)

        # Transform image to birds eye view
        transformed = cv2.warpPerspective(
            img,
            self._calibration._calib_birds_eye["iTb"],
            target_size,
            flags=cv2.INTER_LINEAR,
        )

        return transformed
