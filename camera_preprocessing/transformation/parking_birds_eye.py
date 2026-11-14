# Copyright (c) 2026 Smart Rollerz e.V. All rights reserved.
"""Bird's eye view for parking with a configurable rectangle on the ground.

The regular bird's eye view (Birdseye, /camera/image/bev) only covers about
x = -120..1280 mm and y = -920..1000 mm, so many parking spots are cut off at its
border although they are visible in the camera image. This view is computed from the
undistorted image with the same calibration for a freely chosen rectangle on the
ground. It does not change the regular bird's eye view, which other nodes rely on.

Image axes: up = +x (forward), right = -y (right), like the regular bird's eye view.
The image is a linear map of the ground (vehicle frame of the calibration, mm):

    x = x_range[1] - v * resolution
    y = y_range[1] - u * resolution

with the pixel (u, v) = (column, row). Pixels outside of the camera's field of view
are black.

The optional correction (a homography from the nominal frame of the calibration to
the real vehicle frame, e.g. fitted by the parking package) only changes the
conversion between pixels and mm, not the image.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

from camera_preprocessing.transformation.coordinate_transform import Unit

if TYPE_CHECKING:
    from camera_preprocessing.transformation.coordinate_transform import (
        CoordinateTransform,
    )


class ParkingBirdseye:
    """Bird's eye view of a rectangle on the ground (vehicle frame, mm)."""

    def __init__(
        self,
        transform: CoordinateTransform,
        x_range: tuple[float, float] = (-150.0, 2000.0),
        y_range: tuple[float, float] = (-1200.0, 1200.0),
        resolution: float = 3.0,
        correction: np.ndarray | None = None,
    ):
        """Initialize the ParkingBirdseye.

        Args:
            transform: Coordinate transformation of the calibration.
            x_range: Region on the ground in driving direction (min, max) in mm.
            y_range: Region on the ground to the left (min, max) in mm.
            resolution: Size of a pixel in mm.
            correction: Homography (3, 3) from the nominal to the real vehicle
                frame (mm), None: no correction.

        Raises:
            ValueError: If the region or the resolution is empty.
            RuntimeError: If the camera is not calibrated.
        """
        if x_range[1] <= x_range[0] or y_range[1] <= y_range[0]:
            raise ValueError(f"Invalid region x={x_range}, y={y_range}")
        if resolution <= 0:
            raise ValueError(f"Invalid resolution {resolution}")

        self.x_range = (float(x_range[0]), float(x_range[1]))
        self.y_range = (float(y_range[0]), float(y_range[1]))
        self.resolution = float(resolution)
        self.correction = (
            np.eye(3) if correction is None else np.asarray(correction, dtype=float)
        )
        self.inverse_correction = np.linalg.inv(self.correction)
        self.size = (
            round((self.y_range[1] - self.y_range[0]) / self.resolution),  # width
            round((self.x_range[1] - self.x_range[0]) / self.resolution),  # height
        )

        # Homography from the pixels of this view to the undistorted image, fitted
        # to a grid of ground points (exact for a plane, the grid averages rounding)
        u, v = np.meshgrid(
            np.linspace(0, self.size[0], 9), np.linspace(0, self.size[1], 9)
        )
        pixels = np.stack([u.ravel(), v.ravel()], axis=1)
        world = self.pixel_to_nominal(pixels)
        camera = transform.world_to_camera(
            np.hstack([world, np.zeros((len(world), 1))]), Unit.MILLIMETERS
        )
        if camera is None:
            raise RuntimeError("Camera is not calibrated")
        self.homography, _ = cv2.findHomography(
            pixels.astype(np.float32),
            np.asarray(camera, dtype=np.float32).reshape(-1, 2),
        )
        if self.homography is None:
            raise RuntimeError("Homography of the parking bird's eye view failed")

    @classmethod
    def from_config(
        cls, transform: CoordinateTransform, config: dict
    ) -> ParkingBirdseye:
        """Create the view from a config dict.

        Args:
            transform: Coordinate transformation of the calibration.
            config: Dict with x_range, y_range, resolution and optionally
                ground_correction.

        Returns:
            The view.
        """
        return cls(
            transform,
            x_range=tuple(config["x_range"]),
            y_range=tuple(config["y_range"]),
            resolution=float(config["resolution"]),
            correction=config.get("ground_correction"),
        )

    def warp(self, undistorted: np.ndarray) -> np.ndarray:
        """Transform the undistorted image into this bird's eye view.

        Args:
            undistorted: Undistorted image (target_size of the calibration).

        Returns:
            The bird's eye view, black outside of the camera's field of view.
        """
        return cv2.warpPerspective(
            undistorted,
            self.homography,
            self.size,
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    def pixel_to_world(self, pixels: np.ndarray) -> np.ndarray:
        """Convert pixels of this view to the vehicle frame (with the correction).

        Args:
            pixels: Points in px with shape (N, 2).

        Returns:
            Points in mm with shape (N, 2).
        """
        return apply_homography(self.correction, self.pixel_to_nominal(pixels))

    def world_to_pixel(self, points: np.ndarray) -> np.ndarray:
        """Convert points of the vehicle frame (with the correction) to pixels.

        Args:
            points: Points in mm with shape (N, 2) or (N, 3).

        Returns:
            Points in px with shape (N, 2).
        """
        points = np.asarray(points, dtype=float).reshape(len(points), -1)[:, :2]
        return self.nominal_to_pixel(apply_homography(self.inverse_correction, points))

    def pixel_to_nominal(self, pixels: np.ndarray) -> np.ndarray:
        """Convert pixels to the nominal frame of the calibration (no correction).

        Args:
            pixels: Points in px with shape (N, 2).

        Returns:
            Points in mm with shape (N, 2).
        """
        pixels = np.asarray(pixels, dtype=float).reshape(-1, 2)
        x = self.x_range[1] - pixels[:, 1] * self.resolution
        y = self.y_range[1] - pixels[:, 0] * self.resolution
        return np.stack([x, y], axis=1)

    def nominal_to_pixel(self, points: np.ndarray) -> np.ndarray:
        """Convert points of the nominal frame of the calibration to pixels.

        Args:
            points: Points in mm with shape (N, 2) or (N, 3).

        Returns:
            Points in px with shape (N, 2).
        """
        points = np.asarray(points, dtype=float).reshape(len(points), -1)
        u = (self.y_range[1] - points[:, 1]) / self.resolution
        v = (self.x_range[1] - points[:, 0]) / self.resolution
        return np.stack([u, v], axis=1)


def apply_homography(homography: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a homography (3, 3) to points.

    Args:
        homography: Homography (3, 3).
        points: Points with shape (N, 2).

    Returns:
        Transformed points with shape (N, 2).
    """
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    result = np.hstack([points, np.ones((len(points), 1))]) @ homography.T
    return result[:, :2] / result[:, 2:]
