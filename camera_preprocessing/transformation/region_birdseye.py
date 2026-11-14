# Copyright (c) 2026 Smart Rollerz e.V. All rights reserved.
"""Bird's eye view of a freely chosen rectangle on the ground.

The regular bird's eye view (Birdseye, /camera/image/bev) covers a fixed region of
about x = -120..1280 mm and y = -920..1000 mm. A RegionBirdseye is computed from the
undistorted image with the same calibration for any rectangle on the ground and any
resolution, e.g. a larger region for parking. It does not change the regular bird's
eye view, which other nodes rely on.

camera_preprocessing publishes every configured view on its own topic, whose name
contains the region and the resolution (bev_topic_name), e.g.
/camera/image/bev/x_m150_2000/y_m1200_1200/r3. A subscriber computes the topic from
the same values, so the image and its conversion between px and mm always match.

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

# Topic of the regular bird's eye view, the region views are published below it
BEV_TOPIC_PREFIX = "/camera/image/bev"

if TYPE_CHECKING:
    from camera_preprocessing.transformation.coordinate_transform import (
        CoordinateTransform,
    )


class RegionBirdseye:
    """Bird's eye view of a rectangle on the ground (vehicle frame, mm)."""

    def __init__(
        self,
        transform: CoordinateTransform,
        x_range: tuple[float, float] = (-150.0, 2000.0),
        y_range: tuple[float, float] = (-1200.0, 1200.0),
        resolution: float = 3.0,
        correction: np.ndarray | None = None,
    ):
        """Initialize the RegionBirdseye.

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
            raise RuntimeError("Homography of the bird's eye view failed")

    @classmethod
    def from_config(
        cls, transform: CoordinateTransform, config: dict
    ) -> RegionBirdseye:
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

    def topic_name(self, prefix: str = BEV_TOPIC_PREFIX) -> str:
        """Return the topic of this view (see bev_topic_name).

        Args:
            prefix: Topic of the regular bird's eye view.

        Returns:
            The topic name.
        """
        return bev_topic_name(self.x_range, self.y_range, self.resolution, prefix)

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


def bev_topic_name(
    x_range: tuple[float, float] | list[float],
    y_range: tuple[float, float] | list[float],
    resolution: float,
    prefix: str = BEV_TOPIC_PREFIX,
) -> str:
    """Return the topic of the bird's eye view of a region on the ground.

    The name contains the region and the resolution:
    <prefix>/x_<start>_<end>/y_<start>_<end>/r<resolution>, all in mm. ROS 2 topic
    names allow neither "-" nor "." and no token may start with a digit, so a minus
    is written as "m" and a decimal point as "p", e.g. x_range (-150, 2000), y_range
    (-1200, 1200) and 2.5 mm/px give /camera/image/bev/x_m150_2000/y_m1200_1200/r2p5.

    Args:
        x_range: Region on the ground in driving direction (min, max) in mm.
        y_range: Region on the ground to the left (min, max) in mm.
        resolution: Size of a pixel in mm.
        prefix: Topic of the regular bird's eye view.

    Returns:
        The topic name.
    """
    x = f"x_{_topic_number(x_range[0])}_{_topic_number(x_range[1])}"
    y = f"y_{_topic_number(y_range[0])}_{_topic_number(y_range[1])}"
    return f"{prefix.rstrip('/')}/{x}/{y}/r{_topic_number(resolution)}"


def _topic_number(value: float) -> str:
    """Write a number for a topic name: "-" -> "m", "." -> "p" (up to 3 decimals).

    Args:
        value: The number.

    Returns:
        The number as part of a topic name, e.g. -150 -> "m150", 2.5 -> "2p5".
    """
    text = f"{abs(float(value)):.3f}".rstrip("0").rstrip(".")
    sign = "m" if round(float(value), 3) < 0 else ""
    return sign + text.replace(".", "p")
