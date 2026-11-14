# Copyright (c) 2026 Smart Rollerz e.V. All rights reserved.
"""Tests of the region bird's eye view with a synthetic pinhole camera."""

import numpy as np
import pytest
from camera_preprocessing.transformation.region_birdseye import (
    RegionBirdseye,
    apply_homography,
    bev_topic_name,
)
from rclpy.validate_topic_name import validate_topic_name

IMAGE_SIZE = (800, 640)  # Width, height of the undistorted image


class PinholeTransform:
    """Pinhole camera 250 mm above the ground, tilted 30 deg down, looking forward."""

    def __init__(self):
        """Initialize the camera."""
        self.K = np.array([[400.0, 0.0, 400.0], [0.0, 400.0, 320.0], [0.0, 0.0, 1.0]])
        pitch = np.deg2rad(30.0)
        # Vehicle frame: x forward, y left, z up. Camera frame: x right, y down,
        # z along the optical axis.
        forward = np.array([np.cos(pitch), 0.0, -np.sin(pitch)])
        right = np.array([0.0, -1.0, 0.0])
        down = np.cross(forward, right)
        self.R = np.stack([right, down, forward])  # Vehicle -> camera
        self.position = np.array([0.0, 0.0, 250.0])

    def world_to_camera(self, points, unit=None):
        """Project points of the vehicle frame (mm) into the image (px)."""
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        camera = (points - self.position) @ self.R.T
        image = camera @ self.K.T
        return image[:, :2] / image[:, 2:]


def ground_image(transform):
    """Render a checkerboard on the ground (100 mm squares) into the camera image."""
    u, v = np.meshgrid(np.arange(IMAGE_SIZE[0]), np.arange(IMAGE_SIZE[1]))
    rays = np.stack([u, v, np.ones_like(u)], axis=-1).reshape(-1, 3).astype(float)
    rays = rays @ np.linalg.inv(transform.K).T @ transform.R  # Vehicle frame
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = -transform.position[2] / rays[:, 2]
    ground = transform.position + rays * scale[:, None]
    squares = (np.floor(ground[:, 0] / 100) + np.floor(ground[:, 1] / 100)) % 2
    image = np.where(squares == 0, 60, 200).astype(np.uint8)
    image[~(scale > 0)] = 0  # Above the horizon
    return image.reshape(IMAGE_SIZE[1], IMAGE_SIZE[0])


@pytest.fixture
def transform():
    """Return the synthetic camera."""
    return PinholeTransform()


@pytest.fixture
def birdseye(transform):
    """Return the bird's eye view with the default region."""
    return RegionBirdseye(transform)


def test_size(birdseye):
    """2400 mm x 2150 mm at 3 mm/px -> 800 x 717 px."""
    assert birdseye.size == (800, 717)


def test_axes(birdseye):
    """Image up = +x (forward), image right = -y (right)."""
    np.testing.assert_allclose(birdseye.pixel_to_nominal([[0, 0]]), [[2000, 1200]])
    np.testing.assert_allclose(birdseye.pixel_to_nominal([[800, 717]]), [[-151, -1200]])
    # The ground point straight ahead (y = 0) is in the middle column
    assert birdseye.nominal_to_pixel([[1000.0, 0.0]])[0, 0] == pytest.approx(400)


def test_pixel_world_round_trip(birdseye):
    """Pixel -> mm -> pixel is the identity (with and without the correction)."""
    pixels = np.random.default_rng(0).uniform(0, 800, size=(50, 2))
    np.testing.assert_allclose(
        birdseye.nominal_to_pixel(birdseye.pixel_to_nominal(pixels)), pixels
    )
    np.testing.assert_allclose(
        birdseye.world_to_pixel(birdseye.pixel_to_world(pixels)), pixels
    )


def test_correction(transform):
    """The correction changes the conversion to mm, not the image."""
    correction = np.array([[0.8, -0.05, 88.0], [0.0, 0.82, 0.0], [8e-5, 1.5e-4, 1.0]])
    plain = RegionBirdseye(transform)
    corrected = RegionBirdseye(transform, correction=correction)
    np.testing.assert_allclose(corrected.homography, plain.homography)
    pixels = np.array([[100.0, 200.0], [400.0, 600.0]])
    np.testing.assert_allclose(
        corrected.pixel_to_world(pixels),
        apply_homography(correction, plain.pixel_to_world(pixels)),
    )


def test_homography_matches_projection(transform, birdseye):
    """Every pixel of the view shows the ground point given by pixel_to_nominal."""
    pixels = np.random.default_rng(1).uniform(0, birdseye.size, size=(200, 2))
    world = birdseye.pixel_to_nominal(pixels)
    expected = transform.world_to_camera(np.hstack([world, np.zeros((200, 1))]))
    np.testing.assert_allclose(
        apply_homography(birdseye.homography, pixels), expected, atol=1e-2
    )


def test_warp_shows_ground(transform, birdseye):
    """The warped checkerboard is axis aligned with 100 mm = 33.3 px squares."""
    view = birdseye.warp(ground_image(transform))
    assert view.shape == (birdseye.size[1], birdseye.size[0])
    assert view.dtype == np.uint8

    # Compare with the checkerboard rendered directly in the view, away from the
    # square borders (far away, a camera pixel covers several cm of the ground) and
    # the black area outside of the image. A wrong axis or offset gives ~50 %.
    u, v = np.meshgrid(np.arange(view.shape[1]), np.arange(view.shape[0]))
    world = birdseye.pixel_to_nominal(np.stack([u.ravel(), v.ravel()], axis=1))
    expected = np.where(
        (np.floor(world[:, 0] / 100) + np.floor(world[:, 1] / 100)) % 2 == 0, 60, 200
    ).reshape(view.shape)
    distance = np.abs(world - np.round(world / 100) * 100).min(axis=1)
    valid = (distance.reshape(view.shape) > 25) & (view > 0)
    assert valid.sum() > 0.1 * view.size
    error = np.abs(view[valid].astype(int) - expected[valid])
    assert np.mean(error < 30) > 0.995


def test_invalid_region(transform):
    """An empty region is rejected."""
    with pytest.raises(ValueError, match="Invalid region"):
        RegionBirdseye(transform, x_range=(100.0, 100.0))
    with pytest.raises(ValueError, match="Invalid resolution"):
        RegionBirdseye(transform, resolution=0.0)


@pytest.mark.parametrize(
    ("x_range", "y_range", "resolution", "expected"),
    [
        (
            (-150, 2000),
            (-1200, 1200),
            3.0,
            "/camera/image/bev/x_m150_2000/y_m1200_1200/r3",
        ),
        (
            [-500.0, 2000.0],
            [-1200.0, 1200.0],
            2.5,
            "/camera/image/bev/x_m500_2000/y_m1200_1200/r2p5",
        ),
        ((0, 1000.25), (-0.5, 0.0), 10, "/camera/image/bev/x_0_1000p25/y_m0p5_0/r10"),
    ],
)
def test_topic_name(x_range, y_range, resolution, expected):
    """The topic contains the region and the resolution and is a valid ROS name."""
    topic = bev_topic_name(x_range, y_range, resolution)
    assert topic == expected
    validate_topic_name(topic)  # Raises if invalid


def test_topic_name_prefix(birdseye):
    """The view gives the topic of its region, the prefix is configurable."""
    assert birdseye.topic_name() == "/camera/image/bev/x_m150_2000/y_m1200_1200/r3"
    assert birdseye.topic_name("/sim/bev/") == "/sim/bev/x_m150_2000/y_m1200_1200/r3"
