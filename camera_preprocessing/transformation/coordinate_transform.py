#!/usr/bin/env python3
# Copyright (c) 2024 Smart Rollerz e.V. All rights reserved.

import enum

import numpy as np
import rclpy
from rclpy.logging import get_logger

from camera_preprocessing.transformation.calibration import Calibration
from camera_preprocessing.transformation.helpers import Helpers


class Unit(enum.Enum):
    METERS = ("m",)
    MILLIMETERS = ("mm",)
    PIXELS = "px"


class CoordinateTransform:
    def __init__(
        self, *, calib: Calibration = None, debug: bool = False, **kwargs
    ) -> None:
        self.logger = get_logger("coordinate_transform_logger")
        if kwargs.get("manager", None) is not None:
            self.logger.warn(
                "Using deprecated method. Use the new Calibration class instead."
            )

        if calib is None:
            calib = Calibration(debug=debug)
        self._calib = calib
        self._debug = debug

        try:
            self._calib.setup()
            if not self._calib.all_calibrated:
                self.logger.error("Not calibrated. Please calibrate camera.")
        except ValueError as e:
            self.logger.warn(e.args[0])

    ##################################
    ##### Transformation methods #####
    ##################################

    def camera_to_world(
        self, points: np.ndarray, unit: Unit = Unit.MILLIMETERS, Z_w=0
    ) -> np.ndarray:
        """Transforms the camera coordinates to world coordinates.

        Args:
            points (np.ndarray): Camera coordinates [x, y].
            unit (Unit, optional): The unit of the output data. Defaults to Unit.MILLIMETERS.
            Z_w (float, optional): The z-coordinate of the world. Defaults to 0.

        Raises:
            ValueError: If no camera configuration is found.
            ValueError: Unit is not supported.

        Returns:
            np.ndarray: World coordinates.
        """
        if not self._calib.all_calibrated:
            self.logger.error("No camera configuration found.")
            return None

        points = np.array(points).reshape(-1, 2)

        # Convert the height to meters
        if unit == Unit.METERS:
            Z_w = Z_w
        elif unit == Unit.MILLIMETERS:
            Z_w = Z_w / 1000

        points = np.asarray(points).reshape(-1, 2)
        res: np.ndarray = Helpers.image_to_camera(
            points,
            self._calib._calib_distortion["new_camera_matrix_inv"],
            self._calib.config["focal_length"][0] / 1000,
        )
        res = Helpers.camera_to_world(res, self._calib._calib_extrinsic["cTw"], Z_w=Z_w)

        if unit == Unit.METERS:
            return res
        elif unit == Unit.MILLIMETERS:
            return (res * 1000).round().astype(np.int32)
        else:
            raise ValueError(
                f"Error: {unit.value[0]} is not supported for this transformation."
            )

    def world_to_camera(
        self, points: np.ndarray, input_unit: Unit = Unit.MILLIMETERS, **kwargs
    ) -> np.ndarray:
        """Transforms the world coordinates to camera coordinates.

        Args:
            points (np.ndarray): World coordinates [x, y, 0].
            input_unit (Unit, optional): The unit of the input data. Defaults to Unit.MILLIMETERS.

        Raises:
            ValueError: If no camera configuration is found.
            ValueError: Unit is not supported.

        Returns:
            np.ndarray: Camera coordinates.
        """
        if not self._calib.all_calibrated:
            self.logger.error("No camera configuration found.")
            return None

        input_pts = np.array([])
        points = np.array(points).reshape(-1, 3)

        if input_unit == Unit.METERS:
            input_pts = points
        elif input_unit == Unit.MILLIMETERS:
            input_pts = (points / 1000).astype(np.float64)
        else:
            raise ValueError(
                f"Error: {input_unit.value[0]} is not supported for this transformation."
            )

        res = Helpers.world_to_camera(input_pts, self._calib._calib_extrinsic["wTc"])
        res = Helpers.camera_to_image(
            res, self._calib._calib_distortion["new_camera_matrix"]
        )
        res = np.array(res).reshape(-1, 2)

        return res

    def bird_to_camera(self, points: np.ndarray) -> np.ndarray:
        """Transforms the bird coordinates to camera coordinates.

        Args:
            points (np.ndarray): Bird coordinates [x, y].

        Returns:
            np.ndarray: Camera coordinates.
        """
        if not self._calib.all_calibrated:
            self.logger.error("No camera configuration found.")
            return None

        points = np.array(points).reshape(-1, 2)

        res = Helpers.bird_to_image(points, self._calib._calib_birds_eye["bTi"])
        res = np.array(res).reshape(-1, 2)

        return res

    def camera_to_bird(self, points: np.ndarray) -> np.ndarray:
        """Transforms the camera coordinates to bird coordinates.

        Args:
            points (np.ndarray): Camera coordinates [x, y].

        Returns:
            np.ndarray: Bird coordinates.
        """
        if not self._calib.all_calibrated:
            self.logger.error("No camera configuration found.")
            return None

        points = np.array(points).reshape(-1, 2)

        res = Helpers.image_to_bird(points, self._calib._calib_birds_eye["iTb"])
        res = np.array(res).reshape(-1, 2)

        return res

    def bird_to_world(
        self, points: np.ndarray, unit: Unit = Unit.MILLIMETERS
    ) -> np.ndarray:
        """Transforms the bird coordinates to world coordinates.

        Args:
            points (np.ndarray): Bird coordinates [x, y].
            unit (Unit, optional): The unit of the output data. Defaults to Unit.MILLIMETERS.

        Returns:
            np.ndarray: World coordinates.
        """
        points = np.array(points).reshape(-1, 2)
        return self.camera_to_world(self.bird_to_camera(points), unit)

    def world_to_bird(
        self, points: np.ndarray, input_unit: Unit = Unit.MILLIMETERS, **kwargs
    ) -> np.ndarray:
        """Transforms the world coordinates to bird coordinates.

        Args:
            points (np.ndarray): World coordinates [x, y, 0].
            input_unit (Unit, optional): The unit of the input data. Defaults to Unit.MILLIMETERS.

        Returns:
            np.ndarray: Bird coordinates.
        """
        points = np.array(points).reshape(-1, 3)
        return self.camera_to_bird(self.world_to_camera(points, input_unit, **kwargs))


if __name__ == "__main__":
    rclpy.init()

    world_error = 0
    img_error = 0
    bird_error = 0
    # Test the CoordinateTransform class
    transformation = CoordinateTransform(calib=None, debug=False)
    points = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    world_points_org = np.array(
        [  # m
            [-0.175, 0.62, 0],
            [-0.125, 0.62, 0],
            [-0.075, 0.62, 0],
            [-0.025, 0.62, 0],
            [0.025, 0.62, 0],
            [0.075, 0.62, 0],
            [0.125, 0.62, 0],
            [0.175, 0.62, 0],
            [-0.175, 0.57, 0],
            [-0.125, 0.57, 0],
            [-0.075, 0.57, 0],
            [-0.025, 0.57, 0],
            [0.025, 0.57, 0],
            [0.075, 0.57, 0],
            [0.125, 0.57, 0],
            [0.175, 0.57, 0],
            [-0.175, 0.52, 0],
            [-0.125, 0.52, 0],
            [-0.075, 0.52, 0],
            [-0.025, 0.52, 0],
            [0.025, 0.52, 0],
            [0.075, 0.52, 0],
            [0.125, 0.52, 0],
            [0.175, 0.52, 0],
            [-0.175, 0.47, 0],
            [-0.125, 0.47, 0],
            [-0.075, 0.47, 0],
            [-0.025, 0.47, 0],
            [0.025, 0.47, 0],
            [0.075, 0.47, 0],
            [0.125, 0.47, 0],
            [0.175, 0.47, 0],
            [-0.175, 0.42, 0],
            [-0.125, 0.42, 0],
            [-0.075, 0.42, 0],
            [-0.025, 0.42, 0],
            [0.025, 0.42, 0],
            [0.075, 0.42, 0],
            [0.125, 0.42, 0],
            [0.175, 0.42, 0],
            [-0.175, 0.37, 0],
            [-0.125, 0.37, 0],
            [-0.075, 0.37, 0],
            [-0.025, 0.37, 0],
            [0.025, 0.37, 0],
            [0.075, 0.37, 0],
            [0.125, 0.37, 0],
            [0.175, 0.37, 0],
        ]
    )
    world_points = transformation.camera_to_world(points, Unit.METERS)
    img_points = transformation.world_to_camera(world_points, Unit.METERS)
    img_error = np.sum(np.abs(points - img_points)) / len(points)
    print("World points: ", world_points)
    print("Image points: ", img_points)

    print("Image error: ", img_error)

    img_points = transformation.world_to_camera(world_points_org, Unit.METERS)
    bird_points = transformation.camera_to_bird(points)
    img_points = transformation.bird_to_camera(bird_points)
    img_error = np.sum(np.abs(points - img_points)) / len(points)
    print("\nBird points: ", bird_points)
    print("Image points: ", img_points)

    print("Image error: ", img_error)

    world_points2 = transformation.bird_to_world(bird_points, Unit.METERS)
    bird_points2 = transformation.world_to_bird(world_points, Unit.METERS)
    world_error = np.sum(np.abs(world_points - world_points2)) / len(points)
    bird_error = np.sum(np.abs(bird_points - bird_points2)) / len(points)
    print("\nWorld points: ", world_points2)
    print("Bird points: ", bird_points2)

    print("\nWorld error: ", world_error)
    print("Bird error: ", bird_error)
