#!/usr/bin/env python3
# Copyright (c) 2024 Smart Rollerz e.V. All rights reserved.

import os
from typing import Tuple

import cv2
import numpy as np
import yaml
from ament_index_python.packages import get_package_share_directory
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from rclpy.clock import Clock
from rclpy.logging import get_logger
from scipy.spatial.transform import Rotation

from camera_preprocessing.transformation.helpers import Helpers

share_dir = get_package_share_directory("camera_preprocessing")

DEFAULT_CALIBRATION_PATH = os.path.join(share_dir, "config", "calib.bin")
DEFAULT_CONFIG_PATH = os.path.join(share_dir, "config", "config.yaml")
DEFAULT_DISTORTION_CALIBRATION_DIR = os.path.join(
    share_dir, "img", "calib", "Neue_3MP_Kamera"
)
DEFAULT_EXTRINSIC_CHESSBOARD_FILE = os.path.join(
    share_dir, "img", "position", "chessboard.png"
)
DEBUG = False


class Calibration:
    """Abstract class for camera calibration."""

    def __init__(
        self,
        calibration_path=DEFAULT_CALIBRATION_PATH,
        config_path=DEFAULT_CONFIG_PATH,
        distortion_calibration_dir=DEFAULT_DISTORTION_CALIBRATION_DIR,
        extrinsic_chessboard_file=DEFAULT_EXTRINSIC_CHESSBOARD_FILE,
        debug=DEBUG,
    ):
        """
        Initialize the calibration class.

        Keyword Arguments:
            calibration_path -- Path to the calibration files (default: {DEFAULT_CALIBRATION_PATH})
            config_path -- Path to the configuration file (default: {DEFAULT_CONFIG_PATH})
            distortion_calibration_dir -- Directory containing the images to calibrate the distortion (default: {DEFAULT_DISTORTION_CALIBRATION_DIR})
            extrinsic_chessboard_file -- Chessboard image to calibrate extrinsic (default: {DEFAULT_EXTRINSIC_CHESSBOARD_FILE})
            debug -- Debug mode toggle (default: {DEBUG})
        """
        self.logger = get_logger("calibration_logger")

        # Is calibrated flags
        self.is_extrinsic_calibrated = False
        self.is_birds_eye_calibrated = False
        self.is_distortion_calibrated = False
        self.is_config_loaded = False

        # File paths
        self._calibration_path = calibration_path
        self._config_path = config_path
        self._distortion_calibration_dir = distortion_calibration_dir
        self._extrinsic_chessboard_file = extrinsic_chessboard_file
        self.debug = debug

        # Calibration data
        self._calib_distortion = {
            "internal_camera_matrix": None,
            "new_camera_matrix": None,
            "new_camera_matrix_inv": None,
            "dist_coeffs": None,
            "target_size": None,
        }

        self._calib_extrinsic = {
            "wTc": None,
            "cTw": None,
            "target_size": None,
        }

        self._calib_birds_eye = {
            # Transformation matrix from image to birds eye view (homogeneous)
            "iTb": None,
            "bTi": None,
            "src_points": None,
            "dst_points": None,
            "target_size": None,
        }

        self._timestamps = {
            "distortion": None,
            "extrinsic": None,
            "birds_eye": None,
        }

    def setup(self, force_recalibration: bool = False):
        """
        Setup the calibration.

        Keyword Arguments:
            force_recalibration -- Should the camera be forced to recalibrate (default: {False})
        """
        if not self.is_config_loaded:
            self.load_config(self._config_path)

        if (
            not self.is_extrinsic_calibrated
            or not self.is_birds_eye_calibrated
            or not self.is_distortion_calibrated
        ):
            self.load_calibration(self._calibration_path)

        if (
            not self.is_distortion_calibrated
            or not self.is_extrinsic_calibrated
            or not self.is_birds_eye_calibrated
            or force_recalibration
        ):
            self.calibrate_distortion(resize=False)
            self.calibrate_extrinsic()
            self.calibrate_birds_eye()
            self.calibrate_distortion(resize=True)

        # Check if all calibration parameters are set
        if self.all_calibrated:
            self.save_calibration(self._calibration_path)
            self.logger.info("Camera calibrated successfully.")
        elif not self.is_extrinsic_calibrated:
            self.logger.error("Failed to calibrate extrinsic parameters.")
        elif not self.is_birds_eye_calibrated:
            self.logger.error("Failed to calibrate bird's eye view parameters.")
        elif not self.is_distortion_calibrated:
            self.logger.error("Failed to calibrate distortion parameters.")

    def load_config(self, config_path: str):
        """
        Load configuration from file.

        Arguments:
            config_path -- Path to the configuration file.

        Raises:
            RuntimeError: If the configuration file could not be loaded.
        """
        try:
            with open(config_path, "r") as file:
                self.config = yaml.safe_load(file)
            self.is_config_loaded = True
        except Exception as e:
            self.logger.error("Failed to load configuration file: %s" % str(e))
            raise RuntimeError(str(e))

    def save_config(self, config_path: str):
        """
        Save configuration to file.

        Arguments:
            config_path -- Path to save the configuration file.

        Raises:
            RuntimeError: If the configuration file could not be saved.
        """
        try:
            with open(config_path, "w") as file:
                yaml.dump(self.config, file)
        except Exception as e:
            raise RuntimeError(str(e))

    def load_calibration(self, path):
        """
        Load calibration data from file.

        Arguments:
            path -- Path to the calibration file.
        """
        try:
            if os.path.exists(path):
                fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)

                # Load calibration data
                for name, calib in self._calib_mapping.items():
                    for key in calib.keys():
                        try:
                            calib[key] = fs.getNode(name + "-" + key).mat()
                        except Exception as e:
                            self.logger.warn(
                                "Failed to load calibration data, trying to load as scalar: %s"
                                % str(e)
                            )
                            calib[key] = fs.getNode(name + "-" + key).real()

                # Load timestamps
                for key in self._timestamps.keys():
                    self._timestamps[key] = fs.getNode("timestamps-" + key).real()

                fs.release()

                # Log calibration data
                self.logger.info(self.__str__())

                # Set calibration flags
                self.is_distortion_calibrated = self._set_calibration_flag(
                    self._calib_distortion
                )
                self.is_extrinsic_calibrated = self._set_calibration_flag(
                    self._calib_extrinsic
                )
                self.is_birds_eye_calibrated = self._set_calibration_flag(
                    self._calib_birds_eye
                )

                self.logger.info("Calibration data loaded.")
            else:
                self.logger.warn("Calibration file not found: %s" % path)
        except Exception as e:
            self.logger.warn("Failed to load calibration data: %s" % str(e))

    def save_calibration(self, path):
        """
        Save calibration data to file.

        Arguments:
            path -- Path to save the calibration file.
        """
        if os.path.exists(path):
            os.remove(path)
        fs = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)

        # Save calibration data
        for name, calib in self._calib_mapping.items():
            for key, value in calib.items():
                try:
                    fs.write(name + "-" + key, value)
                except Exception as e:
                    self.logger.warn(f"Failed to save calibration data: {key}-{str(e)}")

        # Save timestamps
        for key, value in self._timestamps.items():
            try:
                fs.write("timestamps-" + key, value)
            except Exception as e:
                self.logger.warn(f"Failed to save timestamp: {key}-{str(e)}")

        fs.release()
        self.logger.info("Calibration data saved to: %s" % path)

    def find_chessboard(self, img: np.ndarray) -> bool:
        """
        Find the chessboard corners in the image.

        Arguments:
            img -- Image.

        Returns:
            bool -- True if the chessboard corners were found, False otherwise.
        """
        if not self.is_config_loaded:
            self.logger.error("No configuration loaded.")
            return False

        board_size = (
            self.config["position_board"]["board_size"][0],
            self.config["position_board"]["board_size"][1],
        )
        img = self._prerpocess_extrinsic_img(img)
        ret, _ = self._get_checkerboard_corners(
            img, board_size, show_failure=False, resize=False
        )
        return ret

    ##############################
    # Calibration methods
    ##############################

    ###
    # Distortion calibration
    ###

    def calibrate_distortion(self, resize=True):
        """
        Calibrate the distortion parameters.

        Keyword Arguments:
            resize -- Should the image be resized to the configured output dimensions (default: {True})
        """
        if not self.is_config_loaded or self.internal_camera_matrix is None:
            self.logger.error("Configuration data not loaded.")
            return

        if not os.path.exists(self._distortion_calibration_dir):
            self.logger.error(
                "Calibration images directory not found: %s"
                % self._distortion_calibration_dir
            )
            return

        # Load calibration board configuration
        board_size = (
            self.config["calibration_board"]["board_size"][0],
            self.config["calibration_board"]["board_size"][1],
        )

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        obj = np.zeros((board_size[0] * board_size[1], 3), np.float32)
        obj[:, :2] = np.mgrid[0 : board_size[0], 0 : board_size[1]].T.reshape(-1, 2)

        object_points = []  # 3d point in real world space
        image_points = []  # 2d points in image plane.

        filenames = []
        for root, dirs, files in os.walk(self._distortion_calibration_dir):
            for file in files:
                if file.endswith((".png", ".jpg", ".jpeg")):
                    filenames.append(os.path.join(root, file))

        for filename in filenames:
            self.logger.info("Processing image: %s" % filename)
            img = cv2.imread(filename)
            if img is None:
                self.logger.warn("Failed to read image: %s" % filename)
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = self.downscale_image(gray) if resize else gray
            ret, corners = self._get_checkerboard_corners(
                gray, board_size, resize=False
            )
            if ret:
                image_points.append(corners)
                object_points.append(obj)
            else:
                self.logger.warn(
                    "Chessboard corners not detected in image: %s" % filename
                )

        if len(image_points) > 0:
            object_points = np.array(object_points)
            ret, new_camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
                object_points, image_points, gray.shape[::-1], None, None
            )

            self.logger.info("Distortion-Error: %f" % ret)

            # Save calibration data
            time_stamp = Clock().now().seconds_nanoseconds()[0]
            self._calib_distortion = {
                "internal_camera_matrix": self.internal_camera_matrix,
                "new_camera_matrix": new_camera_matrix,
                "new_camera_matrix_inv": np.linalg.inv(new_camera_matrix),
                "dist_coeffs": dist_coeffs,
                "error": ret,
                "target_size": np.array(self.config["target_size"]),
            }
            self._timestamps["distortion"] = time_stamp
            if self.debug:
                self._display_final_undistortion(
                    filenames, new_camera_matrix, dist_coeffs, resize=resize
                )

            self.is_distortion_calibrated = self._set_calibration_flag(
                self._calib_distortion
            )

            # Check calibration
            if self.is_distortion_calibrated:
                self.logger.info("Distortion calibration successful.")
            else:
                self.logger.error("Failed to calibrate distortion parameters.")
        else:
            self.logger.error("No calibration images found.")

        cv2.destroyAllWindows()

    def _display_final_undistortion(
        self, filenames, new_camera_matrix, dist_coeffs, resize=True
    ):
        """
        Display the final undistortion result.

        Arguments:
            filenames -- List of calibration images.
            new_camera_matrix -- Camera matrix.
            dist_coeffs -- Distortion coefficients.

        Keyword Arguments:
            resize -- Should the image be resized to the configured output dimensions (default: {True})
        """
        self.logger.info("Displaying calibration results.")
        img = cv2.imread(filenames[0])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = self.downscale_image(gray) if resize else gray
        undistorted = cv2.undistort(gray, new_camera_matrix, dist_coeffs)
        plt.figure(figsize=(10, 10))
        plt.subplot(1, 2, 1)
        plt.imshow(gray, cmap="gray")
        plt.title("Original Image")
        plt.subplot(1, 2, 2)
        plt.imshow(undistorted, cmap="gray")
        plt.title("Undistorted Image")
        plt.show()

    ###
    # Birds eye view calibration
    ###
    def calibrate_birds_eye(self):
        """Calibrate the bird's eye view transformation matrix."""
        if not self.is_config_loaded:
            self.logger.error("No configuration loaded.")
            return
        if not self.is_distortion_calibrated:
            self.logger.error("No distortion calibration data found.")
            return

        # Load calibration image
        if not os.path.exists(self._extrinsic_chessboard_file):
            self.logger.error(
                "Calibration image not found: %s" % self._extrinsic_chessboard_file
            )
            return

        img = cv2.imread(self._extrinsic_chessboard_file)
        if img is None:
            self.logger.error(
                "Failed to load calibration image: %s" % self._extrinsic_chessboard_file
            )
            return

        # Load chessboard configuration
        board_size = (
            self.config["position_board"]["board_size"][0],
            self.config["position_board"]["board_size"][1],
        )

        # Find the chessboard corners
        gray = self._prerpocess_extrinsic_img(img)
        ret, corners = self._get_checkerboard_corners(
            gray, board_size, show_failure=True, resize=True
        )

        # Calculate the transformation matrix
        src_points = np.float32(
            [
                corners[0],
                corners[board_size[0] - 1],
                corners[-1],
                corners[-board_size[0]],
            ]
        )
        # Define your original image size (img_size) and factor
        target_size = (self.config["target_size"][0], self.config["target_size"][1])

        # Calculate the width and height of the source rectangle
        width_src = np.linalg.norm(src_points[0] - src_points[1])
        height_src = np.linalg.norm(src_points[0] - src_points[3])

        # Calculate aspect ratio
        aspect_ratio = target_size[0] / target_size[1]

        # Calculate destination rectangle dimensions maintaining aspect ratio
        if width_src / height_src > aspect_ratio:
            # Width is the limiting dimension
            width_dst = width_src
            height_dst = width_dst / aspect_ratio
        else:
            # Height is the limiting dimension
            height_dst = height_src
            width_dst = height_dst * aspect_ratio

        # Calculate the center of the src_points
        center_src = np.mean(src_points, axis=0)

        # Define destination points centered around the center of src_points
        dst_points = np.float32(
            [
                [center_src[0][0] - width_dst / 2, center_src[0][1] - height_dst / 2],
                [center_src[0][0] + width_dst / 2, center_src[0][1] - height_dst / 2],
                [center_src[0][0] + width_dst / 2, center_src[0][1] + height_dst / 2],
                [center_src[0][0] - width_dst / 2, center_src[0][1] + height_dst / 2],
            ]
        )

        # Move 0.5 * height everything down
        dst_points[:, 1] += 2 * height_dst

        M = cv2.getPerspectiveTransform(src_points, dst_points)
        M_inv = np.linalg.inv(M)

        # Save calibration data
        time_stamp = Clock().now().seconds_nanoseconds()[0]
        self._calib_birds_eye = {
            "iTb": M,
            "bTi": M_inv,
            "src_points": src_points,
            "dst_points": dst_points,
            "target_size": np.array(target_size).reshape(
                2,
            ),
        }
        self._timestamps["birds_eye"] = time_stamp

        if self.debug:
            self._display_result_brids_eye(gray, corners)
        self.is_birds_eye_calibrated = self._set_calibration_flag(self._calib_birds_eye)
        # Check calibration
        if self.is_birds_eye_calibrated:
            self.logger.info("Bird's eye view calibration successful.")
        else:
            self.logger.error("Failed to calibrate bird's eye view parameters.")

    def display_corners(self, img, corners, board_size, resize=True):
        """
        Display the chessboard corners.

        Arguments:
            img -- Image.
            corners -- Corners.
            board_size -- Board size. (rows, cols)

        Keyword Arguments:
            resize -- Should the image be resized to the configured output dimensions (default: {True})
        """
        img = self.downscale_image(img) if resize else img
        img = cv2.drawChessboardCorners(img, board_size, corners, True)
        cv2.imshow("Corners", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def _prerpocess_extrinsic_img(self, img):
        if img is None:
            self.logger.error("Image is None.")
            return None

        if not self.is_distortion_calibrated:
            self.logger.error("Distortion calibration data not found.")
            return img

        # Convert to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img = cv2.undistort(
            img,
            self._calib_distortion["new_camera_matrix"],
            self._calib_distortion["dist_coeffs"],
        )

        # Apply Gaussian Blur
        img = cv2.GaussianBlur(img, (7, 7), 0)

        # Apply Adaptive Thresholding
        img = cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        # Apply Morphological Transformations
        kernel = np.ones((3, 3), np.uint8)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

        return img

    def _display_result_brids_eye(self, img, points=None):
        """
        Display the bird's eye view calibration result.

        Arguments:
            img -- Image.

        Keyword Arguments:
            points -- Points to project (default: {None})
        """
        img = self.downscale_image(img)
        bird = cv2.warpPerspective(
            img, self._calib_birds_eye["iTb"], (img.shape[1], img.shape[0])
        )

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if points is not None:
            points = points.reshape(-1, 2)
            plt.scatter(points[:, 0], points[:, 1], color="r", s=10)
        plt.title("Original Image")
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(bird, cv2.COLOR_BGR2RGB))
        if points is not None:
            points = points.reshape(-1, 2)
            bird_points = Helpers.image_to_bird(points, self._calib_birds_eye["iTb"])
            plt.scatter(bird_points[:, 0], bird_points[:, 1], color="r", s=10)
        plt.title("Birds Eye View")
        plt.axis("off")
        plt.show()

    def calc_reprojection_error_birds_eye(self, image_points, birds_eye_points):
        """
        Calculate the reprojection error for the bird's eye view calibration.

        Arguments:
            image_points -- Points in the image (px).
            birds_eye_points -- Points in the bird's eye view (px).

        Returns:
            float -- Reprojection error.
        """
        # Calculate the projected points img -> birds eye
        img_projected_to_bird = Helpers.image_to_bird(
            image_points, self._calib_birds_eye["iTb"]
        )
        bird_projected_to_img = Helpers.bird_to_image(
            birds_eye_points, self._calib_birds_eye["bTi"]
        )

        # Calculate the reprojection error
        birds_eye_points = np.array(birds_eye_points).astype(np.float64)
        img_projected_to_bird = img_projected_to_bird.astype(np.float64)
        image_points = image_points.astype(np.float64)
        bird_projected_to_img = bird_projected_to_img.astype(np.float64)

        error = cv2.norm(birds_eye_points, img_projected_to_bird, cv2.NORM_L2) / len(
            img_projected_to_bird
        )
        error += cv2.norm(image_points, bird_projected_to_img, cv2.NORM_L2) / len(
            bird_projected_to_img
        )
        return error

    ###
    # Extrinsic calibration
    ###

    def calibrate_extrinsic(self):
        """
        Calibrate the extrinsic parameters.

        Raises:
            RuntimeError: If the extrinsic calibration fails.
        """
        if not self.is_config_loaded:
            self.logger.error("No configuration loaded.")
            return
        if not self.is_distortion_calibrated:
            self.logger.error("No distortion calibration data found.")
            return
        if not os.path.exists(self._extrinsic_chessboard_file):
            self.logger.error(
                "Calibration image not found: %s" % self._extrinsic_chessboard_file
            )
            return

        # Load the calibration image
        img = cv2.imread(self._extrinsic_chessboard_file)
        if img is None:
            self.logger.error("Failed to load the calibration image.")
            return

        img = self._prerpocess_extrinsic_img(img)
        target_size = (self.config["target_size"][0], self.config["target_size"][1])

        # Find the chessboard corners
        ret, corners = self._get_checkerboard_corners(
            img, self.extrinsic_board_size, show_failure=True, resize=False
        )
        if not ret:
            self.logger.error("Failed to find chessboard corners.")
            return

        success, extrinsic_matrix = self._calc_extrinsic_matrix(corners)
        if not success:
            raise RuntimeError("Failed to calculate the extrinsic matrix.")

        # Save the calibration data
        self._calib_extrinsic = {
            "wTc": extrinsic_matrix,
            "cTw": np.linalg.inv(extrinsic_matrix),
            # "target_size": np.array(img.shape)[::-1],
            "target_size": np.array(target_size).reshape(
                2,
            ),
        }
        self._timestamps["extrinsic"] = Clock().now().seconds_nanoseconds()[0]

        self.is_extrinsic_calibrated = self._set_calibration_flag(self._calib_extrinsic)

        err_img_to_world, err_world_to_img = self.calc_reprojection_error_extrinsic(
            np.array(corners).reshape(-1, 2), self.extrinsic_world_points
        )
        self.logger.info("Reprojection error (img -> world): %f" % err_img_to_world)
        self.logger.info("Reprojection error (world -> img): %f" % err_world_to_img)

        try:
            if self.debug:
                self._display_result_extrinsic(img, corners)
        except Exception as e:
            plt.close()
            self.logger.error("Failed to display the calibration result: %s" % str(e))
            return
        # Get the reprojection error
        self.logger.info("Extrinsic calibration completed.")

    def _display_result_extrinsic(self, img, corners):
        """
        Display the extrinsic calibration result.

        Arguments:
            img -- Image.
            corners -- Corners.
        """
        # Define the Roll, Pitch, and Heading (Yaw) angles
        R = self._calib_extrinsic["wTc"][:3, :3]
        c_translation = self._calib_extrinsic["wTc"][:-1, -1]

        # Original coordinate system
        origin = np.array([0, 0, 0])
        x_axis = np.array([1, 0, 0])
        y_axis = np.array([0, 1, 0])
        z_axis = np.array([0, 0, 1])

        # Plotting
        fig = plt.figure(figsize=(12, 6))

        ax = fig.add_subplot(121)

        # Get the new camera matrix (scaled to the target size)
        camera_matrix = self._calib_distortion["new_camera_matrix"]

        # Transform the world points to the camera coordinate system
        camera_points = Helpers.world_to_camera(
            self.extrinsic_world_points, self._calib_extrinsic["wTc"]
        )
        new_image_points = Helpers.camera_to_image(camera_points, camera_matrix)

        # Transform the image points to the world coordinate system
        camera_matrix_inv = np.linalg.inv(camera_matrix)
        corners_camera = Helpers.image_to_camera(
            corners.reshape(-1, 2),
            camera_matrix_inv,
            self.config["focal_length"][0] / 1000,
        )
        corners_world = Helpers.camera_to_world(
            corners_camera, self._calib_extrinsic["cTw"], Z_w=0.0
        )

        # Draw the original corners in green and the transformed corners in red in the image
        ax.imshow(img, cmap="gray")
        ax.scatter(
            corners[:, 0, 0], corners[:, 0, 1], color="g", label="Original Corners", s=4
        )
        ax.scatter(
            np.array(new_image_points.T[0]),
            np.array(new_image_points.T[1]),
            color="r",
            label="Transformed Corners",
            s=3,
        )
        ax.legend()

        # Draw 3 Plot
        ax = fig.add_subplot(122, projection="3d")

        # Plot original coordinate system
        ax.quiver(*origin, *x_axis, color="r", label="X original")
        ax.quiver(*origin, *y_axis, color="g", label="Y original")
        ax.quiver(*origin, *z_axis, color="b", label="Z original")

        # Plot rotated coordinate system
        ax.quiver(
            *c_translation, *(R.T[:, 0]), color="r", label="X rotated", linestyle="--"
        )
        ax.quiver(
            *c_translation, *(R.T[:, 1]), color="g", label="Y rotated", linestyle="--"
        )
        ax.quiver(
            *c_translation, *(R.T[:, 2]), color="b", label="Z rotated", linestyle="--"
        )

        # Draw the world points
        world_points = np.array(self.extrinsic_world_points)
        ax.scatter(
            world_points[:, 0],
            world_points[:, 1],
            world_points[:, 2],
            color="g",
            label="World Points",
            s=15,
        )
        ax.scatter(
            corners_world[:, 0],
            corners_world[:, 1],
            corners_world[:, 2],
            color="r",
            label="Corners",
            s=10,
        )

        # Settings
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend()
        plt.title("Original and Rotated Coordinate Systems")
        plt.show()

    def _calc_extrinsic_matrix(self, image_points):
        """
        Calculate the extrinsic matrix.

        Arguments:
            image_points -- Image points.

        Returns:
            Tuple[bool, np.array] -- Success flag and the extrinsic matrix.
        """
        image_points = image_points.reshape(-1, 2).astype(np.float64)
        world_points = np.array(self.extrinsic_world_points, dtype=np.float64)
        camera_matrix = self._calib_distortion["new_camera_matrix"].astype(np.float64)
        success, rotation_vector, translation_vector = cv2.solvePnP(
            world_points, image_points, camera_matrix, None
        )
        if not success:
            self.logger.error("Failed to calculate the extrinsic matrix.")
            return False, None

        # Convert rotation vector to rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(
            np.asarray(rotation_vector[:, :], np.float64)
        )
        # camera_position = -np.matrix(rotation_matrix).T @ np.matrix(translation_vector)
        camera_position = translation_vector

        # Transform the rotation matrix to match the desired coordinate system
        transformed_rotation_matrix = self._transform_rotation_matrix(rotation_matrix)

        # Form the extrinsic matrix [R|t]
        extrinsic_matrix = np.hstack((transformed_rotation_matrix, camera_position))
        extrinsic_matrix = np.vstack((extrinsic_matrix, [0, 0, 0, 1]))

        return True, extrinsic_matrix

    def _transform_rotation_matrix(self, rotation_matrix):
        """
        Transform the rotation matrix to match the desired coordinate system.

        Arguments:
            rotation_matrix -- Rotation matrix.

        Returns:
            np.array -- Transformed rotation matrix.
        """
        # Define the transformation matrix to convert OpenCV's coordinate system to your coordinate system
        T = np.matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        # Transform the rotation matrix
        transformed_rotation_matrix = rotation_matrix @ T
        return transformed_rotation_matrix

    def calc_reprojection_error_extrinsic(self, image_points, world_points):
        """
        Calculate the reprojection error for the extrinsic calibration.

        Arguments:
            image_points -- Image points. (px)
            world_points -- World points. (m)

        Returns:
            Tuple[float, float] -- Reprojection error (img -> world) and (world -> img)
        """
        # Calculate the projected points img -> world
        img_projected_to_camera = Helpers.image_to_camera(
            image_points,
            self._calib_distortion["new_camera_matrix_inv"],
            self.config["focal_length"][0] / 1000,
        )
        img_projected_to_world = Helpers.camera_to_world(
            img_projected_to_camera, self._calib_extrinsic["cTw"], Z_w=0.0
        )

        # Calculate projected points world -> img
        world_projected_to_camera = Helpers.world_to_camera(
            world_points, self._calib_extrinsic["wTc"]
        )
        world_projected_to_image = Helpers.camera_to_image(
            world_projected_to_camera, self._calib_distortion["new_camera_matrix"]
        )

        # Calculate the reprojection error
        image_points = image_points.astype(np.float64)
        img_projected_to_world = img_projected_to_world.astype(np.float64)
        world_points = np.array(world_points).astype(np.float64)
        world_projected_to_image = world_projected_to_image.astype(np.float64)

        # Calculate the reprojection error
        error_img_to_world = cv2.norm(
            world_points, img_projected_to_world, cv2.NORM_L2
        ) / len(img_projected_to_world)
        error_world_to_img = cv2.norm(
            image_points, world_projected_to_image, cv2.NORM_L2
        ) / len(world_projected_to_image)

        return error_img_to_world, error_world_to_img

    ##############################
    # Properties
    ##############################

    @property
    def all_calibrated(self):
        """
        Check if all calibration parameters are set.

        Returns:
            bool -- True if all calibration parameters are set, False otherwise.
        """
        if (
            self.is_distortion_calibrated
            and self.is_extrinsic_calibrated
            and self.is_birds_eye_calibrated
        ):
            if (
                np.equal(
                    self._calib_distortion.get("target_size", None),
                    self._calib_extrinsic.get("target_size", None),
                ).all()
                and np.equal(
                    self._calib_distortion.get("target_size", None),
                    self._calib_birds_eye.get("target_size", None),
                ).all()
            ):
                return True
            else:
                self.logger.error(
                    "Calibration target sizes do not match: %s (distortion) vs %s (extrinsic) vs %s (birds eye)"
                    % (
                        str(self._calib_distortion["target_size"]),
                        str(self._calib_extrinsic["target_size"]),
                        str(self._calib_birds_eye["target_size"]),
                    )
                )
        return False

    @property
    def internal_camera_matrix(self):
        """
        Get the internal camera matrix.

        Returns:
            np.array -- Internal camera matrix.
        """
        if self.is_distortion_calibrated:
            return self._calib_distortion["internal_camera_matrix"]
        if not self.is_config_loaded:
            return None

        # Calculate internal camera matrix
        image_size = (self.config["target_size"][0], self.config["target_size"][1])
        focal_length_mm = (
            self.config["focal_length"][0],
            self.config["focal_length"][1],
        )
        sensor_size_mm = (
            self.config["sensor_width_mm"],
            self.config["sensor_height_mm"],
        )
        scaling_factor = (
            image_size[0] / sensor_size_mm[0],
            image_size[1] / sensor_size_mm[1],
        )
        principal_point = (image_size[0] / 2.0, image_size[1] / 2.0)

        # Create the camera matrix
        K = np.array(
            [
                [focal_length_mm[0] * scaling_factor[0], 0, principal_point[0]],
                [0, focal_length_mm[1] * scaling_factor[1], principal_point[1]],
                [0, 0, 1],
            ]
        )
        return K

    @property
    def _calib_mapping(self):
        """
        Get the calibration mapping.

        Returns:
            dict -- Calibration mapping.
        """
        return {
            "distortion": self._calib_distortion,
            "extrinsic": self._calib_extrinsic,
            "birds_eye": self._calib_birds_eye,
        }

    @property
    def extrinsic_board_size(self):
        """
        Get the extrinsic board size.

        Returns:
            Tuple[int, int] -- Board size.
        """
        if not self.is_config_loaded:
            return None
        return (
            self.config["position_board"]["board_size"][0],
            self.config["position_board"]["board_size"][1],
        )

    @property
    def extrinsic_world_points(self):
        """
        Get the extrinsic world points.

        Returns:
            np.array -- World points.
        """
        if not self.is_config_loaded:
            return None
        return self.config["position_board"]["world_points"]

    @property
    def rpy_rad(self):
        """
        Get the roll, pitch, yaw angles in radians.

        Returns:
            Tuple[float, float, float] -- Roll, pitch, yaw angles in radians.
        """
        if not self.is_extrinsic_calibrated:
            return None
        extrinsic_matrix = self._calib_extrinsic["wTc"]
        R = extrinsic_matrix[:3, :3]

        rotation = Rotation.from_matrix(R)
        rpy = rotation.as_euler("xyz", degrees=False)
        return rpy

    @property
    def camera_position(self):
        """
        Get the camera position.

        Returns:
            np.array -- Camera position.
        """
        if not self.is_extrinsic_calibrated:
            return None
        extrinsic_matrix = self._calib_extrinsic["wTc"]
        return extrinsic_matrix[:-1, -1]

    ##############################
    # Private methods
    ##############################

    def _scale_matrix_points(
        self, matrix: np.ndarray, scale: Tuple[float, float], c_type: str
    ):
        """
        Scale the matrix points.

        Arguments:
            matrix -- Matrix to scale.
            scale -- Scaling factor.
            c_type -- Type of matrix. (camera, transform, points)

        Returns:
            np.array -- Scaled matrix.
        """
        if c_type == "camera":
            assert matrix.shape == (3, 3), "Invalid matrix shape."
            matrix[0, 0] *= scale[0]
            matrix[1, 1] *= scale[1]
            matrix[0, 2] *= scale[0]
            matrix[1, 2] *= scale[1]
        elif c_type == "transform":
            assert matrix.shape == (3, 3), "Invalid matrix shape."
            transform = np.eye(3)
            transform[0, 0] = scale[0]
            transform[1, 1] = scale[1]
            matrix = transform @ matrix @ np.linalg.inv(transform)
        elif c_type == "points":
            original_shape = matrix.shape
            matrix = matrix.reshape(-1, 2)
            assert matrix.shape[1] == 2, "Invalid point shape."
            matrix[:, 0] *= scale[0]
            matrix[:, 1] *= scale[1]
            matrix = matrix.reshape(original_shape)
        return matrix

    def _compute_scaling_factor(self, image_size, target_size):
        """
        Compute the scaling factor.

        Arguments:
            image_size -- Image dimensions.
            target_size -- Target dimensions.

        Returns:
            Tuple[float, float] -- Scaling factor.
        """
        img_x, img_y = image_size[0], image_size[1]
        target_x, target_y = target_size[0], target_size[1]
        scale_x = target_x / img_x
        scale_y = target_y / img_y
        return scale_x, scale_y

    def downscale_image(self, img):
        """
        Downscale the image to the target size.

        Arguments:
            img -- Image.

        Returns:
            np.array -- Downscaled image
        """
        # Extract and ensure target size is a tuple of integers
        img_size = (
            int(self.config["target_size"][0]),
            int(self.config["target_size"][1]),
        )
        if img.shape[0] != img_size[0] or img.shape[1] != img_size[1]:
            # Resize the image using cv2
            img = cv2.resize(img, img_size, interpolation=cv2.INTER_LINEAR)

        return img

    def _get_checkerboard_corners(
        self, img, board_size, show_failure=False, resize=True
    ):
        """
        Get the checkerboard corners.

        Arguments:
            img -- Image.
            board_size -- Board size.

        Keyword Arguments:
            show_failure -- Show failure img (default: {False})
            resize -- Should the image be resized to the configured output dimensions (default: {True})

        Returns:
            Tuple[bool, np.array] -- Success flag and the corners.
        """
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        ret, corners = cv2.findChessboardCorners(img, board_size, None)
        if ret:
            corners2 = cv2.cornerSubPix(img, corners, (11, 11), (-1, -1), criteria)
            if resize:
                scaling_factor = self._compute_scaling_factor(
                    self.config["image_size"], self.config["target_size"]
                )
                corners2 = self._scale_matrix_points(corners2, scaling_factor, "points")
            if self.debug:
                self.display_corners(img, corners2, board_size, resize=resize)

            return ret, corners2
        elif show_failure:
            self.logger.error("Failed to find chessboard corners.")
            plt.imshow(img, cmap="gray")
            plt.title("Failed to find chessboard corners")
            plt.show()
        return ret, None

    def _set_calibration_flag(self, calib):
        """
        Set the calibration flag.

        Arguments:
            calib -- Calibration data.

        Returns:
            bool -- True if the calibration data is valid, False otherwise.
        """
        for key, value in calib.items():
            if value is None:
                return False
            elif isinstance(value, int) and value == 0:
                return False
            elif isinstance(value, float) and value == 0.0:
                return False
            elif isinstance(value, list) and len(value) == 0:
                return False
            elif isinstance(value, dict) and len(value) == 0:
                return False
        return True

    def __str__(self) -> str:
        """
        Get the calibration data as a string.

        Returns:
            str -- Calibration data as a string.
        """
        string = "################## Calibration Data ##################\n"
        for name, calib in self._calib_mapping.items():
            string += name + ":\n"
            for key, value in calib.items():
                if isinstance(value, np.ndarray):
                    string += "\t" + key + ":\n"
                    string += str(value) + "\n"
                else:
                    string += "\t" + key + ": " + str(value) + "\n"

        string += "Timestamps:\n"
        for key, value in self._timestamps.items():
            string += "\t" + key + ": " + str(value) + "\n"
        string += "#####################################################"
        return string
