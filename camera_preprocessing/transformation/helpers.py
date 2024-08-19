#! /usr/bin/env python3
# Copyright (c) 2024 Smart Rollerz e.V. All rights reserved.
import cv2
import numpy as np


class Helpers:
    @staticmethod
    def reorder_corners(corners, rows, cols):
        """
        Reorders corners from column-wise to row-wise.

        Args:
        corners: list or numpy array of points representing the corners.
        rows: int, number of rows in the chessboard.
        cols: int, number of columns in the chessboard.

        Returns:
        reordered_corners: numpy array of reordered points.
        """
        # Convert corners to a numpy array if it is not already
        corners = np.array(corners)

        # Reshape the corners array to a 2D array with shape (cols, rows)
        column_wise_corners = corners.reshape((cols, rows, -1))

        # Transpose the array to switch from column-wise to row-wise
        row_wise_corners = column_wise_corners.transpose(1, 0, 2)

        # Flatten the 2D array back to a 1D array if needed
        reordered_corners = row_wise_corners.reshape(-1, corners.shape[-1])

        return reordered_corners

    @staticmethod
    def world_to_camera(world_points, wTc, is_hom=False) -> np.ndarray:
        """
        Transform world points to camera points.

        Args:
        world_points: numpy array of world points.
        wTc: numpy array of the world-to-camera transformation matrix.
        is_hom: bool, whether the world points are in homogeneous coordinates.

        Returns:
        camera_points: numpy array of camera points.
        """
        # Ensure np.array
        world_points = np.array(world_points)

        # Convert world points to homogeneous coordinates
        if not is_hom:
            world_points = np.hstack(
                (world_points, np.ones((world_points.shape[0], 1)))
            )

        # Transform world points to camera points
        camera_points = (wTc @ world_points.T).T

        if not is_hom:
            camera_points = camera_points[:, :3]

        return np.array(camera_points)

    @staticmethod
    def camera_to_image(camera_points, K, is_hom=False) -> np.ndarray:
        """
        Transform camera points to image points.

        Args:
        camera_points: numpy array of camera points.
        K: numpy array of the camera intrinsic matrix.
        is_hom: bool, whether the camera points are in homogeneous coordinates.

        Returns:
        image_points: numpy array of image points.
        """
        # Ensure np.array
        camera_points = np.matrix(camera_points)

        # Normalize camera points
        camera_points = camera_points[:, :2] / camera_points[:, 2]
        camera_points = np.hstack((camera_points, np.ones((camera_points.shape[0], 1))))

        # Transform camera points to image points
        image_points = (K @ camera_points.T).T

        if not is_hom:
            image_points = image_points[:, :2]

        return np.array(image_points)

    @staticmethod
    def image_to_camera(
        image_points: np.ndarray,
        K_inv: np.ndarray,
        focal_length: float,
        is_hom: bool = False,
    ) -> np.ndarray:
        """
        Convert 2D image points to 3D camera points assuming the projection plane at Z = focal_length.

        Args:
        image_points: numpy array of image points (Nx2).
        K_inv: numpy array of the inverse of the camera intrinsic matrix (3x3).
        focal_length: float, the focal length of the camera.

        Returns:
        camera_points: numpy array of camera points (Nx3).
        """
        # Convert 2D image points to homogeneous coordinates
        if not is_hom:
            image_points = np.hstack(
                (image_points, np.ones((image_points.shape[0], 1)))
            )

        # Transform image points to normalized camera coordinates
        camera_points_hom = (K_inv @ image_points.T).T

        # Scale by the focal length to get 3D camera coordinates where Z = focal_length
        camera_points = camera_points_hom * focal_length

        if is_hom:
            camera_points = np.hstack(
                (camera_points, np.ones((camera_points.shape[0], 1)))
            )

        return np.array(camera_points)

    @staticmethod
    def camera_to_world(
        camera_points: np.ndarray,
        cTw: np.ndarray,
        Z_w: float = 0.0,
        is_hom: bool = False,
    ) -> np.ndarray:
        """
        Transform camera coordinates to world coordinates using a homogeneous transformation matrix
        and intersect the rays with the plane Z = Z_w.

        Args:
        camera_points: numpy array of camera points (Nx3).
        cTw: numpy array of the homogeneous transformation matrix (4x4).
        Z_w: float, known Z-coordinate in the world frame.

        Returns:
        world_points: numpy array of world points (Nx3).
        """
        flag = False
        if camera_points.shape[0] == 1:
            flag = True
            camera_points = np.array([camera_points[0], [0, 0, 0]])
        camera_points = np.array(camera_points).reshape(-1, 3)

        if not is_hom:
            camera_points = np.hstack(
                (camera_points, np.ones((camera_points.shape[0], 1)))
            )

        # Calculate the scaling factor for each camera point to intersect the Z_w plane
        camera_plane = np.matrix((cTw @ camera_points.T).T[:, :3])

        # Ray from camera origin to the camera points
        camera_pose = cTw[:3, 3]
        camera_ray = None
        try:
            camera_ray = (camera_plane.T - camera_pose).T
        except ValueError:
            camera_ray = camera_plane - camera_pose
        camera_ray = camera_ray / np.linalg.norm(camera_ray, axis=1).reshape(-1, 1)

        # Apply the scale factors to the camera points
        scale_factors = (Z_w - camera_pose[2]) / camera_ray[:, 2]

        # Compute the world points
        world_points = None
        try:
            world_points = (
                np.multiply(camera_ray, scale_factors.reshape(-1, 1)).T + camera_pose
            ).T
        except ValueError:
            world_points = np.multiply(camera_ray, scale_factors) + camera_pose

        if is_hom:
            world_points = world_points[:, :3]

        if flag:
            world_points = world_points[0]

        return np.array(world_points)

    @staticmethod
    def image_to_bird(image_points, iTb, is_hom=False) -> np.ndarray:
        """
        Transform image points to bird's-eye view points.

        Args:
        image_points: numpy array of image points.
        iTb: numpy array of the image-to-bird's-eye view transformation matrix.
        is_hom: bool, whether the image points are in homogeneous coordinates.

        Returns:
        bird_points: numpy array of bird's-eye view points.
        """
        # Ensure np.array
        image_points = np.array(image_points)

        # Convert image points to homogeneous coordinates
        if is_hom:
            image_points = image_points[:, :2]

        # Transform image points to bird's-eye view points
        bird_points = cv2.perspectiveTransform(
            np.array(image_points[:, :2], dtype=np.float64).reshape(-1, 1, 2),
            np.array(iTb, dtype=np.float64),
        )

        bird_points = np.array(bird_points, dtype=image_points.dtype).reshape(-1, 2)

        if is_hom:
            bird_points = np.hstack((bird_points, np.ones((bird_points.shape[0], 1))))

        return np.array(bird_points)

    @staticmethod
    def bird_to_image(bird_points, bTi, is_hom=False) -> np.ndarray:
        """
        Transform bird's-eye view points to image points.

        Args:
        bird_points: numpy array of bird's-eye view points.
        bTi: numpy array of the bird's-eye view-to-image transformation matrix.
        is_hom: bool, whether the bird's-eye view points are in homogeneous coordinates.

        Returns:
        image_points: numpy array of image points.
        """
        # Ensure np.array
        bird_points = np.array(bird_points)

        # Convert bird's-eye view points to homogeneous coordinates
        if is_hom:
            bird_points = bird_points[:, :2]

        # Transform bird's-eye view points to image points
        image_points = cv2.perspectiveTransform(
            np.array(bird_points[:, :2], dtype=np.float64).reshape(-1, 1, 2),
            np.array(bTi, dtype=np.float64),
        )

        image_points = np.array(image_points, dtype=bird_points.dtype).reshape(-1, 2)
        return np.array(image_points)
