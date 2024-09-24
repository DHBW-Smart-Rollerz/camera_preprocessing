# Camera Preprocessing

This package is responsible for calibrating the camera and preprocessing the images. Therefore it provides the following two nodes:

1. `camera_calibration_node`: Calibrates the camera
2. `camera_preprocessing_node`: Preprocess the raw image

Additionally, it provides the following classes to transform points and images:

1. `CoordinateTransform`: Transform points
```python
from camera_preprocessing.transformation.coordinate_transform import CoordinateTransform

cordinate_transform = CoordinateTransform()

# Transform your points with the provided methods
```

2. `Birdseye`: Transform images into birdseye
```python
from camera_preprocessing.transformation.calibration import Calibration
from camera_preprocessing.transformation.birds_eyed_view import Birdseye

calib = Calibration()
bev = Birdseye(calib)

# Transform the img to birdseye view (bev)
bev_img = bev.transform_img(img)
```

Please Note that this package is mandatory for using images in the smarty pipeline.

## Camera Calibration

The camera calibration is done on the full image. The node uses the camera configuration (`config/config.yaml`), the calibration board images and the position board image to calibrate the distortion, intrinsics and extrinsic of the camera.

### Usage

It exists a launchfile to calibrate the camera:

```bash
ros2 launch camera_preprocessing camera_calibration.launch.py
```

- `params_file`: Path to the ROS parameters file (default: `config/ros_params.yaml`).
- `debug`: Enable debug mode to show further images and configs (default: `false`).
- `chessboard_path`: Path to the chessboard image (default: `none`). With `none` the node will subscribe to the raw image topic and wait until it finds the chessboard.

### Settings

The camera parameters are specified in `config/config.yaml` and contains the following parameters:

- **recalibration_interval**: Interval in seconds between recalibration (default: 7 days or 604800 seconds).

- **focal_length**: Focal length of the camera in millimeters. It is specified as a list with two values representing the focal length in the x and y directions.

- **sensor_width_mm**: Width of the camera sensor in millimeters.

- **sensor_height_mm**: Height of the camera sensor in millimeters.

- **image_size**: Width and height of the image in pixels. It is specified as a list with two values.

- **target_size**: Width and height of the target image in pixels. It is specified as a list with two values. This parameter can be used to resize the image to a different resolution.

- **recalibrate**: Boolean flag indicating whether the camera should be recalibrated.

- **calibration_board**:
  - **board_size**: Size of the calibration board in terms of the number of squares along the width and height. It is specified as a list with two values.
  - **square_size**: Size of each square on the calibration board in meters.

- **method**: Method used for external camera calibration (e.g., "chessboard").
    > *Note:  This is the only currently implemented method*

- **position_board**:
  - **board_size**: Size of the position board in terms of the number of squares along the width and height. It is specified as a list with two values.
  - **square_size**: Size of each square on the position board in meters.
  - **world_points**: List of 3D coordinates representing the world points on the position board. Each point is specified as a list with three values representing the x, y, and z coordinates in meters.

Additionally the following topics are specified in the `config/ros_params.yaml` file:

- **calibration_file**: Path to the calibration file where the calibration data will be saved (default: `"config/calib.bin"`).
- **config_file**: Path to the configuration file containing camera parameters (default: `"config/config.yaml"`).
- **calibration_images_path**: Path to the directory containing calibration images (default: `"img/calib/Neue_3MP_Kamera/"`).
- **image_topic**: ROS topic from which raw camera images are received (default: `"/camera/image_raw"`).

## Camera Preprocessing

The camera preprocessing node subscribes to the raw image and publishes the undistorted and birdseye view image on the specified topics.

## Usage

The preprocessing can be started by launching the following command:

```bash
ros2 launch camera_preprocessing camera_preprocessing.launch.py
```

The following parameters exist:

- `params_file`: Path to the ROS parameters file (default: `config/ros_params.yaml`).
- `debug`: Enable debug mode to show further images and configs (default: `false`).

Example for all parameters:
```bash
ros2 launch camera_preprocessing camera_preprocessing.launch.py params_file:=/path/to/ros_params.yaml debug:=true
```

## Settings
The topics are defined as ros_parameters:
- **calibration_file**: Path to the calibration file where the calibration data will be saved (default: `"config/calib.bin"`).
- **config_file**: Path to the configuration file containing camera parameters (default: `"config/config.yaml"`).
- **calibration_images_path**: Path to the directory containing calibration images (default: `"img/calib/Neue_3MP_Kamera/"`).
- **position_calib_img_path**: Path to the image used for position calibration (default: `"img/position/chessboard.png"`).
- **num_skip_frames**: Number of frames to skip between processing (default: `1`).
- **image_topic**: ROS topic from which raw camera images are received (default: `"/camera/image_raw"`).
- **undistorted_publisher_topic**: ROS topic to which the undistorted images are published (default: `"/camera/undistorted"`).
- **birds_eye_publisher_topic**: ROS topic to which the bird's eye view images are published (default: `"/camera/birds_eye"`).
