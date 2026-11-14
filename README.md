# Camera Preprocessing

> Attention: In some ros bags in our NAS the camera_topic is /camera/image_raw instead of /camera/image/raw. Use `ros2 bag play <bag_file> --remap /camera/image_raw /camera/image/raw`

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
- **bev_view_topic_prefix**: Prefix of the topics of the additional bird's eye views (default: `"/camera/image/bev"`).
- **bev_views**: Names of additional bird's eye views of freely chosen regions (default: none, `[""]`: none). Each name needs a section `bev_view.<name>` with `x_range` (mm forward, `[min, max]`), `y_range` (mm to the left, `[min, max]`) and `resolution` (mm per px). The name is only a label in the YAML file, see [Additional bird's eye views](#additional-birds-eye-views).

### Additional bird's eye views

Besides the regular bird's eye view (fixed region of about x = -120..1280 mm, y = -920..1000 mm), the node can publish bird's eye views of any rectangle on the ground with any resolution (`RegionBirdseye`). Every view gets its own topic named after its region and resolution, so that any node can use it and the node stays independent of its subscribers:

```
<bev_view_topic_prefix>/x_<x start>_<x end>/y_<y start>_<y end>/r<resolution>
```

All values in mm. ROS 2 topic names allow neither `-` nor `.` and no part of the name may start with a digit, therefore a minus is written as `m` and a decimal point as `p`. Examples:

| x_range | y_range | resolution | Topic | Size |
| --- | --- | --- | --- | --- |
| `[-150, 2000]` | `[-1200, 1200]` | `3.0` | `/camera/image/bev/x_m150_2000/y_m1200_1200/r3` | 800 x 717 px |
| `[-500, 2000]` | `[-1200, 1200]` | `2.5` | `/camera/image/bev/x_m500_2000/y_m1200_1200/r2p5` | 960 x 1000 px |

Add a view in `config/ros_params.yaml`:

```yaml
bev_views: ["wide", "near"]
bev_view:
  wide: {x_range: [-150.0, 2000.0], y_range: [-1200.0, 1200.0], resolution: 3.0}
  near: {x_range: [0.0, 1000.0], y_range: [-500.0, 500.0], resolution: 2.0}
```

or with one line in the node (`camera_preprocessing_node.py`):

```python
self.add_bev_view(x_range=(0, 1000), y_range=(-500, 500), resolution=2.0)
```

A subscriber computes the topic from the same values, so the image and its conversion between px and mm always match:

```python
from camera_preprocessing.transformation.region_birdseye import (
    RegionBirdseye,
    bev_topic_name,
)

topic = bev_topic_name(x_range=(-150, 2000), y_range=(-1200, 1200), resolution=3.0)

# Conversion between the pixels of the view and the vehicle frame (mm)
view = RegionBirdseye(CoordinateTransform(), (-150, 2000), (-1200, 1200), 3.0)
points_mm = view.pixel_to_world(pixels)
```

Image axes: up = +x (forward), right = -y (right), `x = x_range[1] - row * resolution`, `y = y_range[1] - column * resolution`. Pixels outside of the camera's field of view are black.
