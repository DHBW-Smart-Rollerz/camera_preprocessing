import os

from setup_utils import include_directory
from setuptools import find_packages, setup

package_name = "camera_preprocessing"


setup(
    name=package_name,
    version="1.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        *include_directory(
            install_path=os.path.join("share", package_name, "img/calib"),
            source_path="img/calib",
            exclude=[".pdf"],
        ),
        *include_directory(
            install_path=os.path.join("share", package_name, "img/position"),
            source_path="img/position",
            exclude=[".pdf"],
        ),
        *include_directory(
            install_path=os.path.join("share", package_name, "config"),
            source_path="config",
        ),
        *include_directory(
            install_path=os.path.join("share", package_name, "launch"),
            source_path="launch",
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Tom Freudenmann",
    maintainer_email="75214791+Super-T02@users.noreply.github.com",
    description="The camera_preprocessing package",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "camera_preprocessing_node = camera_preprocessing.camera_preprocessing_node:main",
            "camera_calibration_node = camera_preprocessing.camera_calibration_node:main",
        ],
    },
)
