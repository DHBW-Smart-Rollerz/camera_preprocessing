import os

from setuptools import find_packages, setup

package_name = "camera_preprocessing"


# ToDo: Add this to the utils package
def package_files(directory_list):
    """
    Collect all files in the given directories.

    Arguments:
        directory_list -- List of directories to search for files.

    Returns:
        List of paths to all files in the given directories
    """
    paths = []
    for directory in directory_list:
        for path, directories, filenames in os.walk(directory):
            for filename in filenames:
                if filename.endswith(".pdf"):
                    continue
                paths.append(os.path.join(path, filename))
    return paths


setup(
    name=package_name,
    version="1.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (
            os.path.join("share", package_name, "img/calib/Neue_3MP_Kamera"),
            package_files(["img/calib/Neue_3MP_Kamera"]),
        ),
        (
            os.path.join("share", package_name, "img/position"),
            package_files(["img/position"]),
        ),
        (
            os.path.join("share", package_name, "config"),
            package_files(["config"]),
        ),
        (
            os.path.join("share", package_name, "launch"),
            package_files(["launch"]),
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
