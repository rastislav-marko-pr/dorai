from setuptools import setup

package_name = "bridge_mod"

setup(
    name=package_name,
    version="0.1.0",
    # Flat layout, matching voice_mod/stt_mod/doa_mod/wake_mod.
    py_modules=["bridge_node"],
    data_files=[
        ("share/ament_index/resource_index/packages",
         ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Panza Robotics",
    maintainer_email="rastislav.marko@panzarobotics.com",
    description=(
        "dorai ROS 2 <-> ROS 1 UDP bridge, container half: /dorai/doa and "
        "/dorai/wake out to 127.0.0.1:5801, person and robot_state in on 5802."
    ),
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "bridge = bridge_node:main",
        ],
    },
)
