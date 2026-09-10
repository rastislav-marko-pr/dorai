from setuptools import setup

package_name = "wake_mod"

setup(
    name=package_name,
    version="0.1.0",
    # Flat layout, matching voice_mod/stt_mod/doa_mod.
    py_modules=["wake_node"],
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
        "dorai wake-word gate: a grammar-constrained Vosk recogniser over the "
        "clean audio, publishing /dorai/wake with the bearing already paired."
    ),
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "wake = wake_node:main",
        ],
    },
)
