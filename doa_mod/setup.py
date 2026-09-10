from setuptools import setup

package_name = "doa_mod"

setup(
    name=package_name,
    version="0.1.0",
    # Flat layout, matching voice_mod/stt_mod: the estimator modules are
    # vendored byte-identical from updated_beamformer/doa/ and must stay that
    # way, so they sit at the top level exactly as they do upstream and are
    # imported by their upstream names.
    py_modules=["doa_node", "doa_core", "vad", "auto_calibrate"],
    data_files=[
        ("share/ament_index/resource_index/packages",
         ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name, ["vad.onnx"]),
        # The 21-test self-test ships WITH the package on purpose: TASK.md
        # requires it to pass before any recording session, and a test that
        # only exists in the source tree is a test nobody runs on the robot.
        ("share/" + package_name + "/tests", ["tests/test_doa_core.py"]),
    ],
    scripts=["tools/doa_session.py"],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Panza Robotics",
    maintainer_email="rastislav.marko@panzarobotics.com",
    description=(
        "dorai direction of arrival: pooled SRP-PHAT bearing from the "
        "ReSpeaker's four raw capsules, published on /dorai/doa."
    ),
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "doa = doa_node:main",
        ],
    },
)
