from setuptools import setup

package_name = "web_mod"

setup(
    name=package_name,
    version="0.1.0",
    # Flat layout: web.py is a top-level module. console_scripts below points
    # `ros2 run web_mod web` at web:main. The single-page dashboard ships as a
    # share data file and is located at runtime via the ament share dir.
    py_modules=["web"],
    data_files=[
        ("share/ament_index/resource_index/packages",
         ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/static", ["static/index.html"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Sensifai",
    maintainer_email="info@sensifai.com",
    description=(
        "dorai demo dashboard: a web view of mic-array capture + beamformer "
        "output (level meters, raw-vs-clean spectrograms, live transcript, "
        "per-mic diagnostics) with speaker-side raw/clean auditioning."
    ),
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "web = web:main",
        ],
    },
)
