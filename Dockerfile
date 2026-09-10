# dorai — ROS 2 mic-array speech pipeline
#
# Runs ROS 2 Jazzy in-container, so the host OS is irrelevant: this builds and
# runs identically on Ubuntu 20.04 / 22.04 hosts that only have Docker. The
# same image serves every node (voice_mod, stt_mod, web_mod, doa_mod, wake_mod,
# bridge_mod); docker-compose picks which one each service runs.
FROM ros:jazzy-ros-base

ENV DEBIAN_FRONTEND=noninteractive

# --- System libraries -------------------------------------------------------
# libportaudio2      : PortAudio backend for `sounddevice` (mic capture)
# libsamplerate0-dev : libsamplerate for the variable-ratio drift resampler
# python3-pip        : to install the pure-Python runtime deps below
# numpy/scipy come from apt so they stay consistent with the apt-managed numpy
# the ROS base image already ships (pip cannot uninstall that debian package).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libportaudio2 \
        libsamplerate0-dev \
        python3-pip \
        python3-numpy \
        python3-scipy \
    && rm -rf /var/lib/apt/lists/*

# --- Python runtime dependencies --------------------------------------------
# Not covered by rosdep. --break-system-packages is required because Ubuntu
# 24.04 (Jazzy's base) marks the system Python as externally managed (PEP 668),
# matching the project README. faster-whisper + vosk make BOTH STT engines
# runnable; the engine is chosen at runtime via `-p engine:=vosk|whisper`.
RUN pip3 install --no-cache-dir --break-system-packages \
        onnxruntime \
        sounddevice \
        samplerate \
        vosk \
        faster-whisper \
        aiohttp

# --- Build the ROS 2 workspace ----------------------------------------------
WORKDIR /ros2_ws
COPY voice_mod  src/voice_mod
COPY stt_mod    src/stt_mod
COPY web_mod    src/web_mod
# doa_mod ships vad.onnx (2.3 MB) as package data; wake_mod and bridge_mod are
# pure python and add no dependency — wake_mod reuses the vosk already
# installed for stt_mod, and bridge_mod needs only the standard library.
COPY doa_mod    src/doa_mod
COPY wake_mod   src/wake_mod
COPY bridge_mod src/bridge_mod

# Source the ROS base, then colcon-build every ament_python package.
RUN . /opt/ros/jazzy/setup.sh && \
    colcon build --symlink-install

# Persisted STT model caches (Vosk + Hugging Face for faster-whisper) land here.
ENV HF_HOME=/root/.cache/huggingface \
    VOSK_MODEL_PATH=/root/.cache/vosk

# Entrypoint sources both the ROS base and this workspace overlay before exec.
COPY ros_entrypoint.sh /ros_entrypoint.sh
RUN chmod +x /ros_entrypoint.sh
ENTRYPOINT ["/ros_entrypoint.sh"]

# Default to the capture+beamform node; compose overrides per service.
CMD ["ros2", "run", "voice_mod", "voice"]
