FROM ros:jazzy

ENV DEBIAN_FRONTEND=noninteractive
ENV TMPDIR=/home/idm/tmp

# ensure tmp exists and is writable
RUN mkdir -p /home/idm/tmp && chmod 1777 /home/idm/tmp

# system packages needed for builds
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-venv python3-pip build-essential git cmake vim libgl1 \
    libusb-1.0-0-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install ROS Python bindings and message packages via apt
# (these are NOT available on PyPI — they must come from the ROS apt repo)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-jazzy-rclpy \
    ros-jazzy-std-msgs \
    ros-jazzy-sensor-msgs \
    ros-jazzy-std-srvs \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Create venv that can see system site-packages (ROS bindings installed via apt)
RUN python3 -m venv --system-site-packages /opt/praxis_venv \
    && /opt/praxis_venv/bin/python -m pip install --upgrade pip setuptools wheel

WORKDIR /workspace/vla-safety
# Copy project into image so editable install will point to /workspace/vla-safety inside image
COPY . /workspace/vla-safety

# Pre-warm torch/torchvision/torchaudio cu130 wheels in their own layer so the cache survives
# unrelated dependency changes. Prefix with TMPDIR=/var/tmp so pip uses /var/tmp for build/extract.
RUN TMPDIR=/var/tmp /opt/praxis_venv/bin/pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cu130 \
        "torch==2.10.*" "torchvision==0.25.*" "torchaudio==2.10.*"

# Editable install pulls everything else (incl. transformers fork, lerobot, hydra, av, ...) from pyproject.toml
RUN TMPDIR=/var/tmp /opt/praxis_venv/bin/pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cu130 \
        -e .

# Source ROS and venv for interactive shells and set PS1
RUN echo 'source /opt/ros/jazzy/setup.bash' >> /root/.bashrc && \
    echo 'source /opt/praxis_venv/bin/activate' >> /root/.bashrc && \
    echo 'export PS1="\[\e[1;35m\][\[\e[1;37m\]\W\[\e[1;35m\]]#\[\e[0m\] "' >> /root/.bashrc

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
