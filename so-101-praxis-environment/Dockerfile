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
    ros-jazzy-geometry-msgs \
    ros-jazzy-std-srvs \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Create venv that can see system site-packages (ROS bindings installed via apt)
RUN python3 -m venv --system-site-packages /opt/praxis_venv \
    && /opt/praxis_venv/bin/python -m pip install --upgrade pip setuptools wheel

WORKDIR /workspace/praxis-core
# Copy project into image so editable install will point to /workspace/praxis-core inside image
COPY . /workspace/praxis-core

# Install largest packages first in their own layer so TMP space is freed between layers.
# Prefix each pip invocation with TMPDIR=/var/tmp to force pip to use /var/tmp for build/extract.
RUN TMPDIR=/var/tmp /opt/praxis_venv/bin/pip install --no-cache-dir \
        torch torchvision

# Install other binary dependencies (smaller) in a separate layer
RUN TMPDIR=/var/tmp /opt/praxis_venv/bin/pip install --no-cache-dir \
        feetech-servo-sdk \
        "lerobot==0.4.4" \
        hydra-core \
        omegaconf \
        mujoco

# Editable install of your project (feetech extras) in its own layer
RUN TMPDIR=/var/tmp /opt/praxis_venv/bin/pip install --no-cache-dir -e ".[feetech]"

# vla-safety dependencies (PI0 requires a patched transformers fork)
RUN TMPDIR=/var/tmp /opt/praxis_venv/bin/pip install --no-cache-dir \
        "transformers @ git+https://github.com/huggingface/transformers.git@fix/lerobot_openpi" \
        safetensors \
        einops \
        accelerate \
        diffusers \
        "datasets>=4.0.0,<5.0.0"

# Source ROS and venv for interactive shells and set PS1
RUN echo 'source /opt/ros/jazzy/setup.bash' >> /root/.bashrc && \
    echo 'source /opt/praxis_venv/bin/activate' >> /root/.bashrc && \
    echo 'export PS1="\[\e[1;35m\][\[\e[1;37m\]\W\[\e[1;35m\]]#\[\e[0m\] "' >> /root/.bashrc

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
