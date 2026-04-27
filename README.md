# vla-safety

End-to-end stack for safely deploying a vision-language-action (VLA) policy on a real SO101 follower arm.

The repo combines two halves that talk to each other over ROS 2 topics:

- **Hardware bridge** (`hardware/`) — a ROS 2 Jazzy node that drives the SO101 follower arm and a USB webcam, publishing joint state and camera frames.
- **Safe deployment** (`scripts/deploy_ros.py`) — runs the PI0 base policy, gates each action through a latent safety filter trained on a DINO-based world model, and publishes joint commands back to the bridge.

## Layout

```
hardware/      ROS 2 / hardware bridge for the SO101 follower arm
pi0/           PI0 vision-language base policy (LeRobot-compatible)
dino_wm/       DINOv2-based latent world model
latentsafe/    Hamilton-Jacobi reach-avoid safety filter
data/          LeRobot dataset loaders + augmentation
utils/         Shared utilities, paths, types, training helpers
scripts/       Executable entry points (e.g. deploy_ros.py)
configs/       Hydra configs for the hardware bridge
calibration/   SO101 follower / leader calibration files
assets/        Dataset stats, example tensors
```

## Install

For native (non-Docker) installs on a CUDA 13 host (e.g. RTX 5090, Blackwell sm_120):

```bash
pip install --extra-index-url https://download.pytorch.org/whl/cu130 -e .
```

Repeat the `--extra-index-url` on any subsequent `pip install` (extras, upgrades, re-installs) — without it pip silently falls back to CPU wheels and loses Blackwell support. The Docker image bakes this in.

## Docker

Build and run an interactive container:

```bash
docker compose build
docker compose run --rm vla_safety
```

Inside the container, verify GPUs are visible:

```bash
nvidia-smi
```

If `nvidia-smi` fails inside the container, install and configure the NVIDIA container toolkit:

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo sed -i -e '/experimental/ s/^#//g' /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
export NVIDIA_CONTAINER_TOOLKIT_VERSION=1.19.0-1
sudo apt-get install -y \
    nvidia-container-toolkit=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
    nvidia-container-toolkit-base=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
    libnvidia-container-tools=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
    libnvidia-container1=${NVIDIA_CONTAINER_TOOLKIT_VERSION}

sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

## Run the hardware bridge

From inside the container (or any host with the editable install):

```bash
python -m hardware.ros.scripts.robot_node_launcher robot=so101real &
python hardware/ros/examples/webcam.py &
```

The robot node publishes and subscribes under the `so101real` namespace; the webcam publisher writes 640×480 BGR frames @ 30 Hz to `so101real/camera/image`.

After `pip install -e .`, the launcher is also available as the console script `so101-robot-node`.

## Run safe deployment

In a second shell (with the bridge already running):

```bash
huggingface-cli login

python scripts/deploy_ros.py \
    --pi0-model AndrewNoviello/vla-safety-task-1 \
    --prompt "pick up the middle domino from the three domino row and place it flat on top of the other two dominos to form an arch"

# Or, after pip install -e .:
vla-deploy-ros --pi0-model AndrewNoviello/vla-safety-task-1 --prompt "..."
```

The deployment script subscribes to `so101real/joint_state` and `so101real/camera/image`, runs PI0 to propose an action, optionally gates it through the latent safety filter, and publishes the (possibly overridden) action back on `so101real/joint_command`.

To run PI0 unfiltered (no safety gating):

```bash
python scripts/deploy_ros.py --disable-safety
```

## Training

- `python -m pi0.train` — supervised fine-tune of PI0 on a LeRobot dataset.
- `python -m dino_wm.train` — train the DINO-based latent world model.
- `python -m latentsafe.train_safety_ddpg` — train the reach-avoid safety actor/critic against the frozen world model.
- `python -m latentsafe.train_classifier` — train the failure classifier used to define the safety reward.

## Configuration

- Hydra configs live in `configs/`; the launcher resolves `configs/config.yaml` and pulls in `configs/robot/*.yaml` and `configs/ros_node/*.yaml` via the `defaults:` list.
- Calibration files live in `calibration/`. The Docker compose mount sets `HF_LEROBOT_CALIBRATION=/workspace/vla-safety/calibration`; native installs use `${VLA_SAFETY_ROOT}/calibration` (or the repo root by default).
- Dataset stats and example tensors live in `assets/`. `new_stats_format.json` is the canonical normalization file; `stats.json` is kept for back-compat.
