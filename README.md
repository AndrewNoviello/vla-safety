# vla-safety

End-to-end stack for safely deploying a vision-language-action (VLA) policy on a real SO101 follower arm.

## 1. What this project is

This repository is an approximate replication of the experiment in *Generalizing Safety Beyond Collision-Avoidance via Latent-Space Reachability Analysis* (Nakamura, Peters, and Bajcsy). The paper introduces a **latent safety filter** that stops the main policy whenever it attempts an action that will lead to an unsafe state. A visual world model is trained on RGB observations, a failure classifier is trained in its latent space, and Hamilton–Jacobi reach-avoid analysis is run on those latents to decide, at every step, whether the action the base policy wants to take will cause an irrecoverable state. When it does, a learned safety policy takes over for one step.

In this repository the base policy is the **PI0** vision-language-action (VLA) model, the robot is a real **SO101** follower arm with a USB webcam attached to the wrist, and **ROS 2 Jazzy** is the underlying communication layer.

## 2. Layout

```
hardware/      ROS 2 nodes that drive the SO101 arm and the webcam
pi0/           PI0 base policy (proposes actions from image + prompt + joint state)
dino_wm/       DINOv2-based latent world model
latentsafe/    Failure classifier + reach-avoid safety actor/critic (the safety filter)
scripts/       Teleop, recording, dataset utilities, deployment, analysis tools
launch/        ROS 2 launch files (one per pipeline) — see §6
configs/       Hydra configs for the hardware bridge
calibration/   SO101 follower / leader calibration files
data/          LeRobot dataset loaders + augmentation; recordings land under data/recordings/
utils/         Shared utilities, paths, types, training helpers
assets/        Dataset stats, example tensors
```

## 3. Prerequisites

You need a Linux machine with:

- Docker (with `docker compose`) installed.
- An NVIDIA GPU and the matching host driver.
- The SO101 follower arm and the SO101 leader arm plugged in via USB. The compose file expects them at `/dev/ttyACM0` (follower) and `/dev/ttyACM1` (leader). You can override these — see §6.
- A USB webcam mounted to the robot's gripper (default `/dev/video0`).

Add your user to the `docker` group so that you can run `docker` commands without root:

```bash
sudo usermod -aG docker $USER
```

## 4. Get the repository

```bash
git clone --branch combined --single-branch https://github.com/AndrewNoviello/vlasafety.git
cd vla-safety
```

## 5. Run the Docker container

All dependencies (PyTorch with CUDA 13 wheels, ROS 2 Jazzy, LeRobot, Hydra, a patched fork of `transformers`, etc.) are baked into the Docker image.

```bash
docker compose build
docker compose run --rm vla_safety
```

That drops you into a Bash shell inside the container at `/workspace/vla-safety`, with the Python virtualenv already activated and ROS already sourced.

### 5.1 Verify GPU access

From inside the container:

```bash
nvidia-smi
```

If you don't see your GPU, the host is missing the NVIDIA Container Toolkit. On the host (not inside the container):

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

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

Then re-run `docker compose run --rm vla_safety` and check `nvidia-smi` again.

## 6. Quick start: the launch menu

The easiest way to bring up any pipeline (hardware only, teleop, recording, deploy, safety monitor) is the interactive launcher. It prompts you for ports, episode IDs, checkpoint paths, etc., warns when something looks wrong, and runs the right `ros2 launch` invocation for you.

```bash
python scripts/launch_menu.py
```

```
vla-safety launcher  repo: /workspace/vla-safety

Pick a pipeline:

  1. Hardware only
     Robot node + USB webcam. The base for everything else.
  2. Teleop session
     Hardware + leader teleop. Move the leader, the follower mirrors.
  3. Record dataset
     Hardware + teleop + recorder. 's'/'b' start, 'g'/'f'/'d' save, 'q' quit.
  4. Live safety monitor with teleop
     Hardware + teleop + per-tick SAFE/UNSAFE classifier readout.
  5. Deploy PI0 (no safety filter)  ← default deploy path
     Hardware + PI0 driving the arm directly. Safety filter is OFF.
  6. Deploy PI0 with safety filter
     Hardware + PI0 + latent reach-avoid safety filter gating each action.

  q. quit
```

Power-user shortcuts:

- `python scripts/launch_menu.py --menu 3`   — jump straight to "Record dataset".
- `python scripts/launch_menu.py --dry-run`  — print the assembled `ros2 launch` command without running it.

If you'd rather skip the menu, every pipeline is a regular ROS 2 launch file under [launch/](launch/) and can be invoked directly:

```bash
ros2 launch launch/teleop.launch.py follower_port:=/dev/ttyACM2 leader_port:=/dev/ttyACM3
ros2 launch launch/<file>.launch.py --show-args      # see all args + defaults
```

The available launch files are:

| File | Pipeline |
|------|----------|
| [launch/hardware.launch.py](launch/hardware.launch.py) | Robot node + webcam |
| [launch/teleop.launch.py](launch/teleop.launch.py) | Hardware + leader teleop |
| [launch/safety_monitor.launch.py](launch/safety_monitor.launch.py) | Hardware + teleop + live SAFE/UNSAFE monitor |
| [launch/deploy.launch.py](launch/deploy.launch.py) | Hardware + PI0 (with `enable_safety:=true` to engage the filter) |

## 7. Hardware nodes

Under the hood, the hardware bridge is two processes:

```bash
python -m hardware.ros.scripts.robot_node_launcher robot=so101real
python hardware/ros/examples/webcam.py --device /dev/video0
```

They publish on the `so101real` ROS namespace:

| Topic | Direction | Notes |
|-------|-----------|-------|
| `so101real/joint_state` | published | 6-DOF positions/velocities/efforts at 20 Hz |
| `so101real/joint_command` | subscribed | 6-DOF Float64MultiArray; only honored when connected + enabled + not estopped |
| `so101real/camera/image` | published | 640×480 BGR frames @ 30 Hz |
| `so101real/robot_status` | published | JSON status (connected / enabled / control_mode / estop) |
| `so101real/estop` | subscribed | `Bool` — `true` triggers emergency stop |

Lifecycle services (all in the `so101real` namespace): `connect_robot`, `enable_robot`, `set_control_mode`, `clear_estop`, `shutdown_robot`.

To override the follower's serial port, use either the launch arg or a Hydra override:

```bash
ros2 launch launch/hardware.launch.py follower_port:=/dev/ttyACM2
# or, raw:
python -m hardware.ros.scripts.robot_node_launcher robot=so101real robot.config.port=/dev/ttyACM2
```

After `pip install -e .`, the launcher is also available as the console script `so101-robot-node`.

## 8. Teleoperation and recording training data

To control the follower with the leader, on top of the hardware bridge run:

```bash
python scripts/teleop.py --leader-port /dev/ttyACM1
```

[scripts/teleop.py](scripts/teleop.py) reads the SO101 leader arm and republishes its joint positions on `so101real/joint_command`. Move the leader, the follower mirrors it.

To record training data, run [scripts/recorder.py](scripts/recorder.py) on top of the running teleop:

```bash
python scripts/recorder.py
# or skip the interactive prompts:
python scripts/recorder.py --experiment-id exp_42 --task-name 'pick the middle domino'
```

The recorder syncs joint state, joint command, EE pose, and camera frames to a 20 Hz master clock and writes one Parquet file per episode plus an MP4 of the camera. Every frame is labeled GOOD or BAD. It's an interactive keyboard tool:

| Key | Action |
|-----|--------|
| `s` | start recording, label this episode GOOD |
| `b` | start recording, label this episode BAD |
| `g` | finish a GOOD episode and save it |
| `f` | mark the current moment as the failure point; record `FAILURE_EXTRA` more seconds, then save |
| `d` | stop a BAD recording immediately and save |
| `q` | quit |

Environment overrides:

- `VLA_RECORD_DATA_ROOT` — where episodes go (default `<repo>/data/recordings`).
- `VLA_RECORD_FAILURE_EXTRA` — seconds of post-failure recording after `f` (default `5.0`).

Output layout (one folder per "experiment"):

```
<DATA_ROOT>/<experiment_id>/
  data/episode_NNN.parquet     ← one file per episode
  videos/episode_NNN.mp4       ← one video per episode
  meta/stats.json              ← running min/max/mean/std over every recorded episode
  README.md
```

To use the dataset for training, push it to the Hugging Face Hub as a LeRobot dataset (see §9.4) and reference its repo id in the training scripts.

Pipeline 3 in [scripts/launch_menu.py](scripts/launch_menu.py) wraps the full recording flow: it brings up `launch/teleop.launch.py` as a backgrounded process (logs land under `logs/hardware-<ts>.log`) and runs `scripts/recorder.py` in the foreground attached to your terminal — required because the recorder's keyboard input (`s`/`b`/`g`/`f`/`d`/`q`) and live frame counter need a real TTY, which `ros2 launch` doesn't provide to its children. On exit the hardware is SIGINT'd and waited on. The raw two-shell equivalent:

```bash
# shell 1
ros2 launch launch/teleop.launch.py follower_port:=/dev/ttyACM0 leader_port:=/dev/ttyACM1
# shell 2 (set env vars from the menu's data_root / failure_extra_seconds args)
VLA_RECORD_DATA_ROOT=$(pwd)/data/recordings \
VLA_RECORD_FAILURE_EXTRA=5.0 \
python scripts/recorder.py [--experiment-id X --task-name Y]
```

## 9. Dataset pipeline scripts

After raw recording, datasets typically pass through filtering, merging, and conversion before training. Each script lives in [scripts/](scripts/).

### 9.1 [scripts/cache_latents.py](scripts/cache_latents.py)

Pre-computes `z = model.encode(obs, act)` for every frame in a dataset and writes per-episode `.pt` files. Current world-model configs include a CLS token at `z[:, 0]`; cache manifests record `include_cls_token`, `num_tokens`, `num_visual_patches`, and `cls_token_index` so old patch-only caches fail clearly in CLS-enabled training.

```bash
python scripts/cache_latents.py \
    --wm_checkpoint runs/dino_wm_exp_merged/checkpoints/latest/model.pt \
    --output_dir   runs/dino_wm_exp_merged/latents
```

Key flags: `--dataset_repo_id`, `--store_full_patches`, `--batch_size`, `--num_workers`, `--max_episodes`.

### 9.2 [scripts/filter_recordings.py](scripts/filter_recordings.py)

Selects episodes from a flat-format dataset by predicate and copies them to a new folder, recomputing stats over the kept frames.

```bash
python scripts/filter_recordings.py \
    --out data/exp_success \
    --filter no-failures \
    hf:various-and-sundry/domino-trajectories-on-new-table
```

Predicates: `no-failures` (every frame's `label` is 1.0) or `all`. The source can be a local directory or `hf:<repo_id>`.

### 9.3 [scripts/merge_recordings.py](scripts/merge_recordings.py)

Merges multiple flat-format datasets into one, renumbering episodes and Welford-merging stats.

```bash
python scripts/merge_recordings.py \
    --out data/recordings/exp_merged \
    data/recordings/exp_01 \
    data/recordings/exp_02 \
    hf:various-and-sundry/domino-trajectories-on-new-table
```

### 9.4 [scripts/convert_to_lerobot_v3.py](scripts/convert_to_lerobot_v3.py)

Converts a flat-format dataset to the LeRobot v3.0 chunked format **without re-encoding videos**. Required before training PI0 with upstream LeRobot tooling.

```bash
python scripts/convert_to_lerobot_v3.py \
    --src data/recordings/exp_merged \
    --out data/exp_merged_v3 \
    --task 'pick a domino'
```

### 9.5 [scripts/push_dataset_to_hub.py](scripts/push_dataset_to_hub.py)

Uploads a flat or v3 dataset folder to the Hugging Face Hub. Reads your token from `huggingface-cli login`.

```bash
python scripts/push_dataset_to_hub.py data/recordings/exp_merged your-username/your-dataset
python scripts/push_dataset_to_hub.py data/exp_merged_v3 your-username/your-dataset --v3
```

## 10. Training models

There are four training scripts. The first two read their settings from module-level constants (no CLI args) — edit the file to override. The two `latentsafe` scripts have full CLIs.

### 10.1 PI0 base policy

```bash
python -m pi0.train
```

Edit [pi0/train.py](pi0/train.py) to change `DATASET_REPO_ID`, `BATCH_SIZE`, `STEPS`, `OUTPUT_DIR`, etc.

### 10.2 DINO-based world model

```bash
python -m dino_wm.train
```

Edit the `CFG` dataclass at the top of [dino_wm/train.py](dino_wm/train.py) to change the dataset, encoder, predictor depth, training steps, etc. The default world model rolls out a DINO CLS slot alongside patch tokens, so checkpoints and latent caches are not interchangeable with older patch-only runs. Checkpoints land in `<CFG.output_dir>/checkpoints/`.

### 10.3 Failure classifier (defines the unsafe set)

```bash
python -m latentsafe.train_classifier \
    --wm_checkpoint outputs/dino_wm_v2/checkpoints/latest/model.pt \
    --dataset_repo_id <your-hf-username>/<your-dataset>
```

Fine-tunes a small head on top of the frozen current/teacher-forced world-model latents to predict the GOOD/BAD label that the recorder wrote into each episode. With CLS-enabled world models, the head reads current CLS+proprio state features; during DDPG rollout the same head is applied to predicted CLS+proprio states for imagined future rewards. Useful flags:

| Flag | Default | Notes |
|------|---------|-------|
| `--steps` | `10000` | Training iterations |
| `--batch_size` | `64` | |
| `--lr` | `1e-4` | |
| `--use_cached_latents` | off | Use latents from `cache_latents.py` instead of re-encoding |
| `--cached_latents_dir` | `runs/dino_wm_exp_merged/latents` | |
| `--best_metric` | `recall` | Pick best checkpoint by `val_loss`, `accuracy`, `precision`, `recall`, or `f1` |
| `--output_dir` | `outputs/classifier` | |

### 10.4 Reach-avoid safety actor and critic

```bash
python -m latentsafe.train_safety_ddpg \
    --wm_checkpoint outputs/dino_wm_v2/checkpoints/latest/model.pt \
    --dataset_repo_id <your-hf-username>/<your-dataset>
```

This is the actual reach-avoid step from the paper. It uses the frozen world model as a simulator (no real robot needed) and the classifier to define the failure set, and trains a DDPG-style actor/critic pair on predicted CLS+proprio observations. The critic becomes the safety value V that gates actions at deployment time. Useful flags:

| Flag | Default | Notes |
|------|---------|-------|
| `--warmup_steps` | `10000` | γ=0 phase to bootstrap the value function |
| `--train_steps` | `40000` | Per-epoch length of the γ=0.95 phase |
| `--num_epochs` | `15` | |
| `--gamma` | `0.95` | |
| `--actor_lr` / `--critic_lr` | `1e-4` / `1e-3` | |
| `--p_unsafe_reset` | `0.5` | Probability of starting from an unsafe window |
| `--latent_cache_dir` | `runs/dino_wm_exp_merged/latents` | Pass an empty string to disable |
| `--output_dir` | `outputs/safety_ddpg` | Checkpoints land in `outputs/safety_ddpg/checkpoints/step_NNNNNNNN/{actor,critic}.pt` |

## 11. Analysis tools

### 11.1 [scripts/overlay_safety.py](scripts/overlay_safety.py)

Burns the failure-classifier score onto an episode's video frame-by-frame for post-hoc visualization. Reuses cached latents so DINOv2 isn't re-evaluated.

```bash
python scripts/overlay_safety.py \
    --episode 0 \
    --classifier_ckpt runs/classifier_exp_merged/classifier_best.pt \
    --latents_dir     runs/dino_wm_exp_merged/latents \
    --video_dir       data/exp_merged/videos \
    --threshold 0.0 \
    --show_gt
```

### 11.2 [scripts/safety_monitor.py](scripts/safety_monitor.py)

Live SAFE / UNSAFE printout at 20 Hz, driven by the failure classifier on the current camera + joint-state stream. Optionally also displays V(s) when given a `--policy_dir` containing `actor.pt` and `critic.pt`. **Observation only — does not gate or override commands.**

```bash
python scripts/safety_monitor.py \
    --checkpoint runs/classifier_exp_merged/classifier_best.pt \
    --dataset_root data/exp_merged \
    --policy_dir outputs/safety_ddpg/checkpoints/epoch_0015
```

Wrapped by [launch/safety_monitor.launch.py](launch/safety_monitor.launch.py) (menu item 4), which also brings up the hardware bridge and teleop so you can drive the arm and watch the readout in one shell.

## 12. Deployment

[scripts/deploy.py](scripts/deploy.py) is a single ROS node that subscribes to image and joint-state topics, runs PI0 to propose an action, optionally gates the action through the latent safety filter, and publishes the (possibly overridden) action back to the robot.

Log in to Hugging Face once so the script can pull PI0 weights:

```bash
huggingface-cli login
```

### 12.1 PI0 alone (default — no safety filter)

The default deployment path is **unfiltered** PI0. The safety filter has to be opted into.

```bash
ros2 launch launch/deploy.launch.py prompt:='pick up a domino'
# or, raw:
python scripts/deploy.py \
    --disable-safety \
    --pi0-model AndrewNoviello/vla-safety-task-1 \
    --prompt 'pick up a domino'
```

### 12.2 PI0 with the latent safety filter (opt-in)

```bash
ros2 launch launch/deploy.launch.py enable_safety:=true epsilon:=0.3
# or, raw:
python scripts/deploy.py \
    --pi0-model AndrewNoviello/vla-safety-task-1 \
    --wm-checkpoint outputs/dino_wm_v2/checkpoints/latest/model.pt \
    --actor-checkpoint outputs/safety_ddpg/checkpoints/epoch_0015/actor.pt \
    --critic-checkpoint outputs/safety_ddpg/checkpoints/epoch_0015/critic.pt \
    --epsilon 0.3 \
    --prompt 'pick up a domino'
```

`--epsilon` (default `0.3`) is the safety-value threshold ε from the paper. If V(ẑ_{t+1}) for the proposed action exceeds ε, the action is overridden by the safety policy and the queued PI0 action chunk is flushed.

### 12.3 Live safety monitor over teleoperation

To watch the failure classifier on a human-driven session rather than PI0's: use menu item 4 (or [launch/safety_monitor.launch.py](launch/safety_monitor.launch.py)). The monitor prints SAFE / UNSAFE per tick alongside the leader's commands but does not gate them. A gating node that intercepts the leader's `joint_command` is **not** implemented — the safety filter only intervenes against PI0 today.

## 13. Configuration

- **Hydra**: configs live in [configs/](configs/). The launcher resolves [configs/config.yaml](configs/config.yaml) and pulls in [configs/robot/so101real.yaml](configs/robot/so101real.yaml) and [configs/ros_node/default.yaml](configs/ros_node/default.yaml) via the `defaults:` list. Override anything on the command line, e.g. `robot.config.port=/dev/ttyACM2`.
- **Calibration**: files live in [calibration/](calibration/). The Docker compose mount sets `HF_LEROBOT_CALIBRATION=/workspace/vla-safety/calibration`; native installs use the repo root by default.
- **Dataset stats**: each recorded experiment ships its own `meta/stats.json`. Deployment uses [data/exp_success_v3_clean/meta/stats.json](data/exp_success_v3_clean/meta/stats.json) by default; override with `stats_path:=...` in the launch file or `--stats-path` on `deploy.py`.
- **Console scripts** (after `pip install -e .`):
  - `so101-robot-node` → `python -m hardware.ros.scripts.robot_node_launcher`
  - `vla-deploy` → `python scripts/deploy.py`
  - `vla-launch` → `python scripts/launch_menu.py`
