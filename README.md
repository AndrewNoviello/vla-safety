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
scripts/       Entry point: launch_menu.py — plus teleop, recording, deploy, dataset tools
launch/        ROS 2 launch files (one per pipeline) — see §6
configs/       Hydra configs for the hardware bridge
calibration/   SO101 follower / leader calibration files (see §13.3)
data/          LeRobot dataset loaders + augmentation; recordings land under data/recordings/
utils/         Shared utilities, paths, types, training helpers
assets/        Reference LeRobot v3 stats (see §13.2); not wired into deploy defaults
```

See **§13** for where stats, checkpoints, recordings, and calibration files go.

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

[scripts/launch_menu.py](scripts/launch_menu.py) is the recommended entry point for anything that touches the real robot. It presents an interactive menu, prompts for ports / checkpoint paths / prompts / dataset IDs, validates obvious mistakes (missing `/dev/` nodes, checkpoint files that don't exist), prints the exact command it will run, asks for confirmation, and then launches the pipeline.

After `pip install -e .` the same script is available as the console command `vla-launch`.

```bash
python scripts/launch_menu.py
# or:
vla-launch
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
  7. PI0 inference via lerobot-record
     Drive the follower with PI0 through upstream lerobot-record (no ROS bridge).

  q. quit
```

### 6.1 How the launcher works

For most pipelines the launcher assembles and runs a single foreground `ros2 launch …` command. Two pipelines use **custom runners** instead:

| Pipeline | Runner behavior |
|----------|-----------------|
| **3 — Record dataset** | Backgrounds `launch/teleop.launch.py` (logs → `logs/hardware-<timestamp>.log`), runs [scripts/recorder.py](scripts/recorder.py) in the foreground on your TTY. The recorder needs a real terminal for keyboard input and its live frame counter — neither works as a child of `ros2 launch`. On exit the hardware process group is SIGINT'd. |
| **7 — PI0 via lerobot-record** | Runs [scripts/lerobot_record_with_stub.py](scripts/lerobot_record_with_stub.py) in the foreground. Talks to the follower over serial directly (no ROS bridge). **Do not** start pipeline 1 in parallel — both would fight for the same USB port. |

**Validation:** serial ports and camera devices are checked for `/dev/…` prefix and presence on disk (warns if unplugged). Checkpoint paths must exist. Optional paths (e.g. `policy_dir`) may be left empty.

**CLI flags:**

| Flag | Effect |
|------|--------|
| `--menu N` | Skip the menu and jump straight to pipeline `N` (1–7). |
| `--dry-run` | Print the assembled command(s) without executing. |

Examples:

```bash
python scripts/launch_menu.py --menu 3          # jump to "Record dataset"
python scripts/launch_menu.py --menu 6 --dry-run  # show deploy-with-safety command
```

### 6.2 Pipeline details and prompted arguments

Each menu item maps to a launch file (or custom runner). Press Enter at any prompt to accept the default shown in `[brackets]`.

| # | Pipeline | What it starts | Key prompted args |
|---|----------|----------------|-------------------|
| 1 | Hardware only | [launch/hardware.launch.py](launch/hardware.launch.py) | `follower_port`, `camera_device`, `camera_rate_hz`, `robot_name` |
| 2 | Teleop | [launch/teleop.launch.py](launch/teleop.launch.py) | `follower_port`, `leader_port`, `camera_device` |
| 3 | Record dataset | Background teleop + foreground [recorder.py](scripts/recorder.py) | ports, `data_root`, `experiment_id`, `task_name`, `failure_extra_seconds` |
| 4 | Safety monitor | [launch/safety_monitor.launch.py](launch/safety_monitor.launch.py) | ports, `classifier_checkpoint`, `dataset_root`, `policy_dir` (optional) |
| 5 | Deploy PI0 (no filter) | [launch/deploy.launch.py](launch/deploy.launch.py) with `enable_safety:=false` | `follower_port`, `camera_device`, `pi0_model`, `prompt`, `control_hz`, `device` |
| 6 | Deploy PI0 + safety | [launch/deploy.launch.py](launch/deploy.launch.py) with `enable_safety:=true` | above + `wm_checkpoint`, `actor_checkpoint`, `critic_checkpoint`, `stats_path`, `epsilon` |
| 7 | PI0 via lerobot-record | [lerobot_record_with_stub.py](scripts/lerobot_record_with_stub.py) | `follower_port`, `camera_device`, `pi0_model`, `prompt`, `dataset_repo_id`, `episode_time_s`, `num_episodes`, `device` |

### 6.3 Launching without the menu

Every ROS-based pipeline is also a regular launch file under [launch/](launch/):

```bash
ros2 launch launch/teleop.launch.py follower_port:=/dev/ttyACM2 leader_port:=/dev/ttyACM3
ros2 launch launch/<file>.launch.py --show-args      # see all args + defaults
```

| File | Pipeline |
|------|----------|
| [launch/hardware.launch.py](launch/hardware.launch.py) | Robot node + webcam |
| [launch/teleop.launch.py](launch/teleop.launch.py) | Hardware + leader teleop |
| [launch/safety_monitor.launch.py](launch/safety_monitor.launch.py) | Hardware + teleop + live SAFE/UNSAFE monitor |
| [launch/deploy.launch.py](launch/deploy.launch.py) | Hardware + PI0 (pass `enable_safety:=true` to engage the filter) |

### 6.4 Scripts reference

All executable scripts live in [scripts/](scripts/). The table below is the full inventory; later sections go into detail on the common workflows.

| Script | Category | Purpose |
|--------|----------|---------|
| [launch_menu.py](scripts/launch_menu.py) | **Launcher** | Interactive menu for all robot pipelines (§6). |
| [teleop.py](scripts/teleop.py) | Robot | Leader → follower teleop bridge over ROS (§8). |
| [recorder.py](scripts/recorder.py) | Robot | Keyboard-driven episode recorder; flat Parquet + MP4 output (§8). |
| [deploy.py](scripts/deploy.py) | Robot | PI0 deployment node with optional latent safety filter (§12). Alias: `vla-deploy`. |
| [safety_monitor.py](scripts/safety_monitor.py) | Robot | Live SAFE/UNSAFE classifier readout; observe-only (§11). |
| [lerobot_record_with_stub.py](scripts/lerobot_record_with_stub.py) | Robot | Upstream `lerobot-record` wrapper for PI0 eval without ROS (§8.1). |
| [cache_latents.py](scripts/cache_latents.py) | Dataset | Pre-compute world-model latents for training (§9.1). |
| [filter_recordings.py](scripts/filter_recordings.py) | Dataset | Copy episodes matching a predicate (§9.2). |
| [merge_recordings.py](scripts/merge_recordings.py) | Dataset | Merge multiple flat-format datasets (§9.3). |
| [convert_to_lerobot_v3.py](scripts/convert_to_lerobot_v3.py) | Dataset | Flat format → LeRobot v3 chunked layout (§9.4). |
| [push_dataset_to_hub.py](scripts/push_dataset_to_hub.py) | Dataset | Upload a dataset folder to Hugging Face Hub (§9.5). |
| [overlay_safety.py](scripts/overlay_safety.py) | Analysis | Burn classifier scores onto episode videos (§11). |
| [smoke_cls_world_model.py](scripts/smoke_cls_world_model.py) | Dev | Fast tensor-shape smoke test for CLS-token world model (§11.3). |

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

For day-to-day use, prefer the launch menu: pipeline **2** for teleop and pipeline **3** for recording (§6). The sections below describe the underlying scripts if you want to run them manually or integrate them into your own workflow.

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

### 8.1 PI0 evaluation via lerobot-record (no ROS)

[scripts/lerobot_record_with_stub.py](scripts/lerobot_record_with_stub.py) is a thin wrapper around upstream `lerobot.scripts.lerobot_record`. It drives the SO101 follower **directly over serial** (not through the ROS hardware bridge) and runs PI0 inference while recording evaluation episodes to a local Hugging Face dataset layout.

The wrapper registers no-op stubs for processor steps (`delta_actions_processor`, `absolute_actions_processor`) that appear in saved PI0 checkpoint pipeline JSONs but are missing from the pinned `lerobot==0.4.4` registry. Those steps are `enabled=False` in the checkpoints, so pass-through is correct.

**Do not** run this alongside pipeline 1 or 2 — both would open the same follower serial port.

```bash
# via the launch menu (pipeline 7):
python scripts/launch_menu.py --menu 7

# or directly:
python scripts/lerobot_record_with_stub.py \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=follower_v1 \
    --robot.calibration_dir=calibration/robots/so_follower \
    --robot.cameras='{front: {type: opencv, index_or_path: /dev/video0, width: 640, height: 480, fps: 30}}' \
    --policy.type=pi0 \
    --policy.pretrained_path=AndrewNoviello/vla-safety-task-5 \
    --policy.device=cuda \
    --dataset.repo_id=local/eval_pi0_dominos \
    --dataset.single_task='pick up the middle domino' \
    --dataset.fps=20 \
    --dataset.episode_time_s=20 \
    --dataset.num_episodes=1 \
    --dataset.push_to_hub=false
```

Use a `local/…` repo id to keep recordings on disk only (no Hub push).

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

### 11.3 [scripts/smoke_cls_world_model.py](scripts/smoke_cls_world_model.py)

Developer smoke test for the CLS-token world-model tensor plumbing. Uses dummy encoder/decoder modules (no DINOv2 weights) to verify that `encode`, `predict`, `decode`, and `predict_failure` produce the expected shapes when `include_cls_token=True`. Run after changing [dino_wm/](dino_wm/) model code:

```bash
python scripts/smoke_cls_world_model.py
# prints: CLS world-model smoke test passed.
```

## 12. Deployment

For day-to-day use, prefer the launch menu: pipeline **5** for unfiltered PI0 and pipeline **6** for PI0 with the safety filter (§6).

[scripts/deploy.py](scripts/deploy.py) is a single ROS node that subscribes to image and joint-state topics, runs PI0 to propose an action, optionally gates the action through the latent safety filter, and publishes the (possibly overridden) action back to the robot. After `pip install -e .` it is also available as `vla-deploy`.

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

Useful [deploy.py](scripts/deploy.py) flags beyond what the launch menu exposes:

| Flag | Default | Notes |
|------|---------|-------|
| `--disable-safety` | off | Run PI0 with no latent filter (menu pipeline 5 sets this via launch file). |
| `--no-rtc` | off | Disable real-time chunking prefix guidance; inference still overlaps control but chunk seams may jump. |
| `--execution-horizon` | `10` | Refill PI0 action queue when this many ticks remain. |
| `--num-inference-steps` | `10` | Flow-matching denoising steps. |
| `--no-amp` | off | Disable CUDA autocast for PI0. |

### 12.3 Live safety monitor over teleoperation

To watch the failure classifier on a human-driven session rather than PI0's: use menu item 4 (or [launch/safety_monitor.launch.py](launch/safety_monitor.launch.py)). The monitor prints SAFE / UNSAFE per tick alongside the leader's commands but does not gate them. A gating node that intercepts the leader's `joint_command` is **not** implemented — the safety filter only intervenes against PI0 today.

## 13. Configuration and file locations

### 13.1 Where artifacts live

The repo uses a few recurring directory conventions. You do not need to create most of these by hand — recorders and training scripts write them automatically — but deployment and the launch menu expect checkpoints and stats at predictable paths.

```
<repo>/
├── calibration/                         ← robot arm calibration (see §13.2)
├── data/
│   ├── recordings/<experiment_id>/      ← raw teleop recordings (recorder.py)
│   │   ├── data/episode_*.parquet
│   │   ├── videos/episode_*.mp4
│   │   └── meta/stats.json              ← auto-written; flat scalar-key format
│   ├── exp_merged/                      ← example merged flat dataset
│   ├── exp_merged_v3/                   ← example after convert_to_lerobot_v3.py
│   └── exp_success_v3_clean/            ← bundled v3 dataset + stats used at deploy
├── runs/                                ← typical world-model + latent-cache output
│   └── dino_wm_exp_merged/
│       ├── checkpoints/latest/model.pt
│       └── latents/episode_*.pt         ← from cache_latents.py
├── outputs/                             ← typical classifier + safety-DDPG output
│   ├── classifier/classifier_best.pt
│   └── safety_ddpg/checkpoints/step_NNNNNNNN/{actor,critic}.pt
├── logs/hardware-<timestamp>.log        ← background hardware from launch menu pipeline 3
└── assets/stats.json                    ← reference LeRobot v3-format stats (not used by default)
```

| Artifact | Default location | Created by | Used by |
|----------|------------------|------------|---------|
| Raw recordings | `data/recordings/<experiment_id>/` | [recorder.py](scripts/recorder.py) | filter / merge / convert scripts |
| Dataset stats (flat) | `<dataset>/meta/stats.json` | recorder, filter, merge | training on flat datasets; safety monitor (`--dataset_root`) |
| Dataset stats (v3) | `<dataset>/meta/stats.json` | [convert_to_lerobot_v3.py](scripts/convert_to_lerobot_v3.py) | PI0 deploy (`--stats-path`), safety filter |
| World-model checkpoint | `runs/<exp>/checkpoints/latest/model.pt` | [dino_wm/train.py](dino_wm/train.py) | classifier training, safety DDPG, deploy |
| Cached latents | `runs/<exp>/latents/` | [cache_latents.py](scripts/cache_latents.py) | classifier / safety training (`--use_cached_latents`) |
| Classifier checkpoint | `outputs/classifier/classifier_best.pt` | [latentsafe/train_classifier.py](latentsafe/train_classifier.py) | safety monitor, overlay_safety |
| Safety actor/critic | `outputs/safety_ddpg/checkpoints/step_*/{actor,critic}.pt` | [latentsafe/train_safety_ddpg.py](latentsafe/train_safety_ddpg.py) | deploy with safety filter |
| PI0 weights | Hugging Face model ID (e.g. `AndrewNoviello/vla-safety-task-4`) | upstream training / Hub | deploy, lerobot-record |
| Arm calibration | [calibration/](calibration/) | LeRobot calibrate flow | hardware bridge, teleop, lerobot-record |
| Hydra robot logs | `outputs/<timestamp>/` | `robot_node_launcher` | debugging only |

Override any default path through the launch menu prompts, CLI flags, or launch-file args (`name:=value`).

### 13.2 Dataset stats (`meta/stats.json`)

Every dataset folder carries a `meta/stats.json` with per-feature min / max / mean / std used to normalize joint states and actions at inference time.

**Two formats exist:**

| Format | Keys look like | Produced by |
|--------|----------------|-------------|
| **Flat (scalar)** | `observation.state_j1`, `action_j1`, … | [recorder.py](scripts/recorder.py), [filter_recordings.py](scripts/filter_recordings.py), [merge_recordings.py](scripts/merge_recordings.py) |
| **LeRobot v3 (array)** | `observation.state`, `action` (length-6 arrays) | [convert_to_lerobot_v3.py](scripts/convert_to_lerobot_v3.py) |

**Which stats file to point at:**

- **Deploy (PI0 + safety filter)** — pass the stats from the dataset your PI0 checkpoint and safety stack were trained on. Default: [data/exp_success_v3_clean/meta/stats.json](data/exp_success_v3_clean/meta/stats.json). Override via launch menu pipeline 6 (`stats_path`), `ros2 launch … stats_path:=…`, or `deploy.py --stats-path …`.
- **Safety monitor** — pass the dataset root whose stats match the classifier training data: `--dataset_root data/exp_merged` (reads `<root>/meta/stats.json` internally).
- **Recording** — you do not place stats manually; [recorder.py](scripts/recorder.py) creates and incrementally updates `<experiment>/meta/stats.json` as episodes are saved.

If deploy cannot find the stats file it logs a warning and runs without normalization stats (PI0 falls back to its checkpoint defaults where available).

### 13.3 Calibration files

Calibration JSON files live under [calibration/](calibration/):

```
calibration/
├── robots/so_follower/
│   ├── follower_v1.json      ← used by lerobot-record (pipeline 7)
│   └── follower_arm.json
└── teleoperators/so_leader/
    └── leader_arm.json         ← used by teleop.py / hardware bridge
```

Inside Docker, `HF_LEROBOT_CALIBRATION=/workspace/vla-safety/calibration` is set by compose. On a native install LeRobot looks under the repo root by default. Re-calibrate with upstream LeRobot tooling if you swap arms or the joint zeros drift.

### 13.4 Checkpoints and model paths

When running from the launch menu, checkpoint prompts are validated for existence before launch. Typical workflow:

1. Record → `data/recordings/exp_…/`
2. Filter / merge → `data/exp_merged/`
3. Convert to v3 → `data/exp_merged_v3/` (optional but needed for PI0 training with LeRobot)
4. Train world model → `runs/dino_wm_exp_merged/checkpoints/latest/model.pt`
5. (Optional) Cache latents → `runs/dino_wm_exp_merged/latents/`
6. Train classifier → `outputs/classifier/classifier_best.pt` (launch menu default: `runs/classifier_exp_merged/classifier_best.pt`)
7. Train safety DDPG → `outputs/safety_ddpg/checkpoints/step_*/{actor,critic}.pt`
8. Deploy — point pipeline 6 at the paths from steps 4 and 7, plus the v3 stats from step 3

Edit the `output_dir` / `CFG.output_dir` constants in the training scripts to change where checkpoints land. PI0 base-policy weights are pulled from Hugging Face by model ID unless you train locally with [pi0/train.py](pi0/train.py).

### 13.5 Other configuration

- **Hydra**: configs live in [configs/](configs/). The launcher resolves [configs/config.yaml](configs/config.yaml) and pulls in [configs/robot/so101real.yaml](configs/robot/so101real.yaml) and [configs/ros_node/default.yaml](configs/ros_node/default.yaml) via the `defaults:` list. Override anything on the command line, e.g. `robot.config.port=/dev/ttyACM2`.
- **Console scripts** (after `pip install -e .`):
  - `so101-robot-node` → `python -m hardware.ros.scripts.robot_node_launcher`
  - `vla-deploy` → `python scripts/deploy.py`
  - `vla-launch` → `python scripts/launch_menu.py`
