# Bimanual Recording System

Synchronized data recording for bimanual robot manipulation with tactile sensors.

## Overview

This system records synchronized data at 15fps from:
- **rowlet** robot (left arm) - joint positions, velocities, efforts
- **piplup** robot (right arm) - joint positions, velocities, efforts  
- **digitL** tactile sensor (left) - 240x320 RGB images
- **digitR** tactile sensor (right) - 240x320 RGB images

The **digitL sensor serves as the master clock**, ensuring all data sources are perfectly synchronized.

## Directory Structure

```
/home/Shared/data/
├── exp_01/
│   ├── traj_0/
│   │   ├── timestamps.npy
│   │   ├── rowlet_positions.npy
│   │   ├── rowlet_velocities.npy
│   │   ├── rowlet_efforts.npy
│   │   ├── piplup_positions.npy
│   │   ├── piplup_velocities.npy
│   │   ├── piplup_efforts.npy
│   │   ├── digitL/
│   │   │   ├── frame_000000.png
│   │   │   ├── frame_000001.png
│   │   │   └── ...
│   │   ├── digitR/
│   │   │   ├── frame_000000.png
│   │   │   ├── frame_000001.png
│   │   │   └── ...
│   │   └── metadata.json
│   ├── traj_1/
│   └── traj_2/
├── exp_02/
│   ├── traj_0/
│   └── traj_1/
└── ...
```

## Data Format

All data is synchronized with a single timestamp array. For N frames recorded:

- `timestamps.npy`: shape `[N]` - timestamp for each frame
- `rowlet_positions.npy`: shape `[N, num_joints]` - joint positions
- `piplup_positions.npy`: shape `[N, num_joints]` - joint positions
- `digitL/frame_{i:06d}.png`: left sensor images (i = 0 to N-1)
- `digitR/frame_{i:06d}.png`: right sensor images (i = 0 to N-1)
- `metadata.json`: recording metadata

**Note:** Only joint positions are saved. Velocities and efforts are not recorded.

### Metadata Format

```json
{
  "experiment_id": 1,
  "trajectory_num": 0,
  "num_frames": 150,
  "duration_seconds": 10.0,
  "actual_fps": 15.0,
  "target_fps": 15.0,
  "recording_start": "2025-11-25T12:00:00"
}
```

## Recording Data

### Prerequisites

Make sure all nodes are running:
```bash
# Launch robot nodes (rowlet and piplup)
bash praxis/scripts/launch_bimanual.sh

# Launch sensor nodes (digitL and digitR)
# Make sure they are configured for 15fps
```

### Start Recording

```bash
python -m praxis.ros.examples.bimanual_recorder
```

The script will prompt you for an experiment ID at startup. This experiment ID applies to **all recordings in the session**.

### Workflow

1. **Start the script** - prompts for experiment ID
2. **Press 's'** - start recording (saves to exp_XX/traj_0)
3. **Press 's'** - stop and save
4. **Press 's'** - start next recording (saves to exp_XX/traj_1, auto-incremented!)
5. **Repeat** as needed...
6. **Press 'q'** - quit when done

### Controls

- **Press 's'**: Toggle recording (start/stop)
  - First press: starts recording
  - Second press: stops and saves
  - Trajectory number auto-increments after each save
- **Press 'q'**: Quit application

### Example Session

```
$ python -m praxis.ros.examples.bimanual_recorder

Enter experiment ID for this session (integer): 1

All recordings in this session will be saved to:
  /home/Shared/data/exp_01/traj_X
Proceed? [Y/n]: y

[INFO] Bimanual Recorder initialized
[INFO] Experiment ID: 1
[INFO] Next trajectory: exp_01/traj_0

[Window opens showing live sensor feeds]

[Press 's']
[INFO] RECORDING STARTED (exp_01/traj_0)
[INFO] Save path: /home/Shared/data/exp_01/traj_0
...

[Press 's' again]
[INFO] RECORDING STOPPED (150 frames)
[INFO] Saving 150 synchronized frames to: /home/Shared/data/exp_01/traj_0
[INFO] Recording saved successfully!
[INFO] Duration: 10.00s | Frames: 150 | FPS: 15.00
[INFO] Next recording will be: exp_01/traj_1

[Press 's' again for next recording]
[INFO] RECORDING STARTED (exp_01/traj_1)  # Auto-incremented!
[INFO] Save path: /home/Shared/data/exp_01/traj_1
...

[Press 's' to stop]
[INFO] RECORDING STOPPED (200 frames)
[INFO] Saving 200 synchronized frames to: /home/Shared/data/exp_01/traj_1
[INFO] Next recording will be: exp_01/traj_2

[Can continue recording more trajectories or press 'q' to quit]
```

## Loading and Analyzing Data

### List Experiments

```bash
# List all experiments
python -m praxis.ros.examples.load_bimanual_recording --list

# List trajectories in experiment 1
python -m praxis.ros.examples.load_bimanual_recording --list --exp 1
```

### Load and Verify Data

```bash
# Using exp/traj numbers
python -m praxis.ros.examples.load_bimanual_recording --exp 1 --traj 0

# Using full path
python -m praxis.ros.examples.load_bimanual_recording /home/Shared/data/exp_01/traj_0
```

### Play Back Recording

```bash
# Play at recorded speed (15fps)
python -m praxis.ros.examples.load_bimanual_recording --exp 1 --traj 0 --play

# Play at custom speed
python -m praxis.ros.examples.load_bimanual_recording --exp 1 --traj 0 --play --fps 5.0
```

Playback controls:
- **q**: Quit
- **Space**: Pause/Resume
- **n**: Next frame (when paused)

### Load Data in Python

```python
import numpy as np
import cv2
from pathlib import Path

# Load trajectory
traj_dir = Path("/home/Shared/data/exp_01/traj_0")

# Load timestamps (shape: [N])
timestamps = np.load(traj_dir / "timestamps.npy")

# Load joint positions (shape: [N, num_joints])
rowlet_pos = np.load(traj_dir / "rowlet_positions.npy")
piplup_pos = np.load(traj_dir / "piplup_positions.npy")

# Load a specific frame
frame_idx = 50
digitL_img = cv2.imread(str(traj_dir / f"digitL/frame_{frame_idx:06d}.png"))
digitR_img = cv2.imread(str(traj_dir / f"digitR/frame_{frame_idx:06d}.png"))

# Access synchronized data for frame 50
t = timestamps[frame_idx]
rowlet_joints = rowlet_pos[frame_idx]
piplup_joints = piplup_pos[frame_idx]

print(f"Frame {frame_idx} at time {t:.3f}s")
print(f"Rowlet joints: {rowlet_joints}")
print(f"Piplup joints: {piplup_joints}")
# digitL_img and digitR_img are the corresponding sensor images
```

## Synchronization Details

1. **digitL sensor runs at 15fps** and publishes to `/digitL/sensor_reading`
2. When a digitL frame arrives, the recorder captures:
   - The latest digitR image
   - The latest rowlet joint state
   - The latest piplup joint state
3. All data is stored with the same index, ensuring perfect synchronization
4. This guarantees exactly **1 sample from each source per timestamp**

## Troubleshooting

### "Waiting for all data sources. Missing: ..."

This warning appears if some data sources haven't published yet. Common causes:
- A sensor or robot node isn't running
- Network delays on startup

Wait a few seconds for all nodes to start publishing.

### Recording has 0 frames

Make sure all nodes are publishing before starting the recording:
- Check that all status indicators show "OK" in the live view
- Wait until both sensor images are visible in the window

### Frame count mismatch

If the verification shows different frame counts, check:
- Sensor nodes are configured for the same frame rate (15fps)
- No nodes crashed during recording
- Sufficient disk space for saving images

### Input prompt appears at wrong time

The experiment ID prompt appears **once at startup** in the terminal where you launched the script, not in the OpenCV window. After entering it once, all subsequent recordings in that session use the same experiment ID with auto-incremented trajectory numbers.

## Configuration

To change the recording frequency, update both:
1. **Sensor configurations** (digit.yaml): Set `sample_rate: 15.0`
2. **Robot node update rates**: Set to at least 15Hz or higher

The system uses digitL as the master clock, so all data will be recorded at digitL's actual frame rate.

