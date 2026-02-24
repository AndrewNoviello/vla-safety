#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Hardcoded PI05 architecture and flow-matching constants (lerobot/pi05_base)."""

DEFAULT_IMAGE_SIZE = 224

# Architecture
PI05_DEFAULT_PALIGEMMA = "gemma_2b"
PI05_DEFAULT_ACTION_EXPERT = "gemma_300m"
PI05_DEFAULT_CHUNK_SIZE = 50
PI05_DEFAULT_N_ACTION_STEPS = 50
PI05_DEFAULT_MAX_STATE_DIM = 32
PI05_DEFAULT_MAX_ACTION_DIM = 32
PI05_DEFAULT_IMAGE_RESOLUTION = (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)
PI05_DEFAULT_DTYPE = "float32"

# Flow-matching
PI05_DEFAULT_NUM_INFERENCE_STEPS = 10
PI05_DEFAULT_TIME_SAMPLING_BETA_ALPHA = 1.5
PI05_DEFAULT_TIME_SAMPLING_BETA_BETA = 1.0
PI05_DEFAULT_TIME_SAMPLING_SCALE = 0.999
PI05_DEFAULT_TIME_SAMPLING_OFFSET = 0.001
PI05_DEFAULT_MIN_PERIOD = 4e-3
PI05_DEFAULT_MAX_PERIOD = 4.0

# Training
PI05_DEFAULT_GRADIENT_CHECKPOINTING = False
PI05_DEFAULT_COMPILE_MODEL = False
PI05_DEFAULT_COMPILE_MODE = "max-autotune"
PI05_DEFAULT_FREEZE_VISION_ENCODER = False
PI05_DEFAULT_TRAIN_EXPERT_ONLY = False

# Optimizer / Scheduler
PI05_DEFAULT_OPTIMIZER_LR = 2.5e-5
PI05_DEFAULT_OPTIMIZER_BETAS = (0.9, 0.95)
PI05_DEFAULT_OPTIMIZER_EPS = 1e-8
PI05_DEFAULT_OPTIMIZER_WEIGHT_DECAY = 0.01
PI05_DEFAULT_OPTIMIZER_GRAD_CLIP_NORM = 1.0
PI05_DEFAULT_SCHEDULER_WARMUP_STEPS = 1_000
PI05_DEFAULT_SCHEDULER_DECAY_STEPS = 30_000
PI05_DEFAULT_SCHEDULER_DECAY_LR = 2.5e-6

# Tokenizer
PI05_DEFAULT_TOKENIZER_MAX_LENGTH = 200
