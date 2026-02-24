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

"""Hardcoded PI0 architecture and flow-matching constants (lerobot/pi0_base)."""

DEFAULT_IMAGE_SIZE = 224

# Architecture
PI0_DEFAULT_PALIGEMMA = "gemma_2b"
PI0_DEFAULT_ACTION_EXPERT = "gemma_300m"
PI0_DEFAULT_CHUNK_SIZE = 50
PI0_DEFAULT_N_ACTION_STEPS = 50
PI0_DEFAULT_MAX_STATE_DIM = 32
PI0_DEFAULT_MAX_ACTION_DIM = 32
PI0_DEFAULT_IMAGE_RESOLUTION = (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)
PI0_DEFAULT_DTYPE = "float32"

# Flow-matching
PI0_DEFAULT_NUM_INFERENCE_STEPS = 10
PI0_DEFAULT_TIME_SAMPLING_BETA_ALPHA = 1.5
PI0_DEFAULT_TIME_SAMPLING_BETA_BETA = 1.0
PI0_DEFAULT_TIME_SAMPLING_SCALE = 0.999
PI0_DEFAULT_TIME_SAMPLING_OFFSET = 0.001
PI0_DEFAULT_MIN_PERIOD = 4e-3
PI0_DEFAULT_MAX_PERIOD = 4.0

# Training (hardcoded)
PI0_DEFAULT_GRADIENT_CHECKPOINTING = False
PI0_DEFAULT_COMPILE_MODEL = False
PI0_DEFAULT_COMPILE_MODE = "max-autotune"
PI0_DEFAULT_FREEZE_VISION_ENCODER = False
PI0_DEFAULT_TRAIN_EXPERT_ONLY = False

# Tokenizer
PI0_DEFAULT_TOKENIZER_MAX_LENGTH = 48
