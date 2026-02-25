#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import importlib
import importlib.metadata
import json
import logging
import os
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import TypeVar

import numpy as np
import torch
from accelerate import Accelerator
from datasets.utils.logging import disable_progress_bar, enable_progress_bar

JsonLike = str | int | float | bool | None | list["JsonLike"] | dict[str, "JsonLike"] | tuple["JsonLike", ...]
T = TypeVar("T", bound=JsonLike)


def is_package_available(pkg_name: str, import_name: str | None = None) -> bool:
    """Check if a package is installed and importable."""
    if importlib.util.find_spec(import_name or pkg_name) is None:
        return False
    try:
        importlib.metadata.version(pkg_name)
        return True
    except importlib.metadata.PackageNotFoundError:
        return False


_transformers_available = is_package_available("transformers")


def deserialize_json_into_object(fpath: Path, obj: T) -> T:
    """
    Loads the JSON data from `fpath` and recursively fills `obj` with the
    corresponding values (strictly matching structure and types).
    Tuples in `obj` are expected to be lists in the JSON data, which will be
    converted back into tuples.
    """
    with open(fpath, encoding="utf-8") as f:
        data = json.load(f)

    def _deserialize(target, source):
        if isinstance(target, dict):
            if not isinstance(source, dict):
                raise TypeError(f"Type mismatch: expected dict, got {type(source)}")
            if target.keys() != source.keys():
                raise ValueError(
                    f"Dictionary keys do not match.\nExpected: {target.keys()}, got: {source.keys()}"
                )
            for k in target:
                target[k] = _deserialize(target[k], source[k])
            return target
        elif isinstance(target, list):
            if not isinstance(source, list):
                raise TypeError(f"Type mismatch: expected list, got {type(source)}")
            if len(target) != len(source):
                raise ValueError(f"List length mismatch: expected {len(target)}, got {len(source)}")
            for i in range(len(target)):
                target[i] = _deserialize(target[i], source[i])
            return target
        elif isinstance(target, tuple):
            if not isinstance(source, list):
                raise TypeError(f"Type mismatch: expected list (for tuple), got {type(source)}")
            if len(target) != len(source):
                raise ValueError(f"Tuple length mismatch: expected {len(target)}, got {len(source)}")
            return tuple(_deserialize(t_item, s_item) for t_item, s_item in zip(target, source, strict=False))
        else:
            if type(target) is not type(source):
                raise TypeError(f"Type mismatch: expected {type(target)}, got {type(source)}")
            return source

    return _deserialize(obj, data)


def auto_select_torch_device() -> torch.device:
    """Tries to select automatically a torch device."""
    if torch.cuda.is_available():
        logging.info("Cuda backend detected, using cuda.")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        logging.info("Metal backend detected, using mps.")
        return torch.device("mps")
    elif torch.xpu.is_available():
        logging.info("Intel XPU backend detected, using xpu.")
        return torch.device("xpu")
    else:
        logging.warning("No accelerated backend detected. Using default cpu, this will be slow.")
        return torch.device("cpu")



def is_torch_device_available(try_device: str) -> bool:
    try_device = str(try_device)  # Ensure try_device is a string
    if try_device.startswith("cuda"):
        return torch.cuda.is_available()
    elif try_device == "mps":
        return torch.backends.mps.is_available()
    elif try_device == "xpu":
        return torch.xpu.is_available()
    elif try_device == "cpu":
        return True
    else:
        raise ValueError(f"Unknown device {try_device}. Supported devices are: cuda, mps, xpu or cpu.")


def is_amp_available(device: str):
    if device in ["cuda", "xpu", "cpu"]:
        return True
    elif device == "mps":
        return False
    else:
        raise ValueError(f"Unknown device '{device}.")


def init_logging(
    log_file: Path | None = None,
    display_pid: bool = False,
    console_level: str = "INFO",
    file_level: str = "DEBUG",
    accelerator: Accelerator | None = None,
):
    """Initialize logging configuration for LeRobot.

    In multi-GPU training, only the main process logs to console to avoid duplicate output.
    Non-main processes have console logging suppressed but can still log to file.

    Args:
        log_file: Optional file path to write logs to
        display_pid: Include process ID in log messages (useful for debugging multi-process)
        console_level: Logging level for console output
        file_level: Logging level for file output
        accelerator: Optional Accelerator instance (for multi-GPU detection)
    """

    def custom_format(record: logging.LogRecord) -> str:
        dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fnameline = f"{record.pathname}:{record.lineno}"
        pid_str = f"[PID: {os.getpid()}] " if display_pid else ""
        return f"{record.levelname} {pid_str}{dt} {fnameline[-15:]:>15} {record.getMessage()}"

    formatter = logging.Formatter()
    formatter.format = custom_format

    logger = logging.getLogger()
    logger.setLevel(logging.NOTSET)

    # Clear any existing handlers
    logger.handlers.clear()

    # Determine if this is a non-main process in distributed training
    is_main_process = accelerator.is_main_process if accelerator is not None else True

    # Console logging (main process only)
    if is_main_process:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(console_level.upper())
        logger.addHandler(console_handler)
    else:
        # Suppress console output for non-main processes
        logger.addHandler(logging.NullHandler())
        logger.setLevel(logging.ERROR)

    if log_file is not None:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(file_level.upper())
        logger.addHandler(file_handler)


def format_big_number(num, precision=0):
    suffixes = ["", "K", "M", "B", "T", "Q"]
    divisor = 1000.0

    for suffix in suffixes:
        if abs(num) < divisor:
            return f"{num:.{precision}f}{suffix}"
        num /= divisor

    return num


def has_method(cls: object, method_name: str) -> bool:
    return hasattr(cls, method_name) and callable(getattr(cls, method_name))


def is_valid_numpy_dtype_string(dtype_str: str) -> bool:
    """
    Return True if a given string can be converted to a numpy dtype.
    """
    try:
        # Attempt to convert the string to a numpy dtype
        np.dtype(dtype_str)
        return True
    except TypeError:
        # If a TypeError is raised, the string is not a valid dtype
        return False


@contextmanager
def suppress_progress_bars():
    """Context manager to suppress HuggingFace datasets progress bars."""
    disable_progress_bar()
    try:
        yield
    finally:
        enable_progress_bar()


