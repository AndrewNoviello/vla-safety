import dataclasses
import json
import os
import types
from enum import Enum
from logging import getLogger
from pathlib import Path
from typing import Any, TypeVar, Union, get_args, get_origin, get_type_hints

from huggingface_hub import hf_hub_download
from huggingface_hub.constants import CONFIG_NAME
from huggingface_hub.errors import HfHubHTTPError

logger = getLogger(__name__)

T = TypeVar("T")


def _is_optional(tp) -> tuple[bool, type | None]:
    """Return (True, inner_type) if *tp* is ``X | None`` or ``Optional[X]``."""
    origin = get_origin(tp)
    if origin is Union or origin is types.UnionType:
        args = [a for a in get_args(tp) if a is not type(None)]
        if len(args) == 1:
            return True, args[0]
    return False, None


def _serialize(val: Any) -> Any:
    """Recursively convert a value to a JSON-compatible representation."""
    if val is None:
        return None
    if isinstance(val, Enum):
        return val.value
    if isinstance(val, Path):
        return str(val)
    if dataclasses.is_dataclass(val) and not isinstance(val, type):
        return {f.name: _serialize(getattr(val, f.name)) for f in dataclasses.fields(val)}
    if isinstance(val, dict):
        return {k: _serialize(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_serialize(v) for v in val]
    return val


def _deserialize(val: Any, target_type: Any) -> Any:
    """Recursively convert a JSON value to the expected Python type."""
    if val is None:
        return None

    is_opt, inner = _is_optional(target_type)
    if is_opt:
        return _deserialize(val, inner)

    if isinstance(target_type, type) and issubclass(target_type, Enum):
        return target_type(val)

    if isinstance(target_type, type) and issubclass(target_type, Path):
        return Path(val)

    if dataclasses.is_dataclass(target_type):
        hints = get_type_hints(target_type)
        kwargs = {}
        for f in dataclasses.fields(target_type):
            if f.name in val:
                kwargs[f.name] = _deserialize(val[f.name], hints[f.name])
        return target_type(**kwargs)

    origin = get_origin(target_type)
    args = get_args(target_type)

    if origin is dict and args:
        _, val_type = args
        return {k: _deserialize(v, val_type) for k, v in val.items()}

    if origin is tuple and args:
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_deserialize(v, args[0]) for v in val)
        return tuple(_deserialize(v, t) for v, t in zip(val, args))

    if isinstance(val, list) and origin is not list:
        if target_type is tuple:
            return tuple(val)

    return val


def _config_from_dict(config_cls: type[T], data: dict[str, Any]) -> T:
    """Instantiate a config dataclass from a plain dict, converting types as needed."""
    hints = get_type_hints(config_cls)
    known_fields = {f.name for f in dataclasses.fields(config_cls)}
    kwargs: dict[str, Any] = {}
    for key, val in data.items():
        if key not in known_fields:
            continue
        target_type = hints.get(key)
        if target_type is not None:
            kwargs[key] = _deserialize(val, target_type)
        else:
            kwargs[key] = val
    return config_cls(**kwargs)


def save_config(config: Any, save_directory: Path, type_name: str) -> None:
    """Serialize a config dataclass to config.json in save_directory."""
    save_directory = Path(save_directory)
    save_directory.mkdir(parents=True, exist_ok=True)
    d = _serialize(config)
    d["type"] = type_name
    with open(save_directory / CONFIG_NAME, "w") as f:
        json.dump(d, f, indent=4)


def load_config(config_cls: type[T], path: Path) -> T:
    """Load a config from a local directory (reads config.json, ignores the 'type' field)."""
    path = Path(path)
    with open(path / CONFIG_NAME) as f:
        data = json.load(f)
    data.pop("type", None)
    return _config_from_dict(config_cls, data)


def load_config_from_checkpoint(
    pretrained_name_or_path: str | Path,
    *,
    force_download: bool = False,
    resume_download: bool | None = None,
    proxies: dict[Any, Any] | None = None,
    token: str | bool | None = None,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
    revision: str | None = None,
    **kwargs,
) -> Any:
    """
    Load a config from a local directory or HuggingFace Hub checkpoint.

    Reads the 'type' field from config.json to dispatch to the correct config class
    via the policy registry.
    """
    from lerobot.policies.registry import get_config_class

    model_id = str(pretrained_name_or_path)
    config_file: str | None = None

    if Path(model_id).is_dir():
        if CONFIG_NAME in os.listdir(model_id):
            config_file = os.path.join(model_id, CONFIG_NAME)
        else:
            logger.error(f"{CONFIG_NAME} not found in {Path(model_id).resolve()}")
    else:
        try:
            config_file = hf_hub_download(
                repo_id=model_id,
                filename=CONFIG_NAME,
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                token=token,
                local_files_only=local_files_only,
            )
        except HfHubHTTPError as e:
            raise FileNotFoundError(
                f"{CONFIG_NAME} not found on the HuggingFace Hub in {model_id}"
            ) from e

    if config_file is None:
        raise FileNotFoundError(f"{CONFIG_NAME} not found in {model_id}")

    with open(config_file) as f:
        data = json.load(f)

    type_name = data.pop("type", None)
    if type_name is None:
        raise ValueError(f"'type' field missing from {config_file}")

    config_cls = get_config_class(type_name)
    return _config_from_dict(config_cls, data)
