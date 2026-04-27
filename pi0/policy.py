"""PI0 Policy - LeRobot interface for the PI0 model."""

import builtins
import logging
from collections import deque
from pathlib import Path

import torch
from torch import Tensor
from typing_extensions import Unpack

from utils.types import FeatureType, PolicyFeature
from pi0.config import PI0Config
from pi0.model import PI0Pytorch, ActionSelectKwargs, pad_vector
from pi0.pretrained import PreTrainedPolicy, T
from utils.utils import auto_select_torch_device, is_amp_available, is_torch_device_available
from pi0.rtc_model import RTCProcessor
from utils.constants import (
    ACTION,
    OBS_IMAGES,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)


class PI0Policy(PreTrainedPolicy):
    """PI0 OpenPI Policy for LeRobot."""

    config_class = PI0Config
    name = "pi0"

    def __init__(self, config: PI0Config, **kwargs):
        super().__init__(config)

        if not config.device or not is_torch_device_available(config.device):
            auto_device = auto_select_torch_device()
            logging.getLogger(__name__).warning(
                f"Device '{config.device}' is not available. Switching to '{auto_device}'."
            )
            config.device = auto_device.type

        if config.use_amp and not is_amp_available(config.device):
            logging.getLogger(__name__).warning(
                f"Automatic Mixed Precision (amp) is not available on device '{config.device}'. Deactivating AMP."
            )
            config.use_amp = False

        self.init_rtc_processor()
        self.model = PI0Pytorch(config, rtc_processor=self.rtc_processor)

        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self.model.to(config.device)

        self.reset()

    def init_rtc_processor(self):
        """Initialize RTC processor if RTC is enabled."""
        self.rtc_processor = None
        rtc_config = getattr(self.config, "rtc_config", None)
        if rtc_config is not None:
            self.rtc_processor = RTCProcessor(rtc_config)

            model_value = getattr(self, "model", None)
            if model_value is not None:
                model_value.rtc_processor = self.rtc_processor

    @property
    def image_resolution(self) -> tuple[int, int]:
        return self.config.image_resolution

    @property
    def max_state_dim(self) -> int:
        return self.config.max_state_dim

    @property
    def max_action_dim(self) -> int:
        return self.config.max_action_dim

    @property
    def n_action_steps(self) -> int:
        return self.config.n_action_steps

    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: PI0Config | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        strict: bool = True,
        **kwargs,
    ) -> T:
        """Load PI0 from Hub or local path. Architecture is hardcoded for lerobot/pi0_base."""
        print(
            "The PI0 model is a direct port of the OpenPI implementation. \n"
            "This implementation follows the original OpenPI structure for compatibility. \n"
            "Original implementation: https://github.com/Physical-Intelligence/openpi"
        )
        if pretrained_name_or_path is None:
            raise ValueError("pretrained_name_or_path is required")

        # Load config: use provided config, or load from checkpoint, or build default for old checkpoints
        if config is None:
            try:
                config = PI0Config.from_pretrained(
                    pretrained_name_or_path=pretrained_name_or_path,
                    force_download=force_download,
                    resume_download=resume_download,
                    proxies=proxies,
                    token=token,
                    cache_dir=cache_dir,
                    local_files_only=local_files_only,
                    revision=revision,
                    **kwargs,
                )
            except FileNotFoundError:
                # Old checkpoints without config.json: build default PI0Config
                input_features = kwargs.get("input_features") or {}
                output_features = kwargs.get("output_features") or {}
                if OBS_STATE not in input_features:
                    input_features[OBS_STATE] = PolicyFeature(
                        type=FeatureType.STATE,
                        shape=(32,),
                    )
                if ACTION not in output_features:
                    output_features[ACTION] = PolicyFeature(
                        type=FeatureType.ACTION,
                        shape=(32,),
                    )
                if not any(ft.type == FeatureType.VISUAL for ft in input_features.values()):
                    input_features[f"{OBS_IMAGES}.top"] = PolicyFeature(
                        type=FeatureType.VISUAL,
                        shape=(3, 224, 224),
                    )
                config = PI0Config(
                    input_features=input_features,
                    output_features=output_features,
                    pretrained_path=Path(pretrained_name_or_path),
                )

        model = cls(config, **kwargs)

        # Now manually load and remap the state dict
        try:
            # Try to load the pytorch_model.bin or model.safetensors file
            print(f"Loading model from: {pretrained_name_or_path}")
            try:
                from transformers.utils import cached_file

                # Try safetensors first
                resolved_file = cached_file(
                    pretrained_name_or_path,
                    "model.safetensors",
                    cache_dir=cache_dir,
                    force_download=force_download,
                    resume_download=resume_download,
                    proxies=proxies,
                    use_auth_token=token,
                    revision=revision,
                    local_files_only=local_files_only,
                )
                from safetensors.torch import load_file

                original_state_dict = load_file(resolved_file)
                print("✓ Loaded state dict from model.safetensors")
            except Exception as e:
                print(f"Could not load state dict from remote files: {e}")
                print("Returning model without loading pretrained weights")
                return model

            # First, fix any key differences # see openpi `model.py, _fix_pytorch_state_dict_keys`
            fixed_state_dict = model._fix_pytorch_state_dict_keys(original_state_dict)

            # Then add "model." prefix for all keys that don't already have it
            remapped_state_dict = {}
            remap_count = 0

            for key, value in fixed_state_dict.items():
                if not key.startswith("model."):
                    new_key = f"model.{key}"
                    remapped_state_dict[new_key] = value
                    remap_count += 1
                else:
                    remapped_state_dict[key] = value

            if remap_count > 0:
                print(f"Remapped {remap_count} state dict keys")

            # Load the remapped state dict into the model
            missing_keys, unexpected_keys = model.load_state_dict(remapped_state_dict, strict=strict)

            if missing_keys:
                print(f"Missing keys when loading state dict: {len(missing_keys)} keys")
                if len(missing_keys) <= 5:
                    for key in missing_keys:
                        print(f"  - {key}")
                else:
                    for key in missing_keys[:5]:
                        print(f"  - {key}")
                    print(f"  ... and {len(missing_keys) - 5} more")

            if unexpected_keys:
                print(f"Unexpected keys when loading state dict: {len(unexpected_keys)} keys")
                if len(unexpected_keys) <= 5:
                    for key in unexpected_keys:
                        print(f"  - {key}")
                else:
                    for key in unexpected_keys[:5]:
                        print(f"  - {key}")
                    print(f"  ... and {len(unexpected_keys) - 5} more")

            if not missing_keys and not unexpected_keys:
                print("All keys loaded successfully!")

        except Exception as e:
            print(f"Warning: Could not remap state dict keys: {e}")

        return model

    def _fix_pytorch_state_dict_keys(self, state_dict):  # see openpi `BaseModelConfig, _fix_pytorch_state_dict_keys`
        """Fix state dict keys to match current model architecture."""
        import re

        fixed_state_dict = {}

        for key, value in state_dict.items():
            new_key = key

            # Handle layer norm structure changes: .weight -> .dense.weight + .dense.bias
            # For gemma expert layers
            if re.match(
                r"paligemma_with_expert\.gemma_expert\.model\.layers\.\d+\.(input_layernorm|post_attention_layernorm)\.weight",
                key,
            ):
                # Check if the model actually has adaRMS enabled for the expert
                expert_uses_adarms = getattr(
                    self.model.paligemma_with_expert.gemma_expert.config, "use_adarms", False
                )
                if expert_uses_adarms:
                    logging.warning(f"Skipping layer norm key (adaRMS mismatch): {key}")
                    continue

            if re.match(r"paligemma_with_expert\.gemma_expert\.model\.norm\.weight", key):
                # Check if the model actually has adaRMS enabled for the expert
                expert_uses_adarms = getattr(
                    self.model.paligemma_with_expert.gemma_expert.config, "use_adarms", False
                )
                if expert_uses_adarms:
                    logging.warning(f"Skipping norm key (adaRMS mismatch): {key}")
                    continue

            # Handle MLP naming changes for pi0
            # pi0 model expects action_time_mlp_*, but checkpoint might have time_mlp_*
            if key.startswith("time_mlp_in."):
                new_key = key.replace("time_mlp_in.", "action_time_mlp_in.")
            elif key.startswith("time_mlp_out."):
                new_key = key.replace("time_mlp_out.", "action_time_mlp_out.")

            # Handle vision tower embedding layer potential differences
            if "patch_embedding" in key:
                # Some checkpoints might have this, but current model expects different structure
                logging.warning(f"Vision embedding key might need handling: {key}")

            fixed_state_dict[new_key] = value

        return fixed_state_dict

    def reset(self):
        """Reset internal state - called when environment resets."""
        self._action_queue = deque(maxlen=self.n_action_steps)
        self._queues = {
            ACTION: deque(maxlen=self.n_action_steps),
        }

    def _rtc_enabled(self) -> bool:
        rtc_config = getattr(self.config, "rtc_config", None)
        return rtc_config is not None and rtc_config.enabled

    def _preprocess_images(self, batch: dict[str, Tensor]) -> tuple[list[Tensor], list[Tensor]]:
        """Preprocess images for the model.

        Images from LeRobot are typically in [B, C, H, W] format and normalized to [0, 1].
        PaliGemma expects images in [B, C, H, W] format and normalized to [-1, 1].
        """
        images = []
        img_masks = []

        # Get device from model parameters
        device = next(self.parameters()).device

        image_features = {k: v for k, v in self.config.input_features.items() if v.type is FeatureType.VISUAL}
        present_img_keys = [key for key in image_features if key in batch]
        missing_img_keys = [key for key in image_features if key not in batch]

        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. "
                f"(batch: {batch.keys()}) (image_features: {image_features})"
            )

        for key in present_img_keys:
            img = batch[key]

            # Ensure tensor is on the same device as the model
            if img.device != device:
                img = img.to(device)

            # Ensure float32 dtype for consistency
            if img.dtype != torch.float32:
                img = img.to(torch.float32)

            # from openpi preprocess_observation_pytorch: Handle both [B, C, H, W] and [B, H, W, C] formats
            is_channels_first = img.shape[1] == 3  # Check if channels are in dimension 1

            if is_channels_first:
                # Convert [B, C, H, W] to [B, H, W, C] for processing
                img = img.permute(0, 2, 3, 1)

            # Resize is done in preprocessor; images are expected at image_resolution here

            # Normalize from [0,1] to [-1,1] as expected by siglip
            img = img * 2.0 - 1.0

            # from openpi preprocess_observation_pytorch: Convert back to [B, C, H, W] format if it was originally channels-first
            if is_channels_first:
                img = img.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]

            images.append(img)
            # Create mask (all ones for real images)
            bsize = img.shape[0]
            mask = torch.ones(bsize, dtype=torch.bool, device=device)
            img_masks.append(mask)

        # Create image features not present in the batch as fully 0 padded images
        for _num_empty_cameras in range(len(missing_img_keys)):
            img = torch.ones_like(img) * -1  # padded with -1 for SigLIP
            mask = torch.zeros_like(mask)  # mask is zero for empty cameras
            images.append(img)
            img_masks.append(mask)

        return images, img_masks

    def prepare_state(self, batch):
        """Pad state"""
        state = pad_vector(batch[OBS_STATE], self.max_state_dim)
        return state

    def prepare_action(self, batch):
        """Pad action"""
        actions = pad_vector(batch[ACTION], self.max_action_dim)
        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations."""
        assert not self._rtc_enabled(), (
            "RTC is not supported for select_action, use it with predict_action_chunk"
        )

        self.eval()

        # Action queue logic for n_action_steps > 1
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.n_action_steps]
            # Transpose to get shape (n_action_steps, batch_size, action_dim)
            self._action_queue.extend(actions.transpose(0, 1))

        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        self.eval()

        # Prepare inputs
        images, img_masks = self._preprocess_images(batch)
        lang_tokens, lang_masks = batch[f"{OBS_LANGUAGE_TOKENS}"], batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
        state = self.prepare_state(batch)

        # Sample actions using the model (pass through RTC kwargs)
        actions = self.model.sample_actions(images, img_masks, lang_tokens, lang_masks, state, **kwargs)

        # Unpad actions to actual action dimension
        original_action_dim = self.output_features[ACTION].shape[0]
        actions = actions[:, :, :original_action_dim]

        return actions

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean") -> tuple[Tensor, dict]:
        """Run the batch through the model and compute the loss for training.

        Args:
            batch: Training batch containing observations and actions.
            reduction: How to reduce the loss. Options:
                - "mean": Return scalar mean loss (default, backward compatible)
                - "none": Return per-sample losses of shape (batch_size,) for RA-BC weighting
        """
        # Prepare inputs
        images, img_masks = self._preprocess_images(batch)
        lang_tokens, lang_masks = batch[f"{OBS_LANGUAGE_TOKENS}"], batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
        state = self.prepare_state(batch)
        actions = self.prepare_action(batch)

        # Compute loss
        losses = self.model.forward(images, img_masks, lang_tokens, lang_masks, state, actions)

        # Truncate losses to actual action dimensions
        original_action_dim = self.output_features[ACTION].shape[0]
        losses = losses[:, :, :original_action_dim]

        loss_dict = {
            "loss_per_dim": losses.mean(dim=[0, 1]).detach().cpu().numpy().tolist(),
        }

        if reduction == "none":
            # Return per-sample losses (B,) by averaging over time and action dims
            per_sample_loss = losses.mean(dim=(1, 2))
            loss_dict["loss"] = per_sample_loss.mean().item()
            return per_sample_loss, loss_dict
        else:
            # Default: return scalar mean loss
            loss = losses.mean()
            loss_dict["loss"] = loss.item()
            return loss, loss_dict

    def _get_default_peft_targets(self) -> dict[str, any]:
        """Return default PEFT target modules for PI0 fine-tuning."""
        common_projections = (
            "state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out"
        )
        target_modules = rf"(.*\.gemma_expert\..*\.self_attn\.(q|v)_proj|model\.({common_projections}))"
        return {
            "target_modules": target_modules,
            "modules_to_save": [],
        }
