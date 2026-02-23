#!/usr/bin/env python
"""
FastAPI server for PI0 offline inference: send images + prompt (and optional proprio),
get back predicted actions.

Start (from workspace root):
  uvicorn run_pi0_server:app --host 0.0.0.0 --port 8000

Then POST to /predict with multipart/form-data:
  - prompt: text instruction
  - image_0: image file (required; used for all cameras if only one provided)
  - image_1, image_2: optional; if provided, must match policy's image key order
  - proprio: optional JSON array of floats, e.g. "[0.1, 0.2, 0.0]"

Or POST JSON to /predict with:
  - prompt: str
  - images: optional list of base64-encoded image strings (one or three)
  - proprio: optional list of floats

Requirements: pip install fastapi uvicorn (and lerobot[pi] as for run_pi0_offline).

Latency note: This is suitable for teleoperation, batching, or non-real-time use.
For low-latency robot control (e.g. 10–30 Hz), run the policy on the same machine
as the robot or on a local network; internet RTT to RunPod often adds 50–200ms+ per request.
"""

from __future__ import annotations

import base64
import io
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_LEROBOT_SRC = _SCRIPT_DIR / "lerobot" / "src"
if _LEROBOT_SRC.exists() and str(_LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(_LEROBOT_SRC))

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from lerobot.configs.types import FeatureType
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi0 import PI0Policy
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.utils.constants import OBS_STATE

# -----------------------------------------------------------------------------
# Image and observation helpers (no disk I/O for server path)
# -----------------------------------------------------------------------------


def decode_image_to_numpy(data: bytes, size: tuple[int, int] = (224, 224)) -> np.ndarray:
    """Decode image bytes (JPEG/PNG) to (H, W, 3) uint8, resized to size."""
    try:
        from PIL import Image
    except ImportError:
        import cv2
        arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = np.array(Image.open(io.BytesIO(data)).convert("RGB"))
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Expected RGB image (H, W, 3), got shape {img.shape}")
    if (img.shape[0], img.shape[1]) != size:
        try:
            from PIL import Image as PILImage
            pil_img = PILImage.fromarray(img)
            pil_img = pil_img.resize((size[1], size[0]), PILImage.BILINEAR)
            img = np.array(pil_img)
        except ImportError:
            import cv2
            img = cv2.resize(img, (size[1], size[0]), interpolation=cv2.INTER_LINEAR)
    return img.astype(np.uint8)


def build_observation_from_arrays(
    policy,
    image_arrays: list[np.ndarray],
    proprio: list[float] | None,
    image_size: tuple[int, int],
) -> dict[str, np.ndarray]:
    """Build observation dict from in-memory images (no disk)."""
    observation: dict[str, np.ndarray] = {}
    input_features = policy.config.input_features
    state_key = OBS_STATE
    if state_key in input_features:
        state_dim = input_features[state_key].shape[0]
        if proprio is not None:
            arr = np.asarray(proprio, dtype=np.float32).flatten()
            if len(arr) >= state_dim:
                observation[state_key] = arr[:state_dim].astype(np.float32)
            else:
                padded = np.zeros(state_dim, dtype=np.float32)
                padded[: len(arr)] = arr
                observation[state_key] = padded
        else:
            observation[state_key] = np.zeros(state_dim, dtype=np.float32)

    image_keys = [k for k, v in input_features.items() if v.type == FeatureType.VISUAL]
    if not image_keys:
        raise ValueError("Policy expects at least one image input.")
    if len(image_arrays) == 1:
        single = image_arrays[0]
        for k in image_keys:
            observation[k] = single.copy()
    else:
        if len(image_arrays) != len(image_keys):
            raise ValueError(
                f"Got {len(image_arrays)} images but policy has {len(image_keys)} "
                f"camera keys. Pass 1 image (replicated) or {len(image_keys)} images."
            )
        for arr, k in zip(image_arrays, image_keys):
            observation[k] = arr.copy()
    return observation


def run_inference(
    policy,
    preprocessor,
    postprocessor,
    observation: dict[str, np.ndarray],
    task: str,
    device: torch.device,
    use_amp: bool = False,
):
    observation = dict(observation)
    observation = prepare_observation_for_inference(
        observation, device=device, task=task, robot_type=""
    )
    observation = preprocessor(observation)
    ctx = (
        torch.autocast(device_type="cuda")
        if device.type == "cuda" and use_amp
        else torch.amp.autocast("cpu", enabled=False)
    )
    with torch.inference_mode(), ctx:
        action = policy.select_action(observation)
    action = postprocessor(action)
    return action


# -----------------------------------------------------------------------------
# App and state
# -----------------------------------------------------------------------------

app = FastAPI(
    title="PI0 Offline Inference",
    description="Send images + prompt (and optional proprio), get predicted actions.",
)

# Loaded at startup
MODEL_ID = "lerobot/pi0_base"
policy = None
preprocessor = None
postprocessor = None
device = None
use_amp = False
image_size = (224, 224)


@app.on_event("startup")
def startup():
    global policy, preprocessor, postprocessor, device, use_amp, image_size
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    use_amp = device.type == "cuda"
    print(f"Loading policy: {MODEL_ID} on {device}")
    policy = PI0Policy.from_pretrained(MODEL_ID)
    policy.to(device)
    policy.eval()
    policy.config.device = device
    image_size = tuple(policy.config.image_resolution)
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        MODEL_ID,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    print("Policy loaded. Ready for /predict.")


# -----------------------------------------------------------------------------
# Request/response models
# -----------------------------------------------------------------------------


class PredictJSONRequest(BaseModel):
    prompt: str = Field(..., description="Text instruction for the policy")
    images: list[str] | None = Field(None, description="Base64-encoded images (1 or N)")
    proprio: list[float] | None = Field(None, description="State vector (floats)")


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------


@app.get("/")
def root():
    return {"service": "pi0-offline", "predict": "POST /predict", "health": "GET /health"}


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID, "device": str(device)}


def _run_and_return(prompt: str, image_arrays: list[np.ndarray], proprio: list[float] | None):
    if not image_arrays:
        raise HTTPException(status_code=400, detail="At least one image is required.")
    obs = build_observation_from_arrays(policy, image_arrays, proprio, image_size)
    action = run_inference(
        policy, preprocessor, postprocessor, obs, task=prompt, device=device, use_amp=use_amp
    )
    action_np = action.numpy() if isinstance(action, torch.Tensor) else action
    return {"action": action_np.tolist(), "shape": list(action_np.shape)}


@app.post("/predict")
async def predict(
    prompt: str = Form(..., description="Text instruction"),
    image_0: UploadFile = File(..., description="First image (required)"),
    image_1: UploadFile | None = File(None),
    image_2: UploadFile | None = File(None),
    proprio: str | None = Form(None, description='JSON array of floats, e.g. "[0.1, 0.2]"'),
):
    """Predict action from multipart form: prompt, image_0 (required), optional image_1/image_2, optional proprio."""
    if policy is None:
        raise HTTPException(status_code=503, detail="Policy not loaded.")
    try:
        data_0 = await image_0.read()
        image_arrays = [decode_image_to_numpy(data_0, image_size)]
        if image_1 and image_1.filename:
            data_1 = await image_1.read()
            image_arrays.append(decode_image_to_numpy(data_1, image_size))
        if image_2 and image_2.filename:
            data_2 = await image_2.read()
            image_arrays.append(decode_image_to_numpy(data_2, image_size))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    prop = None
    if proprio:
        import json
        try:
            prop = json.loads(proprio)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="proprio must be a JSON array of numbers.")
    return _run_and_return(prompt, image_arrays, prop)


@app.post("/predict_json")
async def predict_json(body: PredictJSONRequest):
    """Predict action from JSON: prompt, optional images (base64), optional proprio."""
    if policy is None:
        raise HTTPException(status_code=503, detail="Policy not loaded.")
    image_arrays = []
    if body.images:
        for b64 in body.images:
            try:
                raw = base64.b64decode(b64)
            except Exception:
                raise HTTPException(status_code=400, detail="Invalid base64 in images.")
            image_arrays.append(decode_image_to_numpy(raw, image_size))
    if not image_arrays:
        raise HTTPException(status_code=400, detail="At least one image required (use 'images' in JSON).")
    return _run_and_return(body.prompt, image_arrays, body.proprio)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
