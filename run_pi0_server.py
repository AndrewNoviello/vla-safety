from __future__ import annotations

import base64
import io
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from transformers import AutoTokenizer

from lerobot.utils.utils import cast_stats_to_numpy, load_json
from lerobot.types import FeatureType
from lerobot.policies.pi0 import PI0Policy, preprocess_pi0, postprocess_pi0
from lerobot.utils.processor_utils import prepare_stats
from lerobot.policies.pi0.configuration_pi0 import PI0Config
from lerobot.utils.processor_utils import prepare_observation_for_inference
from lerobot.utils.constants import OBS_STATE

STATS_PATH = Path("/workspace/vla-safety/stats.json")
DEFAULT_PROMPT = (
    "pick up the middle domino from the three domino row and place it flat "
    "on top of the other two dominos to form an arch"
)

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
        action = policy.predict_action_chunk(observation)
    action = postprocessor(action)
    return action


app = FastAPI(
    title="PI0 Offline Inference",
    description="Send images + prompt (and optional proprio), get predicted actions.",
)

# Loaded at startup
MODEL_ID = "AndrewNoviello/vla-safety-task-1"
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
    config = PI0Config.from_pretrained(MODEL_ID)
    config.compile_model = False  # Explicitly disable torch.compile for server inference
    policy = PI0Policy.from_pretrained(MODEL_ID, config=config)
    policy.to(device)
    policy.eval()
    policy.config.device = device
    image_size = tuple(policy.config.image_resolution)
    dataset_stats = None
    if STATS_PATH.exists():
        dataset_stats = cast_stats_to_numpy(load_json(STATS_PATH))
        print(f"Loaded dataset stats from {STATS_PATH}")
    else:
        print(f"Stats file not found at {STATS_PATH}, using no normalization")
    stats = prepare_stats(dataset_stats)
    all_features = {**policy.config.input_features, **policy.config.output_features}
    output_features = dict(policy.config.output_features)
    norm_map = dict(policy.config.normalization_mapping)
    tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")

    def preprocessor(obs):
        return preprocess_pi0(
            obs,
            stats=stats,
            all_features=all_features,
            norm_map=norm_map,
            tokenizer=tokenizer,
            device=device,
            max_length=policy.config.tokenizer_max_length,
            add_batch_dim=True,
        )

    def postprocessor(action):
        return postprocess_pi0(action, stats=stats, output_features=output_features, norm_map=norm_map)
    print("Policy loaded. Ready for /predict.")


class PredictJSONRequest(BaseModel):
    prompt: str | None = Field(None, description="Text instruction for the policy (uses default if omitted)")
    images: list[str] | None = Field(None, description="Base64-encoded images (1 or N)")
    proprio: list[float] | None = Field(None, description="State vector (floats)")


@app.get("/")
def root():
    return {"service": "pi0-offline", "predict": "POST /predict", "health": "GET /health"}


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID, "device": str(device)}


def _run_and_return(prompt: str | None, image_arrays: list[np.ndarray], proprio: list[float] | None):
    if not image_arrays:
        raise HTTPException(status_code=400, detail="At least one image is required.")
    task = prompt if prompt else DEFAULT_PROMPT
    obs = build_observation_from_arrays(policy, image_arrays, proprio, image_size)
    action = run_inference(
        policy, preprocessor, postprocessor, obs, task=task, device=device, use_amp=use_amp
    )
    action_np = action.numpy() if isinstance(action, torch.Tensor) else action
    return {"action": action_np.tolist(), "shape": list(action_np.shape)}


@app.post("/predict")
async def predict(
    prompt: str | None = Form(None, description="Text instruction (uses default if omitted)"),
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
