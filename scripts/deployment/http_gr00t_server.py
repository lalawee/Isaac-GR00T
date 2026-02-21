#!/usr/bin/env python3
"""
GR00T N1.6 inference + Flask wrapper (single-process)

This updates the older GR00T N1.5 "inference_and_flask.py" style script to the
GR00T N1.6 Policy API (nested observation dicts: video/state/language).

Key differences vs N1.5:
- Gr00tPolicy import path moved to: gr00t.policy.gr00t_policy.Gr00tPolicy
- Policy expects observations as nested dicts:
    {
      "video": {<video_key>: (B,T,H,W,3) uint8},
      "state": {<state_key>: (B,T,D) float32},
      "language": {<language_key>: [[text]]}
    }
- policy.get_action(obs) returns: (predicted_action, info)

References:
- Policy API guide in Isaac-GR00T repo: getting_started/policy.md
"""

from __future__ import annotations

import argparse
import importlib
import io
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import cv2
import torch
from flask import Flask, jsonify, request
from PIL import Image

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy


import importlib.util
from pathlib import Path

def import_py_file(py_file: str) -> None:
    py_file = str(Path(py_file).resolve())
    spec = importlib.util.spec_from_file_location("custom_modality_config", py_file)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)  # executes register_modality_config(...) in that file


# ----------------------------
# Flask + logging
# ----------------------------
app = Flask(__name__)
logger = logging.getLogger("gr00t_n16_flask")
logger.setLevel(logging.INFO)


def _setup_logging(log_path: Optional[str]) -> None:
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)
    logger.addHandler(console)

    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.INFO)
        fh.setFormatter(fmt)
        logger.addHandler(fh)


def measure_interval(route_fn):
    last = {"t": None}
    def wrapped(*args, **kwargs):
        now = time.perf_counter()
        if last["t"] is not None:
            logger.info("Time since last /predict: %.4fs", now - last["t"])
        last["t"] = now
        return route_fn(*args, **kwargs)
    wrapped.__name__ = route_fn.__name__
    return wrapped


# ----------------------------
# Modality config loader (same idea as N1.6 finetune launcher)
# ----------------------------
def load_modality_config(modality_config_path: str) -> None:
    """
    Ensure a user-provided modality config is registered (e.g. your kuavoV4Pro_config.py).
    """
    path = Path(modality_config_path)
    if not (path.exists() and path.suffix == ".py"):
        raise FileNotFoundError(f"Modality config path does not exist or is not a .py: {path}")

    sys.path.append(str(path.parent))
    importlib.import_module(path.stem)
    logger.info("Loaded modality config: %s", path)


# ----------------------------
# Image preprocessing (match GR00T demo: letterbox to 256×256)
# ----------------------------
def preprocess_image_to_256(file_bytes: bytes) -> np.ndarray:
    """
    Returns: uint8 array shaped (256,256,3) in RGB.
    """
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    arr = np.array(img)  # (H,W,3), uint8

    h, w = arr.shape[:2]
    scale = 256.0 / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_vert = 256 - new_h
    pad_top = pad_vert // 2
    pad_bot = pad_vert - pad_top

    pad_horiz = 256 - new_w
    pad_left = pad_horiz // 2
    pad_right = pad_horiz - pad_left

    padded = cv2.copyMakeBorder(
        resized,
        pad_top, pad_bot,
        pad_left, pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=[0, 0, 0],
    )
    return padded  # (256,256,3)


def _parse_csv_floats(form: Dict[str, Any], name: str, expected_dims: int) -> np.ndarray:
    val = form.get(name)
    if val is None:
        raise ValueError(f"Missing field: {name}")
    parts = [p.strip() for p in val.split(",") if p.strip() != ""]
    if len(parts) != expected_dims:
        raise ValueError(f"{name} must have {expected_dims} values, got {len(parts)}")
    return np.array([float(x) for x in parts], dtype=np.float32)


def _best_effort_match_video_key(expected_keys, provided_key: str) -> Optional[str]:
    """
    If your policy expects keys like "ego_view_bg_crop_pad_res256_freq20" but
    your robot sends "ego_view", try to match by substring.
    """
    if provided_key in expected_keys:
        return provided_key
    # substring match (provided_key is short, expected_keys might be longer)
    candidates = [k for k in expected_keys if provided_key in k]
    if len(candidates) == 1:
        return candidates[0]
    return None


@dataclass
class ServerState:
    policy: Gr00tPolicy
    modality_config: Dict[str, Any]
    video_keys: Tuple[str, ...]
    state_keys: Tuple[str, ...]
    language_key: str
    state_dims: Dict[str, int]  # required for parsing your robot state


STATE: Optional[ServerState] = None


def build_observation(req) -> Tuple[Dict[str, Any], Optional[Tuple[str, int]]]:
    assert STATE is not None

    # --- video ---
    video: Dict[str, np.ndarray] = {}
    if not req.files:
        return {}, ("no camera images provided", 400)

    for file_key in req.files.keys():
        img_file = req.files[file_key]
        frame = preprocess_image_to_256(img_file.read())  # (256,256,3)

        matched = _best_effort_match_video_key(STATE.video_keys, file_key)
        if matched is None:
            logger.warning(
                "Received image key '%s' but policy expects %s. "
                "No unique match found; skipping this image.",
                file_key, list(STATE.video_keys)
            )
            continue

        # Make (B,T,H,W,3) => (1,1,256,256,3)
        video[matched] = frame[None, None, ...]

    if not video:
        return {}, (f"no usable camera images. expected one of: {list(STATE.video_keys)}", 400)

    # --- state ---
    state: Dict[str, np.ndarray] = {}
    for k in STATE.state_keys:
        if k not in STATE.state_dims:
            # If you see this warning, add the correct dims for this key via --state-dims
            logger.warning("No dims configured for state key '%s'. Skipping (may break inference).", k)
            continue
        d = STATE.state_dims[k]
        try:
            vec = _parse_csv_floats(req.form, k, d)
        except ValueError as e:
            return {}, (str(e), 400)
        # (B,T,D) => (1,1,D)
        state[k] = vec[None, None, :]

    # --- language ---
    task = req.form.get("task_description", "")
    language = {STATE.language_key: [[task]]}

    obs = {"video": video, "state": state, "language": language}
    return obs, None


@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"ok": True})


@app.route("/predict", methods=["POST"])
@measure_interval
def predict():
    assert STATE is not None

    obs, error = build_observation(request)
    if error:
        msg, code = error
        return jsonify({"error": msg}), code

    try:
        action, info = STATE.policy.get_action(obs)
    except Exception as e:
        logger.exception("Inference failed: %s", e)
        return jsonify({"error": "inference error", "detail": str(e)}), 500

    # JSON serialize (numpy -> lists)
    action_json = {k: np.asarray(v).tolist() for k, v in action.items()}
    return jsonify(action_json)


def main():
    global STATE

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, help="HF repo or local checkpoint dir (e.g. nvidia/GR00T-N1.6-3B or /path/to/checkpoint)")
    parser.add_argument("--embodiment-tag", required=True, help="EmbodimentTag value (e.g. GR1, NEW_EMBODIMENT, etc.)")
    parser.add_argument("--device", default=None, help="e.g. cuda:0 or cpu. Default: auto-detect.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)

    parser.add_argument("--modality-config-path", default=None, help="Path to your modality config .py (register_modality_config call).")
    parser.add_argument(
        "--state-dims",
        default=None,
        help=(
            "JSON dict mapping state keys to dims, e.g. "
            '\'{"left_arm":7,"right_arm":7,"left_hand":6,"right_hand":6}\' '
            "(keys must match modality_config['state'].modality_keys)."
        ),
    )

    parser.add_argument("--strict", action="store_true", help="Pass strict=True to Gr00tPolicy (recommended).")
    parser.add_argument("--log-path", default=None, help="Optional log file path.")
    args = parser.parse_args()
    _setup_logging(args.log_path)

    if args.modality_config_path:
        load_modality_config(args.modality_config_path)

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    embodiment_tag = None
    try:
        embodiment_tag = EmbodimentTag(args.embodiment_tag)      
    except ValueError:
        embodiment_tag = EmbodimentTag[args.embodiment_tag]     
    if args.modality_config_path:
        import_py_file(args.modality_config_path)
    policy = Gr00tPolicy(
        model_path=args.model_path,
        embodiment_tag=embodiment_tag,                          
        device=device,
        strict=args.strict,
    )

    modality_config = policy.get_modality_config()
    # Expected: keys include "video", "state", "language"
    video_keys = tuple(modality_config["video"].modality_keys)
    state_keys = tuple(modality_config["state"].modality_keys)
    language_key = modality_config["language"].modality_keys[0]

    # State dims: you *must* provide correct dims for each state key your policy expects.
    state_dims: Dict[str, int] = {}
    if args.state_dims:
        state_dims = json.loads(args.state_dims)

    STATE = ServerState(
        policy=policy,
        modality_config=modality_config,
        video_keys=video_keys,
        state_keys=state_keys,
        language_key=language_key,
        state_dims=state_dims,
    )

    logger.info("Loaded GR00T policy: %s", args.model_path)
    logger.info("Embodiment: %s", args.embodiment_tag)
    logger.info("Device: %s", device)
    logger.info("Video keys: %s", list(video_keys))
    logger.info("State keys: %s", list(state_keys))
    logger.info("Language key: %s", language_key)

    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
