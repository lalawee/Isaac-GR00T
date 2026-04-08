# Isaac-GR00T — KuavoV4Pro Fine-Tuning & Deployment

> Modified fork of [NVIDIA/Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) for the **KuavoV4Pro 34-DOF humanoid robot**.  
> Maintained by the Ceredroid team.

This fork adds everything needed to fine-tune GR00T N1.6-3B on KuavoV4Pro demonstration data and deploy the trained policy to the real robot via a Flask inference server.

---

## What's Changed from Upstream

This fork adds **4 files** and patches **1 file** on top of the official NVIDIA release:

| File | What it does |
|------|-------------|
| `examples/kuavoV4Pro/kuavoV4Pro_config.py` | Embodiment modality config — defines video keys (ego + 2 wrist), state/action layout (8 joint groups: arms, hands, legs, waist, neck), action horizon, and language prompt key |
| `scripts/deployment/http_gr00t_server.py` | Flask-based HTTP inference server for real-robot deployment — accepts multipart image + state via `/predict`, returns 34D joint actions |
| `gr00t/policy/gr00t_policy.py` | Patched to fall back to the global modality registry when the processor doesn't recognise a custom embodiment tag (needed for `NEW_EMBODIMENT`) |
| `pyproject.toml` | Added `flask` dependency |
| `.gitignore` | Minor additions |

---

## KuavoV4Pro Embodiment Configuration

The embodiment config at `examples/kuavoV4Pro/kuavoV4Pro_config.py` defines:

- **Video inputs**: `ego_view`, `left_wrist_view`, `right_wrist_view`
- **State/Action groups** (8 modality keys): `left_arm`, `right_arm`, `left_hand`, `right_hand`, `left_leg`, `right_leg`, `waist`, `neck`
- **Action representation**: Absolute joint angles (non-EEF), action horizon of 16 steps
- **Language key**: `annotation.human.action.task_description` (integer index into `tasks.jsonl`)

The config auto-registers under `NEW_EMBODIMENT` when imported.

---

## Setup

### Prerequisites

- Python 3.10
- CUDA 12.4 (recommended)
- GPU with ≥24 GB VRAM (fine-tuning requires ≥40 GB; H100 recommended)
- [uv](https://github.com/astral-sh/uv) v0.8.4+

### Installation

```bash
git clone --recurse-submodules https://github.com/Muslinmin/Isaac-GR00T.git
cd Isaac-GR00T
uv sync --python 3.10
uv pip install -e .
```

---

## Usage

### 1. Prepare Data

Your dataset must be in **LeRobot v2.1 format** (Parquet + MP4 + `meta/`). If converting from Isaac Lab HDF5, use the `hdf5_to_lerobot.py` converter in the [IsaacLab fork](https://github.com/Muslinmin/IsaacLab/tree/capstone).

The dataset should be uploaded to HuggingFace or placed locally. Verify with:

```bash
uv run python getting_started/validate_dataset.py --dataset-path <DATASET_PATH>
```

### 2. Fine-Tune

```bash
export NUM_GPUS=1

CUDA_VISIBLE_DEVICES=0 uv run python \
    gr00t/experiment/launch_finetune.py \
    --base-model-path nvidia/GR00T-N1.6-3B \
    --dataset-path <DATASET_PATH> \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path examples/kuavoV4Pro/kuavoV4Pro_config.py \
    --num-gpus $NUM_GPUS \
    --output-dir <OUTPUT_PATH> \
    --save-total-limit 5 \
    --save-steps 2000 \
    --max-steps 10000 \
    --use-wandb \
    --global-batch-size 32 \
    --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
    --dataloader-num-workers 4
```

### 3. Open-Loop Evaluation

Quick sanity check — compares predicted vs ground-truth actions:

```bash
uv run python gr00t/eval/open_loop_eval.py \
    --dataset-path <DATASET_PATH> \
    --embodiment-tag NEW_EMBODIMENT \
    --model-path <CHECKPOINT_PATH> \
    --traj-ids 0 1 2 \
    --action-horizon 16
```

### 4. Deploy to Real Robot

Start the Flask inference server:

```bash
uv run python scripts/deployment/http_gr00t_server.py \
    --model-path <CHECKPOINT_PATH> \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path examples/kuavoV4Pro/kuavoV4Pro_config.py \
    --device cuda:0 \
    --host 0.0.0.0 \
    --port 5050
```

The server exposes two endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ping` | GET | Health check |
| `/predict` | POST | Accepts multipart form with camera images + joint state CSV, returns predicted action as JSON |

On the robot side, the ROS controller node (`robot_controller_node.py`) sends observations to `/predict` at each control step and executes the returned joint targets.

---

## Deployment Architecture

```
KuavoV4Pro Robot
  ├── 3× RealSense cameras (ego D435 + 2× wrist D405)
  ├── 48D joint state feedback
  │
  │  HTTP POST /predict
  │  (images + state CSV)
  ▼
Flask Inference Server (GPU machine)
  ├── http_gr00t_server.py
  ├── Loads Gr00tPolicy with KuavoV4Pro config
  ├── Preprocesses images → 256×256
  ├── Runs 4-step diffusion denoising
  └── Returns 34D joint action (action_horizon × 34)
  │
  │  JSON response
  ▼
ROS Controller Node
  └── Executes joint targets at fixed control rate
```

---

## Important Notes

- **Embodiment tag**: This fork registers KuavoV4Pro under `NEW_EMBODIMENT` (the GR00T default for custom robots). If NVIDIA adds more built-in tags in future releases, consider registering a dedicated `KUAVO_V4PRO` tag.
- **Task prompt indexing**: The language key `annotation.human.action.task_description` expects an integer index into `tasks.jsonl`, not a raw string. Ensure your dataset's `tasks.jsonl` is correctly populated.
- **Action format**: All 8 joint groups use absolute joint angle representation. If switching to delta/relative actions, update the `ActionRepresentation` in the config.
- **Policy patch**: The `gr00t_policy.py` fallback patch may need rebasing when pulling upstream updates — it's a small 7-line change in `__init__`.

---

## Related Repositories

| Repository | Purpose |
|-----------|---------|
| [Muslinmin/IsaacLab](https://github.com/Muslinmin/IsaacLab) (capstone branch) | Isaac Lab fork — teleoperation, MimicGen synthetic data generation, environment configs |
| [Lusmse/syn_realDataset](https://huggingface.co/datasets/Lusmse/syn_realDataset) | Real and synthetic LeRobot datasets |
| [Lusmse/capstone-vla-checkpoints](https://huggingface.co/Lusmse/capstone-vla-checkpoints) | Fine-tuned GR00T checkpoints |

---

## License

This fork inherits the [Apache 2.0](LICENSE) license from NVIDIA Isaac GR00T.
