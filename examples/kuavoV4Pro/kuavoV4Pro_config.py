# examples/kuavoV4Pro/kuavoV4Pro_config.py

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.types import (
    ModalityConfig,
    ActionConfig,
    ActionRepresentation,
    ActionType,
    ActionFormat,
)
from gr00t.data.embodiment_tags import EmbodimentTag


KUAVO_MODALITY_KEYS = [
    "left_arm",
    "left_hand",
    "left_leg",
    "neck",
    "right_arm",
    "right_hand",
    "right_leg",
    "waist",
]

kuavoV4Pro_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["ego_view", "left_wrist_view", "right_wrist_view"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=KUAVO_MODALITY_KEYS,
        # S:contentReference[oaicite:12]{index=12}cos for joint angles if they are radians:
        # sin_cos_embedding_keys=KUAVO_MODALITY_KEYS,
    ),
    "action": ModalityConfig(
        delta_indices=list(range(0, 16)),  # action horizon (match your training choice)
        modality_keys=KUAVO_MODALITY_KEYS,
        action_configs=[
            ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
            ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
            ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
            ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
            ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
            ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
            ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
            ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.action.task_description"],
    ),
}

from gr00t.configs.data import embodiment_configs

if EmbodimentTag.NEW_EMBODIMENT.value not in embodiment_configs.MODALITY_CONFIGS:
    register_modality_config(kuavoV4Pro_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)