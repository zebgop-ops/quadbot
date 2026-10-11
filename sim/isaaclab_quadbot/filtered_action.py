"""Joint position action with a first-order low-pass on the servo targets:
target = alpha * new + (1 - alpha) * previous target. alpha 0.5 at 50 Hz ~ 8 Hz corner:
passes the ~2 Hz gait, removes step-to-step buzz. The Jetson runs the same filter on the robot."""
from collections.abc import Sequence

import torch

from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.utils import configclass


class FilteredJointPositionAction(JointPositionAction):
    cfg: "FilteredJointPositionActionCfg"

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._filtered = self._offset.clone() if isinstance(self._offset, torch.Tensor) else \
            torch.zeros_like(self._processed_actions)

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)
        a = self.cfg.alpha
        self._filtered = a * self._processed_actions + (1.0 - a) * self._filtered
        self._processed_actions = self._filtered.clone()

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
        off = self._offset if isinstance(self._offset, torch.Tensor) else torch.zeros_like(self._filtered)
        self._filtered[env_ids] = off[env_ids]


@configclass
class FilteredJointPositionActionCfg(JointPositionActionCfg):
    class_type: type = FilteredJointPositionAction
    alpha: float = 0.5
