"""ckpt_to_jit.py model_N.pt out_policy.pt -- rsl_rl checkpoint -> TorchScript actor (normalizer + MLP, ELU)."""
import sys
import torch
from torch import nn

sd = torch.load(sys.argv[1], map_location="cpu")["model_state_dict"]
lin = sorted({int(k.split(".")[1]) for k in sd if k.startswith("actor.") and k.endswith(".weight")})
layers = []
for i, j in enumerate(lin):
    l = nn.Linear(sd[f"actor.{j}.weight"].shape[1], sd[f"actor.{j}.weight"].shape[0])
    l.weight.data, l.bias.data = sd[f"actor.{j}.weight"], sd[f"actor.{j}.bias"]
    layers += [l] + ([nn.ELU()] if i < len(lin) - 1 else [])


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(*layers)
        has = "actor_obs_normalizer._mean" in sd
        self.register_buffer("mean", sd["actor_obs_normalizer._mean"] if has else torch.zeros(1, layers[0].in_features))
        self.register_buffer("std", sd["actor_obs_normalizer._std"] + 1e-2 if has else torch.ones(1, layers[0].in_features))

    def forward(self, x):
        return self.mlp((x - self.mean) / self.std)


torch.jit.script(Policy()).save(sys.argv[2])
print("saved", sys.argv[2])
