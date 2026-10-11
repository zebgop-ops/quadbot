#!/usr/bin/env python3
"""Convert a trained policy to the .npz the Jetson runs with plain numpy (run where torch is installed).

  python3 export_npz.py quadbot_cat_policy.pt  models/quadbot_cat_policy.npz     # TorchScript (ckpt_to_jit)
  python3 export_npz.py model_6600.pt          models/quadbot_cat_policy.npz     # raw rsl_rl checkpoint
"""
import sys

import numpy as np
import torch


def from_state_dict(sd, prefix, mean, std):
    idx = sorted({int(k[len(prefix):].split(".")[0]) for k in sd if k.startswith(prefix) and k.endswith(".weight")})
    out = {"mean": mean.reshape(-1), "std": std.reshape(-1)}
    for i, j in enumerate(idx):
        out["W%d" % i] = sd["%s%d.weight" % (prefix, j)].cpu().numpy().astype(np.float32)
        out["b%d" % i] = sd["%s%d.bias" % (prefix, j)].cpu().numpy().astype(np.float32)
    return out


src, dst = sys.argv[1], sys.argv[2]
try:
    m = torch.jit.load(src, map_location="cpu")
    sd = m.state_dict()
    arrays = from_state_dict(sd, "mlp.", sd["mean"].numpy(), sd["std"].numpy())
    ref = m
except RuntimeError:
    sd = torch.load(src, map_location="cpu")["model_state_dict"]
    n = sd["actor.0.weight"].shape[1]
    if "actor_obs_normalizer._mean" in sd:
        mean, std = sd["actor_obs_normalizer._mean"].numpy(), sd["actor_obs_normalizer._std"].numpy() + 1e-2
    else:
        mean, std = np.zeros(n, np.float32), np.ones(n, np.float32)
    arrays = from_state_dict(sd, "actor.", mean, std)
    ref = None
np.savez(dst, **arrays)
print("saved", dst, {k: v.shape for k, v in arrays.items()})

if ref is not None:          # check numpy == torch
    sys.path.insert(0, __file__.rsplit("/", 1)[0] if "/" in __file__ else ".")
    from controller import Policy
    p = Policy(dst)
    x = np.random.randn(64, 45).astype(np.float32) * 2
    a = np.stack([p(xi) for xi in x])
    b = ref(torch.from_numpy(x)).detach().numpy()
    print("numpy vs torch max |diff| = %.2e" % np.abs(a - b).max())
