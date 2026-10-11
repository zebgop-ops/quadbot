"""export_onnx.py policy_jit.pt out.onnx -- TorchScript actor -> ONNX (obs[1,45] -> actions[1,12])."""
import sys
import torch
m = torch.jit.load(sys.argv[1], map_location="cpu").eval()
torch.onnx.export(m, torch.zeros(1, 45), sys.argv[2], input_names=["obs"], output_names=["actions"], opset_version=17)
print("saved", sys.argv[2])
