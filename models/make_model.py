from pathlib import Path

import torch
import torchvision
from torch.utils.mobile_optimizer import optimize_for_mobile

model = torchvision.models.mobilenet_v3_small(pretrained=True)
model.eval()

scripted_model = torch.jit.script(model)
optimized_model = optimize_for_mobile(scripted_model)
spec = Path("live.spec.json").read_text()
extra_files = {}
extra_files["model/live.spec.json"] = spec
optimized_model._save_for_lite_interpreter("custom_model.ptl", _extra_files=extra_files)

print("model successfully exported")