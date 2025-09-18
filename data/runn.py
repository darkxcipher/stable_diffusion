import torch
import pytorch_lightning as pl

ckpt = torch.load("v1-5-pruned-emaonly.ckpt", map_location="cpu", weights_only=False)
print(ckpt.keys())
