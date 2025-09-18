# import torch

# ckpt = torch.load("v1-5-pruned-emaonly.ckpt", map_location="cpu", weights_only=True)
# state_dict = ckpt["state_dict"]

# convert to FP16
# state_dict = {k: v.half() for k, v in state_dict.items()}

# save smaller checkpoint
# torch.save({"model_state_dict": state_dict}, "sd_inference_fp16.ckpt")

# tiny_sd_ckpt.py
import torch

# Paths
FULL_CKPT = "v1-5-pruned-emaonly.ckpt"
OUTPUT_CKPT = "sd_inference_tiny_fp16.ckpt"

# Allowlist for PyTorch Lightning objects
LIGHTNING_GLOBALS = ["pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint"]


def main():
    # Load checkpoint safely
    with torch.serialization.safe_globals(LIGHTNING_GLOBALS):
        ckpt = torch.load(FULL_CKPT, map_location="cpu")

    # Extract model weights
    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        raise ValueError("No state_dict found in checkpoint!")

    # Remove EMA weights (keys containing 'ema')
    state_dict_no_ema = {k: v for k, v in state_dict.items() if "ema" not in k.lower()}

    # Convert all tensors to FP16
    state_dict_fp16 = {k: v.half() for k, v in state_dict_no_ema.items()}

    # Save tiny checkpoint with only model weights
    torch.save({"model_state_dict": state_dict_fp16}, OUTPUT_CKPT)
    print(f"Tiny FP16 checkpoint saved to {OUTPUT_CKPT}")

    # Optional: print layer names and shapes
    for k, v in state_dict_fp16.items():
        print(k, v.shape)


if __name__ == "__main__":
    main()
