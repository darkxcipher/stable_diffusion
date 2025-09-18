# tiny_sd_ckpt_full.py
import torch
import pytorch_lightning as pl  # Ensure this is installed

# Paths
FULL_CKPT = "v1-5-pruned-emaonly.ckpt"
OUTPUT_CKPT = "sd_inference_tiny_fp16.ckpt"


def main():
    # Load the full checkpoint safely
    ckpt = torch.load(FULL_CKPT, map_location="cpu", weights_only=False)
    print("Top-level keys in checkpoint:", ckpt.keys())

    # Extract the model weights (state_dict)
    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        raise ValueError("No state_dict found in checkpoint!")

    # Remove EMA weights (keys containing 'ema')
    state_dict_no_ema = {k: v for k, v in state_dict.items() if "ema" not in k.lower()}

    # Convert all tensors to FP16 for smaller size
    state_dict_fp16 = {k: v.half() for k, v in state_dict_no_ema.items()}

    # Save minimal checkpoint for inference
    torch.save({"model_state_dict": state_dict_fp16}, OUTPUT_CKPT)
    print(f"Tiny FP16 checkpoint saved to {OUTPUT_CKPT}")

    # Optional: print layer names and shapes for verification
    for k, v in state_dict_fp16.items():
        print(k, v.shape)


if __name__ == "__main__":
    main()
