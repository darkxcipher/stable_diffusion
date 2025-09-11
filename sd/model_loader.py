from decoder import VAE_Decoder
from diffusion import Diffusion
from encoder import VAE_Encoder
from clip import CLIP
import model_converter
import torch


def load_weights_safely(model: torch.nn.Module, checkpoint_dict: dict):
    """
    Load weights safely from checkpoint_dict into model.
    - Only loads layers that exist in the model and match in shape.
    - Skips missing or shape-mismatched layers.
    """
    filtered_state_dict = {}
    for k, v in checkpoint_dict.items():
        if k in model.state_dict():
            if v.shape == model.state_dict()[k].shape:
                filtered_state_dict[k] = v
            else:
                print(
                    f"Skipping {k}: shape mismatch {v.shape} vs {model.state_dict()[k].shape}"
                )
        else:
            print(f"Skipping {k}: key not found in model")

    model.load_state_dict(filtered_state_dict, strict=False)
    return model


def preload_models_from_standard_weights(
    ckpt_path: str, device: str = "cuda", dtype=torch.float32
):
    """
    Load VAE encoder, decoder, diffusion model, and CLIP.
    dtype can be torch.float32 (default) or torch.float16 for FP16.
    """
    print(f"Loading checkpoint: {ckpt_path} on device: {device}")
    state_dict = model_converter.load_from_standard_weights(ckpt_path, device="cpu")

    # -------------------- ENCODER --------------------
    encoder = VAE_Encoder()  # load on CPU first
    encoder = load_weights_safely(encoder, state_dict["encoder"])

    # -------------------- DECODER --------------------
    decoder = VAE_Decoder()
    decoder = load_weights_safely(decoder, state_dict["decoder"])

    # -------------------- DIFFUSION --------------------
    diffusion = Diffusion()
    diffusion = load_weights_safely(diffusion, state_dict["diffusion"])

    # -------------------- CLIP --------------------
    clip = CLIP()
    clip = load_weights_safely(clip, state_dict["clip"])

    # -------------------- MOVE TO DEVICE --------------------
    encoder.to(device, dtype=dtype)
    decoder.to(device, dtype=dtype)
    diffusion.to(device, dtype=dtype)
    clip.to(device, dtype=dtype)

    print(f"All models loaded successfully in {dtype}.")
    return {
        "clip": clip,
        "encoder": encoder,
        "decoder": decoder,
        "diffusion": diffusion,
    }
