from decoder import VAE_Decoder
from diffusion import Diffusion
from encoder import VAE_Encoder
from clip import CLIP
import model_converter


def load_weights_safely(model, checkpoint_dict):
    """
    Load weights from checkpoint_dict into model.
    - Only loads layers that exist in model and match in shape.
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


def preload_models_from_standard_weights(ckpt_path, device):
    """
    Load VAE encoder, decoder, diffusion model, and CLIP from standard checkpoint.
    Handles missing/shape-mismatched layers automatically.
    """
    state_dict = model_converter.load_from_standard_weights(ckpt_path, device)

    # -------------------- ENCODER --------------------
    encoder = VAE_Encoder().to(device)
    encoder = load_weights_safely(encoder, state_dict["encoder"])

    # -------------------- DECODER --------------------
    decoder = VAE_Decoder().to(device)
    decoder = load_weights_safely(decoder, state_dict["decoder"])

    # -------------------- DIFFUSION --------------------
    diffusion = Diffusion().to(device)
    diffusion = load_weights_safely(diffusion, state_dict["diffusion"])

    # -------------------- CLIP --------------------
    clip = CLIP().to(device)
    clip = load_weights_safely(clip, state_dict["clip"])

    return {
        "clip": clip,
        "encoder": encoder,
        "decoder": decoder,
        "diffusion": diffusion,
    }
