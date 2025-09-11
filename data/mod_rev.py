# model_rev.py
import torch


def detect_layer_info(name, tensor):
    """
    Infer layer type and dimensions from a weight tensor.
    """
    shape = tuple(tensor.shape)
    if len(shape) == 4:
        out_c, in_c, k_h, k_w = shape
        return f"{name}: Conv2d(in={in_c}, out={out_c}, kernel=({k_h}, {k_w}))"
    elif len(shape) == 2:
        out_f, in_f = shape
        return f"{name}: Linear(in={in_f}, out={out_f})"
    elif len(shape) == 1:
        return f"{name}: Bias shape={shape}"
    else:
        return f"{name}: Tensor shape={shape}"


def build_unet_skeleton(state_dict):
    unet = {"encoder": [], "middle": [], "decoder": []}
    for k, v in state_dict.items():
        if not isinstance(v, torch.Tensor):
            continue
        if k.startswith("model.diffusion_model.input_blocks"):
            unet["encoder"].append(detect_layer_info(k, v))
        elif k.startswith("model.diffusion_model.middle_block"):
            unet["middle"].append(detect_layer_info(k, v))
        elif k.startswith("model.diffusion_model.output_blocks"):
            unet["decoder"].append(detect_layer_info(k, v))
    return unet


def build_vae_skeleton(state_dict):
    vae = {"encoder": [], "decoder": []}
    for k, v in state_dict.items():
        if not isinstance(v, torch.Tensor):
            continue
        if k.startswith("first_stage_model.encoder"):
            vae["encoder"].append(detect_layer_info(k, v))
        elif k.startswith("first_stage_model.decoder"):
            vae["decoder"].append(detect_layer_info(k, v))
    return vae


def build_clip_skeleton(state_dict):
    clip = []
    for k, v in state_dict.items():
        if not isinstance(v, torch.Tensor):
            continue
        if k.startswith("cond_stage_model"):
            clip.append(detect_layer_info(k, v))
    return clip


def generate_model_skeleton(
    ckpt_path, out_file="model_skeleton.py", dump_file="model_layers.txt"
):
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    print("Checkpoint keys:", list(ckpt.keys()))

    # Build groupings
    unet = build_unet_skeleton(state_dict)
    vae = build_vae_skeleton(state_dict)
    clip = build_clip_skeleton(state_dict)

    # --- Write skeleton.py (truncated for readability) ---
    with open(out_file, "w") as f:
        f.write("import torch\nimport torch.nn as nn\n\n")
        f.write("# ---- Generated skeleton from checkpoint ----\n\n")

        # UNet
        f.write("class UNetSkeleton(nn.Module):\n")
        f.write("    def __init__(self):\n")
        f.write("        super().__init__()\n")
        for section, layers in unet.items():
            f.write(f"        # {section.upper()}\n")
            for line in layers[:15]:  # show only first 15
                f.write(f"        # {line}\n")
        f.write("\n    def forward(self, x, context=None, time=None):\n")
        f.write("        # TODO: Implement forward pass\n")
        f.write("        return x\n\n")

        # VAE
        f.write("class VAESkeleton(nn.Module):\n")
        f.write("    def __init__(self):\n")
        f.write("        super().__init__()\n")
        f.write("        # Encoder\n")
        for line in vae["encoder"][:15]:
            f.write(f"        # {line}\n")
        f.write("        # Decoder\n")
        for line in vae["decoder"][:15]:
            f.write(f"        # {line}\n")
        f.write("\n    def forward(self, x):\n")
        f.write("        # TODO: Implement forward pass\n")
        f.write("        return x\n\n")

        # CLIP
        f.write("class CLIPSkeleton(nn.Module):\n")
        f.write("    def __init__(self):\n")
        f.write("        super().__init__()\n")
        for line in clip[:15]:
            f.write(f"        # {line}\n")
        f.write("\n    def forward(self, tokens):\n")
        f.write("        # TODO: Implement forward pass\n")
        f.write("        return tokens\n\n")

    print(f"Skeleton written to {out_file}")

    # --- Write full dump to model_layers.txt ---
    with open(dump_file, "w") as f:
        f.write("# ---- Full layer dump from checkpoint ----\n\n")

        f.write("[UNet ENCODER]\n")
        f.write("\n".join(unet["encoder"]) + "\n\n")
        f.write("[UNet MIDDLE]\n")
        f.write("\n".join(unet["middle"]) + "\n\n")
        f.write("[UNet DECODER]\n")
        f.write("\n".join(unet["decoder"]) + "\n\n")

        f.write("[VAE ENCODER]\n")
        f.write("\n".join(vae["encoder"]) + "\n\n")
        f.write("[VAE DECODER]\n")
        f.write("\n".join(vae["decoder"]) + "\n\n")

        f.write("[CLIP]\n")
        f.write("\n".join(clip) + "\n\n")

    print(f"Full dump written to {dump_file}")


if __name__ == "__main__":
    ckpt_path = "../data/v1-5-pruned-emaonly.ckpt"
    generate_model_skeleton(ckpt_path)
