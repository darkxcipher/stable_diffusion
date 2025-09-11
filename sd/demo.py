# demo.py
import model_loader
import pipeline
from PIL import Image
from transformers import CLIPTokenizer
import torch

# ----------------------------
# Device selection
# ----------------------------
DEVICE = "cpu"
ALLOW_CUDA = True
ALLOW_MPS = False

if torch.cuda.is_available() and ALLOW_CUDA:
    DEVICE = "cuda"
elif (hasattr(torch, "has_mps") and torch.has_mps) or (
    hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
):
    if ALLOW_MPS:
        DEVICE = "mps"

print(f"Using device: {DEVICE}")

# ----------------------------
# Load models + tokenizer
# ----------------------------
tokenizer = CLIPTokenizer("../data/vocab.json", merges_file="../data/merges.txt")

model_file = "../data/v1-5-pruned-emaonly.ckpt"
# Load models normally, no FP16
models = model_loader.preload_models_from_standard_weights(model_file, DEVICE)

# ----------------------------
# Prompts
# ----------------------------
prompt = "a cat stretching on the floor, highly detailed, ultra sharp, cinematic"
uncond_prompt = " "  # negative prompt
do_cfg = True
cfg_scale = 7

# ----------------------------
# Image-to-Image setup (optional)
# ----------------------------
input_image = None
image_path = "./images/dog.jpg"

# Uncomment to enable img2img:
# input_image = Image.open(image_path)

strength = 0.9

# ----------------------------
# Sampler settings
# ----------------------------
sampler = "ddpm"
num_inference_steps = 50
seed = 42

# ----------------------------
# Run generation
# ----------------------------
output_image = pipeline.generate(
    prompt=prompt,
    uncond_prompt=uncond_prompt,
    input_image=input_image,
    strength=strength,
    do_cfg=do_cfg,
    cfg_scale=cfg_scale,
    sampler_name=sampler,
    n_inference_steps=num_inference_steps,
    seed=seed,
    models=models,
    device=DEVICE,
    idle_device="cpu",
    tokenizer=tokenizer,
)

# ----------------------------
# Save or show result
# ----------------------------
result = Image.fromarray(output_image)
result.save("output.png")
print("Saved generated image as output.png")
