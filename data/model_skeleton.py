import torch
import torch.nn as nn

# ---- Generated skeleton from checkpoint ----

class UNetSkeleton(nn.Module):
    def __init__(self):
        super().__init__()
        # ENCODER
        # model.diffusion_model.input_blocks.0.0.weight: Conv2d(in=4, out=320, kernel=(3, 3))
        # model.diffusion_model.input_blocks.0.0.bias: Bias shape=(320,)
        # model.diffusion_model.input_blocks.1.0.in_layers.0.weight: Bias shape=(320,)
        # model.diffusion_model.input_blocks.1.0.in_layers.0.bias: Bias shape=(320,)
        # model.diffusion_model.input_blocks.1.0.in_layers.2.weight: Conv2d(in=320, out=320, kernel=(3, 3))
        # model.diffusion_model.input_blocks.1.0.in_layers.2.bias: Bias shape=(320,)
        # model.diffusion_model.input_blocks.1.0.emb_layers.1.weight: Linear(in=1280, out=320)
        # model.diffusion_model.input_blocks.1.0.emb_layers.1.bias: Bias shape=(320,)
        # model.diffusion_model.input_blocks.1.0.out_layers.0.weight: Bias shape=(320,)
        # model.diffusion_model.input_blocks.1.0.out_layers.0.bias: Bias shape=(320,)
        # model.diffusion_model.input_blocks.1.0.out_layers.3.weight: Conv2d(in=320, out=320, kernel=(3, 3))
        # model.diffusion_model.input_blocks.1.0.out_layers.3.bias: Bias shape=(320,)
        # model.diffusion_model.input_blocks.1.1.norm.weight: Bias shape=(320,)
        # model.diffusion_model.input_blocks.1.1.norm.bias: Bias shape=(320,)
        # model.diffusion_model.input_blocks.1.1.proj_in.weight: Conv2d(in=320, out=320, kernel=(1, 1))
        # MIDDLE
        # model.diffusion_model.middle_block.0.in_layers.0.weight: Bias shape=(1280,)
        # model.diffusion_model.middle_block.0.in_layers.0.bias: Bias shape=(1280,)
        # model.diffusion_model.middle_block.0.in_layers.2.weight: Conv2d(in=1280, out=1280, kernel=(3, 3))
        # model.diffusion_model.middle_block.0.in_layers.2.bias: Bias shape=(1280,)
        # model.diffusion_model.middle_block.0.emb_layers.1.weight: Linear(in=1280, out=1280)
        # model.diffusion_model.middle_block.0.emb_layers.1.bias: Bias shape=(1280,)
        # model.diffusion_model.middle_block.0.out_layers.0.weight: Bias shape=(1280,)
        # model.diffusion_model.middle_block.0.out_layers.0.bias: Bias shape=(1280,)
        # model.diffusion_model.middle_block.0.out_layers.3.weight: Conv2d(in=1280, out=1280, kernel=(3, 3))
        # model.diffusion_model.middle_block.0.out_layers.3.bias: Bias shape=(1280,)
        # model.diffusion_model.middle_block.1.norm.weight: Bias shape=(1280,)
        # model.diffusion_model.middle_block.1.norm.bias: Bias shape=(1280,)
        # model.diffusion_model.middle_block.1.proj_in.weight: Conv2d(in=1280, out=1280, kernel=(1, 1))
        # model.diffusion_model.middle_block.1.proj_in.bias: Bias shape=(1280,)
        # model.diffusion_model.middle_block.1.transformer_blocks.0.attn1.to_q.weight: Linear(in=1280, out=1280)
        # DECODER
        # model.diffusion_model.output_blocks.0.0.in_layers.0.weight: Bias shape=(2560,)
        # model.diffusion_model.output_blocks.0.0.in_layers.0.bias: Bias shape=(2560,)
        # model.diffusion_model.output_blocks.0.0.in_layers.2.weight: Conv2d(in=2560, out=1280, kernel=(3, 3))
        # model.diffusion_model.output_blocks.0.0.in_layers.2.bias: Bias shape=(1280,)
        # model.diffusion_model.output_blocks.0.0.emb_layers.1.weight: Linear(in=1280, out=1280)
        # model.diffusion_model.output_blocks.0.0.emb_layers.1.bias: Bias shape=(1280,)
        # model.diffusion_model.output_blocks.0.0.out_layers.0.weight: Bias shape=(1280,)
        # model.diffusion_model.output_blocks.0.0.out_layers.0.bias: Bias shape=(1280,)
        # model.diffusion_model.output_blocks.0.0.out_layers.3.weight: Conv2d(in=1280, out=1280, kernel=(3, 3))
        # model.diffusion_model.output_blocks.0.0.out_layers.3.bias: Bias shape=(1280,)
        # model.diffusion_model.output_blocks.0.0.skip_connection.weight: Conv2d(in=2560, out=1280, kernel=(1, 1))
        # model.diffusion_model.output_blocks.0.0.skip_connection.bias: Bias shape=(1280,)
        # model.diffusion_model.output_blocks.1.0.in_layers.0.weight: Bias shape=(2560,)
        # model.diffusion_model.output_blocks.1.0.in_layers.0.bias: Bias shape=(2560,)
        # model.diffusion_model.output_blocks.1.0.in_layers.2.weight: Conv2d(in=2560, out=1280, kernel=(3, 3))

    def forward(self, x, context=None, time=None):
        # TODO: Implement forward pass
        return x

class VAESkeleton(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        # first_stage_model.encoder.conv_in.weight: Conv2d(in=3, out=128, kernel=(3, 3))
        # first_stage_model.encoder.conv_in.bias: Bias shape=(128,)
        # first_stage_model.encoder.down.0.block.0.norm1.weight: Bias shape=(128,)
        # first_stage_model.encoder.down.0.block.0.norm1.bias: Bias shape=(128,)
        # first_stage_model.encoder.down.0.block.0.conv1.weight: Conv2d(in=128, out=128, kernel=(3, 3))
        # first_stage_model.encoder.down.0.block.0.conv1.bias: Bias shape=(128,)
        # first_stage_model.encoder.down.0.block.0.norm2.weight: Bias shape=(128,)
        # first_stage_model.encoder.down.0.block.0.norm2.bias: Bias shape=(128,)
        # first_stage_model.encoder.down.0.block.0.conv2.weight: Conv2d(in=128, out=128, kernel=(3, 3))
        # first_stage_model.encoder.down.0.block.0.conv2.bias: Bias shape=(128,)
        # first_stage_model.encoder.down.0.block.1.norm1.weight: Bias shape=(128,)
        # first_stage_model.encoder.down.0.block.1.norm1.bias: Bias shape=(128,)
        # first_stage_model.encoder.down.0.block.1.conv1.weight: Conv2d(in=128, out=128, kernel=(3, 3))
        # first_stage_model.encoder.down.0.block.1.conv1.bias: Bias shape=(128,)
        # first_stage_model.encoder.down.0.block.1.norm2.weight: Bias shape=(128,)
        # Decoder
        # first_stage_model.decoder.conv_in.weight: Conv2d(in=4, out=512, kernel=(3, 3))
        # first_stage_model.decoder.conv_in.bias: Bias shape=(512,)
        # first_stage_model.decoder.mid.block_1.norm1.weight: Bias shape=(512,)
        # first_stage_model.decoder.mid.block_1.norm1.bias: Bias shape=(512,)
        # first_stage_model.decoder.mid.block_1.conv1.weight: Conv2d(in=512, out=512, kernel=(3, 3))
        # first_stage_model.decoder.mid.block_1.conv1.bias: Bias shape=(512,)
        # first_stage_model.decoder.mid.block_1.norm2.weight: Bias shape=(512,)
        # first_stage_model.decoder.mid.block_1.norm2.bias: Bias shape=(512,)
        # first_stage_model.decoder.mid.block_1.conv2.weight: Conv2d(in=512, out=512, kernel=(3, 3))
        # first_stage_model.decoder.mid.block_1.conv2.bias: Bias shape=(512,)
        # first_stage_model.decoder.mid.attn_1.norm.weight: Bias shape=(512,)
        # first_stage_model.decoder.mid.attn_1.norm.bias: Bias shape=(512,)
        # first_stage_model.decoder.mid.attn_1.q.weight: Conv2d(in=512, out=512, kernel=(1, 1))
        # first_stage_model.decoder.mid.attn_1.q.bias: Bias shape=(512,)
        # first_stage_model.decoder.mid.attn_1.k.weight: Conv2d(in=512, out=512, kernel=(1, 1))

    def forward(self, x):
        # TODO: Implement forward pass
        return x

class CLIPSkeleton(nn.Module):
    def __init__(self):
        super().__init__()
        # cond_stage_model.transformer.text_model.embeddings.position_ids: Linear(in=77, out=1)
        # cond_stage_model.transformer.text_model.embeddings.token_embedding.weight: Linear(in=768, out=49408)
        # cond_stage_model.transformer.text_model.embeddings.position_embedding.weight: Linear(in=768, out=77)
        # cond_stage_model.transformer.text_model.encoder.layers.0.self_attn.k_proj.weight: Linear(in=768, out=768)
        # cond_stage_model.transformer.text_model.encoder.layers.0.self_attn.k_proj.bias: Bias shape=(768,)
        # cond_stage_model.transformer.text_model.encoder.layers.0.self_attn.v_proj.weight: Linear(in=768, out=768)
        # cond_stage_model.transformer.text_model.encoder.layers.0.self_attn.v_proj.bias: Bias shape=(768,)
        # cond_stage_model.transformer.text_model.encoder.layers.0.self_attn.q_proj.weight: Linear(in=768, out=768)
        # cond_stage_model.transformer.text_model.encoder.layers.0.self_attn.q_proj.bias: Bias shape=(768,)
        # cond_stage_model.transformer.text_model.encoder.layers.0.self_attn.out_proj.weight: Linear(in=768, out=768)
        # cond_stage_model.transformer.text_model.encoder.layers.0.self_attn.out_proj.bias: Bias shape=(768,)
        # cond_stage_model.transformer.text_model.encoder.layers.0.layer_norm1.weight: Bias shape=(768,)
        # cond_stage_model.transformer.text_model.encoder.layers.0.layer_norm1.bias: Bias shape=(768,)
        # cond_stage_model.transformer.text_model.encoder.layers.0.mlp.fc1.weight: Linear(in=768, out=3072)
        # cond_stage_model.transformer.text_model.encoder.layers.0.mlp.fc1.bias: Bias shape=(3072,)

    def forward(self, tokens):
        # TODO: Implement forward pass
        return tokens

