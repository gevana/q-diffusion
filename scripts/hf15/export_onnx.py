#!/usr/bin/env python
from typing import Optional, Union

import torch
from diffusers import AutoencoderKL, DDIMScheduler, StableDiffusionPipeline


def get_time_embed(sample: torch.Tensor, timestep: Union[torch.Tensor, float, int]) -> Optional[torch.Tensor]:
    return timestep.to(dtype=sample.dtype)


def main():
    base_model_path = "SG161222/Realistic_Vision_V4.0_noVAE"
    orig_base_model_path = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    vae_model_path = "stabilityai/sd-vae-ft-mse"
    device = "cuda"
    dtype = torch.float32

    scheduler = DDIMScheduler.from_pretrained(base_model_path, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained(vae_model_path, torch_dtype=dtype)
    pipe = StableDiffusionPipeline.from_pretrained(
        orig_base_model_path,
        torch_dtype=dtype,
        scheduler=scheduler,
        vae=vae,
        # feature_extractor=AutoFeatureExtractor.from_pretrained(
        #     orig_base_model_path, subfolder="feature_extractor", torch_dtype=dtype
        # ),
        safety_checker=None,
    )
    pipe = pipe.to(device)

    # generate image
    prompt = "photo of a woman in blue dress in a garden"
    negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality, blurry"
    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
        device=device,
    )

    # UNET export
    latent_model_input = torch.randn((1, 4, 64, 64)).to(device)
    timestamp_input = torch.randn(1, 320).to(device)  # torch.randn((1)).to(device)
    condition_input = prompt_embeds.to(device)
    inputs = (
        {
            "sample": latent_model_input,
            "timestep": timestamp_input,
            "encoder_hidden_states": condition_input,
            "return_dict": False,
        },
    )
    pipe.unet.get_time_embed = get_time_embed
    unet_model_onnx_path = "/genai/users/nadavg/sd/unet.onnx"
    torch.onnx.export(
        pipe.unet,
        inputs,
        unet_model_onnx_path,
        output_names=["output"],
        input_names=["sample", "/time_proj/Concat_1_output_0", "encoder_hidden_states"],
        dynamic_axes={
            "sample": {0: "batch"},
            "/time_proj/Concat_1_output_0": {0: "batch"},
            "encoder_hidden_states": {0: "batch"},
            "output": {0: "batch"},
        },
    )


if __name__ == "__main__":
    main()

#onnxsim /genai/users/nadavg/sd/unet.onnx /genai/users/nadavg/sd/unet.sim.onnx --overwrite-input-shape sample:1,4,64,64 /time_proj/Concat_1_output_0:1,320 encoder_hidden_states:1,77,768
#hailo parser onnx /genai/users/nadavg/sd/unet.sim.onnx --start-node-names sample /time_proj/Concat_1_output_0 encoder_hidden_states
