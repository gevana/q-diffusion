    
import torch
from diffusers import StableDiffusionPipeline
import torch
from diffusers import AutoencoderKL, DDIMScheduler, StableDiffusionPipeline

def init_pipe():

    base_model_path = "SG161222/Realistic_Vision_V4.0_noVAE"
    orig_base_model_path = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    vae_model_path = "stabilityai/sd-vae-ft-mse"
    #device = "cuda"
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
    return pipe