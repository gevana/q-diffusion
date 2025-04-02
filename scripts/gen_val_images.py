
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
import pandas as pd
import torch
import PIL
from PIL import Image
from pathlib import Path
from mo_utils.utils.image_utils import stack_images
from mo_utils.utils.image_utils import gen_stacked_image
import numpy as np

prompts = pd.read_parquet('scripts/eval.parquet')['Prompt']

with open('scripts/prompt_acceleras.yaml', "r") as f:
    prompts = f.readlines()

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
generator = torch.Generator(device).manual_seed(42)

def gen_images(pipe, num_images = 4,num_inference_steps =20,
                output_image_path = './validation_images/validation_images.png',
                negative_prompt = None,):
    if negative_prompt == 'dafualt':
        negative_prompt = ("ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame,"
        "extra limbs, disfigured, deformed, body out of frame, bad anatomy, watermark, signature,"
        "cut off, low contrast, underexposed, overexposed, bad art, beginner, amateur, distorted face")


    pipe = pipe.to(device)
    I =[]
    for prompt in prompts[:num_images]: 
        print(prompt)
        Ip = pipe(prompt,num_inference_steps=num_inference_steps,generator= generator,negative_prompt=negative_prompt).images[0]
        I.append(Ip)
    if False:
        for prompt in prompts[-num_images:]: 
            print(prompt)
            Ip = pipe(prompt,num_inference_steps=num_inference_steps,generator= generator,negative_prompt=negative_prompt).images[0]
            I.append(Ip)
    
    I= [np.array(Ip) for Ip in I]
    out_image = gen_stacked_image(I)
    out_image = Image.fromarray(out_image)
    
    # upper_row = Image.new('RGB',(I[0].width*num_images,I[0].height))
    # lower_row = Image.new('RGB',(I[0].width*num_images,I[0].height))
    # for i in range(num_images):
    #     upper_row.paste(I[i],(I[0].width*i,0))
    #     lower_row.paste(I[-(i+1)],(I[0].width*i,0))
    # out_image = Image.new('RGB',(I[0].width*num_images,I[0].height*2))
    # out_image.paste(upper_row,(0,0))
    # out_image.paste(lower_row,(0,I[0].height))

    if output_image_path is not None:
        Path(output_image_path).parent.mkdir(parents=True, exist_ok=True)
        out_image.save(output_image_path)
    return out_image
    

    