
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
import pandas as pd
import torch
import PIL
from PIL import Image
from pathlib import Path
prompts = pd.read_parquet('scripts/eval.parquet')['Prompt']
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
generator = torch.Generator(device).manual_seed(42)

def gen_images(pipe, num_images = 4,num_inference_steps =20, output_image_path = './validation_images/validation_images.png'):
    I =[]
    for prompt in prompts[:num_images]: 
        Ip = pipe(prompt,num_inference_steps=num_inference_steps,generator= generator).images[0]
        I.append(Ip)
    for prompt in prompts[-num_images:]: 
        Ip = pipe(prompt,num_inference_steps=num_inference_steps,generator= generator).images[0]
        I.append(Ip)
    
    upper_row = Image.new('RGB',(I[0].width*num_images,I[0].height))
    lower_row = Image.new('RGB',(I[0].width*num_images,I[0].height))
    for i in range(num_images):
        upper_row.paste(I[i],(I[0].width*i,0))
        lower_row.paste(I[-(i+1)],(I[0].width*i,0))
    out_image = Image.new('RGB',(I[0].width*num_images,I[0].height*2))
    out_image.paste(upper_row,(0,0))
    out_image.paste(lower_row,(0,I[0].height))

    if output_image_path is not None:
        Path(output_image_path).parent.mkdir(parents=True, exist_ok=True)
        out_image.save(output_image_path)
    return out_image
    

    