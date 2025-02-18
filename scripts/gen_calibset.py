from pathlib import Path
from qdiff import (
    QuantModel, QuantModule,QuantOp ,BaseQuantBlock,
    block_reconstruction, layer_reconstruction,
)
from qdiff.utils import resume_cali_model, get_train_samples
from scripts.gen_image import gen_image_from_prompt
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from txt2img import load_model_from_config
from pytorch_lightning import seed_everything
from ldm.models.diffusion.plms import PLMSSampler
from qdiff.quant_layer import UniformAffineQuantizer
from qdiff.adaptive_rounding import AdaRoundQuantizer
import numpy as np
import matplotlib.pyplot as plt
from mo_utils.utils.plot_aux import plot_hist, myim



def get_inputs_dict_from_hook(saved_inputs,prompt=[]):
    #saved_inputs.pop(1)
    
    inputs_dict = {'xs':[],'ts':[],'cs':[],'ucs':[]}
    for i in range(len(saved_inputs)):
        if i == 1:
            continue
        inputs_dict['xs'].append(saved_inputs[i][0][:1].cpu())
        inputs_dict['ts'].append(saved_inputs[i][1][:1].cpu())
        inputs_dict['cs'].append(saved_inputs[i][2][1:2].cpu())
        inputs_dict['ucs'].append(saved_inputs[i][2][:1].cpu())
    for k in inputs_dict.keys():
        inputs_dict[k] = torch.cat(inputs_dict[k],dim=0).unsqueeze(1)
    inputs_dict['prompts']=[]    
    inputs_dict['prompts'].append(prompt)
    return inputs_dict

def agragate_inputs_dict(inputs_dict,inputs_dict_i):
    for k in inputs_dict.keys():
        if k == 'prompts':
            continue
        inputs_dict[k] = torch.cat([inputs_dict[k],inputs_dict_i[k]],dim=1)
    inputs_dict['prompts'].extend(inputs_dict_i['prompts'])
    return inputs_dict


def save_input_hook(module, input, output):
    if not hasattr(module, "saved_inputs"):
        module.saved_inputs = []  # Create attribute if not exists
    #print(f"Saving input # {len(module.saved_inputs)}")
    module.saved_inputs.append(input)  # Append inputs to the model itself

def get_calib_dict_from_prompt(model,sampler,ddim_steps,prompt,seed=None):
    if seed is not None:
        seed_everything(seed)
    model.model.diffusion_model.saved_inputs =[]
    _ = gen_image_from_prompt(model,sampler,prompt,ddim_steps=ddim_steps,n_samples=1,n_iter=1)
    saved_inputs = model.model.diffusion_model.saved_inputs
    assert len(saved_inputs) == ddim_steps+1

    inputs_dict = get_inputs_dict_from_hook(saved_inputs,prompt)
    return inputs_dict

    
def gen_calibseb(ddim_steps=50,num_propts=128,output_folder='.',seed=None):

    config = OmegaConf.load(f'{Path.home()}/q-diffusion/configs/stable-diffusion/v1-inference.yaml')
    model = load_model_from_config(config, "/fastdata/users/nadavg/sd/qdiff/sd-v1-4.ckpt")
    device = torch.device("cuda")
    model = model.to(device)
    sampler = PLMSSampler(model)


    handle = model.model.diffusion_model.register_forward_hook(save_input_hook)

    orig_dataset = torch.load("/fastdata/users/nadavg/sd/qdiff/sd_coco-s75_sample1024_allst.pt")
    orig_dataset.keys()
    prompts = orig_dataset['prompts'][:num_propts]
    del orig_dataset

    seed_everything(seed) 

    for i,prompt in enumerate(prompts):
        print(f"\n\n ############ Prompt #: {i} ##################")
        inputs_dict = get_calib_dict_from_prompt(model,sampler,ddim_steps,prompt,seed=None)
        if 'calib_dict' not in locals():
            calib_dict = inputs_dict
        else:
            calib_dict = agragate_inputs_dict(calib_dict,inputs_dict)

    handle.remove()

    Path(output_folder).mkdir(parents=True,exist_ok=True)
    output_path = f"{output_folder}/calib_dict_steps{ddim_steps}.pt"
    torch.save(calib_dict,output_path)
    print(f"\n\n ##################### \n\nCalibration data saved to {output_path}")



if __name__ == '__main__':
    gen_calibseb(ddim_steps=20,num_propts=256,
                 output_folder='/fastdata/users/nadavg/sd/qdiff/gen_calib',
                 seed=42)




