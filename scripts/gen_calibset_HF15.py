
from pathlib import Path
import torch
import pandas as pd
from diffusers import StableDiffusionPipeline
import os

def get_inputs_dict_from_hook(saved_inputs,prompt=[]):
    #saved_inputs.pop(1)
    
    inputs_dict = {'xs':[],'ts':[],'cs':[],'ucs':[]}
    for i in range(len(saved_inputs)):
        #if i == 1:
        #    continue
        #inputs_dict['uxs'].append(saved_inputs[i][0][0:1].cpu())
        inputs_dict['xs'].append(saved_inputs[i][0][1:2].cpu())
        inputs_dict['ts'].append(saved_inputs[i][1].cpu().unsqueeze(0))
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

def get_calib_dict_from_prompt(pipe,ddim_steps,prompt,seed=None):
 
    pipe.unet.saved_inputs =[]
    _ = pipe(prompt,num_inference_steps=ddim_steps)
    saved_inputs = pipe.unet.saved_inputs
    assert len(saved_inputs) == ddim_steps

    inputs_dict = get_inputs_dict_from_hook(saved_inputs,prompt)
    return inputs_dict

    
def gen_calibseb(ddim_steps=50,num_propts=128,output_folder='.',seed=None):

    pipe = StableDiffusionPipeline.from_pretrained("SG161222/Realistic_Vision_V4.0_noVAE")
    device = torch.device("cuda")
    pipe = pipe.to(device)
    unet = pipe.unet


    handle = unet.register_forward_hook(save_input_hook)

    prompts = pd.read_parquet('scripts/eval.parquet')['Prompt']

   

    for i,prompt in enumerate(prompts):
        if i >= num_propts:
            break
        print(f"\n\n ############ Prompt #: {i} ##################")
        inputs_dict = get_calib_dict_from_prompt(pipe,ddim_steps,prompt,seed=None)
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
    

    gen_calibseb(ddim_steps=50,num_propts=256,
                 output_folder='/fastdata/users/nadavg/sd/qdiff_hf15/gen_calib',
                 seed=42)




