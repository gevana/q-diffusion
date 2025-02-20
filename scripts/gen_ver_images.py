from pathlib import Path
import os
import sys
import yaml
from qdiff import (
    QuantModel,
    #QuantModule,QuantOp ,BaseQuantBlock,
    #block_reconstruction, layer_reconstruction,
)
from qdiff.utils import resume_cali_model#, get_train_samples
from scripts.gen_image import gen_image_from_prompt
import torch
from omegaconf import OmegaConf
from txt2img import load_model_from_config
from pytorch_lightning import seed_everything
from ldm.models.diffusion.plms import PLMSSampler
from qdiff.quant_layer import UniformAffineQuantizer
from qdiff.adaptive_rounding import AdaRoundQuantizer
import numpy as np

import matplotlib.pyplot as plt
#from mo_utils.utils.plot_aux import plot_hist, myim
from src.utils.torch_utils import add_full_name_to_module
from PIL import Image





def gen_ver_images(cali_ckpt,nbit,symmetric,quant_act_ops,ddim_steps,act_bits,split_to_16bits,naive_quant_weights,
               output_dir='./output',
               quant_act=True,weight_quant=True):

    
    config = OmegaConf.load(f'{Path.home()}/q-diffusion/configs/stable-diffusion/v1-inference.yaml')
    model = load_model_from_config(config, "/fastdata/users/nadavg/sd/qdiff/sd-v1-4.ckpt")
    device = torch.device("cuda")
    model = model.to(device)
    sampler = PLMSSampler(model)
    setattr(sampler.model.model.diffusion_model, "split", True)


    wq_params = {'n_bits': nbit, 'channel_wise': True, 'scale_method': 'max','symmetric':symmetric}
    aq_params = {'n_bits': act_bits, 'channel_wise': False, 'scale_method': 'max', 'leaf_param':  True,'split_to_16bits':split_to_16bits}

    qnn = QuantModel(
            model=sampler.model.model.diffusion_model, weight_quant_params=wq_params, act_quant_params=aq_params,
            act_quant_mode="qdiff", sm_abit=16, quant_act_ops=quant_act_ops)
    
    add_full_name_to_module(qnn)
    qnn.cuda()
    qnn.eval()
    qnn.set_grad_ckpt(False)


    if weight_quant or quant_act:
        cali_data = (torch.randn(1, 4, 64, 64), torch.randint(0, 1000, (1,)), torch.randn(1, 77, 768))
        resume_cali_model(qnn, cali_ckpt, cali_data, quant_act, "qdiff", cond=True)#,naive_weights_quant=naive_quant_weights)
    qnn.set_quant_state(weight_quant=weight_quant, act_quant=quant_act)

    prompts  = yaml.load(
                open(f'{Path.home()}/q-diffusion/scripts/prompt.yaml','r'),Loader=yaml.FullLoader
                )
    Images = []
    for prompt,seed in prompts[:12]:
        seed_everything(seed)
        img =  gen_image_from_prompt(model,sampler,prompt,ddim_steps=ddim_steps,use_autocast=False)
        # PIL image to numpy array
        img = np.array(img)
        Images.append(img)
    
    upper = np.concatenate(Images[:6],axis=1)
    lower = np.concatenate(Images[6:],axis=1)
    final = np.concatenate([upper,lower],axis=0)
    final = Image.fromarray(final.astype(np.uint8))
    if output_dir is not None:
        final.save(f'{output_dir}/gen_images.png',final)
    return final
        
