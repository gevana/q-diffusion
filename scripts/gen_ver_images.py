from pathlib import Path
import os
import sys
import yaml
from qdiff import QuantModel

from qdiff.quant_layer import QuantOp , QuantModule #BaseQuantBlock,
    #block_reconstruction, layer_reconstruction,

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
from src.utils.torch_utils import add_full_name_to_module , add_snr_hook_to_model, collect_snr_hook,add_output_hook_to_model,collect_output_hook
from PIL import Image
import argparse


def set_ffn_act_native(qnn,use_act_quant=True,use_weight_quant=True):
    for i in [1,2,4,5,7,8]:
        print(f'setting ffn input_blocks {i} to {use_act_quant=}, {use_weight_quant=}')
        qnn.model.input_blocks[i][1].transformer_blocks[0].ff.net[2].use_act_quant = use_act_quant
        qnn.model.input_blocks[i][1].transformer_blocks[0].ff.net[2].use_weight_quant = use_weight_quant
    for i in [3,4,5,6,7,8,9,10,11]:
        print(f'setting ffn output_blocks {i} to {use_act_quant=}, {use_weight_quant=}')
        qnn.model.output_blocks[i][1].transformer_blocks[0].ff.net[2].use_act_quant = use_act_quant
        qnn.model.output_blocks[i][1].transformer_blocks[0].ff.net[2].use_weight_quant = use_weight_quant

    print(f'setting ffn middle_block[1] to {use_act_quant=}, {use_weight_quant=}')
    qnn.model.middle_block[1].transformer_blocks[0].ff.net[2].use_act_quant = use_act_quant
    qnn.model.middle_block[1].transformer_blocks[0].ff.net[2].use_weight_quant = use_weight_quant




def set_Silu_act_native(qnn,use_act_quant=True):
    for i in [1,2,4,5,7,8,10,11]:
        print(f'setting Silu input_blocks {i} to {use_act_quant=}')
        qnn.model.input_blocks[i][0].in_layers[1].use_act_quant = use_act_quant
        qnn.model.input_blocks[i][0].out_layers[1].use_act_quant = use_act_quant
    for i in [3,4,5,6,7,8,9,10,11]:
        print(f'setting Silu output_blocks {i} to {use_act_quant=}')
        qnn.model.output_blocks[i][0].in_layers[1].use_act_quant = use_act_quant
        qnn.model.output_blocks[i][0].out_layers[1].use_act_quant = use_act_quant
    qnn.model.out[1].use_act_quant = use_act_quant

def set_split_act_native(qnn,use_act_quant=True,use_weight_quant=True):
    for i in range(11):
        print(f'setting output block {i} skip_connection and in_layers[0]  to {use_act_quant=}, {use_weight_quant=}')
        qnn.model.output_blocks[i][0].skip_connection.use_act_quant = use_act_quant
        qnn.model.output_blocks[i][0].skip_connection.use_weight_quant = use_weight_quant
        qnn.model.output_blocks[i][0].in_layers[0].use_act_quant = use_act_quant


def set_QuantOp_act_native(qnn,use_act_quant):
    for module in qnn.model.modules():
        if isinstance(module,QuantOp):
            print(f'setting QuantOp {module.full_name} to {use_act_quant=}')
            module.use_act_quant = use_act_quant

def set_QuantModule_act_native(qnn,use_act_quant,use_weight_quant):
    for module in qnn.model.modules():
        if isinstance(module,QuantModule) and not isinstance(module,QuantOp):
            print(f'setting QuantModel {module.full_name} to {use_act_quant=} and {use_weight_quant=}')
            module.use_act_quant = use_act_quant
            module.use_weight_quant = use_weight_quant


def gen_ver_images(cali_ckpt,nbit,symmetric,quant_act_ops,ddim_steps,act_bits,
                   split_to_16bits,naive_quant_weights,
                    output_dir='./output',
                    act_quant=True,weight_quant=True,
                    num_images=12,
                    prompt=None,seed=42,
                    ddim_discretize='uniform'):

    
    config = OmegaConf.load(f'{Path.home()}/q-diffusion/configs/stable-diffusion/v1-inference.yaml')
    model = load_model_from_config(config, "/fastdata/users/nadavg/sd/qdiff/sd-v1-4.ckpt")
    device = torch.device("cuda")
    model = model.to(device)
    sampler = PLMSSampler(model,ddim_discretize=ddim_discretize)
    setattr(sampler.model.model.diffusion_model, "split", True)


    wq_params = {'n_bits': nbit, 'channel_wise': True, 'scale_method': 'max','symmetric':symmetric}
    aq_params = {'n_bits': act_bits, 'channel_wise': False, 'scale_method': 'max', 'leaf_param':  True,
                 'split_to_16bits':split_to_16bits,'act_quant_mode':'qdiff'}

    qnn = QuantModel(
            model=sampler.model.model.diffusion_model, weight_quant_params=wq_params, act_quant_params=aq_params,
            act_quant_mode="qdiff", sm_abit=16, quant_act_ops=quant_act_ops)
    
    add_full_name_to_module(qnn)
    qnn.cuda()
    qnn.eval()
    qnn.set_grad_ckpt(False)



    if weight_quant or act_quant:
        cali_data = (torch.randn(1, 4, 64, 64), torch.randint(0, 1000, (1,)), torch.randn(1, 77, 768))
        resume_cali_model(qnn, cali_ckpt, cali_data, act_quant, "qdiff", cond=True,naive_weights_quant=naive_quant_weights)
    qnn.set_quant_state(weight_quant=weight_quant, act_quant=act_quant)

    #set_split_act_native(qnn,use_act_quant=False,use_weight_quant=True)
    #set_QuantOp_act_native(qnn,use_act_quant=False)
    

    #set_QuantModule_act_native(qnn,use_act_quant=True,use_weight_quant=False)
    if True:
        set_ffn_act_native(qnn,use_act_quant=False,use_weight_quant=True)
        set_Silu_act_native(qnn,use_act_quant=False)
        qnn.model.input_blocks[6][0].op.use_act_quant = False
        qnn.model.input_blocks[9][0].op.use_act_quant = False

    if args.snr:
        print('Adding SNR hooks')
        add_snr_hook_to_model(qnn,QuantModule)
    
    if args.save_outputs:
        print('Adding output hooks')
        add_output_hook_to_model(qnn,QuantModule)

    if output_dir is not None:
        os.makedirs(output_dir,exist_ok=True)
    
    
    if prompt is None:
        prompts  = yaml.load(
                    open(f'{Path.home()}/q-diffusion/scripts/prompt.yaml','r'),Loader=yaml.FullLoader
                    )
        
        Images = []
        num_images = 2 * (num_images // 2)
        for ind in range(num_images):
            prompt = prompts[ind]['prompt']
            seed = prompts[ind]['seed']
            seed_everything(seed)
            img =  gen_image_from_prompt(model,sampler,prompt,ddim_steps=ddim_steps,
                                        use_autocast=False,n_samples=1,n_rows=0,n_iter=1)
            if output_dir is not None:
                img.save(f'{output_dir}/gen_image_{ind}.png')
            img = np.array(img)
            Images.append(img)
        
        upper = np.concatenate(Images[:num_images//2],axis=1)
        lower = np.concatenate(Images[num_images//2:],axis=1)
        final = np.concatenate([upper,lower],axis=0)
        final = Image.fromarray(final.astype(np.uint8))
        if output_dir is not None:
            snr = collect_snr_hook(qnn,QuantModule)
            os.makedirs(output_dir,exist_ok=True)
            final.save(f'{output_dir}/gen_images.png')
            print(f'Images saved to {output_dir}/gen_images.png')
            torch.save(snr,f'{output_dir}/snr.pth')
            print(f'SNR saved to {output_dir}/snr.pth')
        return final

    else:
        seed_everything(seed)
        img =  gen_image_from_prompt(model,sampler,prompt,ddim_steps=ddim_steps,
                                    use_autocast=False,n_samples=1,n_rows=0,n_iter=1)
        if output_dir is not None:
            img.save(f'{output_dir}/gen_image.png')
            print(f'Image saved to {output_dir}/gen_image.png')
            snr = collect_snr_hook(qnn,QuantModule)
            if snr:
                torch.save(snr,f'{output_dir}/snr.pth')
                print(f'SNR saved to {output_dir}/snr.pth')
            outputs = collect_output_hook(qnn,QuantModule)
            if outputs:
                print(f'Outputs saved to {output_dir}/outputs_{act_quant=}_{weight_quant=}.npz')
                np.savez(f'{output_dir}/outputs_{act_quant=}_{weight_quant=}.npz',outputs)
                img.save(f'{output_dir}/gen_image_{act_quant=}_{weight_quant=}.png')
        return img ,qnn




def str_to_bool(s):
    if s == 'True' or s == 'true':
        return True
    elif s == 'False' or s == 'false':
        return False
    else:
        raise ValueError(f'Invalid string {type(s)=} ,  {s=}')

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--cali_ckpt',type=str,required=True)
    argparser.add_argument('--nbit',type=int,required=True)
    argparser.add_argument('--symmetric',type=str,required=True)
    argparser.add_argument('--quant_act_ops',type=str,required=True)
    argparser.add_argument('--ddim_steps',type=int,required=True)
    argparser.add_argument('--act_bits',type=int,required=True) 
    argparser.add_argument('--split_to_16bits',type=str,required=True)
    argparser.add_argument('--naive_quant_weights',type=str,required=True)
    argparser.add_argument('--output_dir',type=str,required=False,default='./output')
    argparser.add_argument('--act_quant',type=str,required=False,default='True')
    argparser.add_argument('--weight_quant',type=str,required=False,default='True')
    argparser.add_argument('--num_images',type=int,required=False,default=12)
    argparser.add_argument('--prompt',type=str,required=False,default='None')
    argparser.add_argument('--seed',type=int,required=False,default=42)
    argparser.add_argument('--snr',type=str,required=False,default='False')
    argparser.add_argument('--save_outputs',type=str,required=False,default='False')    
    argparser.add_argument('--ddim_discretize',type=str,choices=["uniform", "quad","uniform+"],default="uniform")    

    args = argparser.parse_args()
    args.symmetric = str_to_bool(args.symmetric)
    args.quant_act_ops = str_to_bool(args.quant_act_ops)
    args.split_to_16bits = str_to_bool(args.split_to_16bits)
    args.naive_quant_weights = str_to_bool(args.naive_quant_weights)
    args.act_quant = str_to_bool(args.act_quant)
    args.weight_quant = str_to_bool(args.weight_quant)
    args.snr = str_to_bool(args.snr)
    args.save_outputs = str_to_bool(args.save_outputs)

    if args.prompt == 'None':
        args.prompt = None

    print(f'\n\n{args=}\n\n')

    gen_ver_images(args.cali_ckpt,args.nbit,args.symmetric,args.quant_act_ops,
                   args.ddim_steps,args.act_bits,
                   args.split_to_16bits,args.naive_quant_weights,
                   args.output_dir,args.act_quant,args.weight_quant,
                   args.num_images,args.prompt,args.seed,args.ddim_discretize)