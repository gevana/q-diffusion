from pathlib import Path
from collections import defaultdict
from qdiff import (
    QuantModel, QuantModule,QuantOp ,BaseQuantBlock,
    block_reconstruction, layer_reconstruction,
)
from qdiff.utils import resume_cali_model
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
from src.utils.torch_utils import add_full_name_to_module


torch.set_grad_enabled(False)

class DynamicStats:
    def __init__(self, per_channel=True, device="cpu",num_channels = -1,name = None):
      
        self.device = torch.device(device)
        self.per_channel = per_channel 
        self.num_channels = num_channels
        self.name = name

        self.stats ={
            "mean": None,
            "var": None,
            "min": None,
            "max": None,
            "num_elements": 0,
        }

        # Define histogram bins
    @staticmethod
    def update(layer, data):
        self = layer.stats_collector
        if isinstance(data, tuple):
            data = data[0]
        # n,c,h,w  > c*h*w ,n        
        N,C,H,W = data.size()
        num_elements = N*H*W
        data = data.to(self.device).permute(1, 0, 2, 3).contiguous().view(C, N * H * W)
        min_vals = data.min(dim=1)[0].reshape(1,-1)
        max_vals = data.max(dim=1)[0].reshape(1,-1)
        mean_vals = data.mean(dim=1).reshape(1,-1)
        var_vals = data.var(dim=1).reshape(1,-1)
        
        if self.stats["num_elements"] == 0:
            self.stats["mean"] = mean_vals
            self.stats["var"] = var_vals
            self.stats["min"] = min_vals
            self.stats["max"] = max_vals
            self.stats["num_elements"] = num_elements

            return

        curr_mean = self.stats["mean"].clone()
        self.stats["mean"] = (self.stats["mean"] * self.stats["num_elements"] + mean_vals * num_elements) / (
            self.stats["num_elements"] + num_elements
        )
        self.stats["var"] = self.update_var(
            self.stats["var"],
            self.stats["num_elements"],
            var_vals,
            num_elements,
            curr_mean,
            mean_vals,
            self.stats["mean"],
        )
        self.stats["min"] = torch.min(self.stats["min"], min_vals)
        self.stats["max"] = torch.max(self.stats["max"], max_vals)
        self.stats["num_elements"] += num_elements

    @staticmethod    
    def update_var(curr_var,num_elem,new_var,new_num_elem,
                   curr_mean,new_mean,updated_mean):

        numerator = ((num_elem - 1) * curr_var + (new_num_elem - 1) * new_var) + num_elem * (curr_mean - updated_mean) ** 2 + new_num_elem * (new_mean - updated_mean) ** 2
        return numerator / (num_elem + new_num_elem - 1)
    
    @classmethod
    def add_as_pre_forward_hook(cls,layer,per_channel=True,device="cpu",num_channels = 1):
        stats = cls(per_channel,device,num_channels,name = layer.full_name)
        layer.stats_collector = stats
        handle = layer.register_forward_pre_hook(layer.stats_collector.update)
        return handle


def get_train_samples(sample_data, num_samples=128,timesteps=None):
        # get the real number of timesteps (especially for DDIM)
    if timesteps is None:
        nsteps = len(sample_data["xs"])
        num_st = nsteps // 2
        timesteps = list(range(0, nsteps, nsteps//num_st))
    #logger.info(f'Selected {len(timesteps)} steps from {nsteps} sampling steps')
    xs_lst = [sample_data["xs"][i][:num_samples] for i in timesteps]
    ts_lst = [sample_data["ts"][i][:num_samples] for i in timesteps]
    if True:#args.cond:
        xs_lst += xs_lst
        ts_lst += ts_lst
        conds_lst = [sample_data["cs"][i][:num_samples] for i in timesteps] + [sample_data["ucs"][i][:num_samples] for i in timesteps]
    xs = torch.cat(xs_lst, dim=0)
    ts = torch.cat(ts_lst, dim=0)
   
    conds = torch.cat(conds_lst, dim=0)
    return xs, ts, conds

def gen_stats(ddim_steps=50,num_samples=128,output_folder=None):

    if output_folder is not None:
        output_path = Path(output_folder) / f'ddim_steps_{ddim_steps}.pt'
        output_path.parent.mkdir(parents=True, exist_ok=True)
       
    config = OmegaConf.load(f'{Path.home()}/q-diffusion/configs/stable-diffusion/v1-inference.yaml')
    model = load_model_from_config(config, "/fastdata/users/nadavg/sd/qdiff/sd-v1-4.ckpt")
    device = torch.device("cuda")
    model = model.to(device)
    
    unet = model.model.diffusion_model
    add_full_name_to_module(unet)


    if ddim_steps == 50 :
        calib_data_path =  "/fastdata/users/nadavg/sd/qdiff/sd_coco-s75_sample1024_allst.pt"
    elif ddim_steps == 20:
        #calib_data_path = "/fastdata/users/nadavg/sd/qdiff/gen_calib/calib_dict_steps50.pt"
        calib_data_path = "/fastdata/users/nadavg/sd/qdiff/gen_calib/calib_dict_steps20.pt"
    
    sample_data = torch.load(calib_data_path)

    Stats_dict = defaultdict(defaultdict)

    for time_step in range(ddim_steps):
        print(f"\n ############# Processing timestep {time_step} ############# \n")
        xs, ts, conds = get_train_samples(sample_data, num_samples=num_samples, timesteps=[time_step])

        h = []
        for b in unet.input_blocks:
            b._forward_pre_hooks.clear()
            hi = DynamicStats.add_as_pre_forward_hook(b,per_channel=True)
            h.append((b.full_name , hi))
        
        
        for b in unet.output_blocks:
            b._forward_pre_hooks.clear()
            hi = DynamicStats.add_as_pre_forward_hook(b,per_channel=True)
            h.append((b.full_name , hi))
        
        for i in range(len(xs)):
            x = xs[i].unsqueeze(0).to(device)
            t = ts[i].unsqueeze(0).to(device)
            c = conds[i].unsqueeze(0).to(device)
            _=unet(x,t,c)
        

        for b in unet.input_blocks:
            Stats_dict[b.stats_collector.name][time_step] =  b.stats_collector.stats    
        
        for b in unet.output_blocks:
            Stats_dict[b.stats_collector.name][time_step] =  b.stats_collector.stats    
    
    if output_folder is not None:
        print(f"Saving stats to {output_path}")
        torch.save(Stats_dict,str(output_path))

    return Stats_dict


if __name__ == "__main__":
    seed_everything(42)
    ddim_steps = 20#50
    num_samples = 128
    output_folder = "/fastdata/users/nadavg/sd/qdiff/gen_stats"
    gen_stats(ddim_steps,num_samples,output_folder)
    print("Done")