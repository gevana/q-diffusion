
import torch 
from qdiff.quant_layer import  QuantOp , QuantModule
    #block_reconstruction, layer_reconstruction,


def add_full_name_to_module(model) -> None:
    # add full name to each module
    for name, module in model.named_modules():
        module.full_name = name


def hook_act_snr(module, input, output):
    use_act_quant = module.use_act_quant
    
    module.use_act_quant = False #not use_act_quant
    module._forward_hooks.clear()
    output_ = module(*input)
    snr =  10 *torch.log10(torch.sum(output**2) / torch.sum((output - output_)**2))
    module.use_act_quant = use_act_quant
    module.register_forward_hook(hook_act_snr)

    snr = snr.cpu().detach().numpy()
    if module.act_snr is None:
        module.act_snr = [snr]
    else:
        module.act_snr.append(snr)

def add_snr_hook_to_model(model, instanceof):
    for name, module in model.named_modules():
        if isinstance(module, instanceof):
            module._forward_hooks.clear()
            module.act_snr = None
            module.register_forward_hook(hook_act_snr)

def collect_snr_hook(model,instanceof):
    snr = {}
    for name, module in model.named_modules():
        if isinstance(module, instanceof):
            if getattr(module,'act_snr',None) is not None:
                snr[module.full_name]=module.act_snr #.cpu().detach()#.numpy()
    return snr

    

