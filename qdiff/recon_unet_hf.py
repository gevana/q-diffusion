import torch
# import linklink as link
import logging
from qdiff.quant_layer import QuantModule,QuantOp, StraightThrough, lp_loss,UniformAffineQuantizer
from qdiff.quant_model import QuantModel
from qdiff.quant_block import BaseQuantBlock
from qdiff.adaptive_rounding import AdaRoundQuantizer
from qdiff.utils import save_grad_data, save_inp_oup_data
import wandb
from qdiff.layer_recon import layer_reconstruction
from qdiff.block_recon import block_reconstruction
logger = logging.getLogger(__name__)



def unetHF_reconstruction(qnn: QuantModel,**kwargs):
    # recon preliminary layers. 
    ## conv_in 
    model = qnn.model
    layer_reconstruction(qnn, model.conv_in, **kwargs)
    # down_blocks 
    for down_block in model.down_blocks[:-1]:
        blocks = list(zip(down_block.resnets, down_block.attentions))
        for i, (resnet, attn) in enumerate(blocks):
            block_reconstruction(qnn, resnet, **kwargs)
            layer_reconstruction(qnn, attn.norm, **kwargs)
            layer_reconstruction(qnn, attn.proj_in, **kwargs)
            for transblock in attn.transformer_blocks:    
                block_reconstruction(qnn, transblock, **kwargs)
            layer_reconstruction(qnn, attn.proj_out, **kwargs)
        if down_block.downsamplers is not None: # moduleList
            for ds in down_block.downsamplers:
                if ds.norm:
                    layer_reconstruction(qnn, ds.norm, **kwargs)
                if ds.conv:
                    layer_reconstruction(qnn, ds.conv, **kwargs)
    
    down_block = model.down_blocks[-1]
    for resnet in down_block.resnets:
        block_reconstruction(qnn, resnet, **kwargs)
    if down_block.downsamplers is not None: # moduleList
        for ds in down_block.downsamplers:
            if ds.norm:
                layer_reconstruction(qnn, ds.norm, **kwargs)
            if ds.conv:
                layer_reconstruction(qnn, ds.conv, **kwargs)
    
    
    # middle_block
    block_reconstruction(qnn, model.mid_block.resnets[0], **kwargs)
    for attn, resnet in zip(model.mid_block.attentions, model.mid_block.resnets[1:]):
        layer_reconstruction(qnn, attn.norm, **kwargs)
        layer_reconstruction(qnn, attn.proj_in, **kwargs)
        for transblock in attn.transformer_blocks:    
                block_reconstruction(qnn, transblock, **kwargs)
        layer_reconstruction(qnn, attn.proj_out, **kwargs)
        block_reconstruction(qnn, resnet, **kwargs)
    
    # up_blocks
    # up_block[0] UpBlock2D
    upblock = model.up_blocks[0]
    for resnet in upblock.resnets:
        block_reconstruction(qnn, resnet, **kwargs)
    if upblock.upsamplers:
         for upsampler in upblock.upsamplers:
            if upsampler.norm:
                layer_reconstruction(qnn, upsampler.norm, **kwargs) 
            if upsampler.conv:
                layer_reconstruction(qnn, upsampler.conv, **kwargs)
    
    # up_block[1:4] CrossAttnUpBlock2D
    for upblock in model.up_blocks[1:]:
        for resnet, attn in zip(upblock.resnets, upblock.attentions):
            block_reconstruction(qnn, resnet, **kwargs)
            layer_reconstruction(qnn, attn.norm, **kwargs)
            layer_reconstruction(qnn, attn.proj_in, **kwargs)
            for transblock in attn.transformer_blocks:    
                block_reconstruction(qnn, transblock, **kwargs)
            layer_reconstruction(qnn, attn.proj_out, **kwargs)
        if upblock.upsamplers:
            for upsampler in upblock.upsamplers:
                if upsampler.norm:
                    layer_reconstruction(qnn, upsampler.norm, **kwargs)
                if upsampler.conv:
                    layer_reconstruction(qnn, upsampler.conv, **kwargs)
    # 6. post-process
    if model.conv_norm_out:
        layer_reconstruction(qnn, model.conv_norm_out, **kwargs)
        layer_reconstruction(qnn, model.conv_act, **kwargs)
    layer_reconstruction(qnn, model.conv_out, **kwargs)

    # check if all quantizers are optimized
    check_all_optimized(qnn, partial_name='act' if kwargs['act_quant'] else 'weight')


def check_all_optimized(qnn,partial_name ):
    for name, module in qnn.model.named_modules():
        if isinstance(module, (UniformAffineQuantizer, AdaRoundQuantizer)):
            if not module.optimized and partial_name in module.full_name:
                logger.warning(f'{module.full_name} is not optimized  type={type(module)}')
            
             




            
