import logging
from pathlib import Path
from types import MethodType
import torch.nn as nn
import torch
from qdiff.quant_block import get_specials, BaseQuantBlock
from qdiff.quant_block import QuantBasicTransformerBlock, QuantResBlock ,TimeStepEmbeddingSilu,QuantResBlockHF15
from qdiff.quant_block import QuantQKMatMul, QuantSMVMatMul, QuantBasicTransformerBlock, QuantAttnBlock,KerenlEwAdd
from qdiff.quant_layer import QuantModule, StraightThrough, QuantOp,UniformAffineQuantizer
#from ldm.modules.attention import BasicTransformerBlock
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.embeddings import TimestepEmbedding
from ldm.modules.diffusionmodules.util import GroupNorm32
from src.utils.torch_utils import add_full_name_to_module
from diffusers.models.transformers.transformer_2d import Transformer2DModel
from scripts.hf15.init_pipe import init_pipe

from mo_utils.utils.stand_alone_utils.har_utils import get_har_files,get_params_from_har
from mo_utils.utils.stand_alone_utils.pytorch2accelras import (
    get_nested_attr,
    get_weight_and_bias_from_layer_name,
    acc_ker_to_pytorch_weight,
    UpdateUnet,
    )
from mo_utils.utils.stand_alone_utils.quant_utils import calc_snr,calc_stats


logger = logging.getLogger(__name__)

PartialSmAbit = {
    'ver0': [],
    'ver1':[
        'down_blocks.0.attentions.0.transformer_blocks.0.attn1', # mm2
        'down_blocks.0.attentions.1.transformer_blocks.0.attn1',# mm6
        'up_blocks.3.attentions.0.transformer_blocks.0.attn1', # mm54
        'up_blocks.3.attentions.1.transformer_blocks.0.attn1', # mm58
        'up_blocks.3.attentions.2.transformer_blocks.0.attn1', # mm62
        ],
    }

class QuantModel(nn.Module):

    def __init__(self, model: nn.Module, weight_quant_params: dict = {}, act_quant_params: dict = {}, **kwargs):
        super().__init__()
        self.model = model
        self.sm_abit = kwargs.get('sm_abit', 8)
        self.quant_act_ops = kwargs.get('quant_act_ops', False)
        self.use_post_act_temb = kwargs.get('use_post_act_temb', False)
        self.unite_kvq_act = kwargs.get('unite_kvq_act', False)
        self.unite_skip_ln = kwargs.get('unite_skip_ln', False)
        self.in_channels = model.in_channels
        add_full_name_to_module(self.model)
        if hasattr(model, 'image_size'):
            self.image_size = model.image_size
        self.specials = get_specials()#act_quant_params['leaf_param'])
        self.refacor_group_norm(self.model)
        self.refactor_time_embedding(self.model)
        self.refactor_Transformer2DModel(self.model)
        self.quant_module_refactor(self.model, weight_quant_params, act_quant_params)
        self.quant_block_refactor(self.model, weight_quant_params, act_quant_params)
        add_full_name_to_module(self.model)
        self.split = kwargs.get('split', False)
        if self.split:
            self.add_spliter()
        add_full_name_to_module(self.model)

    def refactor_Transformer2DModel(self,model):
        for module in self.modules():
            if isinstance(module, Transformer2DModel):
                #print(module.full_name)
                module.ew_add_1 = KerenlEwAdd(in1_name='hidden_states',in2_name='residual')
                module._get_output_for_continuous_inputs = MethodType(_get_output_for_continuous_inputs_ew_add, module)


    def add_spliter(self):
        #up_blocks[0]
        self.model.up_blocks[0].resnets[0].set_split(1280)
        self.model.up_blocks[0].resnets[1].set_split(1280)
        self.model.up_blocks[0].resnets[2].set_split(1280)
        #up_blocks[1]
        self.model.up_blocks[1].resnets[0].set_split(1280)
        self.model.up_blocks[1].resnets[1].set_split(1280)
        self.model.up_blocks[1].resnets[2].set_split(1280)
        #up_blocks[2]
        self.model.up_blocks[2].resnets[0].set_split(1280)
        self.model.up_blocks[2].resnets[1].set_split(640)
        self.model.up_blocks[2].resnets[2].set_split(640)
        #up_blocks[3]
        self.model.up_blocks[3].resnets[0].set_split(640)
        self.model.up_blocks[3].resnets[1].set_split(320)
        self.model.up_blocks[3].resnets[2].set_split(320)
        

    def refactor_time_embedding(self,model: nn.Module):
        model.time_embedding = TimeStepEmbeddingSilu(
                                model.time_embedding,
                                use_post_act=self.use_post_act_temb)


    def refacor_group_norm(self, module: nn.Module):
        for name, child_module in module.named_children():
            if isinstance(child_module, nn.GroupNorm):
                setattr(module, name, GroupNorm32(child_module))
            else:
                self.refacor_group_norm(child_module)
    

    def quant_module_refactor(self, module: nn.Module, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        """
        Recursively replace the normal layers (conv2D, conv1D, Linear etc.) to QuantModule
        :param module: nn.Module with nn.Conv2d, nn.Conv1d, or nn.Linear in its children
        :param weight_quant_params: quantization parameters like n_bits for weight quantizer
        :param act_quant_params: quantization parameters like n_bits for activation quantizer
        """
        prev_quantmodule = None
        act_quant_params_layer_norm = act_quant_params.copy()
        act_quant_params_layer_norm['n_bits'] = 16

        for name, child_module in module.named_children():
            if isinstance(child_module, (nn.Conv2d, nn.Conv1d, nn.Linear)): # nn.Conv1d
                setattr(module, name, QuantModule(
                    child_module, weight_quant_params, act_quant_params))
                prev_quantmodule = getattr(module, name)
            elif self.quant_act_ops and isinstance(child_module,(nn.SiLU,GroupNorm32)):
                if self.use_post_act_temb and  isinstance(module, TimeStepEmbeddingSilu):
                    continue
                if isinstance(child_module, nn.SiLU):
                    setattr(module, name, QuantOp(
                        child_module, act_quant_params))
                elif isinstance(child_module, GroupNorm32):
                    setattr(module, name, QuantOp(
                        child_module,act_quant_params_layer_norm))

            elif isinstance(child_module, StraightThrough):
                continue

            else:
                self.quant_module_refactor(child_module, weight_quant_params, act_quant_params)

    def quant_block_refactor(self, module: nn.Module, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        for name, child_module in module.named_children():
            if type(child_module) in self.specials:
                if self.specials[type(child_module)] in [QuantBasicTransformerBlock, QuantAttnBlock]:
                    setattr(module, name, self.specials[type(child_module)](child_module,
                        act_quant_params, sm_abit=self.sm_abit,unite_kvq_act=self.unite_kvq_act))
                elif self.specials[type(child_module)] == QuantSMVMatMul:
                    setattr(module, name, self.specials[type(child_module)](
                        act_quant_params, sm_abit=self.sm_abit))
                elif self.specials[type(child_module)] == QuantQKMatMul:
                    setattr(module, name, self.specials[type(child_module)](
                        act_quant_params))
                elif self.specials[type(child_module)] == QuantResBlockHF15:
                    setattr(module, name, self.specials[type(child_module)](child_module,
                        act_quant_params, skip_time_act=self.use_post_act_temb,unite_skip_ln=self.unite_skip_ln))
                else:
                    setattr(module, name, self.specials[type(child_module)](child_module, 
                        act_quant_params))
            else:
                self.quant_block_refactor(child_module, weight_quant_params, act_quant_params)

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        for m in self.model.modules():
            if isinstance(m, (QuantModule, BaseQuantBlock)):
                m.set_quant_state(weight_quant, act_quant)

    def forward(self, x, timesteps=None, context=None):
        return self.model(x, timesteps, context)
    
    def set_running_stat(self, running_stat: bool, sm_only=False):
        for m in self.model.modules():
            if isinstance(m, QuantBasicTransformerBlock):
                if sm_only:
                    m.attn1.act_quantizer_w.running_stat = running_stat
                    m.attn2.act_quantizer_w.running_stat = running_stat
                else:
                    m.attn1.act_quantizer_q.running_stat = running_stat
                    m.attn1.act_quantizer_k.running_stat = running_stat
                    m.attn1.act_quantizer_v.running_stat = running_stat
                    m.attn1.act_quantizer_w.running_stat = running_stat
                    m.attn2.act_quantizer_q.running_stat = running_stat
                    m.attn2.act_quantizer_k.running_stat = running_stat
                    m.attn2.act_quantizer_v.running_stat = running_stat
                    m.attn2.act_quantizer_w.running_stat = running_stat
                    if self.unite_kvq_act:
                        m.attn1.act_quantizer_to_kvq.running_stat = running_stat
                        
            if isinstance(m, QuantModule) and not sm_only:
                m.set_running_stat(running_stat)

    def set_grad_ckpt(self, grad_ckpt: bool):
        for name, m in self.model.named_modules():
            if isinstance(m, (QuantBasicTransformerBlock, BasicTransformerBlock)):
                # logger.info(name)
                m.checkpoint = grad_ckpt
            # elif isinstance(m, QuantResBlock):
                # logger.info(name)
                # m.use_checkpoint = grad_ckpt


    def load_from_state_dict(self,saved_model_path,input_batch=None):
        if input_batch is None:
            input_batch = [torch.randn((1, 4, 64, 64)),torch.randn(1),torch.randn((1,77,768))]
    
        self.set_quant_state(weight_quant=True, act_quant=True)
        _=self.model(*input_batch)
        self.set_quant_state(False, False)
    
        for m in self.model.modules():
            if isinstance(m, UniformAffineQuantizer):
                if m.delta is not None:
                    m.delta = nn.Parameter(torch.tensor(m.delta,dtype=torch.float32))
                else: 
                    raise ValueError(f"delta is None for {m.full_name}")
                if m.zero_point is not None:
                    m.zero_point = nn.Parameter(torch.tensor(m.zero_point,dtype=torch.float32))
                else:
                    raise ValueError(f"zero_point is None for {m.full_name}")
        self.load_state_dict(torch.load(saved_model_path),strict=True)           
        




def _get_output_for_continuous_inputs_ew_add(self, hidden_states, residual, batch_size, height, width, inner_dim):
    
    if not self.use_linear_projection:
        hidden_states = (
            hidden_states.reshape(batch_size, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
        )
        hidden_states = self.proj_out(hidden_states)
    else:
        hidden_states = self.proj_out(hidden_states)
        hidden_states = (
            hidden_states.reshape(batch_size, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
        )

    output = self.ew_add_1(hidden_states , residual)
    return output

def init_qnn_from_fp_model(har_path,weight_quant_params: dict = {}, act_quant_params: dict = {}, 
                            input_batch=None,scheduler='ddim',debug=False,out_uu=False,**kwargs) -> QuantModel:
    
    if input_batch is None:
        input_batch = [torch.randn((1, 4, 64, 64)),torch.randn(1),torch.randn((1,77,768))]

    
    params = get_params_from_har(har_path,params_name='unet_sim.fpo.npz',verb=False)
    hn = get_params_from_har(har_path,params_name='unet_sim.hn',verb=False)
    pipe = init_pipe(scheduler=scheduler)
    unet = pipe.unet
    add_full_name_to_module(unet)
    out_org = unet(*input_batch)


    uu= UpdateUnet(unet, hn,params,debug=debug)
    uu.update_unet_convs()
    qnn = QuantModel(model=unet, 
                     weight_quant_params=weight_quant_params,
                    act_quant_params=act_quant_params,**kwargs)

    out_reorg_qnn_temp = qnn.model(*input_batch)

    uu= UpdateUnet(qnn.model, hn,params)
    uu.update_unet_ew_adds()
    out_reorg_qnn = qnn.model(*input_batch)

    snr = calc_snr(out_org[0],out_reorg_qnn[0])
    print (f"SNR of acceleras fp model : {snr} [db]")
    
    if out_uu:
        return qnn,pipe,uu
    return qnn , pipe


def init_qnn_from_opt_params(opt,scale_method,debug=False,):
    

    fp_model_path = opt.fp_model_path
    
    if 'partial_sm_abit' in opt:
        partial_sm_abit=PartialSmAbit[opt.partial_sm_abit]
    else:
        print(f"partial_sm_abit is not in opt, using ver0")
        partial_sm_abit = PartialSmAbit['ver0']


    wq_params = {'n_bits': opt.weight_bit, 'channel_wise': opt.channel_wise_weights, 'scale_method': scale_method,
                'symmetric': opt.symmetric_weight ,'debug':opt.debug or debug,}
    
   
    
    aq_params = {'n_bits': 8, 'channel_wise': False, 'scale_method': scale_method, 
                'leaf_param': True, 'debug':opt.debug or debug,
                'split_to_16bits':opt.split_to_16bits,
                'act_quant_mode' :'qdiff','act16bits_rtn':opt.act16bits_rtn,
                'partial_sm_abit': partial_sm_abit,}
    
    if opt.naive_weights_quant:
        wq_params['scale_method'] = 'max'
    
    split = opt.split
    sm_abit = opt.sm_abit
    quant_act_ops = opt.quant_act_ops
    unite_kvq_act = opt.unite_kvq_act
    unite_skip_ln = opt.unite_skip_ln


    qnn,pipe,uu = init_qnn_from_fp_model(
                    fp_model_path, weight_quant_params=wq_params, 
                    act_quant_params=aq_params,scheduler = 'euler',
                    out_uu=True,
                    act_quant_mode="qdiff", sm_abit=sm_abit, 
                    quant_act_ops = quant_act_ops, split=split,
                    unite_kvq_act = unite_kvq_act,
                    unite_skip_ln= unite_skip_ln
                    )
    
    return qnn,pipe,uu


def init_qnn_from_qdiff_opt(qdiff_opt_path):
    
    qdiff_opt_path = Path(qdiff_opt_path)
    opt_params = qdiff_opt_path / 'opt.pth'
    qdiff_opt_model = qdiff_opt_path / 'ckpt.pth' 

    if not opt_params.exists() or not qdiff_opt_model.exists():
        raise ValueError(f"opt.pth or ckpt.pth not found in {qdiff_opt_path}")
    
    opt = torch.load(opt_params)

    qnn,pipe,uu = init_qnn_from_opt_params(opt,scale_method= 'mse')
    
    qnn.load_from_state_dict(str(qdiff_opt_model))

    return qnn,pipe,uu

