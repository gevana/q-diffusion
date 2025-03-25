import logging
import torch.nn as nn
from qdiff.quant_block import get_specials, BaseQuantBlock
from qdiff.quant_block import QuantBasicTransformerBlock, QuantResBlock ,TimeStepEmbeddingSilu,QuantResBlockHF15
from qdiff.quant_block import QuantQKMatMul, QuantSMVMatMul, QuantBasicTransformerBlock, QuantAttnBlock,KerenlEwAdd
from qdiff.quant_layer import QuantModule, StraightThrough, QuantOp
#from ldm.modules.attention import BasicTransformerBlock
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.embeddings import TimestepEmbedding
from ldm.modules.diffusionmodules.util import GroupNorm32
from src.utils.torch_utils import add_full_name_to_module
from diffusers.models.transformers.transformer_2d import Transformer2DModel


logger = logging.getLogger(__name__)


class QuantModel(nn.Module):

    def __init__(self, model: nn.Module, weight_quant_params: dict = {}, act_quant_params: dict = {}, **kwargs):
        super().__init__()
        self.model = model
        self.sm_abit = kwargs.get('sm_abit', 8)
        self.quant_act_ops = kwargs.get('quant_act_ops', False)
        self.use_post_act_temb = kwargs.get('use_post_act_temb', False)
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
                module._get_output_for_continuous_inputs = _get_output_for_continuous_inputs_ew_add


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
        for name, child_module in module.named_children():
            if isinstance(child_module, (nn.Conv2d, nn.Conv1d, nn.Linear)): # nn.Conv1d
                setattr(module, name, QuantModule(
                    child_module, weight_quant_params, act_quant_params))
                prev_quantmodule = getattr(module, name)
            elif self.quant_act_ops and isinstance(child_module,(nn.SiLU,GroupNorm32)):
                if self.use_post_act_temb and  isinstance(module, TimeStepEmbeddingSilu):
                    continue
                setattr(module, name, QuantOp(
                    child_module, act_quant_params))

            elif isinstance(child_module, StraightThrough):
                continue

            else:
                self.quant_module_refactor(child_module, weight_quant_params, act_quant_params)

    def quant_block_refactor(self, module: nn.Module, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        for name, child_module in module.named_children():
            if type(child_module) in self.specials:
                if self.specials[type(child_module)] in [QuantBasicTransformerBlock, QuantAttnBlock]:
                    setattr(module, name, self.specials[type(child_module)](child_module,
                        act_quant_params, sm_abit=self.sm_abit))
                elif self.specials[type(child_module)] == QuantSMVMatMul:
                    setattr(module, name, self.specials[type(child_module)](
                        act_quant_params, sm_abit=self.sm_abit))
                elif self.specials[type(child_module)] == QuantQKMatMul:
                    setattr(module, name, self.specials[type(child_module)](
                        act_quant_params))
                elif self.specials[type(child_module)] == QuantResBlockHF15:
                    setattr(module, name, self.specials[type(child_module)](child_module,
                        act_quant_params, skip_time_act=self.use_post_act_temb))
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