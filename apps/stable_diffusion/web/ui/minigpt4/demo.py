import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import gradio as gr

######### Import for Generate Method ##########
import copy
import inspect
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import torch.distributed as dist
from torch import nn

######### Import for Generate Method ##########

import gradio as gr

from omegaconf import OmegaConf

from apps.stable_diffusion.web.ui.minigpt4.conversation import Chat, CONV_VISION, MiniGPT4SHARK
from apps.stable_diffusion.web.ui.minigpt4.blip_processors import Blip2ImageEvalProcessor

# ========================================
#             Model Initialization
# ========================================

print('Initializing Chat')
config = OmegaConf.load("apps/stable_diffusion/web/ui/minigpt4/configs/minigpt4_eval.yaml")
model_config = OmegaConf.create()
model_config = OmegaConf.merge(
    model_config,
    OmegaConf.load('apps/stable_diffusion/web/ui/minigpt4/configs/minigpt4.yaml'),
    {"model": config["model"]},
)
model_config = model_config['model']
model_config.device_8bit = 0
model = MiniGPT4SHARK.from_config(model_config).to('cpu')

datasets = config.get("datasets", None)
dataset_config = OmegaConf.create()
for dataset_name in datasets:
    dataset_config_path = 'apps/stable_diffusion/web/ui/minigpt4/configs/cc_sbu_align.yaml'
    dataset_config = OmegaConf.merge(
        dataset_config,
        OmegaConf.load(dataset_config_path),
        {"datasets": {dataset_name: config["datasets"][dataset_name]}},
    )
dataset_config = dataset_config['datasets']
vis_processor_cfg = dataset_config.cc_sbu_align.vis_processor.train
vis_processor = Blip2ImageEvalProcessor.from_config(vis_processor_cfg)

llama = model.llama_model

# class FirstLlamaModel(torch.nn.Module):
#     def __init__(self, model):
#         super().__init__()
#         self.model = model
#         print('SHARK: Loading LLAMA Done')

#     def forward(self, inputs_embeds, position_ids, attention_mask):
#         print("************************************")
#         print("inputs_embeds: ", inputs_embeds.shape, " dtype: ", inputs_embeds.dtype)
#         print("position_ids: ", position_ids.shape, " dtype: ", position_ids.dtype)
#         print("attention_mask: ", attention_mask.shape, " dtype: ", attention_mask.dtype)
#         print("************************************")
#         config = {
#             'inputs_embeds':inputs_embeds,
#             'position_ids':position_ids,
#             'past_key_values':None,
#             'use_cache':True,
#             'attention_mask':attention_mask,        
#         }
#         output = self.model(
#                 **config,
#                 return_dict=True,
#                 output_attentions=False,
#                 output_hidden_states=False,
#             )
#         return_vals = []        
#         return_vals.append(output.logits)
#         temp_past_key_values = output.past_key_values
#         for item in temp_past_key_values:
#             return_vals.append(item[0])
#             return_vals.append(item[1])
#         return tuple(return_vals)

# class SecondLlamaModel(torch.nn.Module):
#     def __init__(self, model):
#         super().__init__()
#         self.model = model
#         print('SHARK: Loading LLAMA Done')

#     def forward(self, input_ids, position_ids, attention_mask,
#         i1,
#         i2,
#         i3,
#         i4,
#         i5,
#         i6,
#         i7,
#         i8,
#         i9,
#         i10,
#         i11,
#         i12,
#         i13,
#         i14,
#         i15,
#         i16,
#         i17,
#         i18,
#         i19,
#         i20,
#         i21,
#         i22,
#         i23,
#         i24,
#         i25,
#         i26,
#         i27,
#         i28,
#         i29,
#         i30,
#         i31,
#         i32,
#         i33,
#         i34,
#         i35,
#         i36,
#         i37,
#         i38,
#         i39,
#         i40,
#         i41,
#         i42,
#         i43,
#         i44,
#         i45,
#         i46,
#         i47,
#         i48,
#         i49,
#         i50,
#         i51,
#         i52,
#         i53,
#         i54,
#         i55,
#         i56,
#         i57,
#         i58,
#         i59,
#         i60,
#         i61,
#         i62,
#         i63,
#         i64):
#         print("************************************")
#         print("input_ids: ", input_ids.shape, " dtype: ", input_ids.dtype)
#         print("position_ids: ", position_ids.shape, " dtype: ", position_ids.dtype)
#         print("attention_mask: ", attention_mask.shape, " dtype: ", attention_mask.dtype)
#         print("past_key_values: ", i1.shape, i2.shape, i63.shape, i64.shape)
#         print("past_key_values dtype: ", i1.dtype)
#         print("************************************")
#         config = {
#             'input_ids':input_ids,
#             'position_ids':position_ids,
#             'past_key_values':(
#             (i1, i2),
#             (
#                 i3,
#                 i4,
#             ),
#             (
#                 i5,
#                 i6,
#             ),
#             (
#                 i7,
#                 i8,
#             ),
#             (
#                 i9,
#                 i10,
#             ),
#             (
#                 i11,
#                 i12,
#             ),
#             (
#                 i13,
#                 i14,
#             ),
#             (
#                 i15,
#                 i16,
#             ),
#             (
#                 i17,
#                 i18,
#             ),
#             (
#                 i19,
#                 i20,
#             ),
#             (
#                 i21,
#                 i22,
#             ),
#             (
#                 i23,
#                 i24,
#             ),
#             (
#                 i25,
#                 i26,
#             ),
#             (
#                 i27,
#                 i28,
#             ),
#             (
#                 i29,
#                 i30,
#             ),
#             (
#                 i31,
#                 i32,
#             ),
#             (
#                 i33,
#                 i34,
#             ),
#             (
#                 i35,
#                 i36,
#             ),
#             (
#                 i37,
#                 i38,
#             ),
#             (
#                 i39,
#                 i40,
#             ),
#             (
#                 i41,
#                 i42,
#             ),
#             (
#                 i43,
#                 i44,
#             ),
#             (
#                 i45,
#                 i46,
#             ),
#             (
#                 i47,
#                 i48,
#             ),
#             (
#                 i49,
#                 i50,
#             ),
#             (
#                 i51,
#                 i52,
#             ),
#             (
#                 i53,
#                 i54,
#             ),
#             (
#                 i55,
#                 i56,
#             ),
#             (
#                 i57,
#                 i58,
#             ),
#             (
#                 i59,
#                 i60,
#             ),
#             (
#                 i61,
#                 i62,
#             ),
#             (
#                 i63,
#                 i64,
#             ),
#         ),
#             'use_cache':True,
#             'attention_mask':attention_mask,        
#         }
#         output = self.model(
#                 **config,
#                 return_dict=True,
#                 output_attentions=False,
#                 output_hidden_states=False,
#             )
#         return_vals = []        
#         return_vals.append(output.logits)
#         temp_past_key_values = output.past_key_values
#         for item in temp_past_key_values:
#             return_vals.append(item[0])
#             return_vals.append(item[1])
#         return tuple(return_vals)

# first_llama_model = FirstLlamaModel(llama)
# second_llama_model = SecondLlamaModel(llama)

########################## SHARK CODE #############################
# import torch
# from torch.fx.experimental.proxy_tensor import make_fx
# from torch._decomp import get_decompositions
# from typing import List
# from io import BytesIO
# from apps.stable_diffusion.src.utils import (
#     _compile_module,
#     args,
# )
# from shark.shark_inference import SharkInference
# import os
# args.load_vmfb = True
# import torch_mlir
# from torch_mlir import TensorPlaceholder


# def compile_llama(
#     llama_model,
#     input_ids=None,
#     inputs_embeds=None,
#     attention_mask=None,
#     position_ids=None,
#     past_key_value=None
# ):
#     attention_mask_placeholder = TensorPlaceholder.like(
#         attention_mask, dynamic_axes=[1]
#     )
#     is_first_llama = False
#     if inputs_embeds is not None:
#         is_first_llama = True
#         extended_model_name = "chatgpt_first_llama_fp32_cpu_completely_dynamic"
#         vmfb_path = "chatgpt_first_llama_fp32_cpu_completely_dynamic.vmfb"
#         inputs_embeds_placeholder = TensorPlaceholder.like(
#             inputs_embeds, dynamic_axes=[1]
#         )
#         position_ids_placeholder = TensorPlaceholder.like(
#             position_ids, dynamic_axes=[1]
#         )
#         fx_g = make_fx(
#             llama_model,
#             decomposition_table=get_decompositions(
#                 [
#                     torch.ops.aten.embedding_dense_backward,
#                     torch.ops.aten.native_layer_norm_backward,
#                     torch.ops.aten.slice_backward,
#                     torch.ops.aten.select_backward,
#                     torch.ops.aten.norm.ScalarOpt_dim,
#                     torch.ops.aten.native_group_norm,
#                     torch.ops.aten.upsample_bilinear2d.vec,
#                     torch.ops.aten.split.Tensor,
#                     torch.ops.aten.split_with_sizes,
#                 ]
#             ),
#         )(inputs_embeds, position_ids, attention_mask)
#         example_inputs = [inputs_embeds, position_ids, attention_mask]
#         # example_inputs = [inputs_embeds_placeholder, position_ids_placeholder, attention_mask_placeholder]
#     else:
#         extended_model_name = "chatgpt_second_llama_fp32_cpu_dynamic_after_broadcast_fix"
#         vmfb_path = "chatgpt_second_llama_fp32_cpu_dynamic_after_broadcast_fix.vmfb"
#         past_key_value_placeholder = []
#         for i in past_key_value:
#             past_key_value_placeholder.append(
#                 TensorPlaceholder.like(
#                     i, dynamic_axes=[2]
#                 )
#             )
#         fx_g = make_fx(
#             llama_model,
#             decomposition_table=get_decompositions(
#                 [
#                     torch.ops.aten.embedding_dense_backward,
#                     torch.ops.aten.native_layer_norm_backward,
#                     torch.ops.aten.slice_backward,
#                     torch.ops.aten.select_backward,
#                     torch.ops.aten.norm.ScalarOpt_dim,
#                     torch.ops.aten.native_group_norm,
#                     torch.ops.aten.upsample_bilinear2d.vec,
#                     torch.ops.aten.split.Tensor,
#                     torch.ops.aten.split_with_sizes,
#                 ]
#             ),
#         )(
#             input_ids, position_ids, attention_mask, *past_key_value
#         )
#         example_inputs = [input_ids, position_ids, attention_mask, *past_key_value]
#         # example_inputs = [input_ids, position_ids, attention_mask_placeholder, *past_key_value_placeholder]

#     def _remove_nones(fx_g: torch.fx.GraphModule) -> List[int]:
#         removed_indexes = []
#         for node in fx_g.graph.nodes:
#             if node.op == "output":
#                 assert (
#                     len(node.args) == 1
#                 ), "Output node must have a single argument"
#                 node_arg = node.args[0]
#                 if isinstance(node_arg, (list, tuple)):
#                     node_arg = list(node_arg)
#                     node_args_len = len(node_arg)
#                     for i in range(node_args_len):
#                         curr_index = node_args_len - (i + 1)
#                         if node_arg[curr_index] is None:
#                             removed_indexes.append(curr_index)
#                             node_arg.pop(curr_index)
#                     node.args = (tuple(node_arg),)
#                     break

#         if len(removed_indexes) > 0:
#             fx_g.graph.lint()
#             fx_g.graph.eliminate_dead_code()
#             fx_g.recompile()
#         removed_indexes.sort()
#         return removed_indexes

#     def _unwrap_single_tuple_return(fx_g: torch.fx.GraphModule) -> bool:
#         """
#         Replace tuple with tuple element in functions that return one-element tuples.
#         Returns true if an unwrapping took place, and false otherwise.
#         """
#         unwrapped_tuple = False
#         for node in fx_g.graph.nodes:
#             if node.op == "output":
#                 assert (
#                     len(node.args) == 1
#                 ), "Output node must have a single argument"
#                 node_arg = node.args[0]
#                 if isinstance(node_arg, tuple):
#                     if len(node_arg) == 1:
#                         node.args = (node_arg[0],)
#                         unwrapped_tuple = True
#                         break

#         if unwrapped_tuple:
#             fx_g.graph.lint()
#             fx_g.recompile()
#         return unwrapped_tuple

#     def transform_fx(fx_g):
#         for node in fx_g.graph.nodes:
#             if node.op == "call_function":
#                 if node.target in [
#                     torch.ops.aten.empty,
#                 ]:
#                     # aten.empty should be filled with zeros.
#                     if node.target in [torch.ops.aten.empty]:
#                         with fx_g.graph.inserting_after(node):
#                             new_node = fx_g.graph.call_function(
#                                 torch.ops.aten.zero_,
#                                 args=(node,),
#                             )
#                             node.append(new_node)
#                             node.replace_all_uses_with(new_node)
#                             new_node.args = (node,)

#         fx_g.graph.lint()

#     transform_fx(fx_g)
#     fx_g.recompile()
#     removed_none_indexes = _remove_nones(fx_g)
#     was_unwrapped = _unwrap_single_tuple_return(fx_g)

#     fx_g.graph.set_codegen(torch.fx.graph.CodeGen())
#     fx_g.recompile()

#     print("FX_G recompile")

#     def strip_overloads(gm):
#         """
#         Modifies the target of graph nodes in :attr:`gm` to strip overloads.
#         Args:
#             gm(fx.GraphModule): The input Fx graph module to be modified
#         """
#         for node in gm.graph.nodes:
#             if isinstance(node.target, torch._ops.OpOverload):
#                 node.target = node.target.overloadpacket
#         gm.recompile()

#     strip_overloads(fx_g)
#     # print(fx_g.graph)
#     ts_g = torch.jit.script(fx_g)
    
#     # need_to_compile = True
#     args.load_vmfb = True
#     if args.load_vmfb:
#         if os.path.isfile(vmfb_path):
#             shark_module = SharkInference(
#                 None,
#                 device="cuda",
#                 mlir_dialect="tm_tensor",
#             )
#             print(f"loading existing vmfb from: {vmfb_path}")
#             shark_module.load_module(vmfb_path, extra_args=[])
#             return shark_module

#     # if need_to_compile:
#     print("Compiling via torch-mlir")
#     mlir_module = torch_mlir.compile(
#         ts_g, example_inputs, output_type="linalg-on-tensors"
#     )
#     print("Compilation success. Will write to file now.")
#     from contextlib import redirect_stdout
#     #import sys
#     #with open('second_llama_torch_fp32_after_broadcast_fix_elided.mlir', 'w') as sys.stdout:
#     #    #with redirect_stdout(f):
#     #    print(mlir_module.operation.get_asm(large_elements_limit=4))
#     #return mlir_module

#     with open('first_llama_minigpt_linalg_ir_elided_with_padding_170.mlir', 'w') as f:
#        with redirect_stdout(f):
#            print(mlir_module.operation.get_asm(large_elements_limit=4))

#     print("Elided IR written into file successfully.")
#     # if is_first_llama:
#     #     with open('first_llama_minigpt_linalg_ir_dynamic.mlir', 'w') as f:
#     #         with redirect_stdout(f):
#     #             print(mlir_module.operation.get_asm())
#     # else:
#     #     with open('second_llama_minigpt_linalg_ir_dynamic_after_broadcast_fix.mlir', 'w') as f:
#     #         with redirect_stdout(f):
#     #             print(mlir_module.operation.get_asm())
#     print("Non-Elided IR written into file successfully.")
#     # import sys
#     # sys.exit()
#     bytecode_stream = BytesIO()
#     mlir_module.operation.write_bytecode(bytecode_stream)
#     bytecode = bytecode_stream.getvalue()

#     shark_module = SharkInference(
#         mlir_module=bytecode, device="cuda", mlir_dialect="tm_tensor"
#     )
#     shark_module = _compile_module(shark_module, extended_model_name, [])
#     return shark_module


first_llama_model_shark = 0
second_llama_model_shark = 0

########################### SHARK CODE #############################


# chat = Chat(model, first_llama_model, second_llama_model, vis_processor, device='cpu')
# chat = Chat(model, first_llama_model_shark, second_llama_model_shark, vis_processor, device='cpu')
# chat = Chat(model, first_llama_model, second_llama_model_shark, vis_processor, device='cpu')
# chat = Chat(model, first_llama_model, vis_processor, device='cpu')
# CPU.
# chat = Chat(model, vis_processor, device='cpu')
# CUDA.
chat = Chat(model, vis_processor, device='cuda')
# chat = Chat(model, lam_func, vis_processor, device='cpu')
print('Initialization Finished')


# ========================================
#             Gradio Setting
# ========================================

def gradio_reset(chat_state, img_list):
    if chat_state is not None:
        chat_state.messages = []
    if img_list is not None:
        img_list = []
    return None, gr.update(value=None, interactive=True), gr.update(placeholder='Please upload your image first', interactive=False),gr.update(value="Upload & Start Chat", interactive=True), chat_state, img_list

def upload_img(gr_img, text_input, chat_state):
    if gr_img is None:
        return None, None, gr.update(interactive=True), chat_state, None
    chat_state = CONV_VISION.copy()
    img_list = []
    llm_message = chat.upload_img(gr_img, chat_state, img_list)
    return gr.update(interactive=False), gr.update(interactive=True, placeholder='Type and press Enter'), gr.update(value="Start Chatting", interactive=False), chat_state, img_list

def gradio_ask(user_message, chatbot, chat_state):
    if len(user_message) == 0:
        return gr.update(interactive=True, placeholder='Input should not be empty!'), chatbot, chat_state
    chat.ask(user_message, chat_state)
    chatbot = chatbot + [[user_message, None]]
    return '', chatbot, chat_state


# def compile_first_llama(isPyTorchVariant = True):
#     if isPyTorchVariant:
#         return first_llama_model
#     inputs_embeds = torch.zeros((1,170, 4096), dtype=torch.float32)
#     position_ids = torch.zeros((1,170), dtype=torch.int64)
#     attention_mask = torch.zeros((1,170), dtype=torch.int32)
#     print("Going to compile First Llama")
#     first_llama_model_shark = compile_llama(first_llama_model, inputs_embeds=inputs_embeds, position_ids=position_ids, attention_mask=attention_mask)
#     print("Compilation complete for First llama. You may check .mlir and .vmfb files")
#     return first_llama_model_shark

# def compile_second_llama(isPyTorchVariant = True):
#     if isPyTorchVariant:
#         return second_llama_model
#     input_ids = torch.zeros((1,1), dtype=torch.int64)
#     position_ids = torch.zeros((1,1), dtype=torch.int64)
#     attention_mask = torch.zeros((1,171), dtype=torch.int32)
#     past_key_value = []
#     for i in range(64):
#         past_key_value.append(torch.zeros(1,32,170,128, dtype=torch.float32))

#     print("Going to compile Second Llama")
#     second_llama_model_shark = compile_llama(second_llama_model, input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask, past_key_value=past_key_value)
#     print("Compilation complete for Second llama. You may check .mlir and .vmfb files")
#     return second_llama_model_shark



def gradio_answer(chatbot, chat_state, img_list, num_beams, temperature):
    global first_llama_model_shark
    global second_llama_model_shark
    print("************")
    print(chat_state)
    print("************")
    print(img_list)
    print("************")
    print(num_beams)
    print("************")
    print(temperature)
    print("************")

    # Set this variable to switch between shark and pytorch variant.
    isPyTorchVariant = True
    # if first_llama_model_shark == 0:
    #     first_llama_model_shark = compile_first_llama(isPyTorchVariant)
    # if second_llama_model_shark == 0:
    #     second_llama_model_shark = compile_second_llama(True)
    llm_message = chat.answer(conv=chat_state,
                              img_list=img_list,
                              num_beams=num_beams,
                              temperature=temperature,
                              max_new_tokens=30,
                              max_length=200,
                              first_llama_model=first_llama_model_shark,
                              second_llama_model=second_llama_model_shark,
                              isPyTorchVariant=isPyTorchVariant)[0]
    print(llm_message)
    print("************")
    chatbot[-1][1] = llm_message
    return chatbot, chat_state, img_list


title = """<h1 align="center">Demo of MiniGPT-4</h1>"""
description = """<h3>This is the demo of MiniGPT-4. Upload your images and start chatting!</h3>"""
article = """<p><a href='https://minigpt-4.github.io'><img src='https://img.shields.io/badge/Project-Page-Green'></a></p><p><a href='https://github.com/Vision-CAIR/MiniGPT-4'><img src='https://img.shields.io/badge/Github-Code-blue'></a></p><p><a href='https://raw.githubusercontent.com/Vision-CAIR/MiniGPT-4/main/MiniGPT_4.pdf'><img src='https://img.shields.io/badge/Paper-PDF-red'></a></p>
"""

#TODO show examples below

with gr.Blocks() as minigpt4_web:
    gr.Markdown(title)
    gr.Markdown(description)

    with gr.Row():
        with gr.Column(scale=0.5):
            image = gr.Image(type="pil")
            upload_button = gr.Button(value="Upload & Start Chat", interactive=True, variant="primary")
            clear = gr.Button("Restart")
            
            num_beams = gr.Slider(
                minimum=1,
                maximum=10,
                value=1,
                step=1,
                interactive=True,
                label="beam search numbers)",
            )
            
            temperature = gr.Slider(
                minimum=0.1,
                maximum=2.0,
                value=1.0,
                step=0.1,
                interactive=True,
                label="Temperature",
            )

        with gr.Column():
            chat_state = gr.State()
            img_list = gr.State()
            chatbot = gr.Chatbot(label='MiniGPT-4')
            text_input = gr.Textbox(label='User', placeholder='Please upload your image first', interactive=False)
    
    upload_button.click(upload_img, [image, text_input, chat_state], [image, text_input, upload_button, chat_state, img_list])
    
    text_input.submit(gradio_ask, [text_input, chatbot, chat_state], [text_input, chatbot, chat_state]).then(
        gradio_answer, [chatbot, chat_state, img_list, num_beams, temperature], [chatbot, chat_state, img_list]
    )
    clear.click(gradio_reset, [chat_state, img_list], [chatbot, image, text_input, upload_button, chat_state, img_list], queue=False)
