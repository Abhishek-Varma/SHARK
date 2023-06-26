from omegaconf import OmegaConf
from apps.language_models.src.pipelines.minigpt4_utils.blip_processors import Blip2ImageEvalProcessor
from apps.language_models.src.pipelines.minigpt4_pipeline import MiniGPT4BaseModel

print('Initializing Chat')
config = OmegaConf.load("apps/language_models/src/pipelines/minigpt4_utils/configs/minigpt4_eval.yaml")
model_config = OmegaConf.create()
model_config = OmegaConf.merge(
    model_config,
    OmegaConf.load('apps/language_models/src/pipelines/minigpt4_utils/configs/minigpt4.yaml'),
    {"model": config["model"]},
)
model_config = model_config['model']
model_config.device_8bit = 0
model = MiniGPT4BaseModel.from_config(model_config).to('cpu')

from apps.language_models.src.model_wrappers.minigpt4 import (
    QformerBertModel,
)

from apps.stable_diffusion.src.utils import (
    compile_through_fx,
    args,
)
import torch
import torch_mlir
import os

def compile_qformer_model(precision="fp16", device="cuda", count=1):
    qformerBertModel = QformerBertModel(model.Qformer.bert).to("cuda")
    extended_model_name = f"minigpt4_qformer_bert_model_{precision}_{device}_{count}"
    print(f"Going to compile {extended_model_name}")
    # Inputs for QFormer.
    # inputs = [torch.randint(3, (1, 32, 768), dtype=torch.float32),
    #           torch.randint(3, (1, 257, 1408), dtype=torch.float32),
    #           torch.randint(3, (1, 257), dtype=torch.int64)]
    inputs = [torch.randint(3, (1, 32, 768), dtype=torch.float32).to("cuda"),
              torch.randint(3, (1, 257, 1408), dtype=torch.float32).to("cuda"),
              torch.randint(3, (1, 257), dtype=torch.int64).to("cuda")]
    is_f16 = False
    f16_input_mask = []
    if precision == "fp16":
        is_f16 = True
        f16_input_mask = [True, True, False]
    shark_QformerBertModel, _ = compile_through_fx(
        qformerBertModel,
        inputs,
        extended_model_name=extended_model_name,
        is_f16=is_f16,
        f16_input_mask=f16_input_mask,
        debug=False,
        generate_vmfb=True,
        save_dir=os.getcwd(),
        extra_args=[],
        base_model_id=None,
        model_name=extended_model_name,
        precision=None,
        return_mlir=True,
        device=device,
    )
    print(f"Generated {extended_model_name}.vmfb")
    return shark_QformerBertModel

precision = "fp16"
device = "cuda"
count = 1

####################################
import torch

import torch_mlir
from shark.shark_importer import import_with_fx
import torchvision.models as models
import copy
import io
import numpy as np


# Custom shark backend.
def shark_backend(fx_g, inputs, device:str = "cuda"):
    ts_graph = torch.jit.script(fx_g)
    mlir_module = torch_mlir.compile(ts_graph, inputs, output_type="linalg-on-tensors")
    bytecode_stream = io.BytesIO()
    mlir_module.operation.write_bytecode(bytecode_stream)
    bytecode = bytecode_stream.getvalue()
    from shark.shark_inference import SharkInference
    shark_module = SharkInference(
            mlir_module=bytecode, device=device, mlir_dialect="tm_tensor",
    )
    shark_module.compile(extra_args=[])
    output = shark_module("forward", inputs)
    return output



# Counts the total no. of callable nodes in the graph.
def count_total_nodes(fx_g):
    count:int = 0

    for node in fx_g.graph.nodes:
        if node.op == "call_function":
            count += 1

    return count



# Breaks the graph at the required position.
def break_at_pos(fx_g, pos: int):
    count:int = 0
    output_node = None

    # First capture the output node since we have to add the new output node before the previous output_node. 
    for node in fx_g.graph.nodes:
        if node.op == "output":
            set_of_nodes_to_break_at = node
            # break

    # Break at the required position given by the search.
    for node in fx_g.graph.nodes:
        if node.op == "call_function":
            # TODO: Check here that the node is not of the form of empty tensor etc.
            if count == pos:
                with fx_g.graph.inserting_before(output_node):
                    fx_g.graph.output(node)
                    break
            count += 1

    fx_g.graph.lint()
    fx_g.recompile()
    return fx_g


def check_output(orig_out, comp_out):
    if type(orig_out) == tuple:
        for i,j in zip(orig_out, comp_out):
            get_val = np.allclose(i.cpu().detach().numpy(), j, rtol=1e-2, atol=1e-3)
            if (get_val == False):
                return get_val
    else:
        get_val =  np.allclose(orig_out.cpu().detach().numpy(), comp_out, rtol=1e-2, atol=1e-3)
    return get_val


def binary_search_faulty_graph(fx_g, inputs):
    orig_out = fx_g(*inputs)
    return orig_out


# resnet18 = models.resnet18(pretrained=True)
# resnet18.train(False)
# input = (torch.randn(1,3,224,224),)

# fx_graph = import_with_fx(resnet18, input)
# total_nodes = count_total_nodes(fx_graph)
# print(f" The total nodes in the graph is: {total_nodes}")


qformerBertModel = QformerBertModel(model.Qformer.bert).to("cuda")
inputs = [torch.randint(3, (1, 32, 768), dtype=torch.float32).to("cuda"),
          torch.randint(3, (1, 257, 1408), dtype=torch.float32).to("cuda"),
          torch.randint(3, (1, 257), dtype=torch.int64).to("cuda")]
fx_graph, inputs = import_with_fx(qformerBertModel, inputs, is_f16=True, f16_input_mask=[True, True, False])

set_of_nodes_to_break_at = []
def rec_backward_traversal(node: torch.fx.node.Node):
    global set_of_nodes_to_break_at
    if not hasattr(node, "name"):
        return
    if node.op == "call_function":
        # print(f"Exploring {node.name}")
        if node in set_of_nodes_to_break_at:
            return
        if "_param_constant" in node.name:
            return
        set_of_nodes_to_break_at.append(node)
        no_of_args = len(node.args)
        # print(f"For Node: {node.name} we have {no_of_args} number of args")
        for i in range(no_of_args):
            rec_backward_traversal(node.args[i])
    return

def capture_nodes_to_break_at(fx_g, return_operand_num):
    global set_of_nodes_to_break_at
    for node in fx_g.graph.nodes:
        if node.op == "output":
            # Perform backward traversal.
            # import pdb
            # pdb.set_trace()
            rec_backward_traversal(node.args[return_operand_num])

def break_at_node(fx_g, node_to_break_at):
    output_node = None

    # First capture the output node since we have to add the new output node before the previous output_node. 
    for node in fx_g.graph.nodes:
        if node.op == "output":
            print("Found output node")
            output_node = node
            break

    # Break at the required position given by the search.
    for node in fx_g.graph.nodes:
        if node.name == node_to_break_at.name:
            print(f"Found {node_to_break_at.name}")
            with fx_g.graph.inserting_before(output_node):
                fx_g.graph.output(node)
                break

    fx_g.graph.lint()
    fx_g.recompile()
    from contextlib import redirect_stdout
    with open('fx_g_insert_before_div.mlir', 'w') as f:
        with redirect_stdout(f):
            print(fx_g.graph)
    return fx_g

capture_nodes_to_break_at(fx_graph, 0)
print(set_of_nodes_to_break_at)

op_list = ["view", "slice", "clone", "permute", "transpose", "getitem", "expand"]
for node in set_of_nodes_to_break_at:
    found_strings = [word for word in op_list if word in node.name]
    if found_strings:
        continue
    # if "div_1" != node.name:
    #     continue
    print(f"Will break for Node {node.name}")
    fx_g = break_at_node(copy.deepcopy(fx_graph), node_to_break_at=node)
    ret = binary_search_faulty_graph(fx_g, inputs)
    # print(ret)
    if isinstance(ret, tuple):
        count = 1
        for r in ret:
            print(f"Return value {count} has :-")
            all_nans = torch.isnan(r).any().item()
            if all_nans:
                print("NAN", end=' ')
            else:
                print("NO NAN", end=' ')
            is_negative_inf = torch.isinf(r).any().item()
            if is_negative_inf:
                print("\t\t\t-INFINITY")
            else:
                print("\t\t\tNO INF")
            count = count + 1
        continue
    all_nans = torch.isnan(ret).all().item()
    if all_nans:
        print("NAN", end=' ')
    else:
        print("NO NAN", end=' ')
    is_negative_inf = torch.isinf(ret).any().item()
    if is_negative_inf:
        print("\t\t\t-INFINITY")
    else:
        print("\t\t\tNO INF")
import sys
sys.exit()
total_nodes = count_total_nodes(fx_graph)
print(f" The total nodes in the graph is: {total_nodes}")
# Break the graph at the position.
l = 0
u = total_nodes
while l<=u:
    mid = (l + u) // 2
    print(f"\nBreaking at {mid}")
    fx_g = break_at_pos(copy.deepcopy(fx_graph), mid)
    break
    ret = binary_search_faulty_graph(fx_g, inputs)
    print(f"At {mid} :-")
    print(ret)
    all_nans = torch.isnan(ret).all().item()

    if all_nans:
        print("The tensor contains all NaN values.")
        u = mid-1
    else:
        print("The tensor does not contain all NaN values.")
        l = mid+1

    # if ret == True:
    #     print("Outputs matched so far")
    #     l = mid+1
    # else:
    #     print("Some outputs didn't match - trying to find a shorter graph")
    #     u = mid-1
    qformerBertModel = QformerBertModel(model.Qformer.bert).to("cuda")
    inputs = [torch.randint(3, (1, 32, 768), dtype=torch.float32).to("cuda"),
              torch.randint(3, (1, 257, 1408), dtype=torch.float32).to("cuda"),
              torch.randint(3, (1, 257), dtype=torch.int64).to("cuda")]
    fx_graph = import_with_fx(qformerBertModel, inputs)


# shark_module = compile_qformer_model(precision, device, count)
