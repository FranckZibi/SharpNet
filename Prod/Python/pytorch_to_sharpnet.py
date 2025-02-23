import torch
import numpy as np
import os
import torch.fx
from typing import List

from pytorch_utils import pytorch_utils
from sharpnet_network_config_utils import sharpnet_network_config_utils
from torch_fx_utils import torch_fx_utils 
from pytorch_module_to_sharpnet  import pytorch_module_to_sharpnet
from pytorch_call_function_to_sharpnet import pytorch_call_function_to_sharpnet

torch.manual_seed(0)
np.random.seed(0)


class pytorch_to_sharpnet:
    
    @staticmethod
    def save_sharpnet(model, model_name:str, directory:str, optimizer, loss_criterion, input_shape: List[int], concrete_args=None, verbose:bool=False) -> List[str]:
        pytorch_to_sharpnet.save_sharpnet_model_description(model, model_name, directory, optimizer, loss_criterion, input_shape, concrete_args=concrete_args, verbose=verbose)
        sharpnet_network_config_utils.save_sharpnet_conf(model, model_name, directory, optimizer, loss_criterion, input_shape)
        pytorch_to_sharpnet.save_sharpnet_model_parameters(model, model_name, directory, True)
        torch.save(model, os.path.join(directory, model_name+'.pth'))

    @staticmethod  
    def to_sharpnet_InputLayer_string(node: torch.fx.node.Node, input_shape: List[int], previous_nodes: List[torch.fx.node.Node]) -> str:
        # Type;Layer;InputLayer;intVector;PreviousLayerIndexes;0;string;LayerName;input;bool;Trainable;True;int;_c;3;int;_h;4;int;_w;5;
        if len(input_shape) <2 or  len(input_shape) > 4:
            raise Exception(f"Invalid input shape {input_shape}")
        c = input_shape[1] if len(input_shape)>=2 else -1
        h = input_shape[2] if len(input_shape)>=3 else -1
        w = input_shape[3] if len(input_shape)>=4 else -1
        result = f"Type;Layer;InputLayer;intVector;PreviousLayerIndexes;0;string;LayerName;{torch_fx_utils.extract_node_name(node)};bool;Trainable;True;int;_c;{c};int;_h;{h};int;_w;{w};"
        return result


    def count_placeholder_nodes(nodes: List[torch.fx.node.Node]) -> int:
        result = 0
        for n in nodes:
            if n.op == 'placeholder':
                result += 1
        return result

    @staticmethod
    def to_sharpnet_string(node: torch.fx.node.Node, loss_criterion, input_shape: List[int], name2module, previous_nodes: List[torch.fx.node.Node]) -> str:

        if node.op == 'placeholder':
            if len(node.args) != 0:
                raise Exception(f"placeholder node {node} must have no args")
            return pytorch_to_sharpnet.to_sharpnet_InputLayer_string(node, input_shape, previous_nodes)


        if node.op == 'output':
            if  isinstance(loss_criterion, torch.nn.CrossEntropyLoss):
                # we need to add a softmax layer at the end
                return pytorch_module_to_sharpnet.activation(node, torch.nn.Softmax(), previous_nodes)
            else:
                return None # we can ignore the output layer

        if node.op == "call_function":
            return pytorch_call_function_to_sharpnet.value_of(node, previous_nodes)

        if node.op == 'call_module':
            if not isinstance(node.target, str):
                raise Exception(f'expecting str type for {node.target} , got {type(node.target)} for {node} , {node.op}')
            if node.target not in name2module:
                raise Exception(f"invalid node {node} / {node.target}: not among {name2module.keys()}")
            return pytorch_module_to_sharpnet.value_of(node, name2module[node.target], previous_nodes)

        '''
        if node.op == 'call_method':
            return f"unknown {node.target} in {node} , {node.op}" #//!D TO CHECK
        if node.op == 'get_attr':
            return f"unknown {node.target} in {node} , {node.op}" #//!D TO CHECK
        '''
        #//!D TO CHECK
        error_msg = f"* * * * unknown {node.target} in {node} , {node.op}"
        print(error_msg)
        return error_msg 
        #raise Exception(f"unknown {node.target} in {node} , {node.op}")
      


    @staticmethod
    def save_sharpnet_model_description(model, model_name:str, directory:str, optimizer, loss_criterion, input_shape: List[int], concrete_args = None, verbose:bool = False) -> List[str]:
    
        previous_nodes = []
        previous_nodes_sharpnet = []
    
        name2module = dict()    
        for module_name, module in model.named_modules():
            if module_name in name2module:
                raise Exception(f"duplicate module_name {module_name}")
            name2module[module_name] = module
    
        # Trace the model with torch.fx
        model.eval()


        if type(model).__module__.startswith('transformers.models.'):
            from transformers.utils.fx import symbolic_trace as transformers_symbolic_trace
            tracer = transformers_symbolic_trace(model, input_names=["input_ids", "attention_mask", "token_type_ids"])
        else:
            if concrete_args:
                tracer = torch.fx.symbolic_trace(model, concrete_args=concrete_args)
            else:
                tracer = torch.fx.symbolic_trace(model)
        nodes  = list(tracer.graph.nodes)
    
        # we remove invalid nodes and update args
        for i in range(len(nodes)-1,-1,-1):
            if not torch_fx_utils.is_valid_node(nodes, i):
                torch_fx_utils.remove_node_and_update_all_nodes_args(nodes, i)
    
        for i in range(0, len(nodes)):
            node = nodes[i]
            if verbose:
                torch_fx_utils.display_node(node, name2module)
            current_string = pytorch_to_sharpnet.to_sharpnet_string(node, loss_criterion, input_shape, name2module, previous_nodes)
            if current_string:
                previous_nodes.append(node)
                previous_nodes_sharpnet.append(current_string)

        previous_nodes_sharpnet.insert(0, f"string;ModelName;{model_name};EpochDataVector;EpochData;0;")
        #print("\n".join(previous_nodes_sharpnet))
        filepath = os.path.join(directory, model_name+'.txt')
        with open(filepath, "w") as f:
            f.write("\n".join(previous_nodes_sharpnet))
        return filepath
    
    @staticmethod
    def save_sharpnet_model_parameters(model, model_name:str, directory:str, save_buffers: bool) -> str:
        import h5py
        filepath = os.path.join(directory, model_name+'.h5')
        with h5py.File(filepath, 'w') as h5file:
            for parameter_name, param in model.named_parameters():
                h5file.create_dataset(pytorch_to_sharpnet.normalize_parameter_name(parameter_name), data=pytorch_utils.to_numpy(param))
            for buffer_name, param in model.named_buffers():
                h5file.create_dataset(pytorch_to_sharpnet.normalize_parameter_name(buffer_name), data=pytorch_utils.to_numpy(param))
        return filepath

    
    @staticmethod
    def normalize_parameter_name(parameter_name:str) -> str:
        if '.' not in parameter_name:
            return pytorch_utils.normalize_module_name(parameter_name)
        else:
            for s in ['.out_proj.bias', '.out_proj.weight']:
                if parameter_name.endswith(s):
                    return pytorch_utils.normalize_module_name(parameter_name[:-len(s)])+s
            left_part, right_part = parameter_name.rsplit('.', 1)
            return f"{pytorch_utils.normalize_module_name(left_part)}.{right_part}"
