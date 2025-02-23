import torch 
from typing import List
from torch_fx_utils import torch_fx_utils

class pytorch_call_function_to_sharpnet:

    @staticmethod
    def value_of(node: torch.fx.node.Node, previous_nodes: List[torch.fx.node.Node]) -> str:
        if "built-in function mul>" in str(node.target):
            return pytorch_call_function_to_sharpnet.mul(node, previous_nodes)
        if "function stochastic_depth" in str(node.target):
            return pytorch_call_function_to_sharpnet.stochastic_depth(node, previous_nodes)
        if "built-in function add>" in str(node.target):
            return pytorch_call_function_to_sharpnet.add(node, previous_nodes)
        if "built-in function scaled_dot_product_attention>" in str(node.target):
            return pytorch_call_function_to_sharpnet.scaled_dot_product_attention(node, previous_nodes)
        if "built-in method cat" in str(node.target):
            return pytorch_call_function_to_sharpnet.cat(node, previous_nodes)
        if "built-in method flatten" in str(node.target):
            return pytorch_call_function_to_sharpnet.flatten(node, previous_nodes)
        error_msg = f"* * * * not implemented {str(node.target)} /  {node.target} in {node} , {node.op}" #//!D TODO
        print(error_msg)
        return error_msg 
        #raise Exception(f"unknown call_function {node.target} in {node} , {node.op}")

    @staticmethod
    def flatten(node: torch.fx.node.Node, previous_nodes: List[torch.fx.node.Node]) -> str:
        if len(node.args) == 0:
            raise Exception(f"error in node {node} : node must have at least 1 args but found {len(node.args)}")
        previous_node_index = torch_fx_utils.node_index(node.args[0], previous_nodes)
        if previous_node_index < 0:
            raise Exception(f"fail to find node {node.args[0].name} among previous nodes {previous_nodes}")
        start_dim = 0 if len(node.args) <2 else node.args[1]
        end_dim = -1 if len(node.args) <3 else node.args[2]
        result = f"Type;Layer;Flatten;intVector;PreviousLayerIndexes;1;{previous_node_index};string;LayerName;{torch_fx_utils.extract_node_name(node)};bool;Trainable;True;int;_start_dim;{start_dim};int;_end_dim;{end_dim};"
        return result

    @staticmethod
    def mul(node: torch.fx.node.Node, previous_nodes: List[torch.fx.node.Node]) -> str:
        previous_node_index1,previous_node_index2 = torch_fx_utils.get_previous_nodes_indexes(node, previous_nodes, mandatory_length=2)
        result = f"Type;Layer;Multiply;intVector;PreviousLayerIndexes;2;{previous_node_index1};{previous_node_index2};string;LayerName;{torch_fx_utils.extract_node_name(node)};bool;Trainable;True;"
        return result  

    @staticmethod
    def add(node: torch.fx.node.Node, previous_nodes: List[torch.fx.node.Node]) -> str:
        previous_node_index1,previous_node_index2 = torch_fx_utils.get_previous_nodes_indexes(node, previous_nodes, mandatory_length=2)
        result = f"Type;Layer;AddLayer;intVector;PreviousLayerIndexes;2;{previous_node_index1};{previous_node_index2};string;LayerName;{torch_fx_utils.extract_node_name(node)};bool;Trainable;True;"
        return result  

    @staticmethod
    def stochastic_depth(node: torch.fx.node.Node, previous_nodes: List[torch.fx.node.Node]) -> str:
        if len(node.args) != 4:
            raise Exception(f"error in node {node} : node must have  4 args but found {len(node.args)}")
        previous_node_index = torch_fx_utils.node_index(node.args[0], previous_nodes)
        if previous_node_index < 0:
            raise Exception(f"fail to find node {node.args[0]} among previous nodes {previous_nodes}")
        p = node.args[1]
        mode = 'row' if node.args[3] else 'batch'
        result = f"Type;Layer;StochasticDepthLayer;intVector;PreviousLayerIndexes;1;{previous_node_index};string;LayerName;{torch_fx_utils.extract_node_name(node)};bool;Trainable;True;double;p;{p};string;mode;{mode};"
        return result  

    @staticmethod
    def scaled_dot_product_attention(node: torch.fx.node.Node, previous_nodes: List[torch.fx.node.Node]) -> str:
        if 'dropout_p' in node.kwargs and node.kwargs['dropout_p'] != 0.0:
            raise Exception(f"in scaled_dot_product_attention only dropout_p = 0 is supported , not {node.kwargs['dropout_p']}")
        if 'attn_mask' in node.kwargs and node.kwargs['attn_mask'] is not None:
            raise Exception(f"in scaled_dot_product_attention only attn_mask = None is supported , not {node.kwargs['attn_mask']}")
        conv1d_q, conv1d_k, conv1d_v = torch_fx_utils.get_previous_nodes_indexes(node, previous_nodes, mandatory_length=3)
        is_causal = 'is_causal' in node.kwargs and node.kwargs['is_causal']
        use_scale = 'scale' not in node.kwargs or node.kwargs['scale'] is None
        result = f"Type;Layer;ScaledDotProductAttentionLayer;intVector;PreviousLayerIndexes;3;{conv1d_q};{conv1d_k};{conv1d_v};string;LayerName;{torch_fx_utils.extract_node_name(node)};bool;Trainable;True;bool;_use_scale;{use_scale};bool;_is_causal;{is_causal};"
        return result  

    @staticmethod
    def cat(node: torch.fx.node.Node, previous_nodes: List[torch.fx.node.Node]) -> str:
        previous_node_index1,previous_node_index2 = torch_fx_utils.get_previous_nodes_indexes(node, previous_nodes, mandatory_length=2)
        result = f"Type;Layer;ConcatenateLayer;intVector;PreviousLayerIndexes;2;{previous_node_index1};{previous_node_index2};string;LayerName;{torch_fx_utils.extract_node_name(node)};bool;Trainable;True;"
        return result  

