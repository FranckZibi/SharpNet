import torch
import torch.fx
from typing import List
from pytorch_utils import pytorch_utils

class torch_fx_utils:

    @staticmethod
    def replace_args_in_node(node: torch.fx.node.Node, old_node_arg: torch.fx.node.Node, new_node_arg: torch.fx.node.Node):
        node_args = list(node.args)
        for i in range(len(node_args)-1,-1,-1):
            if node_args[i] == old_node_arg:
                if new_node_arg is None:
                    node_args.pop(i)
                else:
                    node_args[i] = new_node_arg
        node.args = tuple(node_args)

    @staticmethod
    def extract_node_name(node: torch.fx.node.Node):
        return pytorch_utils.normalize_module_name(node.name)

    @staticmethod
    def remove_node_and_update_all_nodes_args(all_nodes: List[torch.fx.node.Node], idx_node_to_remove: int):
        new_nodes = []
        node_to_remove = all_nodes[idx_node_to_remove]
        for c in node_to_remove.args:
            if isinstance(c, torch.fx.node.Node):
                new_nodes.append(c)            
        if len(new_nodes) != 1:
            raise Exception(f'the node to remove must contain exactly 1 args, not {len(new_nodes)}: {new_nodes}')
        for i in range(0, len(all_nodes)):
            if i != idx_node_to_remove:
                torch_fx_utils.replace_args_in_node(all_nodes[i], node_to_remove, new_nodes[0])
        all_nodes.pop(idx_node_to_remove)

    @staticmethod            
    def is_valid_node(nodes: List[torch.fx.node.Node], idx: int):
        node = nodes[idx]
        if node.op == "call_function" and "built-in function getitem>" in str(node.target):
            return False
        #//?Dif node.op == 'output' and idx == len(nodes)-1:             return False
        return True


    @staticmethod            
    def node_index(node: torch.fx.node.Node, previous_nodes: List[torch.fx.node.Node]) -> int:
        for i in range(0, len(previous_nodes)):
            if previous_nodes[i].name == node.name:
                return i
        return -1

    @staticmethod            
    def get_previous_nodes_indexes(node: torch.fx.node.Node, previous_nodes: List[torch.fx.node.Node], mandatory_length:int = None) -> List[int]:
        result = []
        node_args = node.args
        if len(node_args) == 1 and isinstance(node_args[0], tuple):
            node_args = node_args[0]
        if mandatory_length and len(node_args) != mandatory_length:
            raise Exception(f"error in node {node} : node must have {mandatory_length} args but found {len(node_args)}")
        for previous_node in node_args:
            previous_node_index = torch_fx_utils.node_index(previous_node, previous_nodes)
            if previous_node_index < 0:
                raise Exception(f"fail to find node {previous_node.name} among previous nodes {previous_nodes}")
            result.append(previous_node_index)
        return result

    @staticmethod
    def display_node(node: torch.fx.node.Node, name2module):
        print('node.name', node.name, type(node.name))
        print('node', node, type(node))
        print('node.op', node.op, type(node.op))
        print('node.args', node.args, type(node.args))
        if len(node.kwargs): print('node.kwargs', node.kwargs, type(node.kwargs))
        print('node.target', node.target, type(node.target))
        if isinstance(node.target,str) and node.target in name2module:
            print('module', name2module[node.target], type(name2module[node.target]))
        print('-'*50)

        