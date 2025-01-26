import torch
from sharpnet_network_config_utils import sharpnet_network_config_utils
import utils
from torch_fx_utils import torch_fx_utils
from pytorch_utils import pytorch_utils
from typing import List


class pytorch_module_to_sharpnet:

    def value_of(node: torch.fx.node.Node, module, sharpnet_network_config, previous_nodes: List[torch.fx.node.Node]) -> str:
        if isinstance(module, torch.nn.modules.linear.Linear):
            return pytorch_module_to_sharpnet.Linear(node, module, sharpnet_network_config, previous_nodes)
        if isinstance(module, torch.nn.modules.activation.MultiheadAttention):
            return  pytorch_module_to_sharpnet.MultiheadAttention(node, module, previous_nodes)
        if isinstance(module, torch.nn.modules.rnn.RNNBase):
            return  pytorch_module_to_sharpnet.RNNBase(node, module, previous_nodes)
        if isinstance(module, torch.nn.modules.flatten.Flatten):
            return  pytorch_module_to_sharpnet.Flatten(node, module, previous_nodes)
        if isinstance(module, torch.nn.modules.conv.Conv1d):
            return  pytorch_module_to_sharpnet.Conv1d(node, module, sharpnet_network_config, previous_nodes)
        if isinstance(module, torch.nn.modules.conv.Conv2d):
            return  pytorch_module_to_sharpnet.Conv2d(node, module, sharpnet_network_config, previous_nodes)
        if isinstance(module, torch.nn.modules.pooling.AvgPool2d):
            return  pytorch_module_to_sharpnet.AvgPool2d(node, module, previous_nodes)
        if isinstance(module, torch.nn.modules.pooling.AdaptiveAvgPool2d):
            return  pytorch_module_to_sharpnet.AdaptiveAvgPool2d(node, module, previous_nodes)
        if isinstance(module, torch.nn.modules.pooling.MaxPool2d):
            return  pytorch_module_to_sharpnet.MaxPool2d(node, module, previous_nodes)
        if isinstance(module, torch.nn.modules.pooling.AdaptiveMaxPool2d):
            return  pytorch_module_to_sharpnet.AdaptiveMaxPool2d(node, module, previous_nodes)
        if isinstance(module, torch.nn.modules.dropout.Dropout):
            return  pytorch_module_to_sharpnet.Dropout(node, module, previous_nodes)
        if isinstance(module, torch.nn.modules.normalization.LayerNorm):
            return  pytorch_module_to_sharpnet.LayerNorm(node, module, previous_nodes)
        if isinstance(module, torch.nn.modules.normalization.RMSNorm):
            return  pytorch_module_to_sharpnet.RMSNorm(node, module, previous_nodes)
        if isinstance(module, torch.nn.modules.batchnorm.BatchNorm2d):
            return  pytorch_module_to_sharpnet.BatchNorm2d(node, module, previous_nodes)
        if pytorch_utils.is_activation_module(module):
            return pytorch_module_to_sharpnet.activation(node, module, previous_nodes)
        raise Exception(f"unknown call_module {node.target} in {node} , {node.op}")

    @staticmethod
    def RNNBase(node: torch.fx.node.Node, module: torch.nn.modules.rnn.RNNBase, previous_nodes: List[torch.fx.node.Node]) -> str:
        if module.bias != True:
            raise Exception(f' only bias == True is suppotted in RNN, not {module.bias}')
        if module.batch_first != True:
            raise Exception(f' only batch_first == True is suppotted in RNN, not {module.batch_first}')
        if isinstance(module, torch.nn.modules.rnn.RNN):
            cell_mode = "CUDNN_RNN_RELU" if module.nonlinearity == 'relu' else "CUDNN_RNN_TANH"
            bias_mode = "CUDNN_RNN_DOUBLE_BIAS"
        elif isinstance(module, torch.nn.modules.rnn.GRU):
            cell_mode = "CUDNN_GRU"
            bias_mode = "CUDNN_RNN_DOUBLE_BIAS"
        elif isinstance(module, torch.nn.modules.rnn.LSTM):
            cell_mode = "CUDNN_LSTM"
            bias_mode = "CUDNN_RNN_SINGLE_INP_BIAS"
        else:
             raise Exception(f'not supported module : {type(module)} / {module}')
        return_sequences = "return_sequences" in torch_fx_utils.extract_node_name(node) # TODO
        isEncoderThatWillBeFollowedByDecoder = False  # TODO
        previous_node_index = torch_fx_utils.get_previous_nodes_indexes(node, previous_nodes, mandatory_length=1)[0]
        return f"Type;Layer;RecurrentLayer;intVector;PreviousLayerIndexes;1;{previous_node_index};string;LayerName;{torch_fx_utils.extract_node_name(node)};bool;Trainable;True;int;HiddenSize;{module.hidden_size};string;CellMode;{cell_mode};string;BiasMode;{bias_mode};bool;_returnSequences;{return_sequences};bool;IsBidirectional;{module.bidirectional};int;NumLayers;{module.num_layers};double;DropoutRate;{module.dropout};bool;IsEncoderThatWillBeFollowedByDecoder;{isEncoderThatWillBeFollowedByDecoder};"

    @staticmethod
    def MultiheadAttention(node: torch.fx.node.Node, module: torch.nn.modules.activation.MultiheadAttention, previous_nodes: List[torch.fx.node.Node]) -> str:
        if 'key_padding_mask' in node.kwargs and node.kwargs['key_padding_mask'] is not None:
            raise Exception(f"In MultiheadAttention only key_padding_mask = None is supported , not {node.kwargs['key_padding_mask']}")
        if 'need_weights' in node.kwargs and node.kwargs['need_weights']:
            raise Exception(f"In MultiheadAttention only need_weights = False is supported , not {node.kwargs['need_weights']}")
        if 'average_attn_weights' in node.kwargs and node.kwargs['average_attn_weights']:
            raise Exception(f"In MultiheadAttention only average_attn_weights = False is supported , not {node.kwargs['need_weights']}")
        if module.dropout != 0:
            raise Exception(f' only dropout == 0 is suppotted in MultiHeadAttention, not {module.dropout}')
        if module.bias_k or module.bias_v:
            raise Exception(f' only bias_k == None and bias_v == None are suppotted in MultiHeadAttention, not {module.add_bias_k} / {module.add_bias_v}')
        if module.add_zero_attn:
            raise Exception(f' only add_zero_attn == False is suppotted in MultiHeadAttention, not {module.add_zero_attn}')
        if not module.batch_first:
            raise Exception(f' only batch_first == True is suppotted in MultiHeadAttention, not {module.batch_first}')
        use_bias_O = use_bias_Q_K_V = module.in_proj_bias is not None
        conv1d_q, conv1d_k, conv1d_v = torch_fx_utils.get_previous_nodes_indexes(node, previous_nodes, mandatory_length=3)
        is_causal = 'is_causal' in node.kwargs and node.kwargs['is_causal']
        use_scale = 'scale' not in node.kwargs or node.kwargs['scale'] is None
        value_dim = key_dim = module.embed_dim//module.num_heads
        if module.num_heads*key_dim != module.embed_dim:
            raise Exception(f'embed_dim {module.embed_dim} must be a multiple of num_heads {module.num_heads}')
        result = f"Type;Layer;MultiheadAttention;intVector;PreviousLayerIndexes;3;{conv1d_q};{conv1d_k};{conv1d_v};string;LayerName;{torch_fx_utils.extract_node_name(node)};bool;Trainable;True;int;_num_heads;{module.num_heads};int;_key_dim;{key_dim};int;_value_dim;{value_dim};bool;_use_bias_Q_K_V;{use_bias_Q_K_V};bool;_use_bias_O;{use_bias_O};bool;_is_causal;{is_causal};"
        return result  

    @staticmethod
    def Linear(node: torch.fx.node.Node, module: torch.nn.modules.linear.Linear, sharpnet_network_config, previous_nodes: List[torch.fx.node.Node]) -> str:
        previous_node_index = torch_fx_utils.get_previous_nodes_indexes(node, previous_nodes, mandatory_length=1)[0]
        has_bias = module.bias is not None
        lambdaL2Regularization = sharpnet_network_config_utils.get_lambdaL2Regularization(sharpnet_network_config)
        flattenInputTensorOnLastDimension = True # this is the only supported mode in PyTorch
        result = f"Type;Layer;LinearLayer;intVector;PreviousLayerIndexes;1;{previous_node_index};string;LayerName;{torch_fx_utils.extract_node_name(node)};bool;Trainable;True;int;out_features;{module.out_features};bool;bias;{has_bias};double;LambdaL2Regularization;{lambdaL2Regularization};bool;_flattenInputTensorOnLastDimension;{flattenInputTensorOnLastDimension};"
        return result
    
    @staticmethod
    def AvgPool2d(node: torch.fx.node.Node, module: torch.nn.modules.pooling.AvgPool2d, previous_nodes: List[torch.fx.node.Node]) -> str:
        if module.output_size != 1:
            raise Exception(f"Invalid output_size {module.output_size} in {module}")
        previous_node_index = torch_fx_utils.get_previous_nodes_indexes(node, previous_nodes, mandatory_length=1)[0]
        result = f"Type;Layer;PoolingLayer;intVector;PreviousLayerIndexes;1;{previous_node_index};string;LayerName;{torch_fx_utils.extract_node_name(node)};bool;Trainable;True;string;_poolingMode;CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;int;_poolingHeight;-1;int;_poolingWidth;-1;int;_verticalStride;-1;int;_horizontalStride;-1;"
        return result

    @staticmethod
    def AdaptiveAvgPool2d(node: torch.fx.node.Node, module: torch.nn.modules.pooling.AdaptiveAvgPool2d, previous_nodes: List[torch.fx.node.Node]) -> str:
        if module.output_size != 1:
            raise Exception(f"Invalid output_size {module.output_size} in {module}")
        previous_node_index = torch_fx_utils.get_previous_nodes_indexes(node, previous_nodes, mandatory_length=1)[0]
        result = f"Type;Layer;PoolingLayer;intVector;PreviousLayerIndexes;1;{previous_node_index};string;LayerName;{torch_fx_utils.extract_node_name(node)};bool;Trainable;True;string;_poolingMode;CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;int;_poolingHeight;-1;int;_poolingWidth;-1;int;_verticalStride;-1;int;_horizontalStride;-1;"
        return result

    @staticmethod
    def MaxPool2d(node: torch.fx.node.Node, module: torch.nn.modules.pooling.MaxPool2d, previous_nodes: List[torch.fx.node.Node]) -> str:
        previous_node_index = torch_fx_utils.get_previous_nodes_indexes(node, previous_nodes, mandatory_length=1)[0]
        result = f"Type;Layer;PoolingLayer;intVector;PreviousLayerIndexes;1;{previous_node_index};string;LayerName;{torch_fx_utils.extract_node_name(node)};bool;Trainable;True;string;_poolingMode;CUDNN_POOLING_MAX_DETERMINISTIC;int;_poolingHeight;-1;int;_poolingWidth;-1;int;_verticalStride;-1;int;_horizontalStride;-1;"
        return result

    @staticmethod
    def Flatten(node: torch.fx.node.Node, module: torch.nn.modules.flatten.Flatten, previous_nodes: List[torch.fx.node.Node]) -> str:
        previous_node_index = torch_fx_utils.get_previous_nodes_indexes(node, previous_nodes, mandatory_length=1)[0]
        result = f"Type;Layer;FlattenLayer;intVector;PreviousLayerIndexes;1;{previous_node_index};string;LayerName;{torch_fx_utils.extract_node_name(node)};bool;Trainable;True;int;_start_dim;{module.start_dim};int;_end_dim;{module.end_dim};"
        return result

    @staticmethod
    def AdaptiveMaxPool2d(node: torch.fx.node.Node, module: torch.nn.modules.pooling.AdaptiveMaxPool2d, previous_nodes: List[torch.fx.node.Node]) -> str:
        if module.output_size != 1:
            raise Exception(f"Invalid output_size {module.output_size} in {module}")
        previous_node_index = torch_fx_utils.get_previous_nodes_indexes(node, previous_nodes, mandatory_length=1)[0]
        result = f"Type;Layer;PoolingLayer;intVector;PreviousLayerIndexes;1;{previous_node_index};string;LayerName;{torch_fx_utils.extract_node_name(node)};bool;Trainable;True;string;_poolingMode;CUDNN_POOLING_MAX_DETERMINISTIC;int;_poolingHeight;-1;int;_poolingWidth;-1;int;_verticalStride;-1;int;_horizontalStride;-1;"
        return result

    @staticmethod
    def extract_padding(module_padding, kernel_height, kernel_width) -> str:
        if isinstance(module_padding, str):
            return module_padding.upper()
        if isinstance(module_padding, tuple) and len(module_padding) <= 2:
            (padding_height, padding_width) = (module_padding[0], module_padding[-1])
            if padding_height != padding_width:
                raise Exception(f'padding_height {padding_height} and padding_width {padding_width} must be the same')
            if padding_height!=0 and padding_height !=kernel_height//2:
                raise Exception(f'invalid padding_height {padding_height} should be either 0 or half of kernel_height {kernel_height} must be the same')
            if padding_width!=0 and padding_width !=kernel_width//2:
                raise Exception(f'invalid padding_width {padding_width} should be either 0 or half of kernel_width {kernel_width} must be the same')
            return "SAME" if padding_height>0 else "VALID"
        raise Exception(f"do not know how to manage padding {module_padding}")
    
    @staticmethod
    def Conv2d(node: torch.fx.node.Node, module: torch.nn.modules.conv.Conv2d, sharpnet_network_config, previous_nodes: List[torch.fx.node.Node]) -> str:
        lambdaL2Regularization = sharpnet_network_config_utils.get_lambdaL2Regularization(sharpnet_network_config)
        previous_node_index = torch_fx_utils.get_previous_nodes_indexes(node, previous_nodes, mandatory_length=1)[0]
        has_bias = module.bias is not None
        (kernel_height, kernel_width) = utils.extract_first_second_element(module.kernel_size)
        (stride_height, stride_width) = utils.extract_first_second_element(module.stride)
        if stride_height!=stride_width:
            raise Exception(f"Invalid stride {module.stride} in {module}")
        padding = pytorch_module_to_sharpnet.extract_padding(module.padding, kernel_height, kernel_width)
        if module.groups != 1:
            # depthwise convolution
            if (module.out_channels % module.in_channels) != 0:
                raise Exception(f'in Depthwise convolution, out_channels {module.out_channels} must be a multiple of in_channels {module.in_channels}')
            if module.in_channels != module.groups:
                raise Exception(f'in Depthwise convolution, in_channels {module.in_channels} must be equal to groups {module.groups}')
            depth_multiplier = module.out_channels // module.in_channels
            if depth_multiplier != 1:
                raise Exception(f'in Depthwise convolution, depth_multiplier must be 1, not {depth_multiplier}')
            depthwise_convolution = True
            out_channels = -1
        else:
            # standard convolution
            depth_multiplier = -1
            depthwise_convolution = False
            out_channels = module.out_channels
        result = f"Type;Layer;ConvolutionLayer;intVector;PreviousLayerIndexes;1;{previous_node_index};string;LayerName;{torch_fx_utils.extract_node_name(node)};bool;Trainable;True;"
        result += f"bool;_isDepthwiseConvolution;{depthwise_convolution};bool;_isConv1D;False;int;_out_channels;{out_channels};int;_depthMultiplier;{depth_multiplier};int;_kernelHeight;{kernel_height};int;_kernelWidth;{kernel_width};"
        result += f"int;_stride;{stride_height};string;_paddingType;{padding};double;_lambdaL2Regularization;{lambdaL2Regularization};bool;UseBias;{has_bias};"
        return result

    @staticmethod
    def Conv1d(node: torch.fx.node.Node, module: torch.nn.modules.conv.Conv1d, sharpnet_network_config, previous_nodes: List[torch.fx.node.Node]) -> str:
        lambdaL2Regularization = sharpnet_network_config_utils.get_lambdaL2Regularization(sharpnet_network_config)
        previous_node_index = torch_fx_utils.get_previous_nodes_indexes(node, previous_nodes, mandatory_length=1)[0]
        has_bias = module.bias is not None
        if utils.count_integers(module.kernel_size) != 1:
            raise Exception(f'only 1 kernel size allowed in Conv1d, found: {module.kernel_size}')
        (_, kernel_width) = utils.extract_first_second_element(module.kernel_size)
        (stride_height, stride_width) = utils.extract_first_second_element(module.stride)
        if stride_height!=stride_width:
            raise Exception(f"Invalid stride {module.stride} in {module}")
        padding = pytorch_module_to_sharpnet.extract_padding(module.padding, kernel_width, kernel_width)
        result = f"Type;Layer;ConvolutionLayer;intVector;PreviousLayerIndexes;1;{previous_node_index};string;LayerName;{torch_fx_utils.extract_node_name(node)};bool;Trainable;True;bool;_isDepthwiseConvolution;False;bool;_isConv1D;True;int;_out_channels;{module.out_channels};int;_depthMultiplier;-1;int;_kernelHeight;1;int;_kernelWidth;{kernel_width};int;_stride;{stride_height};string;_paddingType;{padding};double;_lambdaL2Regularization;{lambdaL2Regularization};bool;UseBias;{has_bias};"
        return result

    @staticmethod
    def LayerNorm(node: torch.fx.node.Node, module: torch.nn.modules.normalization.LayerNorm, previous_nodes: List[torch.fx.node.Node]) -> str:
        previous_node_index = torch_fx_utils.get_previous_nodes_indexes(node, previous_nodes, mandatory_length=1)[0]
        _last_D_dimension = utils.count_integers(module.normalized_shape)
        if _last_D_dimension <= 0:
            raise Exception(f"Invalid normalized_shape {module.normalized_shape} in {module} : expecting at least 1 element")
        result = f"Type;Layer;LayerNorm;intVector;PreviousLayerIndexes;1;{previous_node_index};string;LayerName;{torch_fx_utils.extract_node_name(node)};bool;Trainable;True;int;_last_D_dimension;{_last_D_dimension};double;_epsilon;{module.eps};"
        return result
    
    @staticmethod
    def RMSNorm(node: torch.fx.node.Node, module: torch.nn.modules.normalization.RMSNorm, previous_nodes: List[torch.fx.node.Node]) -> str:
        previous_node_index = torch_fx_utils.get_previous_nodes_indexes(node, previous_nodes, mandatory_length=1)[0]
        _last_D_dimension = utils.count_integers(module.normalized_shape)
        if _last_D_dimension <= 0:
            raise Exception(f"Invalid normalized_shape {module.normalized_shape} in {module} : expecting at least 1 element")
        result = f"Type;Layer;RMSNorm;intVector;PreviousLayerIndexes;1;{previous_node_index};string;LayerName;{torch_fx_utils.extract_node_name(node)};bool;Trainable;True;int;_last_D_dimension;1;double;_epsilon;{module.eps};"
        return result

    @staticmethod
    def BatchNorm2d(node: torch.fx.node.Node, module: torch.nn.modules.batchnorm.BatchNorm2d, previous_nodes: List[torch.fx.node.Node]) -> str:
        previous_node_index = torch_fx_utils.get_previous_nodes_indexes(node, previous_nodes, mandatory_length=1)[0]
        result = f"Type;Layer;BatchNormalizationLayer;intVector;PreviousLayerIndexes;1;{previous_node_index};string;LayerName;{torch_fx_utils.extract_node_name(node)};bool;Trainable;True;double;_epsilon;{module.eps};double;_momentum;{1.0-module.momentum};"
        return result

    @staticmethod
    def Dropout(node: torch.fx.node.Node, module: torch.nn.modules.dropout.Dropout, previous_nodes: List[torch.fx.node.Node]) -> str:
        previous_node_index = torch_fx_utils.get_previous_nodes_indexes(node, previous_nodes, mandatory_length=1)[0]
        result = f"Type;Layer;DropoutLayer;intVector;PreviousLayerIndexes;1;{previous_node_index};string;LayerName;{torch_fx_utils.extract_node_name(node)};bool;Trainable;True;double;p;{module.p};"
        return result

    @staticmethod
    def to_sharpnet_activation_name(module: torch.nn.Module) -> str: 
        if isinstance(module, torch.nn.ReLU): return "CUDNN_ACTIVATION_RELU"
        if isinstance(module, torch.nn.Sigmoid): return "CUDNN_ACTIVATION_SIGMOID"
        if isinstance(module, torch.nn.Softmax): return "CUDNN_ACTIVATION_SOFTMAX"
        if isinstance(module, torch.nn.LeakyReLU): return "CUDNN_ACTIVATION_LEAKY_RELU"
        if isinstance(module, torch.nn.ELU): return "CUDNN_ACTIVATION_ELU"
        if isinstance(module, torch.nn.SiLU): return "CUDNN_ACTIVATION_SWISH"
        if isinstance(module, torch.nn.Tanh): return "CUDNN_ACTIVATION_TANH"
        raise Exception(f"Invalid activation {module}, {type(module)}")
        #if isinstance(module, torch.nn.Identity): return ""
        return None
    
    @staticmethod
    def activation(node: torch.fx.node.Node, module: torch.nn.Module, previous_nodes: List[torch.fx.node.Node]) -> str:
        previous_node_index = torch_fx_utils.get_previous_nodes_indexes(node, previous_nodes, mandatory_length=1)[0]
        activation_name = pytorch_module_to_sharpnet.to_sharpnet_activation_name(module)
        result = f"Type;Layer;ActivationLayer;intVector;PreviousLayerIndexes;1;{previous_node_index};string;LayerName;{torch_fx_utils.extract_node_name(node)};bool;Trainable;True;string;ActivationFunction;{activation_name};"
        if isinstance(module, torch.nn.LeakyReLU): 
            result += f"CpuTensor;_activationParameter;Single;1;1;{module.negative_slope};"
        return result



