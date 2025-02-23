import torch 
from typing import List
import os
from pytorch_utils import pytorch_utils


class sharpnet_network_config_utils:

    @staticmethod
    def save_sharpnet_conf(model, model_name:str, directory:str, optimizer, loss_criterion, input_shape: List[int]) -> None:
        sharpnet_network_config = sharpnet_network_config_utils.to_sharpnet_network_config(model, optimizer, loss_criterion, input_shape)
        filepath = os.path.join(directory, model_name+'.NetworkSample.conf')
        # we sort the keys
        sorted_sharpnet_network_config = {key: sharpnet_network_config[key] for key in sorted(sharpnet_network_config )}
        with open(filepath, "w") as file:
            for key, value in sorted_sharpnet_network_config.items():
                file.write(f"{key} = {value}\n")    
        return filepath

    @staticmethod
    def update_sharpnet_newtork_config_with_loss_criterion(network_config, loss_criterion) -> None:
        key = 'LossFunction'
        if isinstance(loss_criterion, torch.nn.CrossEntropyLoss): 
            network_config[key] = "CategoricalCrossentropy"
            return
        if isinstance(loss_criterion, torch.nn.BCELoss): 
            network_config[key] = "BinaryCrossentropy"
            return
        if isinstance(loss_criterion, torch.nn.MSELoss): 
            network_config[key] = "Mse"
            return
        if isinstance(loss_criterion, torch.nn.L1Loss): 
            network_config[key] = "Mae"
            return
        if isinstance(loss_criterion, torch.nn.HuberLoss): 
            network_config[key] = "Huber"
            network_config['Huber_Delta'] = loss_criterion.delta
            return
        if "focalloss" in type(loss_criterion).__name__.lower():
        #if isinstance(loss_criterion, torch.nn.BinaryFocalLoss): 
            network_config[key] = "BCEWithFocalLoss"
            raise Exception('not implemented, need alpha, gamma and check mean')
        raise Exception(f"Invalid loss criterion {loss_criterion}, {type(loss_criterion)}")
    
        
    @staticmethod
    def to_sharpnet_network_config(model, optimizer, loss_criterion, input_shape:List[int]):
        result = dict()
    
        result['BatchSize'] = input_shape[0]
        result['CompatibilityMode'] = 'PyTorch'
        sharpnet_network_config_utils.update_sharpnet_newtork_config_with_loss_criterion(result, loss_criterion)
        
        if pytorch_utils.is_running_on_cuda(model):
            #result['ConvolutionAlgoPreference'] = 'FASTEST_DETERMINIST'
            result['ConvolutionAlgoPreference'] = 'FASTEST'
            result['ResourceIds'] = 0
        else:
            result['ConvolutionAlgoPreference'] = 'FASTEST_DETERMINIST_NO_TRANSFORM'
            result['ResourceIds'] = -1
        
        
        if isinstance(optimizer, torch.optim.SGD):
            result['InitialLearningRate'] = optimizer.param_groups[0]['lr']
            result['SGD_momentum'] = optimizer.param_groups[0]['momentum'] or 0
            result['weight_decay'] = optimizer.param_groups[0]['weight_decay'] or 0
            result['nesterov'] = optimizer.param_groups[0]['nesterov'] if optimizer.param_groups[0]['nesterov'] is not None else False
            if result['SGD_momentum'] == 0 and result['weight_decay'] == 0 and result['nesterov'] == False:
                result['OptimizerType'] = 'VanillaSGD'
            else:
                result['OptimizerType'] = 'SGD'
        elif isinstance(optimizer, torch.optim.AdamW):
            result['InitialLearningRate'] = optimizer.param_groups[0]['lr']
            result['Adam_beta1'] = optimizer.param_groups[0]['betas'][0]
            result['Adam_beta2'] = optimizer.param_groups[0]['betas'][1]
            result['Adam_eps'] = optimizer.param_groups[0]['eps']
            result['weight_decay'] = optimizer.param_groups[0]['weight_decay'] or 0
            result['OptimizerType'] = 'AdamW'
        elif isinstance(optimizer, torch.optim.Adam):
            result['InitialLearningRate'] = optimizer.param_groups[0]['lr']
            result['Adam_beta1'] = optimizer.param_groups[0]['betas'][0]
            result['Adam_beta2'] = optimizer.param_groups[0]['betas'][1]
            result['Adam_eps'] = optimizer.param_groups[0]['eps']
            result['weight_decay'] = optimizer.param_groups[0]['weight_decay'] or 0
        else:
            raise Exception(f"not supported optimizer {type(optimizer)} , {optimizer}")
        return result

    @staticmethod
    def get_with_default(sharpnet_network_config, key:str, default_value):
        if key in sharpnet_network_config:
            return sharpnet_network_config[key]
        return default_value

