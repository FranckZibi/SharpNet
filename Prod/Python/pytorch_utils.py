import torch
import numpy as np
from pathlib import Path
import os
import io
from contextlib import redirect_stdout
from torch.utils.data import DataLoader, TensorDataset
import math
from typing import Tuple


class pytorch_utils:
    log_file_name =  str( Path(os.path.abspath(''))  /  "PyTorch.log" )

    @staticmethod
    def normalize_module_name(module_name: str):
        return module_name.lower().replace('.','_')

    @staticmethod
    def numpy_array_for_tests(shape) -> np.array:
        count = math.prod(shape)
        return np.array([  2*(i+1) /count*((i%2)-0.5) for i in range(count)] ).reshape(shape)
    
    @staticmethod
    def y_numpy_array_for_tests(rows, cols) -> np.array:
        data = [0.0]*rows*cols
        for row in range(rows):
            data[row*cols+ row%cols ] = 1.0
        return np.array(data).reshape([rows, cols])

    @staticmethod
    def to_numpy(t: torch.Tensor) -> np.ndarray:
        return t.detach().cpu().numpy()

    @staticmethod
    def count_model_weights(model):
        total_params = sum(p.numel() for p in model.parameters())
        return total_params

    @staticmethod
    def display_model_weights(model):
        for name, param in model.named_parameters():
            print(name, param.shape)
        
    @staticmethod
    def is_running_on_cuda(model) -> bool:
        return str(next(model.parameters()).device).lower().startswith('cuda')
    
    @staticmethod
    def is_activation_module(module: torch.nn.Module) -> bool:
        return module and module.__class__.__module__.startswith('torch.nn.modules.activation')

    @staticmethod
    def truncate(s, max_size):
        if len(s)<=max_size:
            return s
        return s[:(max_size//2)]+"..."+s[-(max_size//2):]

    @staticmethod
    def log(tensor, tensor_name = "", display_to_screen = True):
        with torch.no_grad():
            if isinstance(tensor, torch.nn.parameter.Parameter) or isinstance(tensor, torch.Tensor):
                return pytorch_utils.log(tensor.detach().to('cpu').numpy(), tensor_name, display_to_screen)
            if isinstance(tensor, list):
                for e in tensor:
                    pytorch_utils.log(e, tensor_name, display_to_screen)
                return
            asString = None
            if isinstance(tensor,str):
                asString = tensor
            elif isinstance(tensor,np.ndarray):
                asString = str(tensor_name) + " " + str(tensor.shape) + "\n" + pytorch_utils.truncate(str(tensor.tolist()), 10000)
            else:
                asString = str(tensor)
            with open(pytorch_utils.log_file_name, "a") as myfile:
                myfile.write(asString + "\n")
            if display_to_screen:
                print(asString)

    @staticmethod
    def print_intermediate_layer_model(layer, layer_name, layer_input: torch.Tensor, display_to_screen) -> torch.Tensor:
        with torch.no_grad():
            try:
                if layer_input is None:
                    return None
                if not isinstance(layer, torch.nn.Module):
                    raise Exception(f'not a layer {type(layer)}  {layer}')            
                layer.eval()
                layer_output = layer.forward(layer_input)
                layer.train()
                pytorch_utils.log(layer_output, layer_name, display_to_screen)
                return layer_output
            except Exception as e:  
                #print(e)
                return None

    @staticmethod
    def print_weights(model: torch.nn.Module) -> None:
        import torchvision
        m = torchvision.models.efficientnet_b0()
        a= dict(m.named_parameters())
        total_weight = 0
        for key,val in a.items():
            key_count = torch.numel(val.data) 
            pytorch_utils.log(key+":"+str(key_count))
            total_weight += key_count
        pytorch_utils.log('total_weight: '+str(total_weight))         
        
    @staticmethod
    def print_model(model, inputs, display_weights = True, display_layers_output = False, display_layers_grads = False, display_to_screen = False):
        if not isinstance(model, torch.nn.Module):
            raise Exception(f'not a model{type(model)}  {model}')
        with torch.no_grad():
            model.eval()
            layers = list(model.named_modules())[1:]

            layer_input = inputs
            for layerIndex in range(len(layers)):
                if layerIndex>3 and layerIndex <  (len(layers)-3 and False)  :
                    continue
                layer_name = layers[layerIndex][0]
                layer = layers[layerIndex][1]
                pytorch_utils.log("--------------------------------------------------------------------\n-","", display_to_screen)           
                pytorch_utils.log("layer#"+ str(layerIndex)+": "+layer_name, "", display_to_screen)
                #pytorch_utils.log(str(layer.get_config()), "get_config:", False)
                if display_layers_grads:
                    for p_name, p in layer.named_parameters():
                        if p.grad is not None:
                            pytorch_utils.log(p.grad, layer_name+"."+p_name+" grad:", display_to_screen)
                if display_weights:
                    for p_name, p in layer.named_parameters():
                        pytorch_utils.log(p, layer_name+"."+p_name, display_to_screen)
                    if hasattr(layer, 'running_mean'):
                        pytorch_utils.log(layer.running_mean, layer_name+".running_mean", display_to_screen)
                    if hasattr(layer, 'running_var'):
                        pytorch_utils.log(layer.running_var, layer_name+".running_var", display_to_screen)
                    
                pytorch_utils.log("-", "", display_to_screen)
                if display_layers_output:
                    pytorch_utils.log("layer#"+ str(layerIndex)+" output:", "", display_to_screen)
                    layer_input = pytorch_utils.print_intermediate_layer_model(layer, layer_name, layer_input, display_to_screen)
            pytorch_utils.log("--------------------------------------------------------------------", display_to_screen=display_to_screen)
            model.train()
        
    @staticmethod
    def Train(model: torch.nn.Module, 
              X_numpy: np.array,
              Y_numpy: np.array,
              device: str,
              loss_criterion,
              optimizer,
              num_epochs: int,
              batch_size: int|None = None,
              verbose: bool = True
             ) -> Tuple[float, float]:
        # by default, the batch size will proceed all elements at once
        model.train()
        if not batch_size or batch_size <= 0:
            batch_size = X_numpy.shape[0]
    
        if os.path.isfile(pytorch_utils.log_file_name):
            try:
                os.remove(pytorch_utils.log_file_name)
            except Exception:
                pass
    
        X = torch.tensor(X_numpy, device=device, dtype=torch.float32)
        Y_true = torch.tensor(Y_numpy, device=device, dtype=torch.float32)
        dataset = TensorDataset(X, Y_true) 
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        pytorch_utils.log( "--------------------------------------------------------------------\n-")
        f = io.StringIO()
        with redirect_stdout(f):
            try:
                pass #summary(model, input_size=(X.shape[1:]), device = device)
            except Exception as e:
                pytorch_utils.log(f'fail to create model summay: {e}')
        pytorch_utils.log(str(model))
            
        model_summary = f.getvalue()
        pytorch_utils.log(str(model_summary), '', True)
        with torch.no_grad():
            model.eval()
            Y_pred_before = model.forward(X)
            train_loss_before = loss_criterion(Y_pred_before, Y_true).to('cpu').item()
            model.train()
        

        pytorch_utils.log( "-\n--------------------------------------------------------------------\n-", "", False)
        pytorch_utils.log(X,"X", False)
        pytorch_utils.log(Y_true,"Y_true", False)
        pytorch_utils.log("--------------------------------------------------------------------\nmodel before trainig","", False)
        #pytorch_utils.log( "--------------------------------------------------------------------\nprint_layers(model) before\n-")
        if verbose:
            pytorch_utils.print_model(model, X, display_weights = True, display_layers_output = True, display_to_screen = False)

        print( "-")
        print( "--------------------------------------------------------------------")
        print( "-")
        print( "training for ",num_epochs," epochs...")

        pytorch_utils.log( "-\n--------------------------------------------------------------------\n-\nmodel after trainig")
        # pytorch_utils.log( "--------------------------------------------------------------------\nprint_layers(model) after\n-")


        model.train()
        for epoch in range(0,num_epochs):
            pytorch_utils.log(f'Epoch#{epoch}', '', False)
            for batch_id, (x_batch,y_batch_true) in enumerate(data_loader):
                if verbose:
                    pytorch_utils.log(f'Epoch#{epoch} / batch_id#{batch_id}', '', False)
                y_batch_pred = model(x_batch)
                l = loss_criterion(y_batch_pred, y_batch_true) # compute loss
                #with torch.no_grad():
                #    pytorch_utils.log(f'loss for epoch#{epoch} / batch_id#{batch_id} : {l.cpu().item()}', '', True)
                l.backward() # compute gradients
                optimizer.step() # update weights

                with torch.no_grad():
                    if verbose:
                        model.eval()
                        pytorch_utils.print_model(model, X, display_weights = True, display_layers_grads = True ,display_layers_output = False, display_to_screen = False)
                        model.train()

                optimizer.zero_grad()

        with torch.no_grad():
            model.eval()
            Y_pref_after = model.forward(X)
            model.train()
            train_loss_after = loss_criterion(Y_pref_after, Y_true).to('cpu').item()

        if verbose:
            pytorch_utils.print_model(model, X, display_weights = True, display_layers_output = True, display_to_screen = False)
        pytorch_utils.log("PyTorch num_epochs = " + str(num_epochs))
        # pytorch_utils.log("TF learningRate = "+str(keras.backend.eval(optim.lr)))
        if 'lr' in optimizer.param_groups[0]:    
            pytorch_utils.log("PyTorch lr = " + str(optimizer.param_groups[0]['lr']))
        if 'momentum' in optimizer.param_groups[0]:    
            pytorch_utils.log("PyTorch momentum = " + str(optimizer.param_groups[0]['momentum']))
        if 'nesterov' in optimizer.param_groups[0] and optimizer.param_groups[0]['nesterov']:
            pytorch_utils.log("PyTorch nesterov = " + str(optimizer.param_groups[0]['nesterov']))
        pytorch_utils.log("PyTorch batch_size = " + str(batch_size))
        if 'weight_decay' in optimizer.param_groups[0]:    
            pytorch_utils.log("PyTorch lambdaL2Regularization = " + str(optimizer.param_groups[0]['weight_decay']))
        #pytorch_utils.log("PyTorch use_bias = " + str(use_bias))
        pytorch_utils.log(Y_pred_before, "PyTorch Y_pred_before", True)
        pytorch_utils.log("PyTorch loss_before="+str(train_loss_before))
        pytorch_utils.log(Y_pref_after, "PyTorch Y_pref_after", True)
        pytorch_utils.log("PyTorch loss_after="+str(train_loss_after))
        return train_loss_before, train_loss_after
