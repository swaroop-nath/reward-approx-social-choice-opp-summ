import torch
import torch.nn as nn

class FFRewardModel(nn.Module):
    def __init__(self, layer_config, num_input_features=7, num_output_features=1, activation='sigmoid'):
        super().__init__()
        '''
            layer_config = {
                'num-layers': *, # number of hidden layers, not including input and output neurons
                'layer-wise-size': [] # num-layers sized array
            }
        '''
        
        self._nn = self._construct_ffnn(layer_config, num_input_features, num_output_features, activation)
        self._num_inputs = num_input_features
        self._num_outputs = num_output_features
        
    def _construct_ffnn(self, layer_config, num_input_features, num_output_features, activation):
        if layer_config['num-layers'] == 0:
            ffn = nn.Linear(in_features=num_input_features, out_features=num_output_features)
            if activation == 'sigmoid': act = nn.Sigmoid()
            return nn.Sequential(ffn, act)
        
        nn_layers = []
        for hidden_layer_idx in range(layer_config['num-layers'] + 1):
            if hidden_layer_idx == 0: 
                hidden_layer_ff = nn.Linear(in_features=num_input_features, out_features=layer_config['layer-wise-size'][hidden_layer_idx])
                if activation == 'sigmoid': hidden_layer_act = nn.Sigmoid()
                elif activation == 'tanh': hidden_layer_act = nn.Tanh()
                elif activation == 'relu': hidden_layer_act = nn.ReLU()
                else: hidden_layer_act = None
                
            elif hidden_layer_idx == layer_config['num-layers']:
                hidden_layer_ff = nn.Linear(in_features=layer_config['layer-wise-size'][hidden_layer_idx - 1], out_features=num_output_features)
                if activation == 'sigmoid': hidden_layer_act = nn.Sigmoid()
                elif activation == 'tanh': hidden_layer_act = nn.Tanh()
                elif activation == 'relu': hidden_layer_act = nn.ReLU()
                else: hidden_layer_act = None
            
            else:
                hidden_layer_ff = nn.Linear(in_features=layer_config['layer-wise-size'][hidden_layer_idx - 1], out_features=layer_config['layer-wise-size'][hidden_layer_idx])
                hidden_layer_act = None
                
            if hidden_layer_act is not None: nn_layers.extend([hidden_layer_ff, hidden_layer_act])
            else: nn_layers.append(hidden_layer_ff)
            
        return nn.Sequential(*nn_layers)
    
    def _compute_loss(self, win, lose):
        return torch.mean(torch.log(1 + torch.exp(lose - win))) # NLL Loss
    
    def forward(self, **batch):
        # X_pref.size() == X_unpref.size() == (bsz, num_input_features)
        # y.size() == (bsz,)
        X_pref = batch['pref']
        X_unpref = batch['unpref']
        outputs_pref = self._nn(X_pref) # (bsz, num_output_features)
        outputs_unpref = self._nn(X_unpref)
        loss = self._compute_loss(outputs_pref, outputs_unpref)
        
        return {'loss': loss}
    
    def get_reward(self, X):
        return self._nn(X)