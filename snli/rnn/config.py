from torch import nn

RNNConfig = {

    'training_params': {
        'lr': 0.001,
        'batch_size': 128
    },

    'model_params': {
        'mlp_num_layers': 1,
        'mlp_hidden_size': 200,
        'mlp_activation': nn.ReLU,
        'mlp_dp': 0.2
    }

}
