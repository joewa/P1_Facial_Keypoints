import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


# Template class for the classifier network, inspired from
# https://github.com/udacity/DSND_Term1/blob/1196aafd48a2278b02eff85510b582fd7e2a9d2d/lessons/DeepLearning/new-intro-to-pytorch/fc_model.py
# Note: The usage of dropout didn't improve the performance with the given datasets - but required a few more training epochs to
# achieve the same performance.

class Network(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_p=0.25):
        ''' Builds a feedforward network with arbitrary hidden layers.
        
            Arguments
            ---------
            input_size: integer, size of the input layer
            output_size: integer, size of the output layer
            hidden_layers: list of integers, the sizes of the hidden layers
        
        '''
        super().__init__()
        # Input to a hidden layer
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        
        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        
        self.output = nn.Linear(hidden_layers[-1], output_size)
        
        # self.dropout = nn.Dropout(p=drop_p)
        
    def forward(self, x):
        ''' Forward pass through the network, returns the output logits '''
        
        for each in self.hidden_layers:
            x = F.relu(each(x))
            # x = self.dropout(x)
        x = self.output(x)
        return x  # F.log_softmax(x, dim=1)
