# torch imports
import torch.nn.functional as F
import torch.nn as nn
import torch


## TODO: Complete this classifier
class BinaryClassifier(nn.Module):
    """
    Define a neural network that performs binary classification.
    The network should accept your number of features as input, and produce 
    a single sigmoid value, that can be rounded to a label: 0 or 1, as output.
    
    Notes on training:
    To train a binary classifier in PyTorch, use BCELoss.
    BCELoss is binary cross entropy loss, documentation: https://pytorch.org/docs/stable/nn.html#torch.nn.BCELoss
    """

    ## TODO: Define the init function, the input params are required (for loading code in train.py to work)
    def __init__(self, input_features, hidden_layers_number, hidden_dim, output_dim):
        """
        Initialize the model by setting up linear layers.
        Use the input parameters to help define the layers of your model.
        :param input_features: the number of input features in your training/test data
        :param hidden_layers_number: the number of hidden layers
        :param hidden_dim: helps define the number of nodes in the hidden layer(s)
        :param output_dim: the number of outputs you want to produce
        """
        super(BinaryClassifier, self).__init__()

        # define any initial layers, here

        # Define hidden layers attribute
        self.hidden = nn.ModuleList()

        self.input_layer = nn.Linear(input_features, hidden_dim)

        for hidden in range(0, hidden_layers_number):
          self.hidden.append(nn.Linear(hidden_dim, hidden_dim))

        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    
    ## TODO: Define the feedforward behavior of the network
    def forward(self, x):
        '''Feedforward behavior of the net.
           :param x: A batch of input features
           :return: A single, sigmoid activated value
         '''
        # Input layer
        x = torch.sigmoid(self.input_layer(x))
        
        # Hidden layers
        for hidden in range(0, len(self.hidden)):
          x = torch.sigmoid(self.hidden[hidden](x))
  
        # Output layer
        x = torch.sigmoid(self.output_layer(x))

        return x
    