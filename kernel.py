import torch
import gensim
import numpy as np
import pickle as cPickle
import torch.optim as optim
import time
import sys
from collections import defaultdict

import torch.nn as nn
import torch.nn.functional as F

class LSTMcell(nn.Module):
    
    def __init__(self,input_size, hidden_size, output_size, cell):
        
        super(LSTMcell, self).__init__()
        
        self.hidden_size = hidden_size
        self.cell = cell
        
        """
        LSTM cell basic operations
        """
        self.i2cdasht = nn.Linear(input_size + hidden_size, hidden_size, bias = True)
        if self.cell == "RKM-LSTM" or self.cell == "LSTM":
            self.i2ft = nn.Linear(input_size + hidden_size, hidden_size, bias = True)
            self.i2it = nn.Linear(input_size + hidden_size, hidden_size, bias = True)
            self.i2o = nn.Linear(input_size+hidden_size, hidden_size, bias=True)
        if self.cell == "RKM-CIFG":
            self.i2ft = nn.Linear(input_size + hidden_size, hidden_size, bias = True)
            self.i2o = nn.Linear(input_size+hidden_size, hidden_size, bias=True)
        if self.cell == "Linear-kernel-wto" or self.cell == "Gated-CNN":
            self.i2o = nn.Linear(input_size+hidden_size, hidden_size, bias=True)
        if self.cell == "Linear-kernel-wto" or self.cell == "Linear-kernel" or self.cell == "Gated-CNN" or self.cell == "CNN":
            self.sigmai = 0.5
        if self.cell == "Linear-kernel-wto" or self.cell == "Linear-kernel":
            self.sigmaf = 0.5


    def forward(self, input, hidden_state, cell_state):
        
        """
        input dimension = (batch size X 300); where 300 is dimension used for word embedding
        
        hidden state dimension = (batch size X 300); where 300 is hidden state dimension as mentioned in the paper
        
        """

        combined = torch.cat((input, hidden_state), axis = 1)
        
        if self.cell == "LSTM" or self.cell == "RKM-LSTM" or self.cell == "RKM-CIFG":
            forget_gate = torch.sigmoid(self.i2ft(combined))
        if self.cell == "LSTM" or self.cell == "RKM-LSTM":
            i_t = torch.sigmoid(self.i2it(combined))
        c_dash = self.i2cdasht(combined)
        
        if self.cell == "LSTM":
            cell_state = forget_gate*cell_state + i_t*torch.tanh(c_dash)
        if self.cell == "RKM-LSTM":
            cell_state = forget_gate*cell_state + i_t*(c_dash)
        if self.cell == "RKM-CIFG":
            cell_state = forget_gate*cell_state + (1 - forget_gate)*c_dash
        if self.cell == "Linear-kernel-wto" or self.cell == "Linear-kernel":
            cell_state = self.sigmai*c_dash + self.sigmaf*cell_state
        if self.cell == "Gated-CNN" or self.cell == "CNN":
            cell_state = self.sigmai*c_dash
        cell_state = torch.nn.functional.layer_norm(cell_state, cell_state.size()[1:])
        
        """
        IMP: Layer normalization [2] to be performed after the computation of the cell state
        """
        if self.cell == "LSTM" or self.cell == "RKM-LSTM" or self.cell == "RKM-CIFG" or self.cell == "Linear-kernel-wto" or self.cell == "Gated-CNN":
            output_state = torch.sigmoid(self.i2o(combined))
        if self.cell == "LSTM":
            hidden_state = output_state*torch.tanh(cell_state)
        if self.cell == "RKM-LSTM" or self.cell == "RKM-CIFG" or self.cell == "Linear-kernel-wto" or self.cell == "Gated-CNN":
            hidden_state = output_state*(cell_state)
        if self.cell == "Linear-kernel" or self.cell == "CNN":
            hidden_state = torch.tanh(cell_state)
        
        
        return hidden_state, cell_state