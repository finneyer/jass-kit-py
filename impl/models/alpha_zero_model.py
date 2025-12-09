import torch
import torch.nn as nn
import torch.nn.functional as F

class JassNet(nn.Module):
    def __init__(self, input_shape=330): 
        """
        Neural Network for Jass AlphaZero.
        Input shape calculation (example):
        - Hand: 36
        - Cards played by each player (history): 4 * 36 = 144
        - Current trick state (who played what): 4 * 36 = 144
        - Trump: 6
        Total: 330
        """
        super(JassNet, self).__init__()
        # Shared layers
        self.fc1 = nn.Linear(input_shape, 512)
        self.fc2 = nn.Linear(512, 256)
        
        # Policy Head (36 cards)
        self.policy_head = nn.Linear(256, 36)
        
        # Value Head (Score estimation)
        self.value_head = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # Policy: probabilities over 36 cards
        policy = F.softmax(self.policy_head(x), dim=1)
        
        # Value: score between -1 and 1
        value = torch.tanh(self.value_head(x))
        
        return policy, value
