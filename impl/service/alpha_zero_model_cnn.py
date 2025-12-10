import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.fc1(x)))
        out = self.bn2(self.fc2(out))
        out += residual
        out = F.relu(out)
        return out

class JassNet(nn.Module):
    def __init__(self, input_shape=332, hidden_dim=256, num_res_blocks=3): 
        """
        ResNet-style Neural Network for Jass AlphaZero.
        """
        super(JassNet, self).__init__()
        
        # Initial embedding
        self.input_fc = nn.Linear(input_shape, hidden_dim)
        self.input_bn = nn.BatchNorm1d(hidden_dim)
        
        # Residual Tower
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(num_res_blocks)
        ])
        
        # Policy Head
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 36)
        )
        
        # Value Head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        # Input processing
        x = F.relu(self.input_bn(self.input_fc(x)))
        
        # Residual blocks
        for block in self.res_blocks:
            x = block(x)
        
        # Heads
        policy = F.softmax(self.policy_head(x), dim=1)
        value = self.value_head(x)
        
        return policy, value
