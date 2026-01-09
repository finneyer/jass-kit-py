import numpy as np
import torch
from jass.game.game_observation import GameObservation
from jass.game.const import card_ids

def convert_obs_to_tensor(obs: GameObservation) -> torch.Tensor:
    """
    Encodes the GameObservation into a tensor for the neural network.
    
    Encoding scheme (332 features):
    - 0-35: My Hand (One-hot)
    - 36-179: Cards played by each player (4 * 36) - History
    - 180-323: Current trick (4 * 36) - Position 0, 1, 2, 3 in trick
    - 324-329: Trump (One-hot 6)
    - 330: Normalized Trick Number
    - 331: Normalized Score Difference
    """
    features = np.zeros(332, dtype=np.float32)
    
    features[0:36] = obs.hand
    
    # Cards played by each player (History) - RELATIVE to current player
    # This makes the network perspective-invariant
    current_player = obs.player
    for t in range(obs.nr_tricks):
        first_player = obs.trick_first_player[t]
        if first_player == -1: continue
        
        for i in range(4):
            card = obs.tricks[t, i]
            if card != -1:
                absolute_player = (first_player + i) % 4
                relative_player = (absolute_player - current_player) % 4
                features[36 + relative_player * 36 + card] = 1.0

    # Encode sequence in the current trick
    for i in range(4):
        card = obs.current_trick[i]
        if card != -1:
            features[180 + i * 36 + card] = 1.0
            
    if obs.trump != -1:
        features[324 + obs.trump] = 1.0
        
    # Global state features
    features[330] = obs.nr_tricks / 9.0 
    
    my_team = 0 if (obs.player % 2 == 0) else 1
    opponent_team = 1 - my_team
    features[331] = (obs.points[my_team] - obs.points[opponent_team]) / 157.0 

    return torch.tensor(features).unsqueeze(0)
