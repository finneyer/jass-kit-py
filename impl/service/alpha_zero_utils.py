import numpy as np
import torch
from jass.game.game_observation import GameObservation
from jass.game.const import card_ids

def convert_obs_to_tensor(obs: GameObservation) -> torch.Tensor:
    """
    Encodes the GameObservation into a tensor for the neural network.
    
    Encoding scheme (330 features):
    - 0-35: My Hand (One-hot)
    - 36-179: Cards played by each player (4 * 36) - History
    - 180-323: Current trick (4 * 36) - Position 0, 1, 2, 3 in trick
    - 324-329: Trump (One-hot 6)
    """
    features = np.zeros(330, dtype=np.float32)
    
    # 1. My Hand
    features[0:36] = obs.hand
    
    # 2. Cards played by each player (History)
    # We need to iterate through tricks to see who played what.
    # obs.tricks is [9, 4]. obs.trick_first_player is [9].
    for t in range(obs.nr_tricks):
        first_player = obs.trick_first_player[t]
        if first_player == -1: continue
        
        for i in range(4):
            card = obs.tricks[t, i]
            if card != -1:
                player = (first_player + i) % 4
                # Offset: 36 + player * 36 + card
                features[36 + player * 36 + card] = 1.0

    # 3. Current Trick
    # obs.current_trick is [4].
    # We want to encode the sequence in the current trick.
    # Position 0, 1, 2, 3 in the trick.
    for i in range(4):
        card = obs.current_trick[i]
        if card != -1:
            # Offset: 180 + i * 36 + card
            features[180 + i * 36 + card] = 1.0
            
    # 4. Trump
    if obs.trump != -1:
        features[324 + obs.trump] = 1.0
        
    return torch.tensor(features).unsqueeze(0) # Add batch dimension
