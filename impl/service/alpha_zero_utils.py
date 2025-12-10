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
    
    # 1. My Hand
    features[0:36] = obs.hand
    
    # 2. Cards played by each player (History) - RELATIVE to current player
    # This makes the network perspective-invariant
    # Position 0 = current player, 1 = next, 2 = partner, 3 = previous
    # obs.tricks is [9, 4]. obs.trick_first_player is [9].
    current_player = obs.player
    for t in range(obs.nr_tricks):
        first_player = obs.trick_first_player[t]
        if first_player == -1: continue
        
        for i in range(4):
            card = obs.tricks[t, i]
            if card != -1:
                absolute_player = (first_player + i) % 4
                # Convert to relative position (0=me, 1=next, 2=partner, 3=previous)
                relative_player = (absolute_player - current_player) % 4
                # Offset: 36 + relative_player * 36 + card
                features[36 + relative_player * 36 + card] = 1.0

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
        
    # 5. Global State features
    # trick number normalized (0 to 8) / 9.0
    features[330] = obs.nr_tricks / 9.0 
    # current score difference from MY TEAM's perspective, normalized
    my_team = 0 if (obs.player % 2 == 0) else 1
    opponent_team = 1 - my_team
    features[331] = (obs.points[my_team] - obs.points[opponent_team]) / 157.0 

    return torch.tensor(features).unsqueeze(0) # Add batch dimension
