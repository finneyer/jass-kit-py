import torch
import numpy as np
import os
from impl.agents.alpha_zero import AlphaZeroAgent, AlphaZeroConfig, NeuralNetwork
from impl.models.alpha_zero_model import JassNet
from impl.service.alpha_zero_utils import convert_obs_to_tensor
from jass.arena.arena import Arena
from jass.agents.agent_random_schieber import AgentRandomSchieber
from typing import Tuple, List
from jass.game.game_observation import GameObservation

class TrainWrapper(NeuralNetwork):
    def __init__(self, model: JassNet, device):
        self.model = model
        self.device = device

    def predict(self, obs: GameObservation) -> Tuple[np.ndarray, float]:
        self.model.eval()
        tensor_in = convert_obs_to_tensor(obs).to(self.device)
        with torch.no_grad():
            policy, value = self.model(tensor_in)
        return policy.cpu().numpy()[0], value.item()

def self_play(agent: AlphaZeroAgent, num_games=10) -> List[Tuple[GameObservation, np.ndarray, float]]:
    """
    Generates training data by self-play.
    Returns a list of (observation, policy_target, value_target).
    """
    dataset = []
    
    # We need to hook into the agent to capture (obs, policy) pairs.
    # Since AlphaZeroAgent doesn't expose this easily, we might need to modify it 
    # or subclass it to record data.
    # For now, let's assume we can access the search root or similar.
    # Actually, the cleanest way is to have the agent return the policy used.
    
    # But AlphaZeroAgent.action_play_card returns an int.
    # We can modify AlphaZeroAgent to store the latest search info.
    
    # Let's run games using Arena, but we need to capture data.
    # Arena doesn't support capturing internal agent state.
    # So we might need to run the game loop manually here or modify the agent to log.
    
    # Let's implement a simple game loop here for self-play.
    from jass.game.game_sim import GameSim
    from jass.game.rule_schieber import RuleSchieber
    from jass.game.game_util import deal_random_hand
    
    rule = RuleSchieber()
    
    for _ in range(num_games):
        game = GameSim(rule=rule)
        dealer = np.random.randint(4)
        game.init_from_cards(hands=deal_random_hand(), dealer=dealer)
        
        # Trump selection (random for now as per agent)
        # We skip trump selection training for now
        # game.action_trump(game.get_observation().trump) # This is wrong, we need to select trump
        
        # Actually, let's just use a random trump for training play phase
        trump = np.random.randint(6)
        game.action_trump(trump)
        
        game_history = [] # (obs, policy, player_to_move)
        
        while game.state.nr_tricks < 9:
            player = game.state.player
            obs = game.get_observation()
            
            # Run MCTS
            # We need to access the root node to get visit counts (policy)
            root = agent._search(obs)
            
            # Calculate policy from visits
            policy = np.zeros(36, dtype=np.float32)
            for action, child in root.children.items():
                policy[action] = child.visits
            policy /= policy.sum()
            
            # Store
            game_history.append((obs, policy, player))
            
            # Sample action from policy (exploration)
            action = np.random.choice(36, p=policy)
            game.action_play_card(action)
            
        # Game over, assign rewards
        # Score is normalized [-1, 1]
        points_team_0 = game.state.points[0]
        points_team_1 = game.state.points[1]
        
        # Who won?
        # In Jass, it's points.
        # Let's use the normalized score difference as value target.
        score_diff = (points_team_0 - points_team_1) / 157.0 # Approx
        
        for obs, policy, player in game_history:
            # Value is from perspective of 'player'
            val = score_diff if (player % 2 == 0) else -score_diff
            dataset.append((obs, policy, val))
            
    return dataset

def train_loop():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = JassNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    os.makedirs("checkpoints", exist_ok=True)
    best_loss = float('inf')
    
    for iteration in range(100):
        print(f"Iteration {iteration}")
        
        # 1. Self Play
        agent = AlphaZeroAgent(
            config=AlphaZeroConfig(iterations=50, time_limit_ms=None), # Fast MCTS for training
            network=TrainWrapper(model, device)
        )
        
        print("Generating data...")
        new_data = self_play(agent, num_games=5) # Small number for demo
        print(f"Generated {len(new_data)} samples.")
        
        # 2. Train
        model.train()
        # Simple batch training
        batch_size = 32
        indices = np.arange(len(new_data))
        np.random.shuffle(indices)
        
        total_loss = 0
        for start_idx in range(0, len(new_data), batch_size):
            end_idx = min(start_idx + batch_size, len(new_data))
            batch_indices = indices[start_idx:end_idx]
            
            batch_obs = []
            batch_policy = []
            batch_value = []
            
            for idx in batch_indices:
                obs, pi, v = new_data[idx]
                batch_obs.append(convert_obs_to_tensor(obs))
                batch_policy.append(pi)
                batch_value.append(v)
            
            # Stack
            # convert_obs_to_tensor returns (1, 330), so cat them
            tensor_obs = torch.cat(batch_obs).to(device)
            tensor_policy = torch.tensor(np.array(batch_policy), dtype=torch.float32).to(device)
            tensor_value = torch.tensor(np.array(batch_value), dtype=torch.float32).unsqueeze(1).to(device)
            
            optimizer.zero_grad()
            pred_policy, pred_value = model(tensor_obs)
            
            # Loss
            # Value loss: MSE
            loss_v = torch.mean((tensor_value - pred_value) ** 2)
            # Policy loss: Cross Entropy (pred_policy is softmaxed)
            # We use -sum(target * log(pred))
            loss_p = -torch.mean(torch.sum(tensor_policy * torch.log(pred_policy + 1e-8), dim=1))
            
            loss = loss_v + loss_p
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / (len(new_data)/batch_size)
        print(f"Loss: {avg_loss}")
        
        # 3. Save if best
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "checkpoints/best_model.pth")
            print(f"New best model saved with loss {best_loss}")

if __name__ == "__main__":
    train_loop()
