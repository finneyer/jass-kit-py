import sys
import os

# Add project root to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import numpy as np
import os
import random
from collections import deque
from dataclasses import dataclass
from typing import Tuple, List
from multiprocessing import Pool, cpu_count

from impl.agents.alpha_zero import AlphaZeroAgent, AlphaZeroConfig, NeuralNetwork
from impl.service.alpha_zero_model_cnn import JassNet
from impl.service.alpha_zero_utils import convert_obs_to_tensor
from jass.game.game_observation import GameObservation
from jass.game.game_sim import GameSim
from jass.game.rule_schieber import RuleSchieber
from jass.game.game_util import deal_random_hand

@dataclass
class TrainConfig:
    iterations: int = 50
    games_per_iteration: int = 20
    mcts_simulations: int = 100  # Reduced for faster generation
    batch_size: int = 32
    learning_rate: float = 0.001
    checkpoint_dir: str = "impl/models"
    model_name: str = "alpha_zero_model_cnn.pth"
    replay_buffer_size: int = 50000
    min_buffer_size: int = 1000
    train_steps: int = 100
    num_workers: int = max(1, cpu_count() - 1)  # Leave 1 core free

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

def self_play(agent: AlphaZeroAgent, num_games: int) -> List[Tuple[GameObservation, np.ndarray, float]]:
    dataset = []
    rule = RuleSchieber()
    
    for _ in range(num_games):
        game = GameSim(rule=rule)
        dealer = np.random.randint(4)
        game.init_from_cards(hands=deal_random_hand(), dealer=dealer)
        
        trump = np.random.randint(6)
        game.action_trump(trump)
        
        game_history = []
        
        while game.state.nr_tricks < 9:
            player = game.state.player
            obs = game.get_observation()
            
            root = agent._search(obs)
            
            # Policy target is the visit count distribution from MCTS
            policy = np.zeros(36, dtype=np.float32)
            for action, child in root.children.items():
                policy[action] = child.visits
            
            # Only normalize - don't mask again! MCTS already only visited valid actions
            if policy.sum() > 0:
                policy /= policy.sum()
            else:
                # Shouldn't happen, but fallback to uniform over valid actions
                valid_actions = rule.get_valid_cards_from_obs(obs)
                valid_mask = np.flatnonzero(valid_actions)
                policy[valid_mask] = 1.0 / len(valid_mask)
            
            game_history.append((obs, policy, player))
            
            action = np.random.choice(36, p=policy)
            game.action_play_card(action)
            
        # Calculate score with match bonus
        points_team_0 = game.state.points[0]
        points_team_1 = game.state.points[1]
        
        # Check for match (all 9 tricks won by one team)
        # Note: game.state.points might not include the 100 bonus depending on implementation,
        # but let's manually check tricks won if needed. 
        # Actually, GameSim updates points correctly including match bonus if configured,
        # but let's be safe and use the same logic as GameStateAdapter.score
        
        # Count tricks per team
        tricks_0 = 0
        tricks_1 = 0
        # We can check trick_winner array
        for t in range(9):
            w = game.state.trick_winner[t]
            if w == 0 or w == 2:
                tricks_0 += 1
            else:
                tricks_1 += 1
                
        if tricks_0 == 9:
            points_team_0 += 100
        elif tricks_1 == 9:
            points_team_1 += 100
            
        score_diff = (points_team_0 - points_team_1) / 257.0 # Normalized with match bonus
        
        for obs, policy, player in game_history:
            val = score_diff if (player % 2 == 0) else -score_diff
            dataset.append((obs, policy, val))
            
    return dataset

def self_play_single_game(args) -> List[Tuple[GameObservation, np.ndarray, float]]:
    """
    Play a single game. This function is designed to be called in parallel.
    Args: (model_state_dict, mcts_simulations, game_idx)
    """
    model_state_dict, mcts_simulations, game_idx = args
    
    # Create model for this worker (CPU to avoid GPU contention)
    device = torch.device("cpu")
    model = JassNet(hidden_dim=256, num_res_blocks=3).to(device)
    model.load_state_dict(model_state_dict)
    model.eval()
    
    # Create agent
    agent = AlphaZeroAgent(
        config=AlphaZeroConfig(
            iterations=mcts_simulations,
            time_limit_ms=None,
            dirichlet_alpha=0.3
        ),
        network=TrainWrapper(model, device)
    )
    
    # Play game
    dataset = []
    rule = RuleSchieber()
    
    game = GameSim(rule=rule)
    dealer = np.random.randint(4)
    game.init_from_cards(hands=deal_random_hand(), dealer=dealer)
    
    trump = np.random.randint(6)
    game.action_trump(trump)
    
    game_history = []
    
    while game.state.nr_tricks < 9:
        player = game.state.player
        obs = game.get_observation()
        
        root = agent._search(obs)
        
        # Policy target is the visit count distribution from MCTS
        policy = np.zeros(36, dtype=np.float32)
        for action, child in root.children.items():
            policy[action] = child.visits
        
        # Only normalize - don't mask again! MCTS already only visited valid actions
        if policy.sum() > 0:
            policy /= policy.sum()
        else:
            # Shouldn't happen, but fallback to uniform over valid actions
            valid_actions = rule.get_valid_cards_from_obs(obs)
            valid_mask = np.flatnonzero(valid_actions)
            policy[valid_mask] = 1.0 / len(valid_mask)
        
        game_history.append((obs, policy, player))
        
        action = np.random.choice(36, p=policy)
        game.action_play_card(action)
    
    # Calculate score with match bonus
    points_team_0 = game.state.points[0]
    points_team_1 = game.state.points[1]
    
    # Count tricks per team for match bonus
    tricks_0 = 0
    tricks_1 = 0
    for t in range(9):
        w = game.state.trick_winner[t]
        if w == 0 or w == 2:
            tricks_0 += 1
        else:
            tricks_1 += 1
            
    if tricks_0 == 9:
        points_team_0 += 100
    elif tricks_1 == 9:
        points_team_1 += 100
        
    score_diff = (points_team_0 - points_team_1) / 257.0
    
    for obs, policy, player in game_history:
        val = score_diff if (player % 2 == 0) else -score_diff
        dataset.append((obs, policy, val))
    
    return dataset

def self_play_parallel(model, num_games: int, mcts_simulations: int, num_workers: int) -> List[Tuple[GameObservation, np.ndarray, float]]:
    """
    Generate self-play games in parallel using multiprocessing.
    """
    print(f"  Using {num_workers} parallel workers")
    
    # Get model state dict (CPU)
    model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
    
    # Prepare arguments for each game
    args_list = [
        (model_state_dict, mcts_simulations, i)
        for i in range(num_games)
    ]
    
    # Run games in parallel
    with Pool(processes=num_workers) as pool:
        results = pool.map(self_play_single_game, args_list)
    
    # Flatten results
    dataset = []
    for game_data in results:
        dataset.extend(game_data)
    
    return dataset

def train_loop():
    config = TrainConfig()
    
    # Device selection: CUDA > MPS > CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    print(f"Using {config.num_workers} workers for parallel self-play")
    
    # Reduced model complexity
    model = JassNet(hidden_dim=256, num_res_blocks=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    
    # Replay Buffer
    replay_buffer = deque(maxlen=config.replay_buffer_size)
    
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    best_loss = float('inf')
    
    for iteration in range(config.iterations):
        print(f"\n{'='*60}")
        print(f"Iteration {iteration + 1}/{config.iterations}")
        print(f"{'='*60}")
        
        print("Generating self-play data...")
        new_data = self_play_parallel(
            model,
            num_games=config.games_per_iteration,
            mcts_simulations=config.mcts_simulations,
            num_workers=config.num_workers
        )
        
        # Add new data to buffer
        replay_buffer.extend(new_data)
        print(f"Generated {len(new_data)} samples. Buffer size: {len(replay_buffer)}")
        
        if len(replay_buffer) < config.min_buffer_size:
            print("Buffer too small, skipping training...")
            continue
            
        model.train()
        
        total_loss = 0
        total_loss_v = 0
        total_loss_p = 0
        
        for _ in range(config.train_steps):
            batch = random.sample(replay_buffer, config.batch_size)
            
            batch_obs = []
            batch_policy = []
            batch_value = []
            
            for obs, pi, v in batch:
                batch_obs.append(convert_obs_to_tensor(obs))
                batch_policy.append(pi)
                batch_value.append(v)
            
            tensor_obs = torch.cat(batch_obs).to(device)
            tensor_policy = torch.tensor(np.array(batch_policy), dtype=torch.float32).to(device)
            tensor_value = torch.tensor(np.array(batch_value), dtype=torch.float32).unsqueeze(1).to(device)
            
            optimizer.zero_grad()
            pred_policy, pred_value = model(tensor_obs)
            
            loss_v = torch.mean((tensor_value - pred_value) ** 2)
            loss_p = -torch.mean(torch.sum(tensor_policy * torch.log(pred_policy + 1e-8), dim=1))
            
            loss = loss_v + loss_p
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_loss_v += loss_v.item()
            total_loss_p += loss_p.item()
            
        avg_loss = total_loss / config.train_steps
        avg_loss_v = total_loss_v / config.train_steps
        avg_loss_p = total_loss_p / config.train_steps
        
        print(f"Training Loss - Total: {avg_loss:.4f} | Value: {avg_loss_v:.4f} | Policy: {avg_loss_p:.4f}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_path = os.path.join(config.checkpoint_dir, "best_" + config.model_name)
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved! Loss: {best_loss:.4f}")
        
    save_path = os.path.join(config.checkpoint_dir, config.model_name)
    torch.save(model.state_dict(), save_path)
    print(f"\nFinal model saved to {save_path}")
    print(f"Best loss achieved: {best_loss:.4f}")

if __name__ == "__main__":
    train_loop()

