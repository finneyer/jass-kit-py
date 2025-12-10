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
import functools

from impl.agents.alpha_zero import AlphaZeroAgent, AlphaZeroConfig, NeuralNetwork
from impl.service.alpha_zero_model_transformer import JassTransformer, JassTransformerSequence
from impl.service.alpha_zero_utils import convert_obs_to_tensor
from jass.game.game_observation import GameObservation
from jass.game.game_sim import GameSim
from jass.game.rule_schieber import RuleSchieber
from jass.game.game_util import deal_random_hand

@dataclass
class TrainConfig:
    iterations: int = 50
    games_per_iteration: int = 20
    mcts_simulations: int = 100  # Reduced from 200 for faster generation
    batch_size: int = 32
    learning_rate: float = 0.0001  # Lower LR for transformer
    checkpoint_dir: str = "impl/models"
    model_name: str = "alpha_zero_transformer_model.pth"
    replay_buffer_size: int = 50000
    min_buffer_size: int = 1000
    train_steps: int = 100
    num_workers: int = max(1, cpu_count() - 1)  # Leave 1 core free
    
    # Transformer architecture config
    use_sequence_model: bool = False  # Use JassTransformerSequence vs JassTransformer
    embed_dim: int = 128
    num_heads: int = 4
    ff_dim: int = 512
    num_layers: int = 3
    dropout: float = 0.1

class TrainWrapper(NeuralNetwork):
    def __init__(self, model: torch.nn.Module, device):
        self.model = model
        self.device = device

    def predict(self, obs: GameObservation) -> Tuple[np.ndarray, float]:
        self.model.eval()
        tensor_in = convert_obs_to_tensor(obs).to(self.device)
        with torch.no_grad():
            policy, value = self.model(tensor_in)
        return policy.cpu().numpy()[0], value.item()

def self_play_single_game(args) -> List[Tuple[GameObservation, np.ndarray, float]]:
    """
    Play a single game. This function is designed to be called in parallel.
    Args: (model_state_dict, device_str, mcts_simulations, model_config, game_idx)
    """
    model_state_dict, device_str, mcts_simulations, model_config, game_idx = args
    
    # Create model for this worker
    device = torch.device("cpu")  # Workers use CPU to avoid GPU contention
    
    if model_config['use_sequence_model']:
        model = JassTransformerSequence(
            embed_dim=model_config['embed_dim'],
            num_heads=model_config['num_heads'],
            ff_dim=model_config['ff_dim'],
            num_layers=model_config['num_layers'],
            dropout=model_config['dropout']
        )
    else:
        model = JassTransformer(
            input_dim=332,
            embed_dim=model_config['embed_dim'],
            num_heads=model_config['num_heads'],
            ff_dim=model_config['ff_dim'],
            num_layers=model_config['num_layers'],
            dropout=model_config['dropout']
        )
    
    model.load_state_dict(model_state_dict)
    model.to(device)
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
        
    score_diff = (points_team_0 - points_team_1) / 257.0  # Normalized with match bonus
    
    for obs, policy, player in game_history:
        val = score_diff if (player % 2 == 0) else -score_diff
        dataset.append((obs, policy, val))
    
    return dataset

def self_play_parallel(model, device, num_games: int, mcts_simulations: int, model_config: dict, num_workers: int) -> List[Tuple[GameObservation, np.ndarray, float]]:
    """
    Generate self-play games in parallel using multiprocessing.
    """
    print(f"  Using {num_workers} parallel workers")
    
    # Get model state dict (CPU)
    model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
    device_str = str(device)
    
    # Prepare arguments for each game
    args_list = [
        (model_state_dict, device_str, mcts_simulations, model_config, i)
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
    
    # Create transformer model
    if config.use_sequence_model:
        print("Using JassTransformerSequence model")
        model = JassTransformerSequence(
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            ff_dim=config.ff_dim,
            num_layers=config.num_layers,
            dropout=config.dropout
        ).to(device)
    else:
        print("Using JassTransformer model")
        model = JassTransformer(
            input_dim=332,
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            ff_dim=config.ff_dim,
            num_layers=config.num_layers,
            dropout=config.dropout
        ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Optimizer with weight decay for transformer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=0.01,
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler (cosine annealing)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.iterations,
        eta_min=config.learning_rate * 0.1
    )
    
    # Replay Buffer
    replay_buffer = deque(maxlen=config.replay_buffer_size)
    
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    best_loss = float('inf')
    
    for iteration in range(config.iterations):
        print(f"\n{'='*60}")
        print(f"Iteration {iteration + 1}/{config.iterations}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"{'='*60}")
        
        print("Generating self-play data...")
        model_config = {
            'use_sequence_model': config.use_sequence_model,
            'embed_dim': config.embed_dim,
            'num_heads': config.num_heads,
            'ff_dim': config.ff_dim,
            'num_layers': config.num_layers,
            'dropout': config.dropout
        }
        
        new_data = self_play_parallel(
            model,
            device,
            num_games=config.games_per_iteration,
            mcts_simulations=config.mcts_simulations,
            model_config=model_config,
            num_workers=config.num_workers
        )
        
        # Add new data to buffer
        replay_buffer.extend(new_data)
        print(f"Generated {len(new_data)} samples. Buffer size: {len(replay_buffer)}")
        
        if len(replay_buffer) < config.min_buffer_size:
            print("Buffer too small, skipping training...")
            continue
        
        # Training phase
        model.train()
        total_loss = 0
        total_loss_v = 0
        total_loss_p = 0
        
        for step in range(config.train_steps):
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
            
            # Value loss (MSE)
            loss_v = torch.mean((tensor_value - pred_value) ** 2)
            
            # Policy loss (Cross-entropy)
            loss_p = -torch.mean(torch.sum(tensor_policy * torch.log(pred_policy + 1e-8), dim=1))
            
            # Combined loss
            loss = loss_v + loss_p
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            total_loss_v += loss_v.item()
            total_loss_p += loss_p.item()
        
        avg_loss = total_loss / config.train_steps
        avg_loss_v = total_loss_v / config.train_steps
        avg_loss_p = total_loss_p / config.train_steps
        
        print(f"Training Loss - Total: {avg_loss:.4f} | Value: {avg_loss_v:.4f} | Policy: {avg_loss_p:.4f}")
        
        # Update learning rate
        scheduler.step()
        
        # Save best model (state_dict only)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_path = os.path.join(config.checkpoint_dir, "best_" + config.model_name)
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved! Loss: {best_loss:.4f}")
    
    # Save final model (state_dict only)
    final_model_path = os.path.join(config.checkpoint_dir, config.model_name)
    torch.save(model.state_dict(), final_model_path)
    print(f"\nFinal model saved to {final_model_path}")
    print(f"Best loss achieved: {best_loss:.4f}")

if __name__ == "__main__":
    train_loop()
