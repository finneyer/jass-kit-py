import sys
import os

# Add project root to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import torch.nn.functional as F
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
    iterations: int = 100
    games_per_iteration: int = 50
    mcts_simulations: int = 400
    batch_size: int = 128
    learning_rate: float = 0.002
    train_steps: int = 100
    
    hidden_dim: int = 256
    num_res_blocks: int = 3
    
    checkpoint_dir: str = "impl/models"
    model_name: str = "alpha_zero_model_cnn.pth"
    replay_buffer_size: int = 50000
    min_buffer_size: int = 2000
    num_workers: int = max(1, cpu_count() - 1)

class TrainWrapper(NeuralNetwork):
    def __init__(self, model: JassNet, device):
        self.model = model
        self.device = device

    def predict(self, obs: GameObservation) -> Tuple[np.ndarray, float]:
        self.model.eval()
        tensor_in = convert_obs_to_tensor(obs).to(self.device)
        with torch.no_grad():
            policy, value, win_prob = self.model(tensor_in)
        return policy.cpu().numpy()[0], value.item()

def self_play_single_game(args) -> List[Tuple[GameObservation, np.ndarray, float]]:
    """
    Play a single game. This function is designed to be called in parallel.
    Args: (model_state_dict, hidden_dim, num_res_blocks, mcts_simulations, game_idx)
    """
    model_state_dict, hidden_dim, num_res_blocks, mcts_simulations, game_idx = args
    
    device = torch.device("cpu")
    model = JassNet(hidden_dim=hidden_dim, num_res_blocks=num_res_blocks).to(device)
    model.load_state_dict(model_state_dict)
    model.eval()
    
    agent = AlphaZeroAgent(
        config=AlphaZeroConfig(
            iterations=mcts_simulations,
            time_limit_ms=None,
            dirichlet_alpha=0.3
        ),
        network=TrainWrapper(model, device)
    )
    
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
        
        policy = np.zeros(36, dtype=np.float32)
        for action, child in root.children.items():
            policy[action] = child.visits
        
        if policy.sum() > 0:
            policy /= policy.sum()
        else:
            valid_actions = rule.get_valid_cards_from_obs(obs)
            valid_mask = np.flatnonzero(valid_actions)
            policy[valid_mask] = 1.0 / len(valid_mask)
        
        game_history.append((obs, policy, player))
        
        action = np.random.choice(36, p=policy)
        game.action_play_card(action)
    
    points_team_0 = game.state.points[0]
    points_team_1 = game.state.points[1]
    
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
    
    team_0_won = 1.0 if points_team_0 > points_team_1 else 0.0
    
    for obs, policy, player in game_history:
        val_score = score_diff if (player % 2 == 0) else -score_diff
        is_team_0 = (player % 2 == 0)
        val_win = team_0_won if is_team_0 else (1.0 - team_0_won)
        dataset.append((obs, policy, val_score, val_win))
    
    return dataset

def self_play_parallel(model, config: TrainConfig, num_games: int) -> List[Tuple[GameObservation, np.ndarray, float]]:
    """
    Generate self-play games in parallel using multiprocessing.
    """
    print(f"  Using {config.num_workers} parallel workers")
    
    model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
    
    args_list = [
        (model_state_dict, config.hidden_dim, config.num_res_blocks, config.mcts_simulations, i)
        for i in range(num_games)
    ]
    
    with Pool(processes=config.num_workers) as pool:
        results = pool.map(self_play_single_game, args_list)
    
    dataset = []
    for game_data in results:
        dataset.extend(game_data)
    
    return dataset

def evaluate_vs_baseline(model, device, num_games=10):
    """Quick evaluation against random MCTS to track progress."""
    from jass.arena.arena import Arena
    from impl.agents.alpha_zero import MCTSAgent, MCTSConfig
    
    az_network = TrainWrapper(model, device)
    az_config = AlphaZeroConfig(iterations=400, time_limit_ms=None, dirichlet_alpha=0.0)
    az_agent = AlphaZeroAgent(config=az_config, network=az_network)
    
    mcts_config = MCTSConfig(iterations=400, time_limit_ms=None)
    mcts_agent = MCTSAgent(config=mcts_config)
    
    arena = Arena(nr_games_to_play=num_games, print_every_x_games=999)
    arena.set_players(az_agent, mcts_agent, az_agent, mcts_agent)
    arena.play_all_games()
    
    points_team_0 = arena.points_team_0
    points_team_1 = arena.points_team_1
    avg_diff = np.mean(points_team_0 - points_team_1)
    win_rate = np.sum(points_team_0 > points_team_1) / num_games
    
    return avg_diff, win_rate

def train_loop():
    config = TrainConfig()
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    print(f"Using {config.num_workers} workers for parallel self-play")
    print(f"Model architecture: hidden_dim={config.hidden_dim}, num_res_blocks={config.num_res_blocks}")
    
    model = JassNet(hidden_dim=config.hidden_dim, num_res_blocks=config.num_res_blocks).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.iterations, eta_min=1e-5)
    
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
            config,
            num_games=config.games_per_iteration
        )
        
        replay_buffer.extend(new_data)
        print(f"Generated {len(new_data)} samples. Buffer size: {len(replay_buffer)}")
        
        if len(replay_buffer) < config.min_buffer_size:
            print("Buffer too small, skipping training...")
            continue
            
        model.train()
        
        total_loss = 0
        total_loss_v = 0
        total_loss_p = 0
        total_loss_w = 0
        
        for _ in range(config.train_steps):
            batch = random.sample(replay_buffer, config.batch_size)
            
            batch_obs = []
            batch_policy = []
            batch_value = []
            batch_win = []
            
            for obs, pi, v_score, v_win in batch:
                batch_obs.append(convert_obs_to_tensor(obs))
                batch_policy.append(pi)
                batch_value.append(v_score)
                batch_win.append(v_win)
            
            tensor_obs = torch.cat(batch_obs).to(device)
            tensor_policy = torch.tensor(np.array(batch_policy), dtype=torch.float32).to(device)
            tensor_value = torch.tensor(np.array(batch_value), dtype=torch.float32).unsqueeze(1).to(device)
            tensor_win = torch.tensor(np.array(batch_win), dtype=torch.float32).unsqueeze(1).to(device)
            
            optimizer.zero_grad()
            pred_policy, pred_value, pred_win_prob = model(tensor_obs)
            
            loss_v = torch.mean((tensor_value - pred_value) ** 2)
            loss_p = -torch.mean(torch.sum(tensor_policy * torch.log(pred_policy + 1e-8), dim=1))
            loss_w = F.binary_cross_entropy(pred_win_prob, tensor_win)
            
            loss = loss_v + loss_p + loss_w
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            total_loss_v += loss_v.item()
            total_loss_p += loss_p.item()
            total_loss_w += loss_w.item()
            
        avg_loss = total_loss / config.train_steps
        avg_loss_v = total_loss_v / config.train_steps
        avg_loss_p = total_loss_p / config.train_steps
        avg_loss_w = total_loss_w / config.train_steps
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Training Loss - Total: {avg_loss:.4f} | Value: {avg_loss_v:.4f} | Policy: {avg_loss_p:.4f} | Win: {avg_loss_w:.4f} | LR: {current_lr:.6f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_path = os.path.join(config.checkpoint_dir, "best_" + config.model_name)
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved! Loss: {best_loss:.4f}")
        
        scheduler.step()
        
        if (iteration + 1) % 10 == 0:
            print(f"\nEvaluating against baseline MCTS...")
            model.eval()
            avg_diff, win_rate = evaluate_vs_baseline(model, device, num_games=10)
            print(f"Evaluation: Avg Score Diff: {avg_diff:+.1f} | Win Rate: {win_rate:.1%}\n")
        
    save_path = os.path.join(config.checkpoint_dir, config.model_name)
    torch.save(model.state_dict(), save_path)
    print(f"\nFinal model saved to {save_path}")
    print(f"Best loss achieved: {best_loss:.4f}")

if __name__ == "__main__":
    train_loop()
