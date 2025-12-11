from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Tuple
import time
import os
import sys

# Add project root to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import torch

from jass.agents.agent import Agent
from jass.game.const import PUSH, next_player, partner_player, card_values
from jass.game.game_observation import GameObservation
from jass.game.game_rule import GameRule
from jass.game.game_util import convert_one_hot_encoded_cards_to_int_encoded_list, \
    convert_int_encoded_cards_to_str_encoded
from jass.game.rule_schieber import RuleSchieber
from jass.agents.agent_random_schieber import AgentRandomSchieber
from jass.arena.arena import Arena
from impl.service.alpha_zero_model_cnn import JassNet
from impl.service.alpha_zero_model_transformer import JassTransformer, JassTransformerSequence
from impl.service.alpha_zero_utils import convert_obs_to_tensor
import logging
from impl.agents.agent_monte_carlo import MCTSAgent, MCTSConfig

# Import TrainConfig to allow unpickling of saved checkpoints
try:
    from impl.service.alpha_zero_train_transformer import TrainConfig as TransformerTrainConfig
except ImportError:
    TransformerTrainConfig = None


class NeuralNetwork:
    """
    Abstract interface for the AlphaZero neural network.
    """
    def predict(self, obs: GameObservation) -> Tuple[np.ndarray, float]:
        """
        Predicts the policy and value for a given observation.
        
        Args:
            obs: The game observation from the perspective of the current player.
            
        Returns:
            policy: A numpy array of size 36 (one for each card) representing action probabilities.
            value: A float in [-1, 1] representing the expected score/win probability.
        """
        raise NotImplementedError


class DummyNetwork(NeuralNetwork):
    """
    A dummy network that returns uniform probabilities and random values.
    Used when no trained model is available.
    """
    def predict(self, obs: GameObservation) -> Tuple[np.ndarray, float]:
        # Uniform policy over all 36 cards (validity will be masked by the search)
        policy = np.ones(36, dtype=np.float32) / 36.0
        # Random value
        value = np.random.uniform(-0.1, 0.1)
        return policy, value


@dataclass
class AlphaZeroConfig:
    """Configuration for the AlphaZero agent."""
    iterations: int = 200
    time_limit_ms: Optional[int] = 150
    c_puct: float = 1.0
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    determinization_samples: int = 8
    use_rollouts: bool = False
    rollout_count: int = 10
    rollout_mixing: float = 0.9


@dataclass
class AlphaZeroNode:
    """A node in the AlphaZero search tree."""
    parent: Optional[AlphaZeroNode] = None
    children: Dict[int, AlphaZeroNode] = field(default_factory=dict)
    visits: int = 0
    value_sum: float = 0.0
    prior: float = 0.0
    player_to_move: int = -1
    is_expanded: bool = False
    # Cached policy from the network, used to instantiate children lazily
    policy_probs: Optional[np.ndarray] = None

    @property
    def value(self) -> float:
        return self.value_sum / self.visits if self.visits > 0 else 0.0


class GameStateAdapter:
    """
    A lightweight adapter for the Jass game state to be used in MCTS.
    It must be cloneable and provide methods to step through the game.
    Crucially, it must enforce that only information available to the current
    player is used.
    """
    def __init__(self, obs: GameObservation, determinized_hands: np.ndarray):
        """
        Initializes the state from a player's observation and a specific
        determinization of hidden cards.
        """
        self._rule = RuleSchieber()
        self.player = obs.player
        self.trump = obs.trump
        self.hands = determinized_hands
        self.tricks = np.copy(obs.tricks)
        self.nr_tricks = obs.nr_tricks
        self.nr_cards_in_trick = obs.nr_cards_in_trick
        self.trick_first_player = np.copy(obs.trick_first_player)
        self.trick_winner = np.copy(obs.trick_winner)
        self.trick_points = np.copy(obs.trick_points)
        self.points = np.copy(obs.points)

        if self.nr_cards_in_trick > 0:
            self.current_trick = self.tricks[self.nr_tricks]
        else:
            self.current_trick = np.full(4, -1, dtype=np.int32)


    def clone(self) -> GameStateAdapter:
        """Return a deep copy of the current game state."""
        # create a new, empty object and copy over the state
        new_state = GameStateAdapter.__new__(GameStateAdapter)
        new_state._rule = self._rule
        new_state.player = self.player
        new_state.trump = self.trump
        new_state.hands = np.copy(self.hands)
        new_state.tricks = np.copy(self.tricks)
        new_state.nr_tricks = self.nr_tricks
        new_state.nr_cards_in_trick = self.nr_cards_in_trick
        new_state.trick_first_player = np.copy(self.trick_first_player)
        new_state.trick_winner = np.copy(self.trick_winner)
        new_state.trick_points = np.copy(self.trick_points)
        new_state.points = np.copy(self.points)
        new_state.current_trick = np.copy(self.current_trick)
        return new_state

    def valid_actions(self) -> List[int]:
        """Return a list of valid actions for the current player."""
        hand = self.hands[self.player]
        valid_cards_mask = self._rule.get_valid_cards(hand, self.current_trick, self.nr_cards_in_trick, self.trump)
        return np.flatnonzero(valid_cards_mask).tolist()

    def step(self, action: int) -> None:
        """
        Apply an action and update the game state.
        """
        # card is played
        self.hands[self.player, action] = 0
        self.current_trick[self.nr_cards_in_trick] = action
        self.nr_cards_in_trick += 1

        if self.nr_cards_in_trick == 4:
            # trick is complete
            self.tricks[self.nr_tricks] = self.current_trick
            self.trick_winner[self.nr_tricks] = self._rule.calc_winner(self.current_trick, self.trick_first_player[self.nr_tricks], self.trump)
            self.trick_points[self.nr_tricks] = self._rule.calc_points(self.current_trick, self.nr_tricks == 8, self.trump)

            winner = self.trick_winner[self.nr_tricks]
            if winner == 0 or winner == 2:
                self.points[0] += self.trick_points[self.nr_tricks]
            else:
                self.points[1] += self.trick_points[self.nr_tricks]

            self.nr_tricks += 1
            self.nr_cards_in_trick = 0
            self.player = winner
            if self.nr_tricks < 9:
                self.trick_first_player[self.nr_tricks] = self.player
                # rebind to the next trick row and clear
                self.current_trick = self.tricks[self.nr_tricks]
                self.current_trick.fill(-1)
        else:
            self.player = next_player[self.player]


    def is_terminal(self) -> bool:
        """Check if the game has ended."""
        return self.nr_tricks == 9

    def score(self, player: int) -> float:
        """
        Return the final score for the given player's team, normalized to [-1, 1].
        Max possible score is 157 + 100 for match.
        """
        player_team = 0 if player == 0 or player == 2 else 1
        opponent_team = 1 - player_team
        
        score = self.points[player_team] - self.points[opponent_team]
        
        # check if a team made the match (only valid at terminal state)
        if self.nr_tricks == 9:
            if self.points[player_team] > 0 and self.points[opponent_team] == 0:
                score += 100
            elif self.points[opponent_team] > 0 and self.points[player_team] == 0:
                score -= 100
            
        return score / 257.0

    def to_observation(self) -> GameObservation:
        """
        Constructs a GameObservation from the current state.
        """
        obs = GameObservation()
        obs.dealer = -1 
        obs.player = self.player
        obs.player_view = self.player
        obs.trump = self.trump
        obs.forehand = -1
        obs.declared_trump = self.trump
        obs.hand = self.hands[self.player]
        obs.tricks = self.tricks
        obs.trick_winner = self.trick_winner
        obs.trick_points = self.trick_points
        obs.trick_first_player = self.trick_first_player
        obs.current_trick = self.current_trick
        obs.nr_tricks = self.nr_tricks
        obs.nr_cards_in_trick = self.nr_cards_in_trick
        obs.nr_played_cards = self.nr_tricks * 4 + self.nr_cards_in_trick
        obs.points = self.points
        return obs


class AlphaZeroAgent(Agent):
    """
    An agent that uses AlphaZero-style MCTS (PUCT + Neural Network) to decide its actions.
    """

    def __init__(self, config: Optional[AlphaZeroConfig] = None, network: Optional[NeuralNetwork] = None):
        super().__init__()
        self._rule = RuleSchieber()
        self.config = config or AlphaZeroConfig()
        self.network = network or DummyNetwork()
        self._rng = np.random.default_rng()

    def action_trump(self, obs: GameObservation) -> int:
        # For simplicity, use a basic heuristic or random for trump selection
        # as AlphaZero is typically applied to the play phase here.
        return AgentRandomSchieber().action_trump(obs)

    def action_play_card(self, obs: GameObservation) -> int:
        valid_cards = self._rule.get_valid_cards_from_obs(obs)
        action_space = np.flatnonzero(valid_cards).tolist()

        if not action_space:
            raise ValueError("No valid cards to play.")
        if len(action_space) == 1:
            return action_space[0]

        root = self._search(obs)

        # Select best action based on visit counts
        best_action = -1
        max_visits = -1
        for action, child in root.children.items():
            if child.visits > max_visits:
                max_visits = child.visits
                best_action = action
        
        return best_action

    def _search(self, obs: GameObservation) -> AlphaZeroNode:
        root = AlphaZeroNode(parent=None, prior=0.0, player_to_move=obs.player)
        
        # Initial expansion of root
        policy, _ = self.network.predict(obs)
        
        # Mask invalid actions
        valid_actions = self._rule.get_valid_cards_from_obs(obs)
        valid_mask = np.flatnonzero(valid_actions)
        masked_policy = np.zeros_like(policy)

        # Copy probabilities only for valid moves
        masked_policy[valid_mask] = policy[valid_mask]

        # Re-normalize to sum to 1
        if masked_policy.sum() > 0:
            masked_policy /= masked_policy.sum()
        else:
            # Fallback if network predicts 0 for all valid moves
            masked_policy[valid_mask] = 1.0 / len(valid_mask)
            
        root.policy_probs = masked_policy
        root.is_expanded = True

        deadline = None
        if self.config.time_limit_ms is not None:
            deadline = time.perf_counter() + (self.config.time_limit_ms / 1000.0)

        iters = 0
        while True:
            if deadline is None:
                if iters >= self.config.iterations:
                    break
            else:
                if time.perf_counter() >= deadline:
                    break

            # Determinize
            hands = self._sample_hidden_state(obs)
            state = GameStateAdapter(obs, hands)
            node = root

            # Selection
            while node.is_expanded:
                valid_actions = state.valid_actions()
                
                if not valid_actions:
                    break
                
                self._ensure_children(node, valid_actions)
                
                action, node = self._select(node, valid_actions, state.player)
                state.step(action)

            # Expansion and Evaluation
            if not state.is_terminal():
                leaf_obs = state.to_observation()
                policy, value = self.network.predict(leaf_obs)
                
                # Mask and renormalize policy for valid actions
                # Use state.valid_actions() which is already computed correctly
                valid_actions_list = state.valid_actions()
                masked_policy_leaf = np.zeros_like(policy)
                
                if valid_actions_list:
                    masked_policy_leaf[valid_actions_list] = policy[valid_actions_list]
                    if masked_policy_leaf.sum() > 0:
                        masked_policy_leaf /= masked_policy_leaf.sum()
                    else:
                        # Network predicted 0 for all valid moves, use uniform
                        masked_policy_leaf[valid_actions_list] = 1.0 / len(valid_actions_list)
                else:
                    # No valid actions but not terminal - shouldn't happen, but handle gracefully
                    # Just use uniform over all actions as fallback
                    masked_policy_leaf = np.ones_like(policy) / len(policy)
                
                if self.config.use_rollouts:
                    rollout_value_sum = 0.0
                    for _ in range(self.config.rollout_count):
                        rollout_value_sum += self._rollout(state)
                    rollout_value = rollout_value_sum / self.config.rollout_count
                    
                    # Mix network value and rollout value
                    value = (1 - self.config.rollout_mixing) * value + self.config.rollout_mixing * rollout_value
                
                node.policy_probs = masked_policy_leaf
                node.is_expanded = True
            else:
                value = state.score(state.player)

            # Backpropagation
            self._backpropagate(node, value, state.player)

            iters += 1

        return root

    def _ensure_children(self, node: AlphaZeroNode, valid_actions: List[int]):
        """Ensures that child nodes exist for all valid actions."""
        for action in valid_actions:
            if action not in node.children:
                prior = node.policy_probs[action] if node.policy_probs is not None else 0.0
                node.children[action] = AlphaZeroNode(parent=node, prior=prior, player_to_move=-1)

    def _select(self, node: AlphaZeroNode, valid_actions: List[int], player_to_move: int) -> Tuple[int, AlphaZeroNode]:
        """Selects the best child according to PUCT."""
        best_score = -np.inf
        best_action = -1
        best_child = None
        
        is_team_0 = (player_to_move % 2 == 0)

        for action in valid_actions:
            child = node.children[action]
            
            if child.visits > 0:
                q_val_team_0 = child.value_sum / child.visits
            else:
                q_val_team_0 = 0.0

            q_value = q_val_team_0 if is_team_0 else -q_val_team_0

            u_value = self.config.c_puct * child.prior * np.sqrt(node.visits) / (1 + child.visits)
            
            score = q_value + u_value
            
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        
        return best_action, best_child

    def _backpropagate(self, node: AlphaZeroNode, value: float, value_perspective_player: int):
        """
        Backpropagates the value up the tree.
        """
        team_0_value = value if (value_perspective_player % 2 == 0) else -value
        current = node
        while current is not None:
            current.visits += 1
            current.value_sum += team_0_value
            current = current.parent

    def _sample_hidden_state(self, obs: GameObservation) -> np.ndarray:
        """
        Generates a random, consistent assignment of cards for hidden hands.
        """
        # Known cards: our hand and all played cards
        known = np.zeros(36, dtype=np.int32)
        known[convert_one_hot_encoded_cards_to_int_encoded_list(obs.hand)] = 1

        for t in range(obs.nr_tricks):
            for card in obs.tricks[t]:
                if card != -1:
                    known[card] = 1
        # include current trick cards
        if obs.nr_tricks < 9 and obs.current_trick is not None:
            for k in range(obs.nr_cards_in_trick):
                card = obs.current_trick[k]
                if card != -1:
                    known[card] = 1

        unknown_cards = np.flatnonzero(1 - known).tolist()
        self._rng.shuffle(unknown_cards)

        hands = np.zeros((4, 36), dtype=np.int32)
        hands[obs.player] = obs.hand

        # Determine how many cards each opponent should have left
        # Each player starts with 9 cards. Each completed trick consumes 1 card per player.
        # In the current trick, some players may already have played.
        counts_needed = [0, 0, 0, 0]
        first = int(obs.trick_first_player[obs.nr_tricks]) if obs.nr_tricks < 9 else 0
        played_this_trick_players = set()
        for i in range(obs.nr_cards_in_trick):
            played_this_trick_players.add(int((first + i) % 4))

        for p in range(4):
            have = int(np.sum(hands[p]))
            # expected remaining cards
            expected = 9 - obs.nr_tricks - (1 if p in played_this_trick_players else 0)
            if p == obs.player:
                # sanity: our observed hand size should equal expected
                # if mismatch due to data, clamp to observed
                expected = have
            need = max(0, expected - have)
            counts_needed[p] = need

        # Assign unknown cards to other players according to required counts
        pos = 0
        for p in range(4):
            if p == obs.player:
                continue
            need = counts_needed[p]
            if need > 0:
                take = unknown_cards[pos:pos + need]
                pos += need
                if take:
                    hands[p, take] = 1

        # Any remaining unknown cards (due to mismatches) are distributed round-robin
        while pos < len(unknown_cards):
            for p in range(4):
                if p == obs.player:
                    continue
                if pos >= len(unknown_cards):
                    break
                if np.sum(hands[p]) < 9 - obs.nr_tricks - (1 if p in played_this_trick_players else 0):
                    hands[p, unknown_cards[pos]] = 1
                    pos += 1

        return hands

    def _rollout(self, state: GameStateAdapter) -> float:
        """
        Performs a random rollout from the given state and returns the score.
        """
        rollout_state = state.clone()
        while not rollout_state.is_terminal():
            actions = rollout_state.valid_actions()
            if not actions:
                break
            action = self._rng.choice(actions)
            rollout_state.step(action)
            
        return rollout_state.score(state.player)

class PytorchNetwork(NeuralNetwork):
    def __init__(self, model_path: str, hidden_dim: int = 256, num_res_blocks: int = 3):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        
        # Match the architecture used in training
        self.model = JassNet(hidden_dim=hidden_dim, num_res_blocks=num_res_blocks) 
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.to(self.device)
        self.model.eval()

    def predict(self, obs: GameObservation) -> Tuple[np.ndarray, float]:
        tensor_in = convert_obs_to_tensor(obs).to(self.device)
        
        with torch.no_grad():
            policy, value, win_prob = self.model(tensor_in)
            
        # Return value (score difference) for MCTS
        return policy.cpu().numpy()[0], value.item()

class PytorchTransformerNetwork(NeuralNetwork):
    def __init__(self, model_path: str, use_sequence_model: bool = False,
                 embed_dim: int = 128, num_heads: int = 4, ff_dim: int = 256,
                 num_layers: int = 3, dropout: float = 0.1):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        
        # Load state dict (weights only, no checkpoints)
        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        
        # Create model with matching architecture
        if use_sequence_model:
            self.model = JassTransformerSequence(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                num_layers=num_layers,
                dropout=dropout
            )
        else:
            self.model = JassTransformer(
                input_dim=332,
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                num_layers=num_layers,
                dropout=dropout
            )
        
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, obs: GameObservation) -> Tuple[np.ndarray, float]:
        tensor_in = convert_obs_to_tensor(obs).to(self.device)
        
        with torch.no_grad():
            policy, value = self.model(tensor_in)
            
        return policy.cpu().numpy()[0], value.item()

def main():
    logging.basicConfig(level=logging.INFO)

    # Load both models
    transformer_model_path = "impl/models/alpha_zero_transformer_model.pth"
    cnn_model_path = "impl/models/best_alpha_zero_model_cnn.pth"
    
    # Check which models are available
    has_transformer = os.path.exists(transformer_model_path)
    has_cnn = os.path.exists(cnn_model_path)
    
    if not has_transformer and not has_cnn:
        print(f"No trained models found. Please train at least one model first.")
        print(f"  Transformer: {transformer_model_path}")
        print(f"  CNN: {cnn_model_path}")
        return
    
    # Initialize AlphaZero config
    # Use fewer simulations since the network provides guidance
    az_config = AlphaZeroConfig(iterations=200, time_limit_ms=500)
    
    # Create agents based on available models
    if has_transformer and has_cnn and False:
        # Both models available - CNN vs Transformer
        print(f"Loading Transformer model from {transformer_model_path}...")
        transformer_network = PytorchTransformerNetwork(transformer_model_path)
        transformer_agent = AlphaZeroAgent(config=az_config, network=transformer_network)
        
        print(f"Loading CNN model from {cnn_model_path}...")
        cnn_network = PytorchNetwork(cnn_model_path)
        cnn_agent = AlphaZeroAgent(config=az_config, network=cnn_network)
        
        # Setup Arena: Transformer (Team 0) vs CNN (Team 1)
        arena = Arena(nr_games_to_play=20)
        arena.set_players(transformer_agent, cnn_agent, transformer_agent, cnn_agent)
        
        print(f"\n{'='*60}")
        print(f"Starting match: Transformer (Team 0) vs CNN (Team 1)")
        print(f"Playing {arena.nr_games_to_play} games...")
        print(f"{'='*60}\n")
        
    elif has_transformer and False:
        # Only transformer available - play against MCTS
        print(f"Loading Transformer model from {transformer_model_path}...")
        transformer_network = PytorchTransformerNetwork(transformer_model_path)
        transformer_agent = AlphaZeroAgent(config=az_config, network=transformer_network)
        
        mcts_config = MCTSConfig(iterations=400, time_limit_ms=500)
        mcts_agent = MCTSAgent(config=mcts_config)
        
        arena = Arena(nr_games_to_play=20)
        arena.set_players(transformer_agent, mcts_agent, transformer_agent, mcts_agent)
        
        print(f"\n{'='*60}")
        print(f"Starting match: Transformer (Team 0) vs MCTS (Team 1)")
        print(f"Playing {arena.nr_games_to_play} games...")
        print(f"{'='*60}\n")
        
    elif has_cnn and True:
        # Only CNN available - play against MCTS
        print(f"Loading CNN model from {cnn_model_path}...")
        cnn_network = PytorchNetwork(cnn_model_path)
        cnn_agent = AlphaZeroAgent(config=az_config, network=cnn_network)
        
        mcts_config = MCTSConfig(iterations=100, time_limit_ms=500)
        mcts_agent = MCTSAgent(config=mcts_config)
        
        arena = Arena(nr_games_to_play=20)
        arena.set_players(cnn_agent, mcts_agent, cnn_agent, mcts_agent)
        
        print(f"\n{'='*60}")
        print(f"Starting match: CNN (Team 0) vs MCTS (Team 1)")
        print(f"Playing {arena.nr_games_to_play} games...")
        print(f"{'='*60}\n")

    else:
        # Play against random agent
        if has_transformer and False:
            print(f"Loading Transformer model from {transformer_model_path}...")
            network = PytorchTransformerNetwork(transformer_model_path)
        else:
            print(f"Loading CNN model from {cnn_model_path}...")
            network = PytorchNetwork(cnn_model_path)
        az_agent = AlphaZeroAgent(config=az_config, network=network)
        random_agent = AgentRandomSchieber()
        arena = Arena(nr_games_to_play=20)
        arena.set_players(az_agent, random_agent, az_agent, random_agent)
        print(f"\n{'='*60}")
        print(f"Starting match: AlphaZero Agent (Team 0) vs Random Agent (Team 1)")
        print(f"Playing {arena.nr_games_to_play} games...")
        print(f"{'='*60}\n")
    
    arena.play_all_games()
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Team 0 Average Points: {arena.points_team_0.mean():.2f}")
    print(f"Team 1 Average Points: {arena.points_team_1.mean():.2f}")
    
    diff = arena.points_team_0.mean() - arena.points_team_1.mean()
    if diff > 0:
        print(f"\n🏆 Team 0 won by {diff:.2f} points on average!")
    elif diff < 0:
        print(f"\n🏆 Team 1 won by {-diff:.2f} points on average!")
    else:
        print(f"\n🤝 It's a tie!")
    print("="*60)


if __name__ == '__main__':
    main()
