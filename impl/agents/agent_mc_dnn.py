from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Tuple

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from jass.agents.agent import Agent
from jass.game.const import PUSH, next_player
from jass.game.game_observation import GameObservation
from jass.game.game_util import convert_one_hot_encoded_cards_to_int_encoded_list
from jass.game.rule_schieber import RuleSchieber
from jass.agents.agent_random_schieber import AgentRandomSchieber
from jass.arena.arena import Arena
import logging

from pathlib import Path

# --- Configuration Constants and Device Setup for DNN ---
INPUT_SIZE = 37
OUTPUT_SIZE = 7
HIDDEN_LAYER_1 = 256
HIDDEN_LAYER_2 = 128
HIDDEN_LAYER_3 = 64
DROPOUT_RATE = 0.3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

class TrumpSelectionDNN(nn.Module):
    """
    Multilayer Perceptron for Jass trump selection.
    """

    def __init__(self):
        super(TrumpSelectionDNN, self).__init__()
        self.fc1 = nn.Linear(INPUT_SIZE, HIDDEN_LAYER_1)
        self.fc2 = nn.Linear(HIDDEN_LAYER_1, HIDDEN_LAYER_2)
        self.fc3 = nn.Linear(HIDDEN_LAYER_2, HIDDEN_LAYER_3)
        self.fc4 = nn.Linear(HIDDEN_LAYER_3, OUTPUT_SIZE)

        self.dropout = nn.Dropout(p=DROPOUT_RATE)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return x


@dataclass
class MCTSConfig:
    """Configuration for the MCTS agent."""
    iterations: int = 200
    time_limit_ms: Optional[int] = 150
    c_uct: float = 1.414  # Exploration constant
    rollout_depth: int = 10  # Max number of moves in a rollout
    determinization_samples: int = 8  # Reuse this many rollouts per sampled world
    use_priors: bool = True  # Whether to use priors from a heuristic (e.g., RuleBasedAgent)


@dataclass
class MCTSNode:
    """A node in the Monte Carlo search tree."""
    parent: Optional[MCTSNode] = None
    children: Dict[int, MCTSNode] = field(default_factory=dict)  # action -> MCTSNode
    visits: int = 0
    value: float = 0.0  # Mean value from the perspective of the player to move
    prior: float = 0.0  # Prior probability of selecting this node's action
    action: Optional[int] = None  # The action that led to this node
    player_to_move: int = -1
    untried_actions: List[int] = field(default_factory=list)
    state_key: Optional[Tuple | int] = None  # For transposition table


class GameStateAdapter:
    """
    A lightweight adapter for the Jass game state to be used in MCTS.
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
            self.trick_winner[self.nr_tricks] = self._rule.calc_winner(self.current_trick,
                                                                       self.trick_first_player[self.nr_tricks],
                                                                       self.trump)
            self.trick_points[self.nr_tricks] = self._rule.calc_points(self.current_trick, self.nr_tricks == 8,
                                                                       self.trump)

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


class MCTSAgent_DNN(Agent):
    """
    An agent that uses Monte Carlo Tree Search for card play and a DNN for trump selection.
    """

    def __init__(self, config: Optional[MCTSConfig] = None, model_path: str = None):
        """
        Initializes the MCTS agent and loads the DNN model.
        """
        super().__init__()
        self._rule = RuleSchieber()
        self.config = config or MCTSConfig()
        self.rollout_policy = AgentRandomSchieber()
        self._rng = np.random.default_rng()

        base_dir = Path(__file__).resolve().parent

        # --- DNN Model Loading ---
        if model_path is None:
            # Default path to your trained model
            model_path = base_dir / "../models/dnn_trump_model_v7.pth"

        # 1. Instantiate the model
        self.trump_model = TrumpSelectionDNN()

        # 2. Load the trained weights
        # IMPORTANT: map_location ensures it loads correctly regardless of CPU/GPU
        try:
            self.trump_model.load_state_dict(
                torch.load(model_path, map_location=torch.device('cpu'))
            )
        except Exception as e:
            print(f"Error loading DNN model from {model_path}: {e}")
            print("Please ensure the model file exists at the specified path.")
            # Fallback or error handling can be added here

        # 3. Set model to evaluation mode (crucial for Dropout layers)
        self.trump_model.eval()
        # --- End DNN Model Loading ---


    def action_trump(self, obs: GameObservation) -> int:
        """
        Predict the trump using the trained DNN model, correctly mapping the PUSH index
        and adhering to the forehand/rearhand rules.
        """
        # ... (Input preparation) ...
        hand_vector = np.array(obs.hand, dtype=np.float32)
        forehand_feature = np.array([1.0 if obs.forehand == 1 else 0.0], dtype=np.float32)

        X_input_np = np.concatenate([hand_vector, forehand_feature]).reshape(1, -1)
        X_input_tensor = torch.from_numpy(X_input_np).float()

        # 1. Make prediction (Output index 0-6)
        with torch.no_grad():
            Y_logits = self.trump_model(X_input_tensor)
            predicted_label_index = torch.argmax(Y_logits, dim=1).item()

        # 2. Map the predicted index (0-6) to the Jass constant (0-5, or 10)

        DNN_PUSH_INDEX = 6  # The index your model was trained to output for PUSH

        trump_value = 0  # Default to a safe, legal choice (DIAMONDS)

        # Case A: Model predicts a standard trump (0-5)
        if predicted_label_index != DNN_PUSH_INDEX:
            trump_value = predicted_label_index  # Directly 0-5. This is always legal.

        # Case B: Model predicts PUSH (index 6)
        else:
            # Subcase B1: It's the first call (forehand has not declared)
            if obs.forehand == -1:
                # The arena expects the forehand player's action.
                # PUSH (10) IS legal as a move, though it passes the right to the partner.
                trump_value = PUSH  # Returns 10. Legal.

            # Subcase B2: It's the second call (forehand passed, obs.forehand == 0)
            elif obs.forehand == 0:
                # The player is now the rearhand player (partner). The rule mandates a non-PUSH choice (0-5).
                # The DNN's PUSH prediction must be overridden to a non-PUSH value (e.g., DIAMONDS=0).
                trump_value = 0  # Cannot return PUSH (10). Illegal move, must choose 0-5.

            # Subcase B3: Trump was already declared (obs.forehand == 1, should not happen in trump selection phase)
            elif obs.forehand == 1:
                # This state implies trump was already declared. If this code runs, it's an API misuse.
                # Safest is to choose 0.
                trump_value = 0

        return trump_value

    def _obs_with_trump(self, obs: GameObservation, trump_action: int) -> GameObservation:
        """Create a shallow copy of obs with the given trump set (used for trump evaluation)."""
        new_obs = GameObservation()
        # Basic fields
        new_obs.dealer = int(obs.dealer)
        new_obs.player = int(obs.player)
        new_obs.player_view = int(obs.player_view)
        new_obs.trump = int(trump_action)
        new_obs.forehand = int(obs.forehand)
        new_obs.declared_trump = int(obs.declared_trump)
        # Copy arrays
        new_obs.hand = np.array(obs.hand, copy=True)
        new_obs.tricks = np.array(obs.tricks, copy=True)
        new_obs.trick_winner = np.array(obs.trick_winner, copy=True)
        new_obs.trick_points = np.array(obs.trick_points, copy=True)
        new_obs.trick_first_player = np.array(obs.trick_first_player, copy=True)
        new_obs.current_trick = None if obs.current_trick is None else np.array(obs.current_trick, copy=True)
        new_obs.nr_tricks = int(obs.nr_tricks)
        new_obs.nr_cards_in_trick = int(obs.nr_cards_in_trick)
        new_obs.nr_played_cards = int(obs.nr_played_cards)
        new_obs.points = np.array(obs.points, copy=True)
        return new_obs

    def action_play_card(self, obs: GameObservation) -> int:
        """
        Determine the card to play using MCTS.
        """
        valid_cards = self._rule.get_valid_cards_from_obs(obs)
        action_space = np.flatnonzero(valid_cards).tolist()

        if not action_space:
            raise ValueError("No valid cards to play.")
        if len(action_space) == 1:
            return action_space[0]

        root = self._search(obs, action_space)

        # choose action with most visits
        best_action = -1
        max_visits = -1
        for action, child in root.children.items():
            if child.visits > max_visits:
                max_visits = child.visits
                best_action = action
        return best_action

    def _search(self, obs: GameObservation, action_space: List[int]) -> MCTSNode:
        """
        Runs the main MCTS algorithm for a given number of iterations or time budget.
        """
        import time

        root = MCTSNode(player_to_move=obs.player, untried_actions=action_space)

        deadline = None
        if self.config.time_limit_ms is not None:
            deadline = time.perf_counter() + (self.config.time_limit_ms / 1000.0)

        iters = 0
        while True:
            # Budget check
            if deadline is None:
                if iters >= self.config.iterations:
                    break
            else:
                if time.perf_counter() >= deadline:
                    break

            # One determinization per full simulation
            hands = self._sample_hidden_state(obs)

            # INIT STATE FOR THIS SIMULATION
            state = GameStateAdapter(obs, hands)
            node = root

            # SELECTION
            while not node.untried_actions and node.children:
                node = self._select(node)
                state.step(node.action)

            # EXPANSION
            if node.untried_actions:
                action = int(self._rng.choice(node.untried_actions))
                state.step(action)
                node = self._expand(node, action, state)

            # SIMULATION (Rollout)
            reward = self._rollout(state, root_player=obs.player)

            # BACKPROPAGATION
            self._backpropagate(node, reward, root_player=obs.player)

            # Count this completed simulation
            iters += 1

        return root

    def _select(self, node: MCTSNode) -> MCTSNode:
        """
        Selects a child node to explore based on a safe UCT formula.
        """
        best_child = None
        best_score = -np.inf

        for child in node.children.values():
            if child.visits == 0:
                score = np.inf
            else:
                # Convert child's value to current node player's team perspective for selection.
                same_team = (child.player_to_move % 2) == (node.player_to_move % 2)
                exploit_raw = child.value / child.visits
                exploit = exploit_raw if same_team else -exploit_raw
                explore = np.sqrt(np.log(max(1, node.visits)) / child.visits)
                score = exploit + self.config.c_uct * explore
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def _expand(self, node: MCTSNode, action: int, state: GameStateAdapter) -> MCTSNode:
        """
        Expands the tree by creating a new child node.
        """
        node.untried_actions.remove(action)
        new_node = MCTSNode(parent=node, action=action, player_to_move=state.player)
        # Initialize child untried actions from the current state
        new_node.untried_actions = state.valid_actions()
        node.children[action] = new_node
        return new_node

    def _rollout(self, state: GameStateAdapter, root_player: int) -> float:
        """
        Simulate to the end of the game and return the final normalized score
        from the root player's team perspective.
        """
        # Continue until terminal state (full game simulated)
        while not state.is_terminal():
            hand = state.hands[state.player]
            valid_cards = self._rule.get_valid_cards(hand, state.current_trick, state.nr_cards_in_trick, state.trump)
            valid_indices = np.flatnonzero(valid_cards)
            if len(valid_indices) == 0:
                # Determinization inconsistency: stop and evaluate current score
                break
            action = self._rng.choice(valid_indices)
            state.step(action)
        return state.score(root_player)

    def _backpropagate(self, node: MCTSNode, reward: float, root_player: int):
        """
        Updates the value and visit counts of nodes up the tree.
        """
        while node is not None:
            node.visits += 1
            # reward is from the root player's team perspective
            if (node.player_to_move % 2) != (root_player % 2):
                node.value -= reward
            else:
                node.value += reward
            node = node.parent

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


def main():
    logging.basicConfig(level=logging.WARNING)

    # setup the arena
    arena = Arena(nr_games_to_play=25)
    player = AgentRandomSchieber()

    # Instantiate your custom agent pointing to the trained DNN model path
    model_path = Path(__file__).resolve().parent / "../models/dnn_trump_model_v7.pth"

    my_player = MCTSAgent_DNN(
        config=MCTSConfig(iterations=500, time_limit_ms=250, determinization_samples=8),
        model_path=str(model_path)
    )

    arena.set_players(my_player, player, my_player, player)
    print('Playing {} games'.format(arena.nr_games_to_play))
    arena.play_all_games()
    print('Average Points Team 0: {:.2f})'.format(arena.points_team_0.mean()))
    print('Average Points Team 1: {:.2f})'.format(arena.points_team_1.mean()))


if __name__ == '__main__':
    main()