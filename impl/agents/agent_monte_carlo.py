from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Tuple

import numpy as np

from jass.agents.agent import Agent
from jass.game.const import PUSH, next_player, partner_player, card_values
from jass.game.game_observation import GameObservation
from jass.game.game_rule import GameRule
from jass.game.game_util import convert_one_hot_encoded_cards_to_int_encoded_list, \
    convert_int_encoded_cards_to_str_encoded
from jass.game.rule_schieber import RuleSchieber
from jass.agents.agent_random_schieber import AgentRandomSchieber
from jass.arena.arena import Arena
import logging


@dataclass
class MCTSConfig:
    """Configuration for the MCTS agent."""
    # TODO: Finalize and document configuration options
    iterations: int = 1000
    time_limit_ms: Optional[int] = None
    c_uct: float = 1.414  # Exploration constant
    rollout_depth: int = 24  # Max number of moves in a rollout
    determinization_samples: int = 16  # Number of random worlds to sample
    use_priors: bool = True  # Whether to use priors from a heuristic (e.g., RuleBasedAgent)


@dataclass
class MCTSNode:
    """A node in the Monte Carlo search tree."""
    # TODO: Implement state_key generation (e.g., Zobrist hashing) for transposition table
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
        
        # check if a team made the match
        if self.points[player_team] > 0 and self.points[opponent_team] == 0:
            score += 100
        elif self.points[opponent_team] > 0 and self.points[player_team] == 0:
            score -= 100
            
        return score / 257.0


class MCTSAgent(Agent):
    """
    An agent that uses Monte Carlo Tree Search to decide its actions.
    """

    def __init__(self, config: Optional[MCTSConfig] = None):
        """
        Initializes the MCTS agent.
        """
        super().__init__()
        self._rule = RuleSchieber()
        self.config = config or MCTSConfig()
        self.rollout_policy = AgentRandomSchieber()
        self._rng = np.random.default_rng()

    def action_trump(self, obs: GameObservation) -> int:
        """
        Determine trump using MCTS. The action space includes suits, obenabe, uneufe, and push.
        """
        action_space = list(range(6)) # 0-3 for suits, 4 for Obe, 5 for Une
        # A player can push if it is their turn to select trump and the first player to choose has not yet made a decision.
        if obs.trump == -1 and obs.forehand == -1:
            action_space.append(PUSH)

        root = self._search(obs, action_space)
        
        # choose action with highest value
        best_action = -1
        max_value = -np.inf
        for action, child in root.children.items():
            if child.value > max_value:
                max_value = child.value
                best_action = action
        return best_action

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
        root = MCTSNode(player_to_move=obs.player, untried_actions=action_space)

        for _ in range(self.config.iterations):
            # Create a determinized world
            hands = self._sample_hidden_state(obs)
            state = GameStateAdapter(obs, hands)

            node = root
            
            # SELECTION
            while not node.untried_actions and node.children:
                node = self._select(node)
                state.step(node.action)

            # EXPANSION
            if node.untried_actions:
                action = self._rng.choice(node.untried_actions)
                state.step(action)
                node = self._expand(node, action, state.player)

            # SIMULATION (Rollout)
            reward = self._rollout(state, self.config.rollout_depth)

            # BACKPROPAGATION
            self._backpropagate(node, reward)
            
        return root

    def _select(self, node: MCTSNode) -> MCTSNode:
        """
        Selects a child node to explore based on the UCT formula.
        """
        best_child = None
        best_score = -np.inf

        for child in node.children.values():
            exploit = child.value / child.visits
            explore = np.sqrt(np.log(node.visits) / child.visits)
            score = exploit + self.config.c_uct * explore
            
            # In MCTS, the value is from the perspective of the parent node's player.
            # If the current node's player is not on the same team, we should negate the value.
            if (node.player_to_move % 2) != (child.player_to_move % 2):
                score = (1 - child.value) / child.visits + self.config.c_uct * explore

            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def _expand(self, node: MCTSNode, action: int, next_player: int) -> MCTSNode:
        """
        Expands the tree by creating a new child node.
        """
        node.untried_actions.remove(action)
        new_node = MCTSNode(parent=node, action=action, player_to_move=next_player)
        node.children[action] = new_node
        return new_node

    def _rollout(self, state: GameStateAdapter, depth_limit: int) -> float:
        """
        Simulates a game from the current state to a terminal state or depth limit.
        """
        for _ in range(depth_limit):
            if state.is_terminal():
                break
            
            hand = state.hands[state.player]
            valid_cards = self._rule.get_valid_cards(hand, state.current_trick, state.nr_cards_in_trick, state.trump)
            
            valid_indices = np.flatnonzero(valid_cards)
            if len(valid_indices) == 0:
                # This can happen if the determinization is inconsistent with the game rules.
                # In this case, we stop the rollout and return the current score.
                break
            
            action = self._rng.choice(valid_indices)
            state.step(action)
            
        return state.score(state.player)

    def _backpropagate(self, node: MCTSNode, reward: float):
        """
        Updates the value and visit counts of nodes up the tree.
        """
        while node is not None:
            node.visits += 1
            # The reward is from the perspective of the player who made the move.
            # We need to update the value from the perspective of the parent node's player.
            if (node.parent is not None) and ((node.parent.player_to_move % 2) != (node.player_to_move % 2)):
                 node.value += (1-reward)
            else:
                node.value += reward
            node = node.parent

    def _sample_hidden_state(self, obs: GameObservation) -> np.ndarray:
        """
        Generates a random, consistent assignment of cards for hidden hands.
        """
        # 1. Get all cards not in player's hand and not in trick history.
        all_cards = np.ones(36, dtype=np.int32)
        all_cards[convert_one_hot_encoded_cards_to_int_encoded_list(obs.hand)] = 0
        
        for trick in obs.tricks:
            for card in trick:
                if card != -1:
                    all_cards[card] = 0
        
        unknown_cards = np.flatnonzero(all_cards).tolist()
        self._rng.shuffle(unknown_cards)
        
        # 2. Shuffle and deal them to the other players.
        hands = np.zeros((4, 36), dtype=np.int32)
        hands[obs.player] = obs.hand
        
        other_players = [p for p in range(4) if p != obs.player]
        
        cards_per_player = (36 - np.sum(obs.hand)) // 3
        
        for i, player_idx in enumerate(other_players):
            start = i * cards_per_player
            end = start + cards_per_player
            player_cards = unknown_cards[start:end]
            hands[player_idx, player_cards] = 1
            
        return hands

def main():
    logging.basicConfig(level=logging.WARNING)

    # setup the arena
    arena = Arena(nr_games_to_play=100)
    player = AgentRandomSchieber()
    my_player = MCTSAgent()

    arena.set_players(my_player, player, my_player, player)
    print('Playing {} games'.format(arena.nr_games_to_play))
    arena.play_all_games()
    print('Average Points Team 0: {:.2f})'.format(arena.points_team_0.mean()))
    print('Average Points Team 1: {:.2f})'.format(arena.points_team_1.mean()))


if __name__ == '__main__':
    main()
