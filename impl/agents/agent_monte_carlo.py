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
    iterations: int = 200
    time_limit_ms: Optional[int] = 150
    c_uct: float = 1.414  # Exploration constant
    rollout_depth: int = 10  # Max number of moves in a rollout
    determinization_samples: int = 8  # Reuse this many rollouts per sampled world
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
        # Simple evaluation: sample worlds and simulate with selected trump, pick best average score
        action_space = list(range(6))  # 0-3 suits, 4 Obe, 5 Une
        if obs.trump == -1 and obs.forehand == -1:
            action_space.append(PUSH)

        # If only one option, return immediately
        if len(action_space) == 1:
            return action_space[0]

        samples_per_trump = max(4, int(self.config.determinization_samples))
        best_action = action_space[0]
        best_value = -np.inf

        for trump_action in action_space:
            # Handle PUSH separately: we don't simulate with an invalid trump value
            if trump_action == PUSH:
                avg = -1e6  # simple baseline to avoid selecting PUSH by default
            else:
                total = 0.0
                count = 0
                obs_t = self._obs_with_trump(obs, trump_action)
                for _ in range(samples_per_trump):
                    hands = self._sample_hidden_state(obs_t)
                    state = GameStateAdapter(obs_t, hands)
                    reward = self._rollout(state, self.config.rollout_depth, root_player=obs.player)
                    total += reward
                    count += 1
                avg = total / max(1, count)
            if avg > best_value:
                best_value = avg
                best_action = trump_action

        return best_action

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
        Uses a time budget if configured; otherwise uses a fixed iteration budget.
        Reuses one determinized world for multiple rollouts to amortize dealing cost.
        """
        import time

        root = MCTSNode(player_to_move=obs.player, untried_actions=action_space)

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
            iters += 1

            # Sample one determinized world
            hands = self._sample_hidden_state(obs)

            # Run several rollouts on the same world
            inner_rollouts = max(1, int(self.config.determinization_samples))
            for _ in range(inner_rollouts):
                if deadline is not None and time.perf_counter() >= deadline:
                    break

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
                reward = self._rollout(state, self.config.rollout_depth, root_player=obs.player)

                # BACKPROPAGATION
                self._backpropagate(node, reward, root_player=obs.player)

        return root

    def _select(self, node: MCTSNode) -> MCTSNode:
        """
        Selects a child node to explore based on a safe UCT formula.
        Unvisited children are prioritized.
        """
        best_child = None
        best_score = -np.inf

        for child in node.children.values():
            if child.visits == 0:
                score = np.inf
            else:
                exploit = child.value / child.visits
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

    def _rollout(self, state: GameStateAdapter, depth_limit: int, root_player: int) -> float:
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
        # Score from the root player's team perspective
        return state.score(root_player)

    def _backpropagate(self, node: MCTSNode, reward: float, root_player: int):
        """
        Updates the value and visit counts of nodes up the tree.
        """
        while node is not None:
            node.visits += 1
            # reward is from the root player's team perspective
            if (node.player_to_move % 2) != (root_player % 2):
                node.value += (1.0 - reward)
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

def main():
    logging.basicConfig(level=logging.WARNING)

    # setup the arena
    arena = Arena(nr_games_to_play=10)
    player = AgentRandomSchieber()
    my_player = MCTSAgent(MCTSConfig(iterations=500, time_limit_ms=500, rollout_depth=10, determinization_samples=8))

    arena.set_players(my_player, player, my_player, player)
    print('Playing {} games'.format(arena.nr_games_to_play))
    arena.play_all_games()
    print('Average Points Team 0: {:.2f})'.format(arena.points_team_0.mean()))
    print('Average Points Team 1: {:.2f})'.format(arena.points_team_1.mean()))


if __name__ == '__main__':
    main()
