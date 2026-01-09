import copy
import logging
import numpy as np
from jass.agents.agent_cheating import AgentCheating
from jass.game.game_state import GameState
from jass.game.rule_schieber import RuleSchieber
from jass.game.game_sim import GameSim
from jass.agents.agent_cheating_random_schieber import AgentCheatingRandomSchieber
from jass.arena.arena import Arena
from jass.game.game_util import *
from jass.game.const import *

class AgentCheatingMinimaxNTricks(AgentCheating):
    def __init__(self, n_tricks=1):
        self.rule = RuleSchieber()
        self.n_tricks = n_tricks

    def action_trump(self, state: GameState) -> int:
        hand = state.hands[state.player, :]
        card_list = convert_one_hot_encoded_cards_to_int_encoded_list(hand)

        trump_scores = [
            self.calculate_trump_selection_score(card_list, current_trump)
            for current_trump in range(4)
        ]
        best_trump = int(np.argmax(trump_scores))
        best_score = trump_scores[best_trump]

        if self.can_push(state) and best_score < 68:
            return PUSH
        else:
            return best_trump

    def can_push(self, state: GameState) -> bool:
        return state.forehand == -1

    def calculate_trump_selection_score(self, hand: list[int], trump: int) -> int:
        trump_score = [15, 10, 7, 25, 6, 19, 5, 5, 5]
        no_trump_score = [9, 7, 5, 2, 1, 0, 0, 0, 0]
        score = 0
        for card in hand:
            color = card // 9
            offset = card % 9
            if color == trump:
                score += trump_score[offset]
            else:
                score += no_trump_score[offset]
        return score

    def action_play_card(self, state: GameState) -> int:
        valid_cards = np.flatnonzero(self.rule.get_valid_cards_from_state(state))
        best_card = None
        best_value = -float('inf')

        for card in valid_cards:
            value = self.minimax(
                state,
                card,
                depth=0,
                start_trick_count=state.nr_tricks,
                alpha=-float('inf'),
                beta=float('inf'),
            )
            if value > best_value:
                best_value = value
                best_card = card

        return best_card

    def minimax(self,state: GameState,card_to_play: int,depth: int,start_trick_count=None,alpha=-float('inf'),beta=float('inf')) -> int:
        sim = GameSim(self.rule)
        sim.init_from_state(copy.deepcopy(state))
        sim.action_play_card(card_to_play)
        new_state = sim.state

        if start_trick_count is None:
            start_trick_count = state.nr_tricks

        tricks_played = new_state.nr_tricks - start_trick_count

        if tricks_played >= self.n_tricks or new_state.nr_played_cards == 36:
            return self.evaluate_heuristic(new_state, state.player)

        next_player = new_state.player
        valid_cards_next = np.flatnonzero(self.rule.get_valid_cards_from_state(new_state))

        is_maximizing = (next_player % 2) == (state.player % 2)

        if is_maximizing:
            value = -float('inf')
            for next_card in valid_cards_next:
                cand = self.minimax(new_state, next_card, depth + 1, start_trick_count, alpha, beta)
                value = max(value, cand)
                alpha = max(alpha, value)
                if beta <= alpha:
                    break
            return value
        else:
            value = float('inf')
            for next_card in valid_cards_next:
                cand = self.minimax(new_state, next_card, depth + 1, start_trick_count, alpha, beta)
                value = min(value, cand)
                beta = min(beta, value)
                if beta <= alpha:
                    break
            return value

    def evaluate_score(self, state: GameState, my_player_id: int, start_trick_count: int) -> int:
        my_team = my_player_id % 2
        total_gain = 0
        for i in range(start_trick_count, state.nr_tricks):
            trick_winner = state.trick_winner[i]
            trick_points = state.trick_points[i]
            winning_team = trick_winner % 2
            if winning_team == my_team:
                total_gain += trick_points
            else:
                total_gain -= trick_points
        return total_gain

    def evaluate_heuristic(self, state: GameState, my_player_id: int) -> float:
        """
        Heuristic to steer towards winning the full round.
        - point_diff: rewards current lead
        - tricks_won: rewards control
        - trumps_left: only when a suit trump is active
        - hand_value: sum of card_values of remaining cards under current trump mode
        """
        my_team = my_player_id % 2
        opp_team = 1 - my_team

        point_diff = int(state.points[my_team]) - int(state.points[opp_team])

        tricks_won = sum(
            1 for w in state.trick_winner[:state.nr_tricks]
            if w != -1 and (w % 2) == my_team
        )

        hand = state.hands[state.player, :]
        if state.trump is not None and 0 <= state.trump < 4:
            trumps_left = int(np.sum(hand * color_masks[state.trump, :]))
        else:
            trumps_left = 0

        hand_cards = np.flatnonzero(hand)
        hand_value = float(np.sum(card_values[state.trump, hand_cards])) if hand_cards.size > 0 else 0.0

        return (1.0 * point_diff) + (8.0 * tricks_won) + (2.0 * trumps_left) + (0.3 * hand_value)



def main():
    logging.basicConfig(level=logging.WARNING)

    arena = Arena(nr_games_to_play=30, save_filename='arena_games', cheating_mode=True)
    random_cheating_player = AgentCheatingRandomSchieber()
    cheating_minimax_player = AgentCheatingMinimaxNTricks(n_tricks=2)

    arena.set_players(cheating_minimax_player, random_cheating_player, cheating_minimax_player, random_cheating_player)
    print('Playing {} games'.format(arena.nr_games_to_play))
    arena.play_all_games()
    print('Average Points Team 0: {:.2f})'.format(arena.points_team_0.mean()))
    print('Average Points Team 1: {:.2f})'.format(arena.points_team_1.mean()))


if __name__ == '__main__':
    main()