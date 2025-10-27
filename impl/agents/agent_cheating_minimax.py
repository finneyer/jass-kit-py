import copy
import logging
import numpy as np
from jass.agents.agent_cheating import AgentCheating
from jass.game.game_state import GameState
from jass.game.rule_schieber import RuleSchieber
from jass.game.game_sim import GameSim
from jass.agents.agent_random_schieber import AgentRandomSchieber
from jass.agents.agent_cheating_random_schieber import AgentCheatingRandomSchieber
from jass.arena.arena import Arena

class AgentCheatingMinimax(AgentCheating):
    def __init__(self):
        self.rule = RuleSchieber()

    def action_trump(self, state: GameState) -> int:
        return np.random.randint(0, 4)


    def action_play_card(self, state: GameState) -> int:
        valid_cards = np.flatnonzero(self.rule.get_valid_cards_from_state(state))

        best_card = None
        best_value = -float('inf')

        for card in valid_cards:
            value = self.minimax(state, card, depth=0)
            if value > best_value:
                best_value = value
                best_card = card

        return best_card

    def minimax(self, state: GameState, card_to_play: int, depth: int) -> int:

        sim = GameSim(self.rule)
        sim.init_from_state(copy.deepcopy(state))
        sim.action_play_card(card_to_play)
        new_state = sim.state

        if new_state.nr_cards_in_trick == 0 and new_state.nr_tricks > state.nr_tricks:
            return self.evaluate_trick(new_state, state.player)

        next_player = new_state.player
        valid_cards_next = np.flatnonzero(self.rule.get_valid_cards_from_state(new_state))

        scores = []
        for next_card in valid_cards_next:
            scores.append(self.minimax(new_state, next_card, depth + 1))

        # Decide if we maximize or minimize depending on team
        if next_player % 2 == state.player % 2:
            return max(scores)
        else:
            return min(scores)

    def evaluate_trick(self, state, my_player_id):
        trick_index = state.nr_tricks - 1
        winner = state.trick_winner[trick_index]
        points = state.trick_points[trick_index]

        my_team = my_player_id % 2
        winning_team = winner % 2

        return points if winning_team == my_team else -points



def main():
    logging.basicConfig(level=logging.WARNING)

    # setup the arena
    arena = Arena(nr_games_to_play=1000, save_filename='arena_games', cheating_mode=True)
    random_cheating_player = AgentCheatingRandomSchieber()
    cheating_minimax_player = AgentCheatingMinimax()

    arena.set_players(cheating_minimax_player, random_cheating_player, cheating_minimax_player, random_cheating_player)
    print('Playing {} games'.format(arena.nr_games_to_play))
    arena.play_all_games()
    print('Average Points Team 0: {:.2f})'.format(arena.points_team_0.mean()))
    print('Average Points Team 1: {:.2f})'.format(arena.points_team_1.mean()))


if __name__ == '__main__':
    main()