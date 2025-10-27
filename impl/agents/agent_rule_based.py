from jass.agents.agent import Agent
from jass.game.game_observation import GameObservation
from jass.game.game_util import *
from jass.game.const import *
from jass.game.rule_schieber import RuleSchieber
from jass.agents.agent_random_schieber import AgentRandomSchieber
from jass.arena.arena import Arena
import logging

class RuleBasedAgent(Agent):

    def __init__(self):
        self.rule = RuleSchieber()

    def action_trump(self, obs: GameObservation):
        hand = obs.hand
        card_list = convert_one_hot_encoded_cards_to_int_encoded_list(hand)
        trump_scores = [self.calculate_trump_selection_score(card_list, current_trump) for current_trump in range (4)]
        best_trump = np.argmax(trump_scores)
        best_score = trump_scores[best_trump]

        if self.can_push(obs) and best_score < 68:
            return PUSH
        else:
            return best_trump

    def action_play_card(self, obs: GameObservation):

        valid_cards = self.rule.get_valid_cards_from_obs(obs)
        valid_indices = np.flatnonzero(valid_cards)
        current_trick = obs.current_trick  # cards played so far in the current trick

        # Determine if we are leading
        if np.sum(current_trick >= 0) == 0:
            return self._play_leading_card(obs, valid_indices)
        else:
            return self._play_following_card(obs, valid_indices)

    def _play_leading_card(self, obs: GameObservation, valid_indices):
        hand = convert_one_hot_encoded_cards_to_int_encoded_list(obs.hand)
        trump = obs.trump

        # Define approximate strength ranking (lower = stronger)
        # Order in each suit is 0..8 -> [6, 7, 8, 9, 10, J, Q, K, A]
        # Trump has different order, so we treat it separately
        normal_strength = [0, 1, 2, 3, 4, 5, 6, 7, 8]  # base order, higher index = stronger card
        trump_strength = [4, 5, 6, 7, 8, 1, 3, 2, 0]  # roughly: Jack strongest, then 9, then A, K, Q, 10...

        # Count cards per suit
        suit_counts = [0, 0, 0, 0]
        for card in hand:
            suit_counts[card // 9] += 1

        # --- Case 1: many trumps
        trumps_in_hand = [c for c in valid_indices if c // 9 == trump]
        if len(trumps_in_hand) >= 3:
            # Play the strongest trump
            best_card = min(trumps_in_hand, key=lambda c: trump_strength[c % 9])
            return best_card

        # --- Case 2: choose a long suit
        longest_suit = int(np.argmax(suit_counts))
        long_suit_cards = [c for c in valid_indices if c // 9 == longest_suit]
        if long_suit_cards:
            # Play strongest card in that suit
            best_card = max(long_suit_cards, key=lambda c: normal_strength[c % 9])
            return best_card

        # --- Fallback: just play highest card overall
        best_card = max(valid_indices, key=lambda c: normal_strength[c % 9])
        return best_card

    def _play_following_card(self, obs: GameObservation, valid_indices):
        trump = obs.trump
        current_trick = obs.current_trick
        hand = convert_one_hot_encoded_cards_to_int_encoded_list(obs.hand)

        # Card strength tables
        normal_strength = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        trump_strength = {0: 8, 1: 7, 2: 6, 3: 5, 4: 4, 5: 0, 6: 3, 7: 2, 8: 1}

        # Find lead card and suit
        first_card = next((c for c in current_trick if c != -1), None)
        if first_card is None:
            # Safety fallback (shouldn't happen)
            return np.random.choice(valid_indices)

        lead_suit = first_card // 9

        # Determine already played cards
        played_cards = [c for c in current_trick if c != -1]
        nr_played = len(played_cards)

        # Determine who is winning (approximation)
        def card_strength(card):
            color = card // 9
            offset = card % 9
            if color == trump:
                return 100 + (9 - trump_strength[offset])  # trump cards highest
            elif color == lead_suit:
                return 50 + offset
            else:
                return offset  # irrelevant suit

        current_winner = max(played_cards, key=card_strength)
        winning_strength = card_strength(current_winner)

        # Determine teammate index
        # We can compute positions relative to obs.player and obs.trick_first_player
        # Since this is from current player's perspective, we know our position in the trick
        player_pos_in_trick = nr_played  # 0-based in this trick
        teammate_pos_in_trick = (player_pos_in_trick - 2) % 4  # teammate plays 2nd after you in rotation

        # Check if teammate is currently winning (approximation)
        teammate_card = current_trick[teammate_pos_in_trick] if teammate_pos_in_trick < len(current_trick) else -1
        teammate_winning = (teammate_card == current_winner)

        # --- Case 1: Can follow suit
        follow_suit_cards = [c for c in valid_indices if c // 9 == lead_suit]
        if follow_suit_cards:
            if teammate_winning:
                # Save strong cards
                return min(follow_suit_cards, key=lambda c: normal_strength[c % 9])
            else:
                # Can we beat the current winner?
                better_cards = [c for c in follow_suit_cards if card_strength(c) > winning_strength]
                if better_cards:
                    return min(better_cards, key=lambda c: normal_strength[c % 9])
                else:
                    return min(follow_suit_cards, key=lambda c: normal_strength[c % 9])

        # --- Case 2: Cannot follow suit
        trumps_in_hand = [c for c in valid_indices if c // 9 == trump]
        if trumps_in_hand:
            better_trumps = [c for c in trumps_in_hand if card_strength(c) > winning_strength]
            if better_trumps:
                return min(better_trumps, key=lambda c: trump_strength[c % 9])
            elif teammate_winning:
                return min(trumps_in_hand, key=lambda c: trump_strength[c % 9])
            else:
                # Can't win and teammate not winning → discard lowest trump
                return min(trumps_in_hand, key=lambda c: trump_strength[c % 9])
        else:
            # No trumps → discard lowest card overall
            return min(valid_indices, key=lambda c: normal_strength[c % 9])


    def can_push(self, obs: GameObservation):
        return obs.trump == -1 and obs.player == next_player[obs.dealer]

    def calculate_trump_selection_score(self, hand: list[int], trump: int) -> int:
        """
        hand: list of 9 integers in [0..35]
        trump: 0–3 = suit, 4 = obenabe, 5 = uneufe
        """
        # score if the color is trump
        trump_score = [15, 10, 7, 25, 6, 19, 5, 5, 5]
        # score if the color is not trump
        no_trump_score = [9, 7, 5, 2, 1, 0, 0, 0, 0]
        # score if obenabe is selected (all colors)
        obenabe_score = [14, 10, 8, 7, 5, 0, 5, 0, 0,]
        # score if uneufe is selected (all colors)
        uneufe_score = [0, 2, 1, 1, 5, 5, 7, 9, 11]


        score = 0
        for card in hand:
            color = card // 9     # 0–3
            offset = card % 9     # 0–8

            if trump == 4:
                score_table = obenabe_score
            elif trump == 5:
                score_table = uneufe_score
            elif color == trump:
                score_table = trump_score
            else:
                score_table = no_trump_score

            score += score_table[offset]

        return score

def main():
    logging.basicConfig(level=logging.WARNING)

    # setup the arena
    arena = Arena(nr_games_to_play=1000, save_filename='arena_games')
    random_player = AgentRandomSchieber()
    rule_based_player = RuleBasedAgent()

    arena.set_players(rule_based_player, random_player, rule_based_player, random_player)
    print('Playing {} games'.format(arena.nr_games_to_play))
    arena.play_all_games()
    print('Average Points Team 0: {:.2f})'.format(arena.points_team_0.mean()))
    print('Average Points Team 1: {:.2f})'.format(arena.points_team_1.mean()))


if __name__ == '__main__':
    main()