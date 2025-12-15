import logging
from pathlib import Path

from jass.arena.arena import Arena

from ..agents.agent_mc_ml import MCTSAgent_ML, MCTSConfig
from ..agents.agent_mc_dnn import MCTSAgent_DNN



def main():
    logging.basicConfig(level=logging.WARNING)

    # setup the arena
    arena = Arena(nr_games_to_play=50)

    svm_model_path = Path(__file__).resolve().parent / "../models/svm_trump_model.pkl"
    svm_player = MCTSAgent_ML(MCTSConfig(iterations=500, time_limit_ms=250, determinization_samples=8), model_path=str(svm_model_path))

    dnn_model_path = Path(__file__).resolve().parent / "../models/dnn_trump_model_v7.pth"
    dnn_player = MCTSAgent_DNN(config=MCTSConfig(iterations=500, time_limit_ms=250, determinization_samples=8), model_path=str(dnn_model_path))

    arena.set_players(svm_player, dnn_player, svm_player, dnn_player)
    print('Playing {} games'.format(arena.nr_games_to_play))
    arena.play_all_games()
    print('Average Points SVM Player: {:.2f})'.format(arena.points_team_0.mean()))
    print('Average Points DNN Player: {:.2f})'.format(arena.points_team_1.mean()))


if __name__ == '__main__':
    main()