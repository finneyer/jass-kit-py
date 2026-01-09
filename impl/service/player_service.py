import logging
from jass.service.player_service_app import PlayerServiceApp
from impl.agents.agent_mc_ml import MCTSAgent_ML, MCTSConfig
from impl.agents.agent_mc_dnn import MCTSAgent_DNN

logging.basicConfig(level=logging.DEBUG)

app = PlayerServiceApp('player_service')

config_1 = MCTSConfig(iterations=2500, time_limit_ms=2500)

app.add_player('mcts_ml_final', MCTSAgent_ML(config_1))
app.add_player('mcts_dnn', MCTSAgent_DNN(config_1))
