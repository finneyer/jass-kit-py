import logging
from jass.service.player_service_app import PlayerServiceApp
from impl.agents.agent_mc_ml import MCTSAgent_ML, MCTSConfig

logging.basicConfig(level=logging.DEBUG)

app = PlayerServiceApp('player_service')

config_1 = MCTSConfig(iterations=1000, time_limit_ms=1000)
config_2 = MCTSConfig(iterations=500, time_limit_ms=250)


app.add_player('mcts_ml_final', MCTSAgent_ML(config_1))
app.add_player('mcts_ml_midterm', MCTSAgent_ML(config_2))

