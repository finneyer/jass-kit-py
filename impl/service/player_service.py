import logging
from jass.service.player_service_app import PlayerServiceApp
from impl.agents.agent_mc_ml import MCTSAgent_ML, MCTSConfig

logging.basicConfig(level=logging.DEBUG)

app = PlayerServiceApp('player_service')
config = MCTSConfig(iterations=1000, time_limit_ms=1000)


app.add_player('mcts_ml', MCTSAgent_ML(config))

