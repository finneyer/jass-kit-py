import logging
from jass.service.player_service_app import PlayerServiceApp
from impl.agents.agent_monte_carlo import MCTSAgent, MCTSConfig


logging.basicConfig(level=logging.DEBUG)

app = PlayerServiceApp('player_service')
config = MCTSConfig(iterations=500, time_limit_ms=250, determinization_samples=8)
app.add_player('mcts1', MCTSAgent(config))
app.add_player('mcts2', MCTSAgent(config))

