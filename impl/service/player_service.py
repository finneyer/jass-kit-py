import logging
from jass.service.player_service_app import PlayerServiceApp
from jass.agents.agent_random_schieber import AgentRandomSchieber
from impl.agents.agent_monte_carlo import MCTSAgent, MCTSConfig

def create_app():
    """
    This is the factory method for flask. It is automatically detected when flask is run, but we must tell flask
    what python file to use:

        export FLASK_APP=player_service.py
        export FLASK_ENV=development
        flask run --host=0.0.0.0 --port=8888
    """
    logging.basicConfig(level=logging.DEBUG)

    # create and configure the app
    app = PlayerServiceApp('player_service')

    # add some players
    app.add_player('mcts1', MCTSAgent(MCTSConfig()))
    app.add_player('mcts2', MCTSAgent(MCTSConfig()))

    return app


if __name__ == '__main__':
   app = create_app()
   app.run(host='0.0.0.0', port=8888)

