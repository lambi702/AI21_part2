# Complete this class for all parts of the project

import numpy as np
from pacman_module.game import Agent
from pacman_module.pacman import Directions
from pacman_module.util import PriorityQueue, manhattanDistance


def keyHash(state, ghostID):
    """
    Returns a unique hash to identifie the game's state

    Arguments:
    ----------
    - `state`: current gameState, see class
                `pacman.gameState`.

    Return:
    -------
    - Returns a unique hash to identifie the game's state.
    """
    return state.getPacmanPosition(), ghostID


class PacmanAgent(Agent):
    def __init__(self, args):
        """
        Arguments:
        ----------
        - `args`: Namespace of arguments from command-line prompt.
        """
        self.args = args

    def get_action(self, state, belief_state):
        """
        Given a pacman game state and a belief state,
                returns a legal move.

        Arguments:
        ----------
        - `state`: the current game state. See FAQ and class
                   `pacman.GameState`.
        - `belief_state`: a list of probability matrices.

        Return:
        -------
        - A legal move as defined in `game.Directions`.
        """

        return Directions.STOP