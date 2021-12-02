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

        w, h = np.shape(belief_state[0])

        ghost = 1

        while belief_state[ghost - 1].max() == 0 and \
                ghost <= len(belief_state):
            ghost += 1

        maxBeliefPos = (0, 0)
        maxBelief = 0

        for i in range(w):
            for j in range(h):
                belief = belief_state[ghost - 1][i][j]

                if belief > maxBelief:
                    maxBeliefPos = (i, j)
                    maxBelief = belief

        aStar = self.aStar(state, maxBeliefPos, ghost)

        if aStar == []:
            return Directions.STOP

        move = aStar.pop(0)

        return move

    def aStar(self, state, maxBeliefPos, ghostID):
        """
        Computes the path to find to go to maxBelief node in the fastest way

        Arguments:
        ----------
        - `state`: the current game state. See FAQ and class
                    `pacman.GameState`.
        - `maxBeliefPos`: Tuple representing the position of the square with
                            max belief of position for the ghost in next state

        Return:
        -------
        - A list of legal moves leading to maxBeliefPos
        """

        closedSet = set()
        toReturn = []

        queue = PriorityQueue()
        queue.push((0, toReturn, state), 0)

        pacPos = state.getPacmanPosition()

        if pacPos == maxBeliefPos:
            return toReturn

        while True:
            if queue.isEmpty():
                return []

            priority, (backCost, toReturn, currentState) = queue.pop()

            if keyHash(currentState, ghostID) not in closedSet:
                closedSet.add(keyHash(currentState, ghostID))

                for nextState, move in currentState.generatePacmanSuccessors():
                    if keyHash(nextState, ghostID) not in closedSet:
                        newBackCost = backCost + 1
                        cost = manhattanDistance(pacPos, maxBeliefPos)
                        priority = cost + newBackCost
                        queue.push((newBackCost, toReturn + [move],
                                    nextState), priority)

                        nextPacPos = nextState.getPacmanPosition()

                        if nextPacPos == maxBeliefPos:
                            return toReturn

