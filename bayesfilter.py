# Complete this class for all parts of the project

import numpy as np

# import matplotlib.pyplot as plt
# only important for record_metrics

from pacman_module.game import Agent
from pacman_module import util
from scipy.stats import binom


class BeliefStateAgent(Agent):
    def __init__(self, args):
        """
        Arguments:
        ----------
        - `args`: Namespace of arguments from command-line prompt.
        """
        self.args = args
        """
            Variables to use in 'update_belief_state' method.
            Initialization occurs in 'get_action' method.
        """
        # Variables used for record_metrics
        self.nbIt = 0

        # Current list of belief states over ghost positions
        self.beliefGhostStates = None

        # Grid of walls (assigned with 'state.getWalls()' method)
        self.walls = None

        # Hyper-parameters
        self.ghost_type = self.args.ghostagent
        self.sensor_variance = self.args.sensorvariance

        self.p = 0.5
        self.n = int(self.sensor_variance/(self.p*(1-self.p)))

    def _get_sensor_model(self, pacman_position, evidence):
        """
        Arguments:
        ----------
        - `pacman_position`: 2D coordinates position
          of pacman at state x_{t}
          where 't' is the current time step

        Return:
        -------
        The sensor model represented as a 2D numpy array of
        size [width, height].
        The element at position (w, h) is the probability
        P(E_t=evidence | X_t=(w, h))
        """
        walls = self.walls
        w = walls.width
        h = walls.height

        sensor = np.zeros((w,h))

        for i in range(w):
          for j in range(h):
            distPacmanIJ = util.manhattanDistance(pacman_position,(i,j))
            sensor[i][j] = binom.pmf(distPacmanIJ - evidence + self.n*self.p, self.n, self.p)

        return sensor

    def _get_transition_model(self, pacman_position):
        """
        Arguments:
        ----------
        - `pacman_position`: 2D coordinates position
          of pacman at state x_{t}
          where 't' is the current time step

        Return:
        -------
        The transition model represented as a 4D numpy array of
        size [width, height, width, height].
        The element at position (w1, h1, w2, h2) is the probability
        P(X_t+1=(w1, h1) | X_t=(w2, h2))
        """
        walls = self.walls

        w = walls.width
        h = walls.height
        ghostType = self.ghost_type

        neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        transition = np.zeros((w, h, w, h))

        if ghostType == "confused":
          mul = 1
        elif ghostType == "afraid":
          mul = 2
        else:
          mul = 8

        for i in range(1, w-1):
          for j in range(1, h-1):

            if not walls[i][j]:
              dist = util.manhattanDistance(pacman_position, (i,j))
              norm = 0

              for (k, l) in neighbors:
                if walls[i + k][j + l]:
                  transition[i + k][j + l][i][j] = 0

                elif dist < util.manhattanDistance(pacman_position, (i + k, j + l)):
                  norm += mul
                  transition[i + k][j + l][i][j] = mul
                
                else:
                  norm += 1
                  transition[i + k][j + l][i][j] = 1
                
              for (k, l) in neighbors:
                if norm != 0:
                  transition[i + k][j + l][i][j] /= norm

        return transition

    def _get_updated_belief(self, belief, evidences, pacman_position,
            ghosts_eaten):
        """
        Given a list of (noised) distances from pacman to ghosts,
        and the previous belief states before receiving the evidences,
        returns the updated list of belief states about ghosts positions

        Arguments:
        ----------
        - `belief`: A list of Z belief states at state x_{t-1}
          as N*M numpy mass probability matrices
          where N and M are respectively width and height
          of the maze layout and Z is the number of ghosts.
        - `evidences`: list of distances between
          pacman and ghosts at state x_{t}
          where 't' is the current time step
        - `pacman_position`: 2D coordinates position
          of pacman at state x_{t}
          where 't' is the current time step
        - `ghosts_eaten`: list of booleans indicating
          whether ghosts have been eaten or not

        Return:
        -------
        - A list of Z belief states at state x_{t}
          as N*M numpy mass probability matrices
          where N and M are respectively width and height
          of the maze layout and Z is the number of ghosts.

        N.B. : [0,0] is the bottom left corner of the maze.
               Matrices filled with zeros must be returned for eaten ghosts.
        """
        walls = self.walls
        w = walls.width
        h = walls.height

        transition = self._get_transition_model(pacman_position)
        belief = []
        
        nGhosts = len(ghosts_eaten)
        ghostsBelief = self.beliefGhostStates

        for ghost in range(nGhosts):
          sumMatrix = np.zeros((w,h))
          sensor = self._get_sensor_model(pacman_position, evidences[ghost])

          for i in range(1, w-1):
            for j in range(1, h-1):
              if not ghosts_eaten[ghost]:
                for k in range(w):
                  for l in range(h):
                    sumElem = ghostsBelief[ghost][i][j] * transition[k][l][i][j]
                    sumMatrix[k][l] += sumElem
          
          matrixProduct = np.multiply(sensor,sumMatrix)

          norm = sum(sum(matrixProduct))

          if norm:
            matrixProduct /= norm
          
          belief.append((matrixProduct))

        return belief

    def update_belief_state(self, evidences, pacman_position, ghosts_eaten):
        """
        Given a list of (noised) distances from pacman to ghosts,
        returns a list of belief states about ghosts positions

        Arguments:
        ----------
        - `evidences`: list of distances between
          pacman and ghosts at state x_{t}
          where 't' is the current time step
        - `pacman_position`: 2D coordinates position
          of pacman at state x_{t}
          where 't' is the current time step
        - `ghosts_eaten`: list of booleans indicating
          whether ghosts have been eaten or not

        Return:
        -------
        - A list of Z belief states at state x_{t}
          as N*M numpy mass probability matrices
          where N and M are respectively width and height
          of the maze layout and Z is the number of ghosts.

        XXX: DO NOT MODIFY THIS FUNCTION !!!
        Doing so will result in a 0 grade.
        """
        belief = self._get_updated_belief(self.beliefGhostStates, evidences,
                                          pacman_position, ghosts_eaten)
        self.beliefGhostStates = belief
        return belief

    def _get_evidence(self, state):
        """
        Computes noisy distances between pacman and ghosts.

        Arguments:
        ----------
        - `state`: The current game state s_t
                   where 't' is the current time step.
                   See FAQ and class `pacman.GameState`.


        Return:
        -------
        - A list of Z noised distances in real numbers
          where Z is the number of ghosts.

        XXX: DO NOT MODIFY THIS FUNCTION !!!
        Doing so will result in a 0 grade.
        """
        positions = state.getGhostPositions()
        pacman_position = state.getPacmanPosition()
        noisy_distances = []

        for pos in positions:
            true_distance = util.manhattanDistance(pos, pacman_position)
            noise = binom.rvs(self.n, self.p) - self.n*self.p
            noisy_distances.append(true_distance + noise)

        return noisy_distances

    def _record_metrics(self, belief_states, state):
        """
        Use this function to record your metrics
        related to true and belief states.
        Won't be part of specification grading.

        Arguments:
        ----------
        - `state`: The current game state s_t
                   where 't' is the current time step.
                   See FAQ and class `pacman.GameState`.
        - `belief_states`: A list of Z
           N*M numpy matrices of probabilities
           where N and M are respectively width and height
           of the maze layout and Z is the number of ghosts.

        N.B. : [0,0] is the bottom left corner of the maze
        """
        """
        walls = self.walls

        w = walls.width
        h = walls.height

        for ghostID in range(1, len(belief_states) + 1):
          ghostPos = state.getGhostPosition(ghostID)

          returnBelief = 0.0
          returnQuality = 0

          if ghostPos[0] >= 0:
            maxBeliefPos = (0,0)
            maxBelief = 0.0

            for i in range(w):
              for j in range(h):
                belief = belief_states[ghostID - 1][i][j]

                if belief > maxBelief:
                  maxBeliefPos = (i, j)
                  maxBelief = belief

            returnBelief = maxBelief
            returnQuality = util.manhattanDistance(ghostPos, maxBeliefPos)

          file = open("values.txt", "w")
          file.write("%f \n" %returnBelief)
          file.write("%d \n" %returnQuality)
          file.close()
        
        if self.nbIt == 100:
          exit()
        
        self.nbIt += 1
        """
        pass

    def get_action(self, state):
        """
        Given a pacman game state, returns a belief state.

        Arguments:
        ----------
        - `state`: the current game state.
                   See FAQ and class `pacman.GameState`.

        Return:
        -------
        - A belief state.
        """

        """
           XXX: DO NOT MODIFY THAT FUNCTION !!!
                Doing so will result in a 0 grade.
        """
        # Variables are specified in constructor.
        if self.beliefGhostStates is None:
            self.beliefGhostStates = state.getGhostBeliefStates()
        if self.walls is None:
            self.walls = state.getWalls()

        evidence = self._get_evidence(state)
        newBeliefStates = self.update_belief_state(evidence,
                                                   state.getPacmanPosition(),
                                                   state.data._eaten[1:])
        self._record_metrics(self.beliefGhostStates, state)

        return newBeliefStates, evidence
