# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).
from operator import itemgetter
import sys

from util import manhattanDistance, PriorityQueue
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        score = 9999999

        foodScore = calcMinFoodDistance(currentGameState) - calcMinFoodDistance(successorGameState)
        if currentGameState.getScore() < successorGameState.getScore():
            foodScore = successorGameState.getScore() - currentGameState.getScore()


        score = calcGhostDistance(successorGameState) - calcGhostDistance(currentGameState)

        #print action
        #print "Food", foodScore
        #print "Score", score

        return   foodScore + ( score)

def calcGhostDistance(gameState):
        newPos = gameState.getPacmanPosition()
        newGhostStates = gameState.getGhostStates()
        score = 9999999
        for ghost in newGhostStates:
            score = min(score, manhattanDistance(newPos, ghost.getPosition()))

        return score

def calcMinFoodDistance(gameState):
        newPos = gameState.getPacmanPosition()
        newFood = gameState.getFood()
        foodScore = 999999
        for food in newFood.asList():
            foodScore = min(manhattanDistance(newPos, food), foodScore)

        return foodScore

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """


        self.move = None
        val =  self.minMax(gameState, 0, 0)

        #print val

        return self.move

    def minMax(self, gameState, depth, agentIndex):
        #print agentIndex
        actions = gameState.getLegalActions(agentIndex)
        if (depth == self.depth * gameState.getNumAgents() or gameState.isLose() or gameState.isWin() or len(actions) == 0) :
            return self.evaluationFunction(gameState)

        if agentIndex == 0:
            max = - sys.maxint
            nextAgentIndex = 1
            for action in actions:
                successor = gameState.generateSuccessor(agentIndex, action)
                val = self.minMax(successor, depth + 1, nextAgentIndex)
                if val > max:
                    max = val
                    if 0 == depth:
                        self.move = action

            return max

        else:
            min = sys.maxint
            actions = gameState.getLegalActions(agentIndex)
            for action in actions:
                successor = gameState.generateSuccessor(agentIndex, action)
                val = self.minMax(successor, depth + 1, (agentIndex  + 1) % gameState.getNumAgents())
                if val < min:
                    min = val

            return min







class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """

        self.move = None
        val =  self.minMax(gameState, 0, 0, - sys.maxint, sys.maxint)

        return self.move

    def minMax(self, gameState, depth, agentIndex, alfa, beta):
        actions = gameState.getLegalActions(agentIndex)
        if (depth == self.depth * gameState.getNumAgents() or gameState.isLose() or gameState.isWin() or len(actions) == 0) :
            return self.evaluationFunction(gameState)

        if agentIndex == 0:
            maxVal = - sys.maxint

            for action in actions:
                successor = gameState.generateSuccessor(agentIndex, action)
                val = self.minMax(successor, depth + 1, 1, alfa, beta)
                if val > maxVal:
                    maxVal = val
                    if 0 == depth:
                        self.move = action

                if maxVal > beta:
                    return maxVal
                alfa = max(alfa, maxVal)

            return maxVal

        else:
            minVal = sys.maxint

            for action in actions:
                successor = gameState.generateSuccessor(agentIndex, action)
                val = self.minMax(successor, depth + 1, (agentIndex  + 1) % gameState.getNumAgents(), alfa, beta)
                if val < minVal:
                    minVal = val

                if minVal < alfa:
                    return minVal
                beta = min(beta, minVal)

            return minVal



class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        self.move = None
        val =  self.minMax(gameState, 0, 0)

        #print val

        return self.move

    def minMax(self, gameState, depth, agentIndex):
        actions = gameState.getLegalActions(agentIndex)
        if (depth == self.depth * gameState.getNumAgents() or gameState.isLose() or gameState.isWin() or len(actions) == 0) :
            return self.evaluationFunction(gameState)

        if agentIndex == 0:
            nextAgentIndex = gameState.getNumAgents() - 1
            max = - sys.maxint

            for action in actions:
                successor = gameState.generateSuccessor(agentIndex, action)
                val = self.minMax(successor, depth + 1, 1)
                if val > max:
                    max = val
                    if 0 == depth:
                        self.move = action

            return max

        else:
            val = 0
            for action in actions:
                successor = gameState.generateSuccessor(agentIndex, action)
                val += self.minMax(successor, depth + 1, (agentIndex + 1)%gameState.getNumAgents())

            return float(val)/float(len(actions))


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """

    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]


    foodScore = calcMinFoodDistance(currentGameState)

    score = calcGhostDistance(currentGameState)

    #print action
    #print "Food", foodScore
    #print "Score", score
    scared = 0
    for scaredTime in newScaredTimes:
        scared += scaredTime
    return  10 * (1.0/float(foodScore)) + score +  (2 *currentGameState.getScore()) + (5 * scared)

# Abbreviation
better = betterEvaluationFunction

