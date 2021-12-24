# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
import time

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """

    startTime = time.time()

    dfsStack = util.Stack()
    exploredNodes = util.Counter()

    # state = (nodePosition, path) #
    initialState = (problem.getStartState(), list())
    dfsStack.push(initialState)

    while True:
        if dfsStack.isEmpty():
            endTime = time.time() - startTime
            # print("DFS took %s seconds to execute" % endTime)
            return []

        currentNode = dfsStack.pop()

        currentNodePosition = currentNode[0]
        currentNodePath = currentNode[1]

        if problem.isGoalState(currentNodePosition):
            endTime = time.time() - startTime
            # print("DFS took %s seconds to execute" % endTime)
            return currentNodePath

        exploredNodes[currentNodePosition] = 1

        for successorNode in problem.getSuccessors(currentNodePosition):
            successorNodePostion = successorNode[0]
            successorNodeAction = successorNode[1]

            pathToSuccessorNode = currentNodePath.copy()
            pathToSuccessorNode.append(successorNodeAction)

            if not exploredNodes[successorNodePostion]:
                dfsStack.push((successorNodePostion, pathToSuccessorNode))

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""

    startTime = time.time()
    
    bfsQueue = util.Queue()
    exploredNodes = util.Counter()
    elementsInQueue = util.Counter()

    # state = (nodePosition, path) #
    initialState = (problem.getStartState(), list())
    bfsQueue.push(initialState)
    elementsInQueue[initialState[0]] = 1

    while True:
        if bfsQueue.isEmpty():
            endTime = time.time() - startTime
            # print("BFS took %s seconds to execute" % endTime)
            return []

        currentNode = bfsQueue.pop()

        currentNodePosition = currentNode[0]
        currentNodePath = currentNode[1]

        elementsInQueue[currentNodePosition] = 0

        if problem.isGoalState(currentNodePosition):
            endTime = time.time() - startTime
            # print("BFS took %s seconds to execute" % endTime)
            return currentNodePath

        exploredNodes[currentNodePosition] = 1

        for successorNode in problem.getSuccessors(currentNodePosition):
            successorNodePostion = successorNode[0]
            successorNodeAction = successorNode[1]

            pathToSuccessorNode = currentNodePath.copy()
            pathToSuccessorNode.append(successorNodeAction)

            if not exploredNodes[successorNodePostion] and not elementsInQueue[successorNodePostion]:
                bfsQueue.push((successorNodePostion, pathToSuccessorNode))
                elementsInQueue[successorNodePostion] = 1


def uniformCostSearch(problem):
    """Search the node of least total cost first."""

    startTime = time.time()
    
    ucsQueue = util.PriorityQueue()
    exploredNodes = util.Counter()
    elementsInQueue = util.Counter()

    # state = (nodePosition) #

    # util.PriorityQueue does not work as expected with a tuple (nodePosition, path, priority)
    # as a item because the update() function checks if the whole tuple is equal the item passed instead of
    # only the position. To work around this situation, it was created a util.Counter to keep track of the priorities
    # and a dictionary to keep track of the actions
    initialState = problem.getStartState()
    accumulatedPrioritiesPerNode = util.Counter()
    actionsToTake = {}

    ucsQueue.push(initialState, 0)
    elementsInQueue[initialState[0]] = 1
    accumulatedPrioritiesPerNode[initialState] = 0
    actionsToTake[initialState] = list()

    while True:
        if ucsQueue.isEmpty():
            endTime = time.time() - startTime
            # print("UCS took %s seconds to execute" % endTime)
            return []

        currentNode = ucsQueue.pop()

        if problem.isGoalState(currentNode):
            endTime = time.time() - startTime
            # print("UCS took %s seconds to execute" % endTime)
            return actionsToTake[currentNode]

        exploredNodes[currentNode] = 1

        for successorNode in problem.getSuccessors(currentNode):
            successorNodePostion = successorNode[0]
            successorNodeAction = successorNode[1]
            successorNodePriority = successorNode[2]

            pathToSuccessorNode = actionsToTake[currentNode].copy()
            pathToSuccessorNode.append(successorNodeAction)

            accumulatedPriority = (accumulatedPrioritiesPerNode[currentNode] + successorNodePriority)

            if not exploredNodes[successorNodePostion]:
                ucsQueue.update(successorNodePostion, accumulatedPriority)

                if accumulatedPrioritiesPerNode[successorNodePostion] and accumulatedPriority >= accumulatedPrioritiesPerNode[successorNodePostion]:
                    continue

                accumulatedPrioritiesPerNode[successorNodePostion] = accumulatedPriority
                actionsToTake[successorNodePostion] = pathToSuccessorNode

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def greedySearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest heuristic first."""

    startTime =time.time()

    greedyQueue = util.PriorityQueue()
    exploredNodes = util.Counter()
    elementsInQueue = util.Counter()

    initialState = problem.getStartState()
    actionsToTake = {}

    greedyQueue.push(initialState, heuristic(initialState, problem))
    elementsInQueue[initialState[0]] = 1
    actionsToTake[initialState] = list()

    while True:
        if greedyQueue.isEmpty():
            endTime = time.time() - startTime
            # print("GreedySearch took %s seconds to execute" % endTime)
            return []

        currentNode = greedyQueue.pop()

        if problem.isGoalState(currentNode):
            endTime = time.time() - startTime
            # print("GreedySearch took %s seconds to execute" % endTime)
            return actionsToTake[currentNode]

        exploredNodes[currentNode] = 1

        for successorNode in problem.getSuccessors(currentNode):
            successorNodePostion = successorNode[0]
            successorNodeAction = successorNode[1]
            successorNodePriority = heuristic(successorNodePostion, problem)

            pathToSuccessorNode = actionsToTake[currentNode].copy()
            pathToSuccessorNode.append(successorNodeAction)

            if not exploredNodes[successorNodePostion]:
                greedyQueue.update(successorNodePostion, successorNodePriority)

                if successorNodePostion in actionsToTake.keys() and len(pathToSuccessorNode) > len(actionsToTake[successorNodePostion]):
                    continue

                actionsToTake[successorNodePostion] = pathToSuccessorNode


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    """Search the node of least total cost first."""

    startTime = time.time()

    astarQueue = util.PriorityQueue()
    exploredNodes = util.Counter()
    elementsInQueue = util.Counter()

    initialState = problem.getStartState()
    accumulatedPrioritiesPerNode = util.Counter()
    actionsToTake = {}

    astarQueue.push(initialState, 0 + heuristic(initialState, problem))
    elementsInQueue[initialState[0]] = 1
    accumulatedPrioritiesPerNode[initialState] = 0
    actionsToTake[initialState] = list()

    while True:
        if astarQueue.isEmpty():
            endTime = time.time() - startTime
            # print("A* took %s seconds to execute" % endTime)
            return []

        currentNode = astarQueue.pop()

        if problem.isGoalState(currentNode):
            endTime = time.time() - startTime
            # print("A* took %s seconds to execute" % endTime)
            return actionsToTake[currentNode]

        exploredNodes[currentNode] = 1

        for successorNode in problem.getSuccessors(currentNode):
            successorNodePostion = successorNode[0]
            successorNodeAction = successorNode[1]
            successorNodePriority = successorNode[2]
            successorNodePriorityHeuristic = heuristic(successorNodePostion, problem)

            pathToSuccessorNode = actionsToTake[currentNode].copy()
            pathToSuccessorNode.append(successorNodeAction)

            accumulatedPriority = (accumulatedPrioritiesPerNode[currentNode] + successorNodePriority)
            accumulatedPriorityWithHeuristic = accumulatedPriority + successorNodePriorityHeuristic

            if not exploredNodes[successorNodePostion]:
                astarQueue.update(successorNodePostion, accumulatedPriorityWithHeuristic)

                if accumulatedPrioritiesPerNode[successorNodePostion] and accumulatedPriority >= accumulatedPrioritiesPerNode[successorNodePostion]:
                    continue

                accumulatedPrioritiesPerNode[successorNodePostion] = accumulatedPriority
                actionsToTake[successorNodePostion] = pathToSuccessorNode


def foodHeuristic(state, problem):
    """
    Your heuristic for the FoodSearchProblem goes here.

    This heuristic must be consistent to ensure correctness.  First, try to come
    up with an admissible heuristic; almost all admissible heuristics will be
    consistent as well.

    If using A* ever finds a solution that is worse uniform cost search finds,
    your heuristic is *not* consistent, and probably not admissible!  On the
    other hand, inadmissible or inconsistent heuristics may find optimal
    solutions, so be careful.

    The state is a tuple ( pacmanPosition, foodGrid ) where foodGrid is a Grid
    (see game.py) of either True or False. You can call foodGrid.asList() to get
    a list of food coordinates instead.

    If you want access to info like walls, capsules, etc., you can query the
    problem.  For example, problem.walls gives you a Grid of where the walls
    are.

    If you want to *store* information to be reused in other calls to the
    heuristic, there is a dictionary called problem.heuristicInfo that you can
    use. For example, if you only want to count the walls once and store that
    value, try: problem.heuristicInfo['wallCount'] = problem.walls.count()
    Subsequent calls to this heuristic can access
    problem.heuristicInfo['wallCount']
    """
    position, foodGrid = state
    
    distanceFromPosition = list()
    distanceFromFood = list()

    distanceFromFood.append(0)

    for food in foodGrid.asList():
        distancePositionFood = util.manhattanDistance(position, food)
        distanceFromPosition.append(distancePositionFood)

        for _food in foodGrid.asList():
            distanceFoodFood = util.manhattanDistance(food, _food)
            distanceFromFood.append(distanceFoodFood)

    if (len(distanceFromPosition)):
        return min(distanceFromPosition) + max(distanceFromFood)
    else:
        return max(distanceFromFood)

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
ucs = uniformCostSearch
gs = greedySearch
astar = aStarSearch
