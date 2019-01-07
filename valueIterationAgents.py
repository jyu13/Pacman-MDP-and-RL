# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0

        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for i in range(self.iterations):
          v = self.values.copy()  # get value for v_{k-1}
          for state in self.mdp.getStates():
            
            if False == self.mdp.isTerminal(state):    
              valueForA = util.Counter();         
              for possibleA in self.mdp.getPossibleActions(state):
                for transition, prob in self.mdp.getTransitionStatesAndProbs(state, possibleA):
                  # sum T(s,a,s')[ R(s,a,s') + discount * V_{k-1}(s')]
                  valueForA[possibleA] += prob*(self.mdp.getReward(state, possibleA, transition)+ (self.discount*v[transition]))
              # max of {sum T(s,a,s')[ R(s,a,s') + discount * V_k(s')]}
              self.values[state] = valueForA[valueForA.argMax()]

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        qValue = 0
        if False == self.mdp.isTerminal(state):
          for transition, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            qValue += prob*(self.mdp.getReward(state,action,transition)+ (self.discount*self.values[transition]))

        return qValue
        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        if True == self.mdp.isTerminal(state):
          return None

        qValue = float("-inf")
        optimalA = None
        possibleA = self.mdp.getPossibleActions(state)

        for action in possibleA:
          temp_q = self.getQValue(state, action) 
          if temp_q > qValue:
            optimalA = action
            qValue = temp_q

        return optimalA

        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
