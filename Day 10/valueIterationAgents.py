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
        all_states = self.mdp.getStates()

        for _ in range(self.iterations):
            new_values = util.Counter()
            
            for state in all_states:
                if self.mdp.isTerminal(state):
                    continue
                else:
                    new_values[state] = self.getStateValue(state)

            self.values = new_values

        self.values['TERMINAL_STATE'] = self.mdp.getReward('TERMINAL_STATE', None, None)

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
        result = 0

        curr_state_dynamics = self.mdp.getTransitionStatesAndProbs(state, action)

        for next_state, prob in curr_state_dynamics:
            result += prob * (self.mdp.getReward(state, action, next_state) + (self.discount * self.values[next_state]))

        #util.raiseNotDefined()

        # The function getReward does not use its 2nd and 3rd parameter
        return result

    # This function returns the value of the given state
    def getStateValue(self, state):
        actions = self.mdp.getPossibleActions(state)
        max_act = None
        max_val = None

        for action in actions:
            # v(s) = max_a{q(s,a)}
            curr_val = self.computeQValueFromValues(state, action)

            if (max_act == None) or (curr_val > max_val):
                max_act = action
                max_val = curr_val

        #util.raiseNotDefined()
        return max_val

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        actions = self.mdp.getPossibleActions(state)
        max_act = None
        max_val = None

        for action in actions:
            # v(s) = argmax_a{q(s,a)}
            curr_val = self.computeQValueFromValues(state, action)

            if (max_act == None) or (curr_val > max_val):
                max_act = action
                max_val = curr_val

        #util.raiseNotDefined()
        return max_act

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
