# analysis.py
# -----------
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


######################
# ANALYSIS QUESTIONS #
######################

# Set the given parameters to obtain the specified policies through
# value iteration.

def question2():
    # I change the noise parameter to zero, because I want always to do the optimal
    answerDiscount = 0.9
    answerNoise = 0.0
    return answerDiscount, answerNoise

"""
For Question 3


    a. Prefer the close exit (+1), risking the cliff (-10)
    b. Prefer the close exit (+1), but avoiding the cliff (-10)
    c. Prefer the distant exit (+10), risking the cliff (-10)
    d. Prefer the distant exit (+10), avoiding the cliff (-10)
    e. Avoid both exits and the cliff (so an episode should never terminate)
"""

def question3a():
    answerDiscount = 0.3       # Will have myopic vision, thus the algorithm will be more greedy
    answerNoise = 0.0
    answerLivingReward = -1     # Will choose the closest exit as I increase that 
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3b():
    answerDiscount = 0.3
    answerNoise = 0.1   # Noise helps the agent understand that walking near the cliff has a risk
                        # However noise cannot be too big because then the agent will just roam around
    answerLivingReward = -1
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3c():
    answerDiscount = 1
    answerNoise = 0.0
    answerLivingReward = -1

    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3d():
    # Key = higher reward, noise and no discount
    answerDiscount = 1
    answerNoise = 0.2
    answerLivingReward = -0.5
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3e():
    # The key for this is to have high living reward
    answerDiscount = 0
    answerNoise = 0.0
    answerLivingReward = 10

    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question6():
    answerEpsilon = None
    answerLearningRate = None
    return 'NOT POSSIBLE' # because even if we have high exploration rate it will be unlikely
                          # to reach the high reward after 50 episodes
    #return answerEpsilon, answerLearningRate
    # If not possible, return 'NOT POSSIBLE'

if __name__ == '__main__':
    print 'Answers to analysis questions:'
    import analysis
    for q in [q for q in dir(analysis) if q.startswith('question')]:
        response = getattr(analysis, q)()
        print '  Question %s:\t%s' % (q, str(response))
