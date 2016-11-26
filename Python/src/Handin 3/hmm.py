import sys
import math

observables = {'A':0, 'C':1, 'G':2, 'T':3}
states = {'1':0, '2': 1, '3':2, '4':3, '5':4, '6':5, '7':6}

init_probs = [0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.00]

trans_probs = [[0.00, 0.00, 0.90, 0.10, 0.00, 0.00, 0.00],
               [1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
               [0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00],
               [0.00, 0.00, 0.05, 0.90, 0.05, 0.00, 0.00],
               [0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00],
               [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00],
               [0.00, 0.00, 0.00, 0.10, 0.90, 0.00, 0.00]]

emit_probs = [[0.30, 0.25, 0.25, 0.20],
              [0.20, 0.35, 0.15, 0.30],
              [0.40, 0.15, 0.20, 0.25],
              [0.25, 0.25, 0.25, 0.25],
              [0.20, 0.40, 0.30, 0.10],
              [0.30, 0.20, 0.30, 0.20],
              [0.15, 0.30, 0.20, 0.35]]

# Function for computing the joint probability
def compute_joint_prob(x, z):
    p = init_probs[z[0]] * emit_probs[z[0]][x[0]]
    for i in range(1, len(x)):
        p = p * trans_probs[z[i-1]][z[i]] * emit_probs[z[i]][x[i]]
    return p

# Function for computing the joint probability in log space
def compute_joint_log_prob(x, z):
    logp = math.log(init_probs[z[0]]) + math.log(emit_probs[z[0]][x[0]])
    for i in range(1, len(x)):
        logp = logp + math.log(trans_probs[z[i-1]][z[i]]) + math.log(emit_probs[z[i]][x[i]])
    return logp

def calculateJointProbability(obs, hid):
    # Get sequence of observables and hidden states from the commandline
    x = [observables[c] for c in obs]
    z = [states[c] for c in hid]

    if len(x) != len(z):
        print("The two sequences are of different length!")
        sys.exit(1)

    print("    P(x,z) = ", compute_joint_prob(x, z))
    print("log P(x,z) = ", compute_joint_log_prob(x, z))



obs = ['A', 'C', 'A', 'C', 'A', 'G', 'T', 'C']
hid = ['4', '4', '5', '6', '7', '4', '4', '4']

#calculateJointProbability(obs, hid)

def log(number):
    if(number == 0):
        return -math.inf
    else:
        return math.log(number)

def logspace_backtrack_most_likely(w, observations):
    probability = -math.inf
    Z = ['E' for x in range(len(observations))] #E for empty
    for state in states.keys():
        stateIndex = states.get(state)
        probEndingWithThisState = w[stateIndex][-1]
        if(probability < probEndingWithThisState):
            probability = probEndingWithThisState
            Z[len(observations) - 1] = state
    probOfHidden = probability
    for column in reversed(range(0, len(observations) - 1)):
        maxProb = -math.inf
        for state in states.keys():
            stateIndex = states.get(state)
            nextStateIndex = states[Z[column + 1]]
            probability = w[stateIndex][column] + \
                          log(trans_probs[stateIndex][nextStateIndex]) + \
                          log(emit_probs[nextStateIndex][observables[observations[column + 1]]])
            if probability > maxProb:
                Z[column] = state
                maxProb = probability
    return probOfHidden, ''.join(Z)



def backtrack_most_likely(w, observations):
    probability = 0
    Z = ['E' for x in range(len(observations))] #E for empty
    for state in states.keys():
        stateIndex = states.get(state)
        probEndingWithThisState = w[stateIndex][-1]
        if(probability < probEndingWithThisState):
            probability = probEndingWithThisState
            Z[len(observations) - 1] = state
    probOfHidden = probability
    for column in reversed(range(0, len(observations))):
        maxProb = 0
        for state in states.keys():
            stateIndex = states.get(state)
            nextStateIndex = states[Z[column + 1]]
            probability = w[stateIndex][column] * \
                          trans_probs[stateIndex][nextStateIndex] * \
                          emit_probs[nextStateIndex][observables[observations[column + 1]]]
            if probability > maxProb:
                Z[column] = state
                maxProb = probability
    return probOfHidden, ''.join(Z)


def create_matrix(observations):
    w = [[0 for x in range(len(observations))] for y in range(len(states))]
    for state in states.keys():
        stateIndex = states.get(state)
        w[stateIndex][0] = init_probs[stateIndex] * emit_probs[stateIndex][observables[observations[0]]]
    for column in range(1, len(observations)):
        observation = observations[column]
        observationIndex = observables[observation]
        for currentState in states.keys():
            currentStateIndex = states.get(currentState)
            maxProb = 0
            for lastState in states.keys():
                lastStateIndex = states.get(lastState)
                probability = w[lastStateIndex][column - 1] * \
                              trans_probs[lastStateIndex][currentStateIndex] * \
                              emit_probs[currentStateIndex][observationIndex]
                maxProb = max(maxProb, probability)
            print(column)
            w[currentStateIndex][column] = maxProb
    return w

def create_logspace_matrix(observations):
    w = [[0 for x in range(len(observations))] for y in range(len(states))]
    for state in states.keys():
        stateIndex = states.get(state)
        w[stateIndex][0] = log(init_probs[stateIndex]) + log(emit_probs[stateIndex][observables[observations[0]]])
    for column in range(1, len(observations)):
        observation = observations[column]
        observationIndex = observables[observation]
        for currentState in states.keys():
            currentStateIndex = states.get(currentState)
            maxProb = -math.inf
            for lastState in states.keys():
                lastStateIndex = states.get(lastState)
                probability = w[lastStateIndex][column - 1] + \
                              log(trans_probs[lastStateIndex][currentStateIndex]) + \
                              log(emit_probs[currentStateIndex][observationIndex])
                maxProb = max(maxProb, probability)
            print(column)
            w[currentStateIndex][column] = maxProb
    return w

def viterbi_backtrack(observations):
    w = create_matrix(observations)
    return backtrack_most_likely(w, observations)

def logspace_viterbi_backtrack(observations):
    w = create_logspace_matrix(observations)
    return logspace_backtrack_most_likely(w, observations)

#maxProb, hidden = viterbi_backtrack(obs)
#print(maxProb)
#print(hidden)