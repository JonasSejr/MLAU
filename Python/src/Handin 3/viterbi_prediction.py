import sys
import math

class ViterbiPredictor:
    def __init__(self, observables, states, init_probs, trans_probs, emit_probs):
        self.observables = observables
        self.states = states
        self.init_probs = init_probs
        self.trans_probs = trans_probs
        self.emit_probs = emit_probs

    # Function for computing the joint probability
    def compute_joint_prob(self, x, z):
        p = self.init_probs[z[0]] * self.emit_probs[z[0]][x[0]]
        for i in range(1, len(x)):
            p = p * self.trans_probs[z[i-1]][z[i]] * self.emit_probs[z[i]][x[i]]
        return p

    # Function for computing the joint probability in log space
    def compute_joint_log_prob(self, x, z):
        logp = math.log(self.init_probs[z[0]]) + math.log(self.emit_probs[z[0]][x[0]])
        for i in range(1, len(x)):
            logp = logp + math.log(self.trans_probs[z[i-1]][z[i]]) + math.log(self.emit_probs[z[i]][x[i]])
        return logp

    def calculateJointProbability(self, obs, hid):
        # Get sequence of observables and hidden states from the commandline
        x = [self.observables[c] for c in obs]
        z = [self.states[c] for c in hid]

        if len(x) != len(z):
            print("The two sequences are of different length!")
            sys.exit(1)

        print("    P(x,z) = ", self.compute_joint_prob(x, z))
        print("log P(x,z) = ", self.compute_joint_log_prob(x, z))


    def log(self, number):
        if(number == 0):
            return -math.inf
        else:
            return math.log(number)

    def logspace_backtrack_most_likely(self, w, observations):
        probability = -math.inf
        Z = ['E' for x in range(len(observations))] #E for empty
        for state in self.states.keys():
            stateIndex = self.states.get(state)
            probEndingWithThisState = w[stateIndex][-1]
            if(probability < probEndingWithThisState):
                probability = probEndingWithThisState
                Z[len(observations) - 1] = state
        probOfHidden = probability
        for column in reversed(range(0, len(observations) - 1)):
            maxProb = -math.inf
            for state in self.states.keys():
                stateIndex = self.states.get(state)
                nextStateIndex = self.states[Z[column + 1]]
                probability = w[stateIndex][column] + \
                              self.log(self.trans_probs[stateIndex][nextStateIndex]) + \
                              self.log(self.emit_probs[nextStateIndex][self.observables[observations[column + 1]]])
                if probability > maxProb:
                    Z[column] = state
                    maxProb = probability
        return probOfHidden, Z



    def backtrack_most_likely(self, w, observations):
        probability = 0
        Z = ['E' for x in range(len(observations))] #E for empty
        for state in self.states.keys():
            stateIndex = self.states.get(state)
            probEndingWithThisState = w[stateIndex][-1]
            if(probability < probEndingWithThisState):
                probability = probEndingWithThisState
                Z[len(observations) - 1] = state
        probOfHidden = probability
        for column in reversed(range(0, len(observations))):
            maxProb = 0
            for state in self.states.keys():
                stateIndex = self.states.get(state)
                nextStateIndex = self.states[Z[column + 1]]
                probability = w[stateIndex][column] * \
                              self.trans_probs[stateIndex][nextStateIndex] * \
                              self.emit_probs[nextStateIndex][self.observables[observations[column + 1]]]
                if probability > maxProb:
                    Z[column] = state
                    maxProb = probability
        return probOfHidden, ''.join(Z)


    def create_matrix(self, observations):
        w = [[0 for x in range(len(observations))] for y in range(len(self.states))]
        for state in self.states.keys():
            stateIndex = self.states.get(state)
            w[stateIndex][0] = self.init_probs[stateIndex] * self.emit_probs[stateIndex][self.observables[observations[0]]]
        for column in range(1, len(observations)):
            observation = observations[column]
            observationIndex = self.observables[observation]
            for currentState in self.states.keys():
                currentStateIndex = self.states.get(currentState)
                maxProb = 0
                for lastState in self.states.keys():
                    lastStateIndex = self.states.get(lastState)
                    probability = w[lastStateIndex][column - 1] * \
                                  self.trans_probs[lastStateIndex][currentStateIndex] * \
                                  self.emit_probs[currentStateIndex][observationIndex]
                    maxProb = max(maxProb, probability)
                print(column)
                w[currentStateIndex][column] = maxProb
        return w

    def create_logspace_matrix(self, observations):
        w = [[0 for x in range(len(observations))] for y in range(len(self.states))]
        for state in self.states.keys():
            stateIndex = self.states.get(state)
            w[stateIndex][0] = self.log(self.init_probs[stateIndex]) + self.log(self.emit_probs[stateIndex][self.observables[observations[0]]])
        for column in range(1, len(observations)):
            if(column%100000 == 0):
                print(column)
            observation = observations[column]
            observationIndex = self.observables[observation]
            for currentState in self.states.keys():
                currentStateIndex = self.states.get(currentState)
                maxProb = -math.inf
                for lastState in self.states.keys():
                    lastStateIndex = self.states.get(lastState)
                    probability = w[lastStateIndex][column - 1] + \
                                  self.log(self.trans_probs[lastStateIndex][currentStateIndex]) + \
                                  self.log(self.emit_probs[currentStateIndex][observationIndex])
                    maxProb = max(maxProb, probability)
                w[currentStateIndex][column] = maxProb
        return w

    def viterbi_backtrack(self, observations):
        w = self.create_matrix(observations)
        return self.backtrack_most_likely(w, observations)

    def logspace_viterbi_backtrack(self, observations):
        w = self.create_logspace_matrix(observations)
        return self.logspace_backtrack_most_likely(w, observations)

#maxProb, hidden = viterbi_backtrack(obs)
#print(maxProb)
#print(hidden)