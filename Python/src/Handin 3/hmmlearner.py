class HmmLearner:

    def __init__(self, observables, states):
        self.observables = observables
        self.states = states

    def calculate_init_probs(self, sequences, training_pairs):
        # Init count
        init_probs = [0 for x in range(len(self.states))]
        init_count = [0 for x in range(len(self.states))]
        for training_pair in training_pairs:
            training_annotations = sequences[training_pair[1]]
            init_state_index = self.states[training_annotations[0]]
            init_count[init_state_index] = init_count[init_state_index] + 1
        total_init_count = sum(init_count)
        for i in range(len(init_probs)):
            init_probs[i] = init_count[i] / total_init_count
        return init_probs


    def calculate_emit_probs(self, sequences, training_pairs):
        emit_probs = [[0 for x in range(len(self.observables))] for y in range(len(self.states))]
        emit_count = [[0 for x in range(len(self.observables))] for y in range(len(self.states))]
        for training_pair in training_pairs:
            training_annotations = sequences[training_pair[1]]
            training_observations = sequences[training_pair[0]]
            for i in range(len(training_annotations)):
                state_index = self.states[training_annotations[i]]
                observation_index = self.observables[training_observations[i]]
                emit_count[state_index][observation_index] = emit_count[state_index][observation_index] + 1
        for i in range(len(emit_count)):
            total_emissions = sum(emit_count[i])
            for j in range(len(emit_count[i])):
                if total_emissions == 0:
                    emit_probs[i][j] = 0
                else:
                    emit_probs[i][j] = emit_count[i][j]/total_emissions
        return emit_probs

    def calculate_transition_probs(self, sequences, training_pairs):
        trans_probs = [[0 for x in range(len(self.states))] for y in range(len(self.states))]
        trans_count = [[0 for x in range(len(self.states))] for y in range(len(self.states))]
        for training_pair in training_pairs:
            training_annotations = sequences[training_pair[1]]
            for i in range(len(training_annotations) - 1):
                state_index = self.states[training_annotations[i]]
                next_state_index = self.states[training_annotations[i + 1]]
                trans_count[state_index][next_state_index] = trans_count[state_index][next_state_index] + 1
        for i in range(len(trans_count)):
            total_transitions_from_state = sum(trans_count[i])
            for j in range(len(trans_count[i])):
                if total_transitions_from_state == 0:
                    trans_probs[i][j] = 0
                else:
                    trans_probs[i][j] = trans_count[i][j]/total_transitions_from_state
        return trans_probs



