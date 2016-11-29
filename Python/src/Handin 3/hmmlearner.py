import faste_file_reader as fr

observables = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
states = {'S1': 0, 'S2': 1, 'S3': 2, 'S4': 3, 'S5': 4, 'S6': 5, 'S7': 6}


def convert_to_states(annotations):
    states = [0 for x in range(len(annotations))]
    for i in range(len(annotations)):
        annotation = annotations[i]
        last_state = "S1" if i == 0 else states[i-1]
        state_trans_map = {
            "S1": {"N": "S1", "C": "S2", "R": "S5"},
            "S2": {"C": "S3"},
            "S3": {"C": "S4"},
            "S4": {"N": "S1", "C": "S2", "R": "S5"},
            "S5": {"R": "S6"},
            "S6": {"R": "S7"},
            "S7": {"N": "S1", "C": "S2", "R": "S5"},
        }
        this_state = state_trans_map[last_state][annotation]
        states[i] = this_state
    return states


def transform_annotations(sequences, training_pairs):
    for training_pair in training_pairs:
        annotation_data = sequences[training_pair[1]]
        converted_annotations = convert_to_states(annotation_data)
        sequences[training_pair[1]] = converted_annotations
        sequences[training_pair[0]] = list(sequences[training_pair[0]])
    return sequences


def calculate_init_probs(sequences, training_pairs):
    # Init count
    init_probs = [0 for x in range(len(states))]
    init_count = [0 for x in range(len(states))]
    for training_pair in training_pairs:
        training_annotations = sequences[training_pair[1]]
        init_state_index = states[training_annotations[0]]
        init_count[init_state_index] = init_count[init_state_index] + 1
    total_init_count = sum(init_count)
    for i in range(len(init_probs)):
        init_probs[i] = init_count[i] / total_init_count
    return init_probs


def calculate_emit_probs(sequences, training_pairs):
    emit_probs = [[0 for x in range(len(observables))] for y in range(len(states))]
    emit_count = [[0 for x in range(len(observables))] for y in range(len(states))]
    for training_pair in training_pairs:
        training_annotations = sequences[training_pair[1]]
        training_observations = sequences[training_pair[0]]
        for i in range(len(training_annotations)):
            state_index = states[training_annotations[i]]
            observation_index = observables[training_observations[i]]
            emit_count[state_index][observation_index] = emit_count[state_index][observation_index] + 1
    for i in range(len(emit_count)):
        total_emissions = sum(emit_count[i])
        for j in range(len(emit_count[i])):
            if total_emissions == 0:
                emit_probs[i][j] = 0
            else:
                emit_probs[i][j] = emit_count[i][j]/total_emissions
    return emit_probs

def calculate_transition_probs(sequences, training_pairs):
    trans_probs = [[0 for x in range(len(states))] for y in range(len(states))]
    trans_count = [[0 for x in range(len(states))] for y in range(len(states))]
    for training_pair in training_pairs:
        training_annotations = sequences[training_pair[1]]
        for i in range(len(training_annotations) - 1):
            state_index = states[training_annotations[i]]
            next_state_index = states[training_annotations[i + 1]]
            trans_count[state_index][next_state_index] = trans_count[state_index][next_state_index] + 1
    for i in range(len(trans_count)):
        total_transitions_from_state = sum(trans_count[i])
        for j in range(len(trans_count[i])):
            if total_transitions_from_state == 0:
                trans_probs[i][j] = 0
            else:
                trans_probs[i][j] = trans_count[i][j]/total_transitions_from_state
    return trans_probs


def learn_hmm(filenames, training_pairs):
    sequences = {}
    for filename in filenames:
        newsequence = fr.read_fasta_file(filename=filename)
        sequences = {**sequences, **newsequence}

    sequences = transform_annotations(sequences, training_pairs)
    init_probs = calculate_init_probs(sequences, training_pairs)
    emit_probs = calculate_emit_probs(sequences, training_pairs)
    transition_probs = calculate_transition_probs(sequences, training_pairs)
    return init_probs, transition_probs, emit_probs

filenames = ["./data/genome1.fa", "./data/genome2.fa", "./data/genome3.fa", "./data/genome4.fa",
             "./data/genome5.fa",
             "./data/annotation1.fa", "./data/annotation2.fa", "./data/annotation3.fa", "./data/annotation4.fa",
             "./data/annotation5.fa"]

training_pairs = [
    ["genome1", "annotation1"],
    ["genome2", "annotation2"],
    ["genome3", "annotation3"],
    ["genome4", "annotation4"],
    ["genome5", "annotation5"]]
