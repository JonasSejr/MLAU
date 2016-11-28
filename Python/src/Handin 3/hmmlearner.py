import faste_file_reader as fr

observables = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
states = {'S1': 0, 'S1': 1, 'S1': 2, 'S1': 3, 'S1': 4, 'S1': 5, 'S1': 6}


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
    pass

def calculate_init_probs(sequences, training_pairs):
    global init_probs
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
    return(init_probs)



def learn_hmm():
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
    sequences = {}
    for filename in filenames:
        newsequence = fr.read_fasta_file(filename=filename)
        sequences = {**sequences, **newsequence}

    transform_annotations(sequences, training_pairs)
    for training_pair in training_pairs:
        annotation_data = sequences[training_pair[1]]
        converted_annotations = convert_to_states(annotation_data)
        sequences[training_pair[1]] = converted_annotations


    emit_probs = [[0 for x in range(len(states))] for y in range(len(observables))]
    emit_count = [[0 for x in range(len(states))] for y in range(len(observables))]
    trans_probs = [[0 for x in range(len(states))] for y in range(len(states))]
    trans_count = [[0 for x in range(len(states))] for y in range(len(states))]

    # counting
    # init_probs = calculate_init_probs()

    print(init_probs)



