import faste_file_reader as fr


filenames = ["./data/genome1.fa", "./data/genome2.fa", "./data/genome3.fa", "./data/genome4.fa", "./data/genome5.fa",
             "./data/annotation1.fa", "./data/annotation2.fa", "./data/annotation3.fa", "./data/annotation4.fa", "./data/annotation5.fa"]
sequences = {}
for filename in filenames:
    newsequence = fr.read_fasta_file(filename=filename)
    sequences = {**sequences, **newsequence}

#Coud be a list of maps...
training_pairs = [
    ["genome1", "annotation1"],
    ["genome2", "annotation2"],
    ["genome3", "annotation3"],
    ["genome4", "annotation4"],
    ["genome5", "annotation5"]]

#Counting
observables = {'A':0, 'C':1, 'G':2, 'T':3}
states = {'N':0, 'C': 1, 'R':2}

emit_probs = [[0 for x in range(len(states))] for y in range(len(observables))]

emit_count = [[0 for x in range(len(states))] for y in range(len(observables))]


def calculate_init_probs():
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


#Cals transitions
trans_probs = [[0 for x in range(len(states))] for y in range(len(states))]
trans_count = [[0 for x in range(len(states))] for y in range(len(states))]
for training_pair in training_pairs:
    for row 


init_probs = calculate_init_probs()

print(init_probs)