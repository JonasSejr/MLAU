from graphviz import Digraph
from matplotlib._image import from_images

from hmmlearner import learn_hmm
from hmmlearner import convert_to_annotations
from viterbi_prediction import ViterbiPredictor
from compare_anns import print_all
from faste_file_reader import read_fasta_file

observables = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
states = {'S1': 0, 'S2': 1, 'S3': 2, 'S4': 3, 'S5': 4, 'S6': 5, 'S7': 6}

training_filenames_test = ["./data/test_genome1.fa", "./data/test_genome2.fa", "./data/test_genome3.fa", "./data/test_genome4.fa",
             "./data/test_genome5.fa",
             "./data/test_annotation1.fa", "./data/test_annotation2.fa", "./data/test_annotation3.fa", "./data/test_annotation4.fa",
             "./data/test_annotation5.fa"]


training_filenames = ["./data/genome1.fa", "./data/genome2.fa", "./data/genome3.fa", "./data/genome4.fa",
             "./data/genome5.fa",
             "./data/annotation1.fa", "./data/annotation2.fa", "./data/annotation3.fa", "./data/annotation4.fa",
             "./data/annotation5.fa"]


training_pairs = [
    ["genome1", "annotation1"],
    ["genome2", "annotation2"],
    ["genome3", "annotation3"],
    ["genome4", "annotation4"],
    ["genome5", "annotation5"]]


def run_cross_validation():
    sequences = {}
    for filename in training_filenames:
        newsequence = read_fasta_file(filename=filename)
        sequences = {**sequences, **newsequence}

    for i in range(len(training_pairs)):
        test_pair = training_pairs[i]
        cross_training_pairs = training_pairs[:i] + training_pairs[i + 1:]

        init_probs, transition_probs, emit_probs = learn_hmm(training_filenames, cross_training_pairs)
        predictor = ViterbiPredictor(
            observables=observables,
            states=states,
            emit_probs=emit_probs,
            init_probs=init_probs,
            trans_probs=transition_probs)

        test_gnome = sequences[test_pair[0]]
        test_annotation = sequences[test_pair[1]]
        probOfHidden, hidden_states = predictor.logspace_viterbi_backtrack(test_gnome)

        annotations_list = convert_to_annotations(hidden_states)
        print_all(test_annotation, ''.join(annotations_list))


def create_model_params():
    sequences = {}
    for filename in training_filenames:
        newsequence = read_fasta_file(filename=filename)
        sequences = {**sequences, **newsequence}
    return learn_hmm(training_filenames, training_pairs)


#init_probs, transition_probs, emit_probs = create_model_params()
#print(emit_probs)
#print(init_probs)
#print(transition_probs)


def draw_states():
    dot = Digraph(comment='The Round Table')
    for state in states.keys():
        label = state + "\n"
        for observable in observables.keys():
            oberservable_prob = emit_probs[states[state]][observables[observable]]
            if oberservable_prob > 0.0001:
                label = label + observable + ":" + "{0:.4f}".format(oberservable_prob) + "\n"
        dot.node(name=str(states[state]), label=label)
    for from_state in states.keys():
        for to_state in states.keys():
            from_index = states[from_state]
            to_index = states[to_state]
            transition_prob = transition_probs[from_index][to_index]
            if transition_prob > 0.0001:
                dot.edge(head_name=str(from_index), tail_name=str(to_index), label="{0:.4f}".format(transition_prob),
                         constraint='false')
    dot.render('model.gv', view=True)


emit_probs = [[0.3343431467742144, 0.1647945258526105, 0.1661272788031685, 0.33473504857000663], [0.32120539545052706, 0.15902748633903424, 0.32264828763814196, 0.19711883057229676], [0.3525726169204167, 0.20005148550834367, 0.13622459795875103, 0.3111512996124886], [0.3392152281731009, 0.12997639165905922, 0.13122386966234448, 0.39958451050549537], [0.3994628035006402, 0.13240714425930755, 0.12861146351253555, 0.3395185887275167], [0.31271296684236277, 0.13711402277701437, 0.1975614472516107, 0.35261156312901215], [0.19904836965916514, 0.3182585281958577, 0.1605063153916246, 0.32218678675335255]]
init_probs = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
transition_probs = [[0.9966607612374366, 0.0017035029916180194, 0.0, 0.0, 0.00163573577094536, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0034244352818507094, 0.9965734014614962, 0.0, 0.0, 2.1632566530958366e-06, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [0.0033430117576741312, 7.324740923913521e-07, 0.0, 0.0, 0.9966562557682335, 0.0, 0.0]]
draw_states()