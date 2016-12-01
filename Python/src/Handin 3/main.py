from graphviz import Digraph
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


sequences = {}
for filename in training_filenames:
    newsequence = read_fasta_file(filename=filename)
    sequences = {**sequences, **newsequence}

#cross validation
for i in range(len(training_pairs)):
    test_pair = training_pairs[i]
    cross_training_pairs = training_pairs[:i] + training_pairs[i + 1:]

    init_probs, transition_probs, emit_probs = learn_hmm(training_filenames_test, cross_training_pairs)
    predictor = ViterbiPredictor(
        observables=observables,
        states=states,
        emit_probs=emit_probs,
        init_probs=init_probs,
        trans_probs=transition_probs)

    test_gnome = sequences[test_pair[0]]
    test_annotation = sequences[test_pair[1]]
    probOfHidden, hidden_states = predictor.logspace_viterbi_backtrack(test_gnome)

    #print(emit_probs)
    #print(init_probs)
    #print(transition_probs)
    annotations_list = convert_to_annotations(hidden_states)
    print_all(test_annotation, ''.join(annotations_list))
