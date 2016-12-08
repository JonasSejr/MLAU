from hmmlearner import HmmLearner
from AnnotationConverter import AnnotationConverter
from viterbi_prediction import ViterbiPredictor
from compare_anns import print_all
from faste_file_reader import read_fasta_file
from faste_file_reader import write_fasta
from visualiser import draw_states


training_filenames_test = ["./data/test_genome1.fa", "./data/test_genome2.fa", "./data/test_genome3.fa", "./data/test_genome4.fa",
             "./data/test_genome5.fa",
             "./data/test_annotation1.fa", "./data/test_annotation2.fa", "./data/test_annotation3.fa", "./data/test_annotation4.fa",
             "./data/test_annotation5.fa"]

training_filenames = ["./data/genome1.fa", "./data/genome2.fa", "./data/genome3.fa", "./data/genome4.fa",
             "./data/genome5.fa",
             "./data/annotation1.fa", "./data/annotation2.fa", "./data/annotation3.fa", "./data/annotation4.fa",
             "./data/annotation5.fa"]

test_filenames = ["./data/genome6.fa", "./data/genome7.fa", "./data/genome8.fa", "./data/genome9.fa",
             "./data/genome10.fa"]

training_pairs = [
    ["genome1", "annotation1"],
    ["genome2", "annotation2"],
    ["genome3", "annotation3"],
    ["genome4", "annotation4"],
    ["genome5", "annotation5"]]

test_pairs = [
    ["genome6", "annotation6"],
    ["genome7", "annotation7"],
    ["genome8", "annotation8"],
    ["genome9", "annotation9"],
    ["genome10", "annotatio10"]]

def learn_evaluate_print(sequences, training_pairs, test_pair):

    converter = AnnotationConverter()
    observables, states, state_to_annotation, sequences = converter.transform_all_annotations_and_observations(sequences, training_pairs)

    learner = HmmLearner(observables, states)
    init_probs = learner.calculate_init_probs(sequences, training_pairs)
    emit_probs = learner.calculate_emit_probs(sequences, training_pairs)
    transition_probs = learner.calculate_transition_probs(sequences, training_pairs)

    predictor = ViterbiPredictor(
        observables=observables,
        states=states,
        emit_probs=emit_probs,
        init_probs=init_probs,
        trans_probs=transition_probs)

    test_gnome = sequences[test_pair[0]]
    test_annotation = sequences[test_pair[1]]
    probOfHidden, hidden_states = predictor.logspace_viterbi_backtrack(test_gnome)
    annotations_list = converter.convert_to_annotations(state_to_annotation, hidden_states)
    print_all(test_annotation, ''.join(annotations_list))


def run_cross_validation(training_pairs):
    for i in range(len(training_pairs)):
        # Bad section. Should not use global vars. Reason is the sequence is manipulated and cannot be reused.
        sequences = {}
        for filename in training_filenames:
            newsequence = read_fasta_file(filename=filename)
            sequences = {**sequences, **newsequence}

        cross_test_pair = training_pairs[i]
        cross_training_pairs = training_pairs[:i] + training_pairs[i + 1:]
        learn_evaluate_print(sequences, cross_training_pairs, cross_test_pair)



converter = AnnotationConverter()

def train_and_draw():
    sequences = {}
    for filename in training_filenames:
        newsequence = read_fasta_file(filename=filename)
        sequences = {**sequences, **newsequence}

    observables, states, state_to_annotation, sequences = converter.transform_all_annotations_and_observations(sequences, training_pairs)
    learner = HmmLearner(observables, states)
    init_probs = learner.calculate_init_probs(sequences, training_pairs)
    emit_probs = learner.calculate_emit_probs(sequences, training_pairs)
    transition_probs = learner.calculate_transition_probs(sequences, training_pairs)

    draw_states(states, observables, emit_probs, transition_probs)


#run_cross_validation(training_pairs)
#train_and_draw()

def train_predics_and_write_fasta():
    traning_sequences = {}
    for filename in training_filenames:
        newsequence = read_fasta_file(filename=filename)
        traning_sequences = {**traning_sequences, **newsequence}

    test_sequences = {}
    for filename in test_filenames:
        newsequence = read_fasta_file(filename=filename)
        test_sequences = {**test_sequences, **newsequence}

    observables, states, state_to_annotation, sequences = converter.transform_all_annotations_and_observations(
        traning_sequences, training_pairs)

    learner = HmmLearner(observables, states)
    init_probs = learner.calculate_init_probs(traning_sequences, training_pairs)
    emit_probs = learner.calculate_emit_probs(traning_sequences, training_pairs)
    transition_probs = learner.calculate_transition_probs(traning_sequences, training_pairs)
    predictor = ViterbiPredictor(
        observables=observables,
        states=states,
        emit_probs=emit_probs,
        init_probs=init_probs,
        trans_probs=transition_probs)

    for test_pair in test_pairs:
        test_gnome = test_sequences[test_pair[0]]
        test_annotation_name = test_pair[1]
        probOfHidden, hidden_states = predictor.logspace_viterbi_backtrack(test_gnome)
        annotations_list = converter.convert_to_annotations(state_to_annotation, hidden_states)
        write_fasta(filename=test_annotation_name + ".fa", name=test_annotation_name, sequence=''.join(annotations_list))

train_predics_and_write_fasta()

#write_fasta("testfile.fa", "Jonas test", "NCNNNNNNNCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN")
#print(read_fasta_file("testfile.fa"))
#sequences = {}
#for filename in training_filenames:
#    newsequence = read_fasta_file(filename=filename)
#    sequences = {**sequences, **newsequence}
#codons = converter.find_start_codons(sequences, training_pairs)
#print(codons)

