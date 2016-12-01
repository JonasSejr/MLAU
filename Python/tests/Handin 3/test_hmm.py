import unittest

import faste_file_reader as fr
from viterbi_prediction import ViterbiPredictor
from hmmlearner import convert_to_states
from hmmlearner import learn_hmm
from hmmlearner import convert_to_annotations


class ViterbiTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(ViterbiTest, self).__init__(*args, **kwargs)
        self.observables = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        self.states = {'1': 0, '2': 1, '3': 2, '4': 3, '5': 4, '6': 5, '7': 6}

        self.init_probs = [0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.00]
        self.trans_probs = [[0.00, 0.00, 0.90, 0.10, 0.00, 0.00, 0.00],
                      [1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
                      [0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00],
                      [0.00, 0.00, 0.05, 0.90, 0.05, 0.00, 0.00],
                      [0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00],
                      [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00],
                      [0.00, 0.00, 0.00, 0.10, 0.90, 0.00, 0.00]]
        self.emit_probs = [[0.30, 0.25, 0.25, 0.20],
                     [0.20, 0.35, 0.15, 0.30],
                     [0.40, 0.15, 0.20, 0.25],
                     [0.25, 0.25, 0.25, 0.25],
                     [0.20, 0.40, 0.30, 0.10],
                     [0.30, 0.20, 0.30, 0.20],
                     [0.15, 0.30, 0.20, 0.35]]

        self.predictor = ViterbiPredictor(self.observables, self.states, self.init_probs, self.trans_probs, self.emit_probs)

    def test_viterbi_small(self):
        obsAsString = "GTTTCCCAGTGTATATCGAGGGATACTACGTGCATAGTAACATCGGCCAA"
        expectedVitAsString = "44444444444432132132132132132132132132132132132132"
        obs = list(obsAsString)
        expectedLogProb = -70.73228857440488
        prob, hidden = self.predictor.logspace_viterbi_backtrack(obs)
        self.assertEqual(expectedLogProb, prob, 0.001)
        self.assertEqual(expectedVitAsString, hidden)

    def test_viterbi_huge(self):
        obsAsString = "TGAGTATCACTTAGGTCTATGTCTAGTCGTCTTTCGTAATGTTTGGTCTTGTCACCAGTTATCCTATGGCGCTCCGAGTCTGGTTCTCGAAATAAGCATCCCCGCCCAAGTCATGCACCCGTTTGTGTTCTTCGCCGACTTGAGCGACTTAATGAGGATGCCACTCGTCACCATCTTGAACATGCCACCAACGAGGTTGCCGCCGTCCATTATAACTACAACCTAGACAATTTTCGCTTTAGGTCCATTCACTAGGCCGAAATCCGCTGGAGTAAGCACAAAGCTCGTATAGGCAAAACCGACTCCATGAGTCTGCCTCCCGACCATTCCCATCAAAATACGCTATCAATACTAAAAAAATGACGGTTCAGCCTCACCCGGATGCTCGAGACAGCACACGGACATGATAGCGAACGTGACCAGTGTAGTGGCCCAGGGGAACCGCCGCGCCATTTTGTTCATGGCCCCGCTGCCGAATATTTCGATCCCAGCTAGAGTAATGACCTGTAGCTTAAACCCACTTTTGGCCCAAACTAGAGCAACAATCGGAATGGCTGAAGTGAATGCCGGCATGCCCTCAGCTCTAAGCGCCTCGATCGCAGTAATGACCGTCTTAACATTAGCTCTCAACGCTATGCAGTGGCTTTGGTGTCGCTTACTACCAGTTCCGAACGTCTCGGGGGTCTTGATGCAGCGCACCACGATGCCAAGCCACGCTGAATCGGGCAGCCAGCAGGATCGTTACAGTCGAGCCCACGGCAATGCGAGCCGTCACGTTGCCGAATATGCACTGCGGGACTACGGACGCAGGGCCGCCAACCATCTGGTTGACGATAGCCAAACACGGTCCAGAGGTGCCCCATCTCGGTTATTTGGATCGTAATTTTTGTGAAGAACACTGCAAACGCAAGTGGCTTTCCAGACTTTACGACTATGTGCCATCATTTAAGGCTACGACCCGGCTTTTAAGACCCCCACCACTAAATAGAGGTACATCTGA"
        expectedVitAsString = "4444432132132132132132132132132132132132132132132132132132132132132132144444444445675675675675675675675675675675675675675675675675675675675675675675675675675675675675675675675675675675675675675675675675675675674321321321321321321321321321321321321321321321321321321321321321321321321321321321321321321321321321321321321321321321321321321321432132132132132132132132144445675675675675675675675675675675675675675675675675675675675675675675675675675675675675675675675675675675675675675675675675675675675675675675675675675675675675675675675675675675675675675675675675675675675675675675675675675675675675675675675675675675675675675675675675675675675675675675675675675675675675675675675675675675675675675675675675675675675674444444567567567567567567567567567567567567567567567567567567567567567567567567567567567567567567567567567567567567567567443213213213213213213213213213213213213213213213213213213213213213213213213213213213213213213213214321321321321321321321321321321321321321321321321321321321321321"
        obs = list(obsAsString)
        expectedLogProb = -1406.7209253880144
        prob, hidden = self.predictor.logspace_viterbi_backtrack(obs)
        self.assertEqual(expectedLogProb, prob, 0.001)
        self.assertEqual(expectedVitAsString, hidden)


class TrainingTest(unittest.TestCase):
    def test_convert_to_states_simple(self):
        annotation = ["N", "C", "C", "C", "N"]
        expected_states = ["S1", "S2","S3","S4","S1"]
        self.assertEqual(expected_states, convert_to_states(annotation))


    def test_convert_to_states_simple_reverse(self):
        annotation = ["N", "R", "R", "R", "N"]
        expected_states = ["S1", "S5", "S6", "S7", "S1"]
        self.assertEqual(expected_states, convert_to_states(annotation))


    def test_convert_to_states_bulk(self):
        annotation = ["N", "C", "C", "C", "N", "R", "R", "R", "N"]
        expected_states = ["S1", "S2", "S3", "S4", "S1", "S5", "S6", "S7", "S1"]
        self.assertEqual(expected_states, convert_to_states(annotation))

class EndToEndTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(EndToEndTest, self).__init__(*args, **kwargs)
        self.observables = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        self.states = {'S1': 0, 'S2': 1, 'S3': 2, 'S4': 3, 'S5': 4, 'S6': 5, 'S7': 6}

    def test_complete_prediction(self):
        training_filenames = ["./data/test_genome1.fa", "./data/test_annotation1.fa"]
        training_pairs = [["genome1", "annotation1"]]
        init_probs, transition_probs, emit_probs= learn_hmm(training_filenames, training_pairs)

        predictor = ViterbiPredictor(self.observables, self.states, init_probs, transition_probs, emit_probs)

        test_gnome = fr.read_fasta_file(filename="./data/test_genome2.fa")["genome2"]
        test_annotation = fr.read_fasta_file(filename="./data/test_annotation2.fa")["annotation2"]

        prob, hidden = predictor.logspace_viterbi_backtrack(test_gnome)
        annotations = convert_to_annotations(hidden)

        print(annotations)
        print(test_annotation)
        print(prob)
