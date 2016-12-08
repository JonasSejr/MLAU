import unittest
from viterbi_prediction import ViterbiPredictor
from AnnotationConverter import AnnotationConverter
from annotation_converter_with_codon import AnnotationConverterWithCodons

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


class TestConversion(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestConversion, self).__init__(*args, **kwargs)
        self.converter = AnnotationConverter()
        self.converter_with_codon = AnnotationConverterWithCodons()

    def test_convert_to_states_simple(self):
        annotations = ["N", "C", "C", "C", "N"]
        observations = ["A", "C", "G", "T", "A"]
        expected_states = ["S1", "S2","S3","S4","S1"]
        expected_all_states = {"S1":0, "S2":1, "S3":2,"S4":3}
        expected_observables = {"A": 0, "C": 1, "G": 2, "T": 3}
        expected_state_to_annotation = {"S1": "N", "S2": "C", "S3": "C", "S4": "C"}
        expected_observations = ["A", "C", "G", "T", "A"]
        all_states, observables, state_to_annotation, states, observations = self.converter.convert_to_states({}, {}, {}, annotations, observations)
        self.assertEqual(expected_observations, observations)
        self.assertEqual(expected_states, states)
        self.assertEqual(expected_observables, observables)
        self.assertEqual(expected_all_states, all_states)
        self.assertEqual(expected_state_to_annotation, state_to_annotation)

    def test_convert_to_states_simple(self):
        annotations = ["N", "C", "C", "C", "N"]
        observations = ["A", "C", "G", "T", "A"]
        expected_states = ["S1", "CS_CGT_1","CS_CGT_2","CS_CGT_3","S1"]
        expected_all_states = {"S1":0, "CS_CGT_1":1, "CS_CGT_2":2,"CS_CGT_3":3}
        expected_observables = {"A": 0, "C": 1, "G": 2, "T": 3}
        expected_state_to_annotation = {"S1": "N", "CS_CGT_1": "C", "CS_CGT_2": "C", "CS_CGT_3": "C"}
        expected_observations = ["A", "C", "G", "T", "A"]
        all_states, observables, state_to_annotation, states, observations = self.converter_with_codon.convert_to_states({}, {}, {}, annotations, observations)
        self.assertEqual(expected_observations, observations)
        self.assertEqual(expected_states, states)
        self.assertEqual(expected_observables, observables)
        self.assertEqual(expected_all_states, all_states)
        self.assertEqual(expected_state_to_annotation, state_to_annotation)

    def test_convert_to_annotation_simple(self):
        states = ["S1", "S2", "S3", "S4", "S1"]
        state_to_annotation = {"S1": "N", "S2": "C", "S3": "C", "S4": "C"}
        expected_annotations = ["N", "C", "C", "C", "N"]
        annotations = self.converter.convert_to_annotations(state_to_annotation, states)
        self.assertEqual(expected_annotations, annotations)