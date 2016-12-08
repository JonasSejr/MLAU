class AnnotationConverter:
    def convert_to_annotations(self, state_to_annotations, hidden_list):
        annotations_list = [0 for x in range(len(hidden_list))]
        for i in range(len(hidden_list)):
            state = hidden_list[i]
            annotations_list[i] = state_to_annotations[state]
        return annotations_list


    def convert_to_states(self, all_states, observservables, state_to_annotation, annotations, observations):
        states = [0 for x in range(len(annotations))]
        for i in range(len(annotations)):
            annotation = annotations[i]
            observation = observations[i]
            if(observation not in observservables.keys()):
                observservables[observation] = len(observservables.keys())
            last_state = "ST" if i == 0 else states[i-1]
            #state_trans_map = {
            #    "S1": {"N": "S1", "C": "S2", "R": "S5"},
            #    "S2": {"C": "S3"},
            #    "S3": {"C": "S4"},
            #    "S4": {"N": "S1", "C": "S2", "R": "S5"},
            #    "S5": {"R": "S6"},
            #    "S6": {"R": "S7"},
            #    "S7": {"N": "S1", "C": "S2", "R": "S5"},
            #}
            #{'ATT': 3, 'ATC': 3, 'GTT': 1, }
            state_trans_map = {
                "ST": {
                    "N": "ST",
                    "C": {
                        "ATG":"CS11",
                        "TTG":"CS21",
                        "GTG":"CS31"
                     },
                    "R": {
                        "TTA":"RS11",
                        "TCA":"RS21",
                        "CTA":"RS31"
                    }},
                "CS11":{"C":"CS12"},
                "CS12": {"C": "CS13"},
                "CS13": {"C": "C1"},

                "CS21": {"C": "CS22"},
                "CS22": {"C": "CS23"},
                "CS23": {"C": "C1"},

                "CS31": {"C": "CS32"},
                "CS32": {"C": "CS33"},
                "CS33": {"C": "C1"},

                "RS11": {"R": "RS12"},
                "RS12": {"R": "RS13"},
                "RS13": {"R": "R1"},

                "RS21": {"R": "RS22"},
                "RS22": {"R": "RS23"},
                "RS23": {"R": "R1"},

                "RS31": {"R": "RS32"},
                "RS32": {"R": "RS33"},
                "RS33": {"R": "R1"},

                "C1": {"C": "C2"},
                "C2": {"C": "C3"},
                "C3": {"N": "ST", "C": "C1",
                       "R": {
                           "TTA": "RS11",
                           "TCA": "RS21",
                           "CTA": "RS31"
                       }
                       },

                "R1": {"R": "R2"},
                "R2": {"R": "R3"},
                "R3": {"N": "ST", "R": "R1",
                       "C": {
                           "ATG": "CS11",
                           "TTG": "CS21",
                           "GTG": "CS31"
                       }
                       },
            }
            if isinstance(state_trans_map[last_state][annotation],str) :
                this_state = state_trans_map[last_state][annotation]
            else:
                codon = observations[i] + observations[i + 1] + observations[i + 2]
                if codon in state_trans_map[last_state][annotation].keys():
                    this_state = state_trans_map[last_state][annotation][codon]
                elif annotation == "R":
                    this_state = "R1"
                elif annotation == "C":
                    this_state = "C1"

            if (this_state not in state_to_annotation.keys()):
                state_to_annotation[this_state] = annotation
            if(this_state not in all_states):
                all_states[this_state] = len(all_states.keys())
            states[i] = this_state
        return all_states, observservables, state_to_annotation, states, observations

    def transform_all_annotations_and_observations(self, sequences, training_pairs):
        all_states = {}
        observservables = {}
        state_to_annotation = {}
        for training_pair in training_pairs:
            print(training_pair)
            annotation_data = sequences[training_pair[1]]
            observation_data = sequences[training_pair[0]]
            all_states, observservables, state_to_annotation, converted_annotations, converted_observation = \
                self.convert_to_states(all_states, observservables, state_to_annotation, annotation_data, observation_data)
            sequences[training_pair[1]] = converted_annotations
            sequences[training_pair[0]] = converted_observation
        return observservables, all_states, state_to_annotation, sequences

    def find_start_codons(self, sequences, training_pairs):
        start_codon_map = {}
        for pair in training_pairs:
            annotations = sequences[pair[1]]
            observations = sequences[pair[0]]
            for i in range(len(annotations)):
                if (annotations[i] == 'C') and (i == 0 or annotations[i - 1] == 'N'):
                    codon = observations[i]
                    if i + 1 < len(annotations):
                        codon = codon + observations[i + 1]
                    if i + 2 < len(annotations):
                        codon = codon + observations[i + 2]
                    if (codon not in start_codon_map.keys()):
                        start_codon_map[codon] = 1
                    else:
                        start_codon_map[codon] += 1
        return start_codon_map
