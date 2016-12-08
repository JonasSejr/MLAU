class AnnotationConverterWithCodons:
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
            last_state = "S1" if i == 0 else states[i-1]

            state_trans_map = {
                "S": {"N": "S1", "C": "S2", "R": "S5"},
                "EC": {"C": "S3"},
                "G2": {"C": "S3"},
                "G3": {"C": "S4"},
                "G4": {"N": "S1", "C": "S2", "R": "S5"},
                "G5": {"R": "S6"},
                "G6": {"R": "S7"},
                "EC"
                "E": {"N": "S1", "C": "S2", "R": "S5"},
            }
            this_state = state_trans_map[last_state][annotation]

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
            annotation_data = sequences[training_pair[1]]
            observation_data = sequences[training_pair[0]]
            all_states, observservables, state_to_annotation, converted_annotations, converted_observation = \
                self.convert_to_states(all_states, observservables, state_to_annotation, annotation_data, observation_data)
            sequences[training_pair[1]] = converted_annotations
            sequences[training_pair[0]] = converted_observation
        return observservables, all_states, state_to_annotation, sequences

    def find_start_codons(sequences, training_pairs):
        global start_codon_map

        for pair in training_pairs:
            annotations = sequences[pair[1]]
            observations = sequences[pair[0]]
            start_codon_map = {}
            for i in range(len(annotations)):
                if (annotations[i] == 'N' or annotations[i] == 'R') and (i == 0 or annotations[i - 1] == 'C'):
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
