from graphviz import Digraph

#Test data
#emit_probs = [[0.3343431467742144, 0.1647945258526105, 0.1661272788031685, 0.33473504857000663], [0.32120539545052706, 0.15902748633903424, 0.32264828763814196, 0.19711883057229676], [0.3525726169204167, 0.20005148550834367, 0.13622459795875103, 0.3111512996124886], [0.3392152281731009, 0.12997639165905922, 0.13122386966234448, 0.39958451050549537], [0.3994628035006402, 0.13240714425930755, 0.12861146351253555, 0.3395185887275167], [0.31271296684236277, 0.13711402277701437, 0.1975614472516107, 0.35261156312901215], [0.19904836965916514, 0.3182585281958577, 0.1605063153916246, 0.32218678675335255]]
#init_probs = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
#transition_probs = [[0.9966607612374366, 0.0017035029916180194, 0.0, 0.0, 0.00163573577094536, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0034244352818507094, 0.9965734014614962, 0.0, 0.0, 2.1632566530958366e-06, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [0.0033430117576741312, 7.324740923913521e-07, 0.0, 0.0, 0.9966562557682335, 0.0, 0.0]]
#draw_states()


def draw_states(states, observables, emit_probs, transition_probs):
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
                dot.edge(tail_name=str(from_index), head_name=str(to_index), label="{0:.4f}".format(transition_prob),
                         constraint='false')
    print(dot)
    dot.render('model.gv', view=True, )
