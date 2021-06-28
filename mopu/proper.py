import numpy as np
from tunimi.proper import ProperGenerator

class ProperTransformer:
    def __init__(self):
        self.generator = ProperGenerator()

    def __call__(self, sent, name_list = None):
        indices = [i for i, x in enumerate(sent) if x == '<proper>']
        proper_list = [self.generator() for _ in indices]

        if name_list is not None and len(name_list) > 0:
            for i, _ in enumerate(proper_list):
                if np.random.rand() < 0.5:
                    proper_list[i] = np.random.choice(name_list)

        for index in indices:
            n = np.random.binomial(len(indices) - 1, 0.5)
            sent[index] = proper_list[n]
        return sent

