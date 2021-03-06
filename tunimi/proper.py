import numpy as np

output0 = [(1, 'k'), (1, 'l'), (1, 'm'), (1, 'n'), (1, 'p'), (1, 's'), (2, 'j'), (2, 't'), (3, 'w'), (4, 'a'), (4, 'e'), (4, 'i'), (4, 'o'), (4, 'u')]
probs0 = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.02, 0.02, 0.02, 0.02, 0.02]

output1 = [(4, 'a'), (4, 'e'), (4, 'i'), (4, 'o'), (4, 'u')]
output2 = [(4, 'a'), (4, 'e'), (4, 'o'), (4, 'u')]
output3 = [(4, 'a'), (4, 'e'), (4, 'i')]

output4 = [(1, 'k'), (1, 'l'), (1, 'm'), (1, 'p'), (1, 's'), (2, 'j'), (2, 't'), (3, 'w'), (5, 'n')]

output5 = [(1, 'k'), (1, 'l'), (1, 'p'), (1, 's'), (2, 'j'), (2, 't'), (3, 'w'), (4, 'a'), (4, 'e'), (4, 'i'), (4, 'o'), (4, 'u')]

class ProperGenerator:
    def change(self):
        if self.state == 0:
            output, probs = output0, probs0
        elif self.state == 1:
            output, probs = output1, None
        elif self.state == 2:
            output, probs = output2, None
        elif self.state == 3:
            output, probs = output3, None
        elif self.state == 4:
            output, probs = output4, None
        elif self.state == 5:
            output, probs = output5, None
        index = np.random.choice(range(len(output)), p = probs)
        self.state, char = output[index] 
        return char

    def cond(self, lst):
        if self.state not in {4, 5}:
            return True

        return len(lst) < np.random.poisson(5)

    def __call__(self):
        self.state = 0
        lst = []
        while self.cond(lst):
            char = self.change()
            lst.append(char)
        return ''.join(lst).title()

