from pathlib import Path

def load_tokipona_vocabulary():
    path = Path(__file__).parent / 'vocabulary.txt'
    with open(path) as f:
        lst = [x.strip() for x in f]
    return lst

class Vocabulary(list):
    def __init__(self):
        self.token_list = self.init_special_tokens()
        self.token_list += self.init_replaced_tokens()
        self.token_list += self.init_punctuation_tokens()
        self.token_list += load_tokipona_vocabulary()
        super().__init__(self.token_list)
        self.indices = {token: index for index, token in enumerate(self)}

    def init_special_tokens(self):
        self.pad, self.pad_id = '<pad>', 0
        self.eos, self.eos_id = '<eos>', 1
        self.unk, self.unk_id = '<unk>', 2
        return [self.pad, self.eos, self.unk]

    def init_replaced_tokens(self):
        index = len(self.token_list)
        self.number, self.number_id = '<number>', index
        self.proper, self.proper_id = '<proper>', index + 1
        return [self.number, self.proper]

    def init_punctuation_tokens(self):
        self.punctuation_list = list('!",.:?')
        self.punctuation_set = set(self.punctuation_list)
        return self.punctuation_list

